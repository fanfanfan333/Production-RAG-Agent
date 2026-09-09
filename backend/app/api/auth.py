"""
Authentication API router (企业落地第一阶段).

POST /auth/register — create an account (first user becomes admin)
POST /auth/login    — obtain a JWT bearer token
GET  /auth/me       — current user profile

Login is rate-limited per client IP to slow brute-force attempts, and both
success and failure are recorded in the audit log.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.api.deps import client_ip, enforce_rate_limit, get_current_user
from app.config import get_settings
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.services.auth_service import (
    AuthError,
    USERNAME_ALPHABET,
    authenticate_user,
    change_password,
    create_access_token,
    register_user,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


# ── Schemas（字段校验报错全部为中文）───────────────────────────────────────────
# 注意：长度等规则放在 field_validator 里（Field 的 min_length 会在校验器之前
# 生效并返回英文报错），Field 只保留宽松的防御性上限。


def _validate_password_cn(v: str, label: str) -> str:
    if len(v) < 8:
        raise ValueError(f"{label}至少需要 8 个字符")
    if len(v) > 128:
        raise ValueError(f"{label}最多 128 个字符")
    return v


class RegisterRequest(BaseModel):
    username: str = Field(..., max_length=512)
    password: str = Field(..., max_length=512)

    @field_validator("username")
    @classmethod
    def username_rules(cls, v: str) -> str:
        v = v.strip()
        if not (3 <= len(v) <= 64):
            raise ValueError("用户名长度需在 3–64 个字符之间")
        if not set(v) <= USERNAME_ALPHABET:
            raise ValueError("用户名只能包含字母、数字、点、下划线和连字符")
        return v

    @field_validator("password")
    @classmethod
    def password_rules(cls, v: str) -> str:
        return _validate_password_cn(v, "密码")


class LoginRequest(BaseModel):
    username: str = Field(..., max_length=512)
    password: str = Field(..., max_length=512)


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., max_length=512)
    new_password: str = Field(..., max_length=512)

    @field_validator("new_password")
    @classmethod
    def new_password_rules(cls, v: str) -> str:
        return _validate_password_cn(v, "新密码")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserProfile"


class UserProfile(BaseModel):
    id: str
    username: str
    role: str

    model_config = {"from_attributes": True}


def _profile(user: User) -> UserProfile:
    return UserProfile(id=str(user.id), username=user.username, role=user.role)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=201,
    summary="Create an account",
    description=(
        "The **first** registered account automatically becomes the admin. "
        "After that, registration may be disabled via ALLOW_SELF_REGISTRATION."
    ),
)
async def register_endpoint(
    body: RegisterRequest,
    request: Request,
) -> TokenResponse:
    settings = get_settings()
    await enforce_rate_limit(
        request,
        scope="register",
        key=client_ip(request),
        limit=settings.LOGIN_RATE_LIMIT,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    )

    try:
        user = await register_user(body.username, body.password)
    except AuthError as exc:
        await record_audit(
            "auth.register.failed",
            username=body.username,
            detail=str(exc),
            ip=client_ip(request),
        )
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None

    await record_audit(
        "auth.register",
        user_id=user.id,
        username=user.username,
        resource_type="user",
        resource_id=str(user.id),
        ip=client_ip(request),
    )
    return TokenResponse(access_token=create_access_token(user), user=_profile(user))


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and obtain a JWT",
)
async def login_endpoint(
    body: LoginRequest,
    request: Request,
) -> TokenResponse:
    settings = get_settings()
    ip = client_ip(request)
    await enforce_rate_limit(
        request,
        scope="login",
        key=ip,
        limit=settings.LOGIN_RATE_LIMIT,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    )

    try:
        user = await authenticate_user(body.username, body.password)
    except AuthError as exc:
        await record_audit(
            "auth.login.failed",
            username=body.username,
            detail=str(exc),
            ip=ip,
        )
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None

    await record_audit(
        "auth.login",
        user_id=user.id,
        username=user.username,
        ip=ip,
    )
    return TokenResponse(access_token=create_access_token(user), user=_profile(user))


@router.get(
    "/me",
    response_model=UserProfile,
    summary="Current user profile",
)
async def me_endpoint(user: User = Depends(get_current_user)) -> UserProfile:
    return _profile(user)


@router.post(
    "/change-password",
    summary="修改密码",
    description=(
        "验证当前密码后设置新密码。所有校验失败均返回中文错误信息。"
        "修改成功后当前会话保持有效。"
    ),
)
async def change_password_endpoint(
    body: ChangePasswordRequest,
    request: Request,
    user: User = Depends(get_current_user),
) -> dict:
    settings = get_settings()
    # 复用登录限流阈值，防止暴力尝试旧密码
    await enforce_rate_limit(
        request,
        scope="change-password",
        key=str(user.id),
        limit=settings.LOGIN_RATE_LIMIT,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    )

    try:
        await change_password(user.id, body.old_password, body.new_password)
    except AuthError as exc:
        await record_audit(
            "auth.change_password.failed",
            user_id=user.id,
            username=user.username,
            detail=str(exc),
            ip=client_ip(request),
        )
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None

    await record_audit(
        "auth.change_password",
        user_id=user.id,
        username=user.username,
        ip=client_ip(request),
    )
    return {"changed": True, "message": "密码修改成功"}
