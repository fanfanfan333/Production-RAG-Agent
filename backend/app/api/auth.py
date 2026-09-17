"""
Authentication API router (企业落地第一阶段).

POST /auth/register — create an account (first user becomes admin)
POST /auth/login    — obtain a JWT bearer token
GET  /auth/me       — current user profile

Login is rate-limited per client IP to slow brute-force attempts, and both
success and failure are recorded in the audit log.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from app.api.deps import client_ip, enforce_rate_limit, get_current_user
from app.config import get_settings
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.services.auth_service import (
    AuthError,
    authenticate_unified,
    authenticate_user,
    change_password,
    create_access_token,
    normalize_username,
    register_user,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Auth"])


# ── 身份提供方配置（公开）─────────────────────────────────────────────────────
# 前端登录页在渲染前需要知道"要不要显示企业统一身份登录入口、指向哪个 realm"。
# 这里只暴露公开信息（URL / realm / client_id），没有任何密钥。

@router.get(
    "/config",
    summary="身份认证配置（公开）",
    description="返回 Keycloak/OIDC 的公开配置与是否允许本地账号登录，供登录页渲染。",
)
async def auth_config_endpoint() -> dict:
    from app.services.keycloak_auth import public_config

    return public_config()


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
        # 只做「归一化 + 上限」这类无争议的预处理，邮箱格式由
        # auth_service._validate_registration 判定 —— 那里能返回一句
        # 干净的中文提示，而 Pydantic 抛错会变成 422 的字段错误数组。
        v = normalize_username(v)
        if not v:
            raise ValueError("请输入邮箱地址")
        if len(v) > 254:
            raise ValueError("邮箱地址过长（最多 254 个字符）")
        return v

    @field_validator("password")
    @classmethod
    def password_rules(cls, v: str) -> str:
        return _validate_password_cn(v, "密码")


class LoginRequest(BaseModel):
    username: str = Field(..., max_length=512)
    password: str = Field(..., max_length=512)


class UnifiedLoginRequest(BaseModel):
    """企业统一身份登录：企业名称（= 邮箱）、企业职责、密码。"""

    username: str = Field(..., max_length=512)
    duty: str = Field(..., max_length=128)
    password: str = Field(..., max_length=512)

    @field_validator("username")
    @classmethod
    def username_rules(cls, v: str) -> str:
        return normalize_username(v)

    @field_validator("duty")
    @classmethod
    def duty_rules(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("请填写企业职责")
        return v


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
    # ── 身份信息（前端"我属于哪个公司/部门、什么角色"直接读这里）──────────────
    display_name: str | None = None
    role_label: str = ""
    tenant_id: str = "default"
    department_id: str | None = None
    auth_source: str = "local"
    permissions: list[str] = Field(default_factory=list)
    # ── 企业身份展示字段（个人主页与首页拦截据此渲染）────────────────────────
    # 公司/部门都下发**可读原文**：内部 ID 是安全映射（可能是 c1a2b3…），
    # 不该出现在用户界面上。
    company_name: str = "default"
    department_name: str | None = None
    job_title: str | None = None
    # none（去验证）| pending（审核中）| approved（已通过）| rejected（未通过）
    identity_status: str = "approved"

    model_config = {"from_attributes": True}


async def _profile(user: User) -> UserProfile:
    from app.services.permissions import permissions_of, role_label
    from app.services.staff_service import identity_status
    from app.services.tenancy import (
        company_display_name,
        department_display_name,
    )

    return UserProfile(
        id=str(user.id),
        username=user.username,
        role=user.role,
        display_name=user.display_name,
        role_label=role_label(user.role),
        tenant_id=user.tenant_id or "default",
        department_id=user.department_id,
        auth_source=user.auth_source or "local",
        permissions=permissions_of(user),
        company_name=company_display_name(user),
        department_name=department_display_name(user),
        job_title=user.job_title,
        identity_status=await identity_status(user),
    )


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
    return TokenResponse(
        access_token=create_access_token(user), user=await _profile(user)
    )


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

    if not settings.ALLOW_LOCAL_LOGIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="本地账号登录已关闭，请使用企业统一身份（Keycloak）登录",
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
    return TokenResponse(
        access_token=create_access_token(user), user=await _profile(user)
    )


@router.post(
    "/unified-login",
    response_model=TokenResponse,
    summary="企业统一身份登录（企业名称 + 企业职责 + 密码）",
    description=(
        "登录页「使用企业统一身份登录」入口。用**已有账号**的邮箱、"
        "**已登记的部门职责**与密码登入，由本系统账号体系校验（不跳转 Keycloak）。\n\n"
        "- 企业名称 = 账号邮箱（大小写不敏感）\n"
        "- 企业职责必须与账号在身份验证流程中登记的职责一致\n"
        "- 尚未登记职责的账号请先完成身份验证，或改用邮箱密码登录"
    ),
)
async def unified_login_endpoint(
    body: UnifiedLoginRequest,
    request: Request,
) -> TokenResponse:
    settings = get_settings()
    ip = client_ip(request)
    await enforce_rate_limit(
        request,
        scope="unified_login",
        key=ip,
        limit=settings.LOGIN_RATE_LIMIT,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    )

    # 这条链路同样是对本地密码哈希做校验，因此与本地登录同开关。
    if not settings.ALLOW_LOCAL_LOGIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号登录已关闭，请使用企业统一身份（Keycloak）登录",
        )

    try:
        user = await authenticate_unified(body.username, body.duty, body.password)
    except AuthError as exc:
        await record_audit(
            "auth.unified_login.failed",
            username=body.username,
            detail=str(exc),
            ip=ip,
        )
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None

    await record_audit(
        "auth.unified_login",
        user_id=user.id,
        username=user.username,
        ip=ip,
    )
    return TokenResponse(
        access_token=create_access_token(user), user=await _profile(user)
    )


@router.get(
    "/me",
    response_model=UserProfile,
    summary="Current user profile",
)
async def me_endpoint(user: User = Depends(get_current_user)) -> UserProfile:
    return await _profile(user)


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
