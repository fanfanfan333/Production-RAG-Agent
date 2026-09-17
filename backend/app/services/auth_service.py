"""
Authentication service (企业落地第一阶段).

Responsibilities:
  - bcrypt password hashing / verification
  - JWT issue / decode (HS256, configurable expiry)
  - user registration (first user becomes admin) and lookup

Deliberately dependency-light: PyJWT + bcrypt only — no extra auth framework,
so the security-sensitive code stays short and auditable.
"""

import re
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from sqlalchemy import func, select

from app.config import get_settings
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 账号策略：企业账号**就是公司内邮箱**（如 zhangsan@company.com）。
# 邮箱格式：本地部分@域名.顶级域。刻意不引 email_validator —— 为了一个正则
# 拉进整套依赖（还带 DNS 解析）不划算。
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)+$")


def normalize_username(username: str) -> str:
    """
    账号归一化.

    邮箱不区分大小写（各邮件服务商都如此），统一转小写，否则
    ``ZhangSan@Corp.com`` 与 ``zhangsan@corp.com`` 会被当成两个账号。
    存量非邮箱账号（都是小写）原样返回，不受影响。
    """
    value = (username or "").strip()
    return value.lower() if "@" in value else value


def is_email(username: str) -> bool:
    """是否形如邮箱（企业账号的判定口径）。"""
    return bool(EMAIL_PATTERN.fullmatch(username))


class AuthError(Exception):
    """Raised with a user-facing message for any auth flow failure."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# ── Passwords ─────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(user: User) -> str:
    settings = get_settings()
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and verify a JWT. Raises AuthError(401) on any failure."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("登录已过期，请重新登录", status_code=401) from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("无效的登录凭证", status_code=401) from exc

    if not payload.get("sub"):
        raise AuthError("无效的登录凭证", status_code=401)
    return payload


# ── Users ─────────────────────────────────────────────────────────────────────

def _validate_registration(username: str, password: str) -> None:
    settings = get_settings()

    # 产品要求：首次使用即以邮箱注册（账号 = 公司内邮箱）。
    if not is_email(username):
        raise AuthError("请输入有效的企业邮箱地址，例如 zhangsan@company.com")
    if len(username) > 254:
        raise AuthError("邮箱地址过长（最多 254 个字符）")

    if len(password) < settings.PASSWORD_MIN_LENGTH:
        raise AuthError(f"密码至少需要 {settings.PASSWORD_MIN_LENGTH} 个字符")


async def count_users() -> int:
    async with get_db_session() as session:
        result = await session.execute(select(func.count()).select_from(User))
        return int(result.scalar_one())


async def register_user(username: str, password: str) -> User:
    """
    Create a new account. The FIRST registered user becomes admin — this is
    the only admin bootstrap path, so self-registration must stay enabled
    until at least one admin exists.

    账号即公司内邮箱；邮箱大小写不敏感，入库前统一小写。
    """
    username = normalize_username(username)
    _validate_registration(username, password)

    settings = get_settings()
    total = await count_users()

    if total == 0:
        role = User.ROLE_ADMIN
        logger.info("First user '%s' registered — granted admin role", username)
    elif not settings.ALLOW_SELF_REGISTRATION:
        raise AuthError("系统已关闭自主注册，请联系管理员开通账号", status_code=403)
    else:
        role = User.ROLE_USER

    async with get_db_session() as session:
        existing = await session.scalar(
            select(User).where(User.username == username).limit(1)
        )
        if existing is not None:
            raise AuthError("用户名已被占用", status_code=409)

        user = User(
            id=uuid.uuid4(),
            username=username,
            password_hash=hash_password(password),
            role=role,
        )
        session.add(user)
        await session.flush()
        await session.refresh(user)
        logger.info("Registered user '%s' (role=%s, id=%s)", username, role, user.id)
        return user


async def authenticate_user(username: str, password: str) -> User:
    """
    Verify credentials. Raises AuthError(401) with a generic message on failure.

    登录**不做**邮箱格式校验：存量的非邮箱账号（admin、lisi…）仍要能登，
    只在查询前把邮箱归一化成小写，保证与注册时一致。
    """
    username = normalize_username(username)
    async with get_db_session() as session:
        user = await session.scalar(
            select(User).where(User.username == username).limit(1)
        )

    # 账号不存在与密码错，用同一句话（不暴露账号是否存在）。
    if user is None:
        raise AuthError("邮箱或密码错误", status_code=401)
    if not user.password_hash:
        # SSO 账号（auth_source=keycloak）没有本地密码，password_hash 为 NULL。
        # 不拦住的话会掉进 verify_password → None.encode()，抛 AttributeError 变 500。
        raise AuthError(
            "该账号使用企业统一身份登录，请点击「使用企业统一身份登录」",
            status_code=401,
        )
    if not verify_password(password, user.password_hash):
        raise AuthError("邮箱或密码错误", status_code=401)
    if not user.is_active:
        raise AuthError("账号已被禁用，请联系管理员", status_code=403)

    return user


async def get_user_by_id(user_id: uuid.UUID) -> User | None:
    async with get_db_session() as session:
        return await session.get(User, user_id)


def _normalize_duty(duty: str) -> str:
    """职责比对口径：去首尾空白 + 大小写不敏感（中文不受影响）。"""
    return (duty or "").strip().casefold()


async def authenticate_unified(username: str, duty: str, password: str) -> User:
    """
    企业统一身份登录：**邮箱 + 已登记职责 + 密码**，三者都要对。

    职责取自 ``users.job_title`` —— 那是由身份验证流程（上级审核通过）写入的
    权威值，不另开一份事实来源。

    校验顺序刻意是"先账号密码、后比对职责"：
    职责不符的提示只会给到**已经掌握密码**的人，否则这个接口就变成了
    "探测某账号的职责是什么"的枚举入口。
    """
    user = await authenticate_user(username, password)

    registered = (user.job_title or "").strip()
    if not registered:
        raise AuthError(
            "该账号尚未登记部门职责，请先完成身份验证，或改用邮箱密码登录",
            status_code=403,
        )
    if _normalize_duty(duty) != _normalize_duty(registered):
        raise AuthError("企业职责与账号登记的职责不一致", status_code=401)
    return user


async def change_password(
    user_id: uuid.UUID,
    old_password: str,
    new_password: str,
) -> None:
    """
    Change the user's password after verifying the current one.

    Raises AuthError with a user-facing Chinese message on any failure.
    """
    settings = get_settings()

    if len(new_password) < settings.PASSWORD_MIN_LENGTH:
        raise AuthError(f"新密码至少需要 {settings.PASSWORD_MIN_LENGTH} 个字符")
    if new_password == old_password:
        raise AuthError("新密码不能与当前密码相同")

    async with get_db_session() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise AuthError("账号不存在", status_code=401)
        if not user.password_hash:
            # SSO 账号没有本地密码，改密应去企业统一身份平台（Identity Provider）。
            raise AuthError(
                "该账号使用企业统一身份登录，无法在此修改密码",
                status_code=400,
            )
        if not verify_password(old_password, user.password_hash):
            raise AuthError("当前密码不正确", status_code=400)

        user.password_hash = hash_password(new_password)

    logger.info("Password changed for user id=%s", user_id)
