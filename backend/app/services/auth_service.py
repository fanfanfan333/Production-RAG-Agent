"""
Authentication service (企业落地第一阶段).

Responsibilities:
  - bcrypt password hashing / verification
  - JWT issue / decode (HS256, configurable expiry)
  - user registration (first user becomes admin) and lookup

Deliberately dependency-light: PyJWT + bcrypt only — no extra auth framework,
so the security-sensitive code stays short and auditable.
"""

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

# Username policy: 3–64 chars, letters/digits/dot/underscore/hyphen.
USERNAME_ALPHABET = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


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

    if not (3 <= len(username) <= 64):
        raise AuthError("用户名长度需在 3–64 个字符之间")
    if not set(username) <= USERNAME_ALPHABET:
        raise AuthError("用户名只能包含字母、数字、点、下划线和连字符")

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
    """
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
    """Verify credentials. Raises AuthError(401) with a generic message on failure."""
    async with get_db_session() as session:
        user = await session.scalar(
            select(User).where(User.username == username).limit(1)
        )

    # Generic error message — never reveal whether the username exists.
    if user is None or not verify_password(password, user.password_hash):
        raise AuthError("用户名或密码错误", status_code=401)
    if not user.is_active:
        raise AuthError("账号已被禁用，请联系管理员", status_code=403)

    return user


async def get_user_by_id(user_id: uuid.UUID) -> User | None:
    async with get_db_session() as session:
        return await session.get(User, user_id)


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
        if not verify_password(old_password, user.password_hash):
            raise AuthError("当前密码不正确", status_code=400)

        user.password_hash = hash_password(new_password)

    logger.info("Password changed for user id=%s", user_id)
