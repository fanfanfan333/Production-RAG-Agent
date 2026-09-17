"""
Shared FastAPI dependencies (企业落地第一阶段).

Provides:
  - get_current_user   — Bearer-JWT authentication for every protected router
  - require_admin      — admin-role guard
  - limiter            — in-process sliding-window rate limiter with
                         `enforce_rate_limit()` helper (per-user / per-IP keys)

The limiter is intentionally in-memory: it protects a single backend instance
(the deployment shape of docker-compose). For horizontal scaling, swap its
backend for Redis without changing call sites.
"""

import time
import uuid
from asyncio import Lock
from collections import defaultdict, deque

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.db.user_models import User
from app.services.auth_service import decode_token, get_user_by_id
from app.utils.logging import get_logger

logger = get_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> User:
    """Resolve the request's Bearer token to an active User row."""
    return await _resolve_user(credentials.credentials if credentials else None)


def _token_kind(token: str) -> str:
    """
    按 JWT header 的 alg 判断令牌来自哪条通路。

    HS* → 本服务自签（本地账号登录）
    其余非对称算法（RS*/PS*/ES*）→ Keycloak 等 OIDC 提供方签发

    用 header 而不是"先试本地再试 Keycloak"：前者一次判定，失败原因清晰
    （"Keycloak 未启用" vs "签名错误"），也不会因为本地验签异常而误导排查。
    """
    try:
        header = jwt.get_unverified_header(token)
    except Exception:      # noqa: BLE001 — 解析失败按本地通路处理，让后续报标准错误
        return "local"
    alg = str(header.get("alg") or "").upper()
    return "local" if alg.startswith("HS") else "keycloak"


async def _resolve_user(token: str | None) -> User:
    """
    共享的 token → User 解析（两条身份通路）：

      1. 本地令牌（HS256，/auth/login 签发）  → 直接按 sub 查用户
      2. Keycloak 令牌（RS256）              → JWKS 验签 → 身份信息
                                              → 落地/同步本地用户行

    无论走哪条通路，返回的都是**本地 User 行**：检索链路上的 tenant_id /
    department / role 只有一个来源，权限矩阵不需要知道身份来自哪里。
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录或缺少访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if _token_kind(token) == "keycloak":
        return await _resolve_keycloak_user(token)

    payload = decode_token(token)
    try:
        user_id = uuid.UUID(payload["sub"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的登录凭证",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user = await get_user_by_id(user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="账号不存在或已被禁用",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def _resolve_keycloak_user(token: str) -> User:
    """Keycloak 通路：验签 → 解析身份 → 落地本地用户行。"""
    from app.services.keycloak_auth import (
        KeycloakError,
        identity_from_claims,
        resolve_or_provision_user,
        verify_keycloak_token,
    )

    try:
        claims = await verify_keycloak_token(token)
        identity = identity_from_claims(claims)
        user = await resolve_or_provision_user(identity)
    except KeycloakError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="账号已被禁用，请联系管理员",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.debug(
        "Keycloak auth ok: user=%s tenant=%s role=%s",
        user.username, user.tenant_id, user.role,
    )
    return user


async def get_current_user_media(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> User:
    """
    Auth for resource URLs loaded directly by the browser.

    ``<img src>`` and ``<a download>`` cannot attach an Authorization header, so
    these endpoints additionally accept ``?token=<jwt>``. The token is the same
    short-lived JWT used by the API; access is still owner-checked downstream.
    """
    token = credentials.credentials if credentials else None
    if not token:
        token = request.query_params.get("token")
    return await _resolve_user(token)


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="该操作需要管理员权限",
        )
    return user


# ── Rate limiting ─────────────────────────────────────────────────────────────

class SlidingWindowLimiter:
    """
    Per-key sliding-window counter. Keys are namespaced strings such as
    "query:<user_id>" or "login:<ip>".
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    async def allow(self, key: str, limit: int, window_seconds: float) -> bool:
        """Consume one slot for *key*; return False when over the limit."""
        now = time.monotonic()
        async with self._lock:
            dq = self._hits[key]
            cutoff = now - window_seconds
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= limit:
                return False
            dq.append(now)
            # Opportunistic GC so long-tail keys don't accumulate forever
            if len(self._hits) > 10_000:
                self._hits = defaultdict(
                    deque,
                    {k: v for k, v in self._hits.items() if v},
                )
            return True


limiter = SlidingWindowLimiter()


def client_ip(request: Request) -> str:
    """
    Resolve the client IP without trusting spoofable forwarding headers.

    ``X-Forwarded-For`` is used only when the direct peer address is explicitly
    listed in ``TRUSTED_PROXY_IPS``. Direct deployments should leave that list
    empty; reverse-proxy deployments should add the proxy's private IP.
    """
    peer_ip = request.client.host if request.client else "unknown"
    trusted_proxies = set(get_settings().TRUSTED_PROXY_IPS)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and peer_ip in trusted_proxies:
        return forwarded.split(",")[0].strip()
    return peer_ip


async def enforce_rate_limit(
    request: Request,
    *,
    scope: str,
    key: str,
    limit: int,
    window_seconds: int,
) -> None:
    """Raise 429 (with Retry-After) when *key* exceeds *limit* in the window."""
    allowed = await limiter.allow(f"{scope}:{key}", limit, window_seconds)
    if not allowed:
        logger.warning("Rate limit hit: scope=%s key=%s limit=%d/%ds", scope, key, limit, window_seconds)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后再试",
            headers={"Retry-After": str(window_seconds)},
        )
