"""
Keycloak 身份认证（JWT 验签 → 用户身份信息 → 权限层 → 检索）.

整条链路（与架构图一致）::

    用户 → 前端 → Keycloak 登录 → JWT Token → FastAPI
                                                  │
                                       Auth Middleware / Dependency
                                                  │  验证签名 + 过期
                                                  ▼
                                        用户身份信息
                                        user_id / tenant_id(company_a)
                                        department / roles
                                                  │
                                            Permission Layer
                                                  │
                                        RAG Retrieval（前置过滤）
                                                  ▼
                                                 LLM

设计要点
--------
1. **非对称验签**：Keycloak 用 RS256 签发，公钥从 realm 的 JWKS 端点拉取并按
   ``kid`` 缓存。签名验证交给 PyJWT（依赖 cryptography），本模块不自己实现
   密码学。``import jwt.algorithms`` 需要 cryptography，缺失时给出明确报错。
2. **claim 容错**：不同 Keycloak 版本 / mapper 配置下，角色可能出现在
   ``roles`` / ``realm_access.roles`` / ``resource_access.<client>.roles``；
   tenant 与 department 可能是 ``tenant_id``/``company``/``department`` 等。
   这里做"多来源取第一个非空"，而不是要求对方必须按某一种格式配置。
3. **身份落地**：验签成功后按 ``sub`` 找到本地 ``users`` 行（找不到则自动建档），
   之后所有权限判断、租户过滤仍然读本地行 —— 检索链路上 tenant_id 只有
   一个来源，不会出现"token 说 A 公司、数据库说 B 公司"的分叉。
4. **不降级平台管理员**：Keycloak 同步角色时保护本地 ``admin``（跨租户平台
   管理员），避免运维账号被 IdP 里的普通角色覆盖而丢失后台入口。
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

import httpx
import jwt
from sqlalchemy import select

from app.config import get_settings
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.services.tenancy import normalize_tenant_id
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 角色优先级（数字越大越强）——用于把一组 Keycloak 角色收敛成一个应用角色
_ROLE_PRIORITY: dict[str, int] = {
    User.ROLE_ADMIN: 100,
    User.ROLE_COMPANY_ADMIN: 90,
    User.ROLE_KB_ADMIN: 80,
    User.ROLE_DEPT_MANAGER: 70,
    User.ROLE_MANAGER: 70,
    User.ROLE_EMPLOYEE: 60,
    User.ROLE_EDITOR: 60,
    User.ROLE_USER: 60,
    User.ROLE_VIEWER: 50,
}

# Keycloak 侧可能使用的别名 → 应用角色
_ROLE_ALIASES: dict[str, str] = {
    "company_admin": User.ROLE_COMPANY_ADMIN,
    "companyadmin": User.ROLE_COMPANY_ADMIN,
    "enterprise_admin": User.ROLE_COMPANY_ADMIN,
    "admin": User.ROLE_ADMIN,
    "kb_admin": User.ROLE_KB_ADMIN,
    "kbadmin": User.ROLE_KB_ADMIN,
    "knowledge_admin": User.ROLE_KB_ADMIN,
    "dept_manager": User.ROLE_DEPT_MANAGER,
    "department_manager": User.ROLE_DEPT_MANAGER,
    "manager": User.ROLE_MANAGER,
    "employee": User.ROLE_EMPLOYEE,
    "engineer": User.ROLE_EMPLOYEE,      # 架构图示例里 engineer 就是普通员工
    "staff": User.ROLE_EMPLOYEE,
    "editor": User.ROLE_EDITOR,
    "user": User.ROLE_USER,
    "viewer": User.ROLE_VIEWER,
}

_USERNAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")


class KeycloakError(Exception):
    """Keycloak 验签/身份解析失败（携带用户可读中文 + HTTP 状态码）。"""

    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class KeycloakIdentity:
    """从 JWT 里读出的身份信息（Permission Layer 的输入）。"""

    sub: str
    username: str
    tenant_id: str
    department: str | None
    role: str
    roles: list[str] = field(default_factory=list)
    email: str | None = None
    display_name: str | None = None


# ── JWKS 缓存 ─────────────────────────────────────────────────────────────────

_jwks_cache: dict[str, object] = {"keys": {}, "fetched_at": 0.0}


def keycloak_realm_urls() -> dict[str, str]:
    """返回 Keycloak 相关端点（服务端 JWKS 用内部地址，前端用对外地址）。"""
    settings = get_settings()
    base = settings.KEYCLOAK_URL.rstrip("/")
    realm = settings.KEYCLOAK_REALM
    issuer = f"{base}/realms/{realm}"
    return {
        "issuer": issuer,
        "jwks": f"{issuer}/protocol/openid-connect/certs",
        "token": f"{issuer}/protocol/openid-connect/token",
        "auth": f"{issuer}/protocol/openid-connect/auth",
        "logout": f"{issuer}/protocol/openid-connect/logout",
        "public_issuer": f"{settings.KEYCLOAK_PUBLIC_URL.rstrip('/')}/realms/{realm}",
    }


async def _fetch_jwks(force: bool = False) -> dict:
    """拉取并缓存 realm 的公钥集合（JWKS），按 kid 建索引。"""
    settings = get_settings()
    now = time.monotonic()
    if (
        not force
        and _jwks_cache["keys"]
        and now - float(_jwks_cache["fetched_at"]) < settings.KEYCLOAK_JWKS_CACHE_SECONDS
    ):
        return _jwks_cache["keys"]  # type: ignore[return-value]

    url = keycloak_realm_urls()["jwks"]
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        payload = resp.json()

    keys = {k["kid"]: k for k in payload.get("keys", []) if k.get("kid")}
    _jwks_cache["keys"] = keys
    _jwks_cache["fetched_at"] = now
    logger.info("Loaded %d JWKS key(s) from %s", len(keys), url)
    return keys


def reset_jwks_cache() -> None:
    _jwks_cache["keys"] = {}
    _jwks_cache["fetched_at"] = 0.0


# ── 验签 ──────────────────────────────────────────────────────────────────────

async def verify_keycloak_token(token: str) -> dict:
    """
    验证 Keycloak 签发的 access token，返回其 claims。

    Raises:
        KeycloakError: 签名无效 / 已过期 / 签发方不匹配 / 公钥拉取失败。
    """
    settings = get_settings()
    if not settings.KEYCLOAK_ENABLED:
        raise KeycloakError("服务端未启用 Keycloak 认证，请使用本地账号登录")

    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as exc:
        raise KeycloakError("无效的登录凭证") from exc

    kid = header.get("kid")
    if not kid:
        raise KeycloakError("登录凭证缺少密钥标识（kid）")

    try:
        keys = await _fetch_jwks()
    except Exception as exc:      # noqa: BLE001
        raise KeycloakError(
            f"无法获取 Keycloak 公钥（{exc}）；请确认 Keycloak 已启动且 "
            f"KEYCLOAK_URL/REALM 配置正确",
            status_code=503,
        ) from exc

    jwk = keys.get(kid)
    if jwk is None:
        # 密钥轮换：强制刷新一次再试
        try:
            keys = await _fetch_jwks(force=True)
        except Exception as exc:      # noqa: BLE001
            raise KeycloakError(f"无法获取 Keycloak 公钥（{exc}）", status_code=503) from exc
        jwk = keys.get(kid)
    if jwk is None:
        raise KeycloakError("登录凭证的签名密钥已失效，请重新登录")

    try:
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(jwk)  # type: ignore[attr-defined]
    except Exception as exc:      # noqa: BLE001
        raise KeycloakError("Keycloak 公钥解析失败（服务端缺少 cryptography 依赖）") from exc

    issuer = keycloak_realm_urls()["issuer"]
    public_issuer = keycloak_realm_urls()["public_issuer"]
    audience = settings.KEYCLOAK_AUDIENCE
    try:
        claims = jwt.decode(
            token,
            key=public_key,
            algorithms=["RS256", "RS384", "RS512", "PS256", "ES256"],
            options={
                "verify_aud": bool(audience and settings.KEYCLOAK_VERIFY_AUDIENCE),
                # iss 手动校验：见下方 allowed_issuers
                "verify_iss": False,
            },
            audience=audience if (audience and settings.KEYCLOAK_VERIFY_AUDIENCE) else None,
        )
    except jwt.ExpiredSignatureError as exc:
        raise KeycloakError("登录已过期，请重新登录") from exc
    except jwt.InvalidAudienceError as exc:
        raise KeycloakError("登录凭证的受众（aud）不匹配 KEYCLOAK_AUDIENCE") from exc
    except jwt.InvalidTokenError as exc:
        raise KeycloakError(f"登录凭证校验失败：{exc}") from exc

    # 签发方手动校验：同一个 realm 在容器网络里是 http://keycloak:8080，
    # 在浏览器里是 http://localhost:8080 —— 两者都必须被接受，否则要么
    # 前端拿到的 token 被拒，要么只能牺牲服务端的内网直连。
    if settings.KEYCLOAK_VERIFY_ISSUER:
        got_iss = str(claims.get("iss") or "")
        allowed = {issuer, public_issuer}
        same_realm = got_iss.rstrip("/").endswith(f"/realms/{settings.KEYCLOAK_REALM}")
        if got_iss not in allowed and not same_realm:
            raise KeycloakError(
                f"登录凭证的签发方（{got_iss or '缺失'}）不属于本 realm，已拒绝"
            )
    return claims


# ── claims → 身份信息 ──────────────────────────────────────────────────────────

def _collect_roles(claims: dict) -> list[str]:
    """把三种常见来源的角色合成一个去重列表。"""
    settings = get_settings()
    found: list[str] = []

    def _push(value) -> None:
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, (list, tuple, set)):
            found.extend(str(v) for v in value)

    _push(claims.get("roles"))
    realm_access = claims.get("realm_access") or {}
    if isinstance(realm_access, dict):
        _push(realm_access.get("roles"))

    resource_access = claims.get("resource_access") or {}
    if isinstance(resource_access, dict):
        target = resource_access.get(settings.KEYCLOAK_CLIENT_ID)
        if isinstance(target, dict):
            _push(target.get("roles"))

    seen: set[str] = set()
    result: list[str] = []
    for role in found:
        name = str(role).strip().lower()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def map_role(raw_roles: list[str]) -> str:
    """把 Keycloak 角色集合收敛成应用角色（取权限最高者）。"""
    best_role = get_settings().KEYCLOAK_DEFAULT_ROLE
    best_priority = -1
    for raw in raw_roles:
        app_role = _ROLE_ALIASES.get(raw)
        if app_role is None:
            continue
        priority = _ROLE_PRIORITY.get(app_role, 0)
        if priority > best_priority:
            best_priority = priority
            best_role = app_role
    return best_role


def _first_claim(claims: dict, *names: str) -> str | None:
    for name in names:
        value = claims.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)) and value:
            return str(value[0]).strip()
    return None


def identity_from_claims(claims: dict) -> KeycloakIdentity:
    """把已验证的 claims 解析成 KeycloakIdentity。"""
    settings = get_settings()

    sub = str(claims.get("sub") or "").strip()
    if not sub:
        raise KeycloakError("登录凭证缺少用户标识（sub）")

    username = _first_claim(
        claims, "preferred_username", "username", "email", "name", "sub"
    ) or sub
    username = _USERNAME_SANITIZE_RE.sub("-", username)[:64]

    tenant_raw = _first_claim(
        claims,
        settings.KEYCLOAK_TENANT_CLAIM,
        "tenant_id", "tenant", "company", "org", "organization",
    )
    tenant_id = normalize_tenant_id(tenant_raw) if tenant_raw else ""

    department = _first_claim(
        claims,
        settings.KEYCLOAK_DEPARTMENT_CLAIM,
        "department", "department_id", "dept",
    )

    raw_roles = _collect_roles(claims)
    role = map_role(raw_roles)

    return KeycloakIdentity(
        sub=sub,
        username=username,
        tenant_id=tenant_id,
        department=department or None,
        role=role,
        roles=raw_roles,
        email=claims.get("email"),
        display_name=_first_claim(claims, "name", "display_name"),
    )


# ── 身份落地（本地建档 / 同步）─────────────────────────────────────────────────

async def resolve_or_provision_user(identity: KeycloakIdentity) -> User:
    """
    按 Keycloak 身份找到（或创建）本地用户行，并同步租户/部门/角色。

    匹配顺序：keycloak_sub → username（老本地账号首次接入时自动绑定）。
    同步规则：tenant/department/role 以 IdP 为准；唯一例外是**不把平台管理员
    降级**（运维入口不能被 IdP 里的普通角色覆盖）。
    """
    settings = get_settings()

    async with get_db_session() as session:
        user: User | None = await session.scalar(
            select(User).where(User.keycloak_sub == identity.sub).limit(1)
        )

        if user is None:
            user = await session.scalar(
                select(User).where(User.username == identity.username).limit(1)
            )
            if user is not None:
                if not settings.KEYCLOAK_LINK_EXISTING_USERS:
                    raise KeycloakError(
                        "该用户名已存在本地账号，且未开启 Keycloak 账号绑定；"
                        "请联系管理员处理",
                        status_code=409,
                    )
                logger.warning(
                    "Linking existing local account '%s' to Keycloak sub=%s",
                    user.username, identity.sub,
                )

        if user is None:
            if not settings.KEYCLOAK_AUTO_PROVISION:
                raise KeycloakError(
                    "账号尚未在系统中开通，请联系管理员", status_code=403
                )
            user = User(
                id=uuid.uuid4(),
                username=identity.username,
                password_hash=None,          # 联邦账号没有本地密码
                role=identity.role,
                tenant_id=identity.tenant_id or "default",
                department_id=identity.department,
                display_name=identity.display_name,
                keycloak_sub=identity.sub,
                auth_source="keycloak",
            )
            session.add(user)
            await session.flush()
            await session.refresh(user)
            logger.info(
                "Provisioned Keycloak user '%s' (tenant=%s dept=%s role=%s)",
                user.username, user.tenant_id, user.department_id, user.role,
            )
            return user

        # ── 已存在：同步 IdP 的权威信息 ────────────────────────────────────
        if user.keycloak_sub != identity.sub:
            user.keycloak_sub = identity.sub
        user.auth_source = "keycloak" if user.password_hash is None else user.auth_source
        if identity.display_name:
            user.display_name = identity.display_name
        if identity.tenant_id:
            user.tenant_id = identity.tenant_id
        if identity.department:
            user.department_id = identity.department
        # 平台管理员不因 IdP 角色而降级
        if user.role == User.ROLE_ADMIN and identity.role != User.ROLE_ADMIN:
            logger.info(
                "Keeping platform-admin role for '%s' (IdP role=%s ignored)",
                user.username, identity.role,
            )
        else:
            user.role = identity.role

        await session.flush()
        await session.refresh(user)
        return user


def public_config() -> dict:
    """给前端的 Keycloak 配置（不含任何密钥）。"""
    settings = get_settings()
    public_base = settings.KEYCLOAK_PUBLIC_URL.rstrip("/")
    realm_base = f"{public_base}/realms/{settings.KEYCLOAK_REALM}"
    oidc = f"{realm_base}/protocol/openid-connect"
    return {
        "enabled": settings.KEYCLOAK_ENABLED,
        "url": public_base,
        "realm": settings.KEYCLOAK_REALM,
        "client_id": settings.KEYCLOAK_CLIENT_ID,
        "issuer": realm_base,
        "auth_endpoint": f"{oidc}/auth",
        "token_endpoint": f"{oidc}/token",
        "logout_endpoint": f"{oidc}/logout",
        "local_login_enabled": settings.ALLOW_LOCAL_LOGIN,
    }
