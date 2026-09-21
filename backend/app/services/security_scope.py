"""
请求级五维权限 Scope（决策 1：**组合**内嵌既有 ``DocumentScope``，不改它一个字段）.

    UserScope = DocumentScope(既有三维：租户 / 层级 / 个人库)
              + clearance（密级） + project_ids（项目） + principals（ACL 主体）

为什么是组合而不是改造 ``DocumentScope``
────────────────────────────────────────
``DocumentScope`` 已被 ``tenancy`` / ``api/query`` / ``master_graph`` /
``rag_graph`` / ``document_query_service`` / ``relation_service`` 等十余处消费，
且 ``acl_kwargs()`` 是"列表与检索同源"的结构保证。把它重写成五维要一次性触碰
十余个调用点 —— 任一处漏改就是"静默降级为不过滤"。上游 Rev2 已经吃过一次
``rag_graph._retrieve_node`` 漏传权限参数的亏，不再制造第二次。

继承也不行：``replace(scope, ...)`` 与 ``isinstance`` 判定会混入五维语义，
而 ``document_scope_clause`` 只应该看到三维部分。

不可变性
────────
``frozen=True`` + 集合字段一律 ``frozenset``：任何"链路内补权限"的写法都会
``FrozenInstanceError``，而不是悄悄生效（决策 10-②）。

签发点唯一
──────────
:func:`request_security_scope` 是**唯一**签发函数，只允许被 ``app/api/**`` 调用
—— 由 ``tests/test_no_unscoped_retrieval.py`` 的 AST 门禁强制（决策 10-④）：
链路内不得重新查库放宽 scope。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.db.security_models import (
    SECURITY_LEVEL_MAX,
    SECURITY_LEVEL_MIN,
)
from app.db.user_models import User
from app.services.security_policy import (
    CLEARANCE_ON_FAILURE,
    DEFAULT_CLEARANCE_BY_ROLE,
    ScopePredicate,
    build_predicate,
)
from app.services.tenancy import DocumentScope, content_scope
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: 指纹版本。字段集或格式变更时必须 +1 —— 否则新旧指纹会撞键（串缓存）。
SCOPE_FINGERPRINT_VERSION = "v1"

#: 未登录 / 匿名主体的占位 id
ANONYMOUS_USER_ID = "anonymous"


def _fp_set(values: Any) -> str:
    """
    集合指纹（**三态互异**：None / 空集 / 非空）—— 沿用上游
    ``tenancy.tenant_scope_fingerprint`` 的三分支口径。

    ``None``（诊断"不限制"）与 ``frozenset()``（fail-closed 空集）是**两种完全不同
    的权限语义**，绝不能都折叠成空串。
    """
    if values is None:
        return "all"
    items = sorted(str(v) for v in values)
    if not items:
        return "none"
    return "s:" + hashlib.sha1(",".join(items).encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class UserScope:
    """一个请求的**完整**数据范围（五维），签发后不可变。"""

    base: DocumentScope                  # 既有三维，一个字段都不改
    user_id: str
    role: str                            # 只用于动作能力与 principals 组装，不参与判定
    clearance: int                       # 0..3
    project_ids: frozenset[str] = frozenset()
    principals: frozenset[str] = frozenset()
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strict: bool = False                 # 未标注密级按最高档处理（settings.SECURITY_STRICT_MODE）

    # ── 转发既有三维（零破坏：老调用点拿到的 kwargs 一个键都不变）──────────────
    def acl_kwargs(self) -> dict:
        """= ``base.acl_kwargs()``。老调用点不需要感知五维。"""
        return self.base.acl_kwargs()

    # ── 新增维度 ──────────────────────────────────────────────────────────────
    def security_kwargs(self) -> dict:
        """新增五维部分（给第 7 / 11 / 12 环的编译器消费）。"""
        return {
            "user_id": self.user_id,
            "role": self.role,
            "clearance": self.clearance,
            "project_ids": self.project_ids,
            "principals": self.principals,
            "strict": self.strict,
            "scope_fingerprint": self.scope_fingerprint,
        }

    @property
    def scope_fingerprint(self) -> str:
        """
        五维指纹（决策 9）—— 缓存分区与日志的共同主键.

        对**排序后的集合**做哈希 ⇒ 不同 clearance / 不同 project 集合 / 不同
        principals 必然产生不同指纹；``c:`` 一位之差就换键。
        **不同 Scope 不共用缓存**因此是哈希的性质，不是约定。

        反过来：改名 / 改展示名**不影响**指纹（沿用上游决策 10 的论证）——
        它们不进任何维度。
        """
        parts = [
            SCOPE_FINGERPRINT_VERSION,
            f"u:{self.user_id or '-'}",
            f"T:{_fp_set(self.base.tenant_ids)}",
            f"O:{_fp_set(self.base.owns_tenant_ids)}",
            f"d:{self.base.department_id or '-'}",
            f"w:{'1' if self.base.tenant_wide else '0'}",
            f"c:{self.clearance}",
            f"p:{_fp_set(self.project_ids)}",
            f"a:{_fp_set(self.principals)}",
            f"s:{'1' if self.strict else '0'}",
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]

    def predicate(self, *, now: datetime | None = None) -> ScopePredicate:
        """→ :class:`ScopePredicate`（三个编译器的唯一输入）。"""
        return build_predicate(self, now=now)

    # ── 构造辅助 ──────────────────────────────────────────────────────────────
    @classmethod
    def anonymous(cls, *, strict: bool = False) -> "UserScope":
        """未登录主体：空租户集合 + clearance=0 ⇒ **全拒**（fail-closed）。"""
        return cls(
            base=DocumentScope(
                owner_id=None,
                tenant_ids=frozenset(),
                owns_tenant_ids=frozenset(),
                department_id=None,
                tenant_wide=False,
            ),
            user_id=ANONYMOUS_USER_ID,
            role="anonymous",
            clearance=CLEARANCE_ON_FAILURE,
            project_ids=frozenset(),
            principals=frozenset(),
            issued_at=datetime.now(timezone.utc),
            strict=strict,
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"<UserScope u={self.user_id} role={self.role} c={self.clearance} "
            f"fp={self.scope_fingerprint}>"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# principals —— **唯一组装函数**（共享知识 10：禁止各处手拼）
# ═══════════════════════════════════════════════════════════════════════════════


def principals_of(
    user: User | None,
    project_ids: Any = frozenset(),
    *,
    extra_groups: Any = frozenset(),
) -> frozenset[str]:
    """
    组装 ACL 主体集合（格式：``user:<id>`` / ``dept:<id>`` / ``role:<r>`` /
    ``project:<p>`` / ``group:<g>``）。

    五维任一缺失 → **不加该项**（而不是塞一个空串进去）—— 空串会与"授予给
    空主体"巧合命中，那是静默越权。
    """
    if user is None:
        return frozenset()
    items: set[str] = set()
    uid = getattr(user, "id", None)
    if uid is not None:
        items.add(f"user:{uid}")
    dept = getattr(user, "department_id", None)
    if dept:
        items.add(f"dept:{dept}")
    role = getattr(user, "role", None)
    if role:
        items.add(f"role:{role}")
    for pid in (project_ids or frozenset()):
        if pid:
            items.add(f"project:{pid}")
    for gid in (extra_groups or frozenset()):
        if gid:
            items.add(f"group:{gid}")
    return frozenset(items)


# ═══════════════════════════════════════════════════════════════════════════════
# 唯一签发点
# ═══════════════════════════════════════════════════════════════════════════════


async def request_security_scope(
    user: User | None,
    *,
    strict: bool | None = None,
) -> UserScope:
    """
    **唯一签发点**（只允许被 ``app/api/**`` 调用 —— AST 门禁强制）.

    五维任一解析失败 → 按**最小权限**处理（PRD 环节 1.1），绝不抛异常让请求
    整体失败，也绝不放宽：

        clearance   解析失败 → 0
        project_ids 查询失败 → frozenset()（``logger.exception`` 留痕）
        principals  缺 dept  → 不加 dept 项

    仅当 ``user is None`` 时返回 :meth:`UserScope.anonymous`（全拒）。
    """
    if user is None:
        return UserScope.anonymous(strict=bool(strict))

    if strict is None:
        strict = _strict_mode()

    base = await content_scope(user)     # 内容消费口径（含测试公司剔除，与检索一致）

    clearance = _resolve_clearance(user)
    project_ids = await _resolve_project_ids(user)
    principals = principals_of(user, project_ids)

    return UserScope(
        base=base,
        user_id=str(getattr(user, "id", "") or ""),
        role=str(getattr(user, "role", "") or ""),
        clearance=clearance,
        project_ids=project_ids,
        principals=principals,
        issued_at=datetime.now(timezone.utc),
        strict=bool(strict),
    )


def _strict_mode() -> bool:
    """
    读取 ``settings.SECURITY_STRICT_MODE``（"未标注密级是否按最高档处理"）.

    ⚠️ **fail-CLOSED**（T5 上线前修复）：旧实现在读取失败时**静默返回 ``False``**
    —— 等于把"未标注密级的文档被当作公开"这个更宽松的口径悄悄生效，而且日志里
    没有任何痕迹（与 ``except Exception: pass`` 同级的静默降级）。配置读取失败
    必须按**更严**的方向走（``strict=True``），并留下 exception 日志。
    """
    try:
        from app.config import get_settings

        return bool(get_settings().SECURITY_STRICT_MODE)
    except Exception:      # noqa: BLE001 — 读不到配置 ⇒ 取严，绝不取宽
        logger.exception(
            "security_scope: 读取 SECURITY_STRICT_MODE 失败 —— "
            "按 fail-closed 口径取 strict=True（未标注密级按最高档处理）"
        )
        return True


def _resolve_clearance(user: User) -> int:
    """
    ``users.clearance`` 优先；为 NULL 时按角色推导；再失败 → 0（最小权限）.

    ⚠️ 角色只是**初值来源**，不是豁免：判定式里没有任何 role 分支。
    """
    raw = getattr(user, "clearance", None)
    if raw is not None:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = CLEARANCE_ON_FAILURE
        else:
            value = max(SECURITY_LEVEL_MIN, min(SECURITY_LEVEL_MAX, value))
        return value

    role = getattr(user, "role", None)
    value = DEFAULT_CLEARANCE_BY_ROLE.get(str(role or "").strip())
    if value is None:
        logger.warning(
            "security_scope: 用户 %s 的角色 %r 无默认密级映射 —— 按最小权限 clearance=0",
            getattr(user, "id", "-"), role,
        )
        return CLEARANCE_ON_FAILURE
    return value


async def _resolve_project_ids(user: User) -> frozenset[str]:
    """``project_members`` 里该用户的**未过期**项目集合（查询失败 → 空集）。"""
    uid = getattr(user, "id", None)
    if uid is None:
        return frozenset()
    try:
        from sqlalchemy import or_, select

        from app.db.postgres import get_db_session
        from app.db.security_models import ProjectMember

        now = datetime.now(timezone.utc)
        stmt = select(ProjectMember.project_id).where(
            ProjectMember.user_id == uid,
            or_(ProjectMember.expires_at.is_(None), ProjectMember.expires_at > now),
        )
        async with get_db_session() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return frozenset(str(r) for r in rows)
    except Exception:      # noqa: BLE001 — 查不到项目 = 不属于任何项目（最小权限）
        logger.exception(
            "security_scope: 解析 project_ids 失败 —— 按空集处理（最小权限）"
        )
        return frozenset()


# ═══════════════════════════════════════════════════════════════════════════════
# 缓存分区
# ═══════════════════════════════════════════════════════════════════════════════


def cache_key_for_scope(scope: UserScope | None, raw_key: str) -> str:
    """
    权限相关缓存键 = ``scope_fingerprint`` + 原始键（共享知识 11）.

    任何权限相关的缓存（BM25 语料 / 检索结果 / ...）都必须经过它；
    **禁止**手拼 ``perm_ctx`` 字符串。``scope=None`` 走独立的 ``anon::`` 分区，
    绝不退化成"无分区"。
    """
    if scope is None:
        return f"anon::{raw_key}"
    return f"{scope.scope_fingerprint}::{raw_key}"


__all__ = [
    "ANONYMOUS_USER_ID",
    "SCOPE_FINGERPRINT_VERSION",
    "UserScope",
    "cache_key_for_scope",
    "principals_of",
    "request_security_scope",
]
