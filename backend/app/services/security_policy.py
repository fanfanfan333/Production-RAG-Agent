"""
权限判定内核 —— **一个 IR，三个编译器**（设计文档决策 2，本轮的核心）。

    UserScope ──predicate()──▶ ScopePredicate ──┬─▶ to_sql(pred, model)    → ColumnElement   (PG：第 7 环关键词腿 / 第 11 环)
                                                ├─▶ to_qdrant(pred)        → qmodels.Filter  (Qdrant：第 7 环向量腿 / 内存 BM25)
                                                └─▶ allows(pred, view)     → Decision        (Python：第 11 / 12 环逐对象)

为什么必须是这个形态
────────────────────
PRD P0-5 要求「第 7 环两路 Scope 表达式语义等价」且「第 7/11/12 三处复用同一份
判定」。如果"复用"是靠"两处代码写一样"，它必然在某次改动后分叉 —— 事实上
``retrieval_service._visibility_conditions`` 与 ``tenancy.document_scope_clause``
已经这样并存了三轮，每加一个维度都要人肉对齐一次。

这里把它变成**结构保证**：三个编译器的唯一输入都是 :class:`ScopePredicate`
（frozen 纯数据），判定逻辑只写在这三个函数里。分叉的唯一方式是新增第四个
编译器 —— 而 ``tests/test_no_unscoped_retrieval.py`` 的 AST 门禁会拦住任何绕过
它们的手拼条件。

三条不可动摇的口径（共享知识）
──────────────────────────────
1. **PG 是权限权威源**：``allows()`` 吃的是 PG 行（或它的最小字段视图）；
   Qdrant 侧的一切都是**前置剪枝**，不是判定。
2. **fail-closed**：``tenant_ids=frozenset()`` 与 ``tenant_ids=None``（非
   unrestricted）都必须是"全拒"，**绝不**因 ``if tenant_ids:`` 判 falsy 而退化
   成不限制；编译器抛异常 → 调用方返回空，不降级为不过滤。
3. **取严**：派生对象恒取 ``max(自身, 父)``，即使物化的
   ``effective_security_level`` 被写错，判定侧也再取一次 ``max``。
   角色不产生隐式豁免 —— 判定式里**不存在** ``if role == "admin": return True``。

⚠️ 命名口径：PG 列叫 ``owner_id``，Qdrant payload 叫 ``user_id``。映射只写在
:meth:`ObjectACLView.from_payload` 一处（共享知识 15）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import sqlalchemy as sa
from sqlalchemy import ARRAY, String, and_, false, not_, or_, true
from sqlalchemy.sql.elements import ColumnElement

from app.db.security_models import (
    DEFAULT_SECURITY_LEVEL,
    SECURITY_LEVEL_MAX,
    SECURITY_LEVEL_MIN,
    VISIBILITY_MODE_PROJECT,
    VISIBILITY_MODE_TIER,
)
from app.services.tenancy import (
    ACCESS_DEPARTMENT,
    ACCESS_PRIVATE,
    ACCESS_TENANT,
    document_scope_clause,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查，避免 security_scope ⇄ policy 循环导入
    from app.services.security_scope import UserScope


# ═══════════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════════

#: 角色 → 默认密级（决策 14 / 已裁决 Q7 = **admin 不豁免**）。
#: 只在 ``users.clearance IS NULL`` 时作为初值；管理员可显式下调该列（A8）。
#: 判定式里**没有任何**基于 role 的放行分支 —— ``role`` 只用于动作能力与
#: ``principals`` 组装。
DEFAULT_CLEARANCE_BY_ROLE: dict[str, int] = {
    "employee": 1,
    "user": 1,
    "editor": 1,
    "viewer": 1,
    "dept_manager": 2,
    "manager": 2,
    "kb_admin": 3,
    "company_admin": 3,
    "admin": 3,          # 有上限的"高"，不是"无限"
}

#: 密级解析失败 / 未知角色时的兜底（最小权限原则 → 0）
CLEARANCE_ON_FAILURE = 0

#: Qdrant payload 里的字段名（与 §4.6 对齐；**改动必须同步 vector_service**）
P_OBJECT_ID = "object_id"
P_OBJECT_TYPE = "object_type"
P_PARENT_OBJECT_ID = "parent_object_id"
P_VISIBILITY_MODE = "visibility_mode"
P_PROJECT_IDS = "project_ids"
P_SECURITY_LEVEL = "security_level"
P_PARENT_SECURITY_LEVEL = "parent_security_level"
P_EFFECTIVE_SECURITY_LEVEL = "effective_security_level"
P_ACL_ALLOW = "acl_allow"
P_ACL_DENY = "acl_deny"
P_ACL_EXPIRES_AT = "acl_expires_at"
#: 【实现补字段】Qdrant 无法对 keyword 字段做时间比较，日期区间过滤必须走数值。
#: 这是为了让「acl_allow 命中但已过期」在**前置过滤**这一层就能被排除（A2 矩阵
#: 要求的"两路都排除"），否则只能靠第 11 环 PG 复核 —— 那样候选池会被过期项占满。
#: 由 T3 的 payload 写入 / 回填脚本负责落盘；判定时缺失即视为"未过期"（fail-open）。
P_ACL_EXPIRES_AT_TS = "acl_expires_at_ts"
P_EXCLUDED = "excluded"
P_ACCESS_LEVEL = "access_level"
P_TENANT_ID = "tenant_id"
P_DEPARTMENT_ID = "department_id"
P_USER_ID = "user_id"          # ⚠️ payload 侧的所有者字段名（PG 侧叫 owner_id）
P_DOCUMENT_ID = "document_id"


class ScopeCompileError(RuntimeError):
    """
    编译器无法产出条件（缺列 / 类型不兼容 / 模型不支持）.

    调用方**必须 fail-closed**：捕获后返回空结果并 ``logger.error``，
    **绝不**降级为"不过滤"（共享知识 7）。
    """


# ═══════════════════════════════════════════════════════════════════════════════
# IR：ScopePredicate / ObjectACLView / Decision
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ScopePredicate:
    """
    :class:`UserScope` 的**可执行形态** —— 三个编译器的唯一输入.

    刻意做成纯数据（frozen + frozenset）：任何"链路内补权限"的写法都会
    ``FrozenInstanceError``，而不是悄悄生效。

    ``tenant_ids`` 的三态（**绝不写 `if tenant_ids:`**）：
        ``None``             → 无公司上下文；``unrestricted=True`` 才是不限制，
                               否则 fail-closed（全拒）
        ``frozenset()``      → 空集，fail-closed（全拒）
        非空 frozenset       → ``tenant_id IN (...)``
    """

    user_id: str | None
    tenant_ids: frozenset[str] | None
    owns_tenant_ids: frozenset[str] = frozenset()
    department_id: str | None = None
    tenant_wide: bool = False
    clearance: int = SECURITY_LEVEL_MIN
    project_ids: frozenset[str] = frozenset()
    principals: frozenset[str] = frozenset()
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    unrestricted: bool = False
    strict: bool = False

    def __post_init__(self) -> None:
        # 归一化：调用方可能传 set/list；frozen dataclass 不能直接赋值，用 object.__setattr__
        object.__setattr__(self, "owns_tenant_ids", _as_frozenset(self.owns_tenant_ids))
        object.__setattr__(self, "project_ids", _as_frozenset(self.project_ids))
        object.__setattr__(self, "principals", _as_frozenset(self.principals))
        if self.tenant_ids is not None:
            object.__setattr__(self, "tenant_ids", _as_frozenset(self.tenant_ids))
        # 密级钳到合法档位（配置写错时不至于放行一切，也不至于拒掉一切）
        clr = int(self.clearance)
        clr = max(SECURITY_LEVEL_MIN, min(SECURITY_LEVEL_MAX, clr))
        object.__setattr__(self, "clearance", clr)


@dataclass(frozen=True)
class ObjectACLView:
    """
    对象权限视图 —— PG 行与 Qdrant payload 的**共同最小字段集**.

    ``allows()`` 只认这个视图，不认 ORM 对象、也不认裸 payload：
    两种来源各自在 ``from_row`` / ``from_payload`` 里完成归一化，
    判定逻辑因此只有一份。
    """

    object_id: str = ""
    object_type: str = "doc"
    document_id: str = ""
    tenant_id: str = "default"
    owner_id: str | None = None
    user_id: str | None = None          # = owner_id 的 payload 侧副本
    department_id: str | None = None
    access_level: str | None = ACCESS_PRIVATE
    visibility_mode: str = VISIBILITY_MODE_TIER
    project_ids: frozenset[str] = frozenset()
    security_level: int | None = None
    parent_security_level: int | None = None
    effective_security_level: int | None = None
    acl_allow: frozenset[str] = frozenset()
    acl_deny: frozenset[str] = frozenset()
    acl_expires_at: datetime | None = None
    excluded: bool = False
    acl_sync_state: str = "synced"

    # ── 来源归一化（**唯一**的两处命名映射点）─────────────────────────────────
    @classmethod
    def from_row(cls, row: Any) -> "ObjectACLView":
        """PG 行 / ORM 对象 → 视图。缺列一律 getattr 兜底（老行没有新列）。"""
        get = (lambda k, d=None: getattr(row, k, d)) if not isinstance(row, Mapping) \
            else (lambda k, d=None: row.get(k, d))
        return cls(
            object_id=str(get("object_id") or ""),
            object_type=str(get("object_type") or "doc"),
            document_id=str(get("document_id") or ""),
            tenant_id=str(get("tenant_id") or "default"),
            owner_id=_opt_str(get("owner_id")),
            user_id=_opt_str(get("owner_id")),
            department_id=_opt_str(get("department_id")),
            access_level=get("access_level"),
            visibility_mode=str(get("visibility_mode") or VISIBILITY_MODE_TIER),
            project_ids=_as_frozenset(get("project_ids")),
            security_level=_opt_int(get("security_level")),
            parent_security_level=_opt_int(get("parent_security_level")),
            effective_security_level=_opt_int(get("effective_security_level")),
            acl_allow=_as_frozenset(get("acl_allow")),
            acl_deny=_as_frozenset(get("acl_deny")),
            acl_expires_at=get("acl_expires_at"),
            excluded=bool(get("excluded", False)),
            acl_sync_state=str(get("acl_sync_state") or "synced"),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "ObjectACLView":
        """
        Qdrant payload → 视图（**``user_id`` ⇄ ``owner_id`` 的唯一映射点**）.

        字段缺失一律按"最宽松但安全"的默认值：密级缺失 → ``None``（判定侧按
        ``strict`` 取 1 或 3），``visibility_mode`` 缺失 → ``tier``（存量行为），
        ``project_ids`` 缺失 → 空集。这与 Qdrant 侧「字段缺失 ⇒ 条件不匹配 ⇒
        不排除」的 fail-open 形态**逐格对齐**，所以 A2 等价性矩阵才成立。
        """
        payload = payload or {}
        owner = _opt_str(payload.get(P_USER_ID))
        return cls(
            object_id=str(payload.get(P_OBJECT_ID) or ""),
            object_type=str(payload.get(P_OBJECT_TYPE) or "doc"),
            document_id=str(payload.get(P_DOCUMENT_ID) or ""),
            tenant_id=str(payload.get(P_TENANT_ID) or "default"),
            owner_id=owner,
            user_id=owner,                       # ← 映射只在这里发生一次
            department_id=_opt_str(payload.get(P_DEPARTMENT_ID)),
            access_level=payload.get(P_ACCESS_LEVEL),
            visibility_mode=str(payload.get(P_VISIBILITY_MODE) or VISIBILITY_MODE_TIER),
            project_ids=_as_frozenset(payload.get(P_PROJECT_IDS)),
            security_level=_opt_int(payload.get(P_SECURITY_LEVEL)),
            parent_security_level=_opt_int(payload.get(P_PARENT_SECURITY_LEVEL)),
            effective_security_level=_opt_int(payload.get(P_EFFECTIVE_SECURITY_LEVEL)),
            acl_allow=_as_frozenset(payload.get(P_ACL_ALLOW)),
            acl_deny=_as_frozenset(payload.get(P_ACL_DENY)),
            acl_expires_at=_parse_dt(payload.get(P_ACL_EXPIRES_AT)),
            excluded=bool(payload.get(P_EXCLUDED, False)),
            acl_sync_state=str(payload.get("acl_sync_state") or "synced"),
        )

    def to_payload(self) -> dict[str, Any]:
        """视图 → 扁平 payload（测试与 payload 推送共用；字段名与 §4.6 一致）。"""
        return {
            P_OBJECT_ID: self.object_id,
            P_OBJECT_TYPE: self.object_type,
            P_DOCUMENT_ID: self.document_id,
            P_TENANT_ID: self.tenant_id,
            P_USER_ID: self.owner_id,
            P_DEPARTMENT_ID: self.department_id,
            P_ACCESS_LEVEL: self.access_level,
            P_VISIBILITY_MODE: self.visibility_mode,
            P_PROJECT_IDS: sorted(self.project_ids),
            P_SECURITY_LEVEL: self.security_level,
            P_PARENT_SECURITY_LEVEL: self.parent_security_level,
            P_EFFECTIVE_SECURITY_LEVEL: self.effective_security_level,
            P_ACL_ALLOW: sorted(self.acl_allow),
            P_ACL_DENY: sorted(self.acl_deny),
            P_ACL_EXPIRES_AT: _iso(self.acl_expires_at),
            P_ACL_EXPIRES_AT_TS: _epoch(self.acl_expires_at),
            P_EXCLUDED: self.excluded,
            "acl_sync_state": self.acl_sync_state,
        }


@dataclass(frozen=True)
class Decision:
    """一次判定的结果（带原因，供审计日志的 ``reason`` / ``gate`` 字段直接消费）。"""

    allowed: bool
    reason: str
    gate: str          # tenant|security|project|source|deny|ok


# ═══════════════════════════════════════════════════════════════════════════════
# UserScope → ScopePredicate
# ═══════════════════════════════════════════════════════════════════════════════


def build_predicate(scope: "UserScope", *, now: datetime | None = None) -> ScopePredicate:
    """
    把 ``UserScope`` 编译成 :class:`ScopePredicate`（三个编译器的唯一入口）.

    五维里的三维来自 ``scope.base``（既有 ``DocumentScope``，**一个字段都不改**），
    密级 / 项目 / principals 来自新增维度。``now`` 可注入，便于测试"已过期"。
    """
    return ScopePredicate(
        user_id=scope.user_id,
        tenant_ids=scope.base.tenant_ids,
        owns_tenant_ids=scope.base.owns_tenant_ids,
        department_id=scope.base.department_id,
        tenant_wide=scope.base.tenant_wide,
        clearance=scope.clearance,
        project_ids=scope.project_ids,
        principals=scope.principals,
        now=now or datetime.now(timezone.utc),
        unrestricted=False,
        strict=scope.strict,
    )


def clearance_for_role(role: str | None) -> int:
    """
    角色 → 默认密级（未知角色返回 :data:`CLEARANCE_ON_FAILURE`，最小权限）.

    ⚠️ 这不是"角色豁免"：它只是 ``users.clearance IS NULL`` 时的初值；
    判定式里没有任何基于 role 的放行分支。
    """
    if not role:
        return CLEARANCE_ON_FAILURE
    return DEFAULT_CLEARANCE_BY_ROLE.get(str(role).strip(), CLEARANCE_ON_FAILURE)


# ═══════════════════════════════════════════════════════════════════════════════
# 编译器 ③：Python 逐对象判定（第 11 / 12 环）
# ═══════════════════════════════════════════════════════════════════════════════


def allows(pred: ScopePredicate, obj: ObjectACLView | None) -> Decision:
    """
    单个对象是否对该 Scope 可见（纯函数，可单测）.

    判定顺序（**与 ``to_sql`` / ``to_qdrant`` 逐条同构**）：

        0. 对象视图缺失 → 拒（fail-closed）
        1. 硬闸门 A：租户（private 与租户无关 —— 沿用上游 I1 不变式）
        2. deny 一票否决（**优先于一切，含 admin 与 owner 本人**）+ excluded
        3. 硬闸门 B：密级（need-to-know 例外可越过，但必须未过期）
        4. 硬闸门 C：项目模式（visibility_mode=project 必须命中 project_ids）
        5. 范围来源（OR）：owner / dept / tenant / project / acl
    """
    # 0. fail-closed
    if pred is None or obj is None or not obj.object_id:
        return Decision(False, "missing_object_view", "tenant")

    # 1. 租户闸门
    if not _tenant_gate(pred, obj):
        return Decision(False, "tenant_mismatch", "tenant")

    # 2. deny 一票否决（含 owner 本人、含 admin —— 没有任何角色例外）
    if pred.principals & obj.acl_deny:
        return Decision(False, "acl_deny_hit", "deny")
    if obj.excluded:
        return Decision(False, "object_excluded", "deny")

    # 3. 密级闸门（need-to-know 例外必须未过期）
    eff = _effective_level(obj, pred)
    ntu_ok = _need_to_know_ok(pred, obj)
    if eff > pred.clearance and not ntu_ok:
        return Decision(
            False, f"clearance_short(eff={eff},clr={pred.clearance})", "security",
        )

    # 4. 项目模式闸门（与 SQL 的 project 条件、Qdrant 的 Deny-5 同源）
    if obj.visibility_mode == VISIBILITY_MODE_PROJECT and not (obj.project_ids & pred.project_ids):
        return Decision(False, "project_miss", "project")

    # 5. 范围来源（OR）
    if not _source_gate(pred, obj):
        return Decision(False, "no_source_hit", "source")

    return Decision(True, "ok", "ok")


def _tenant_gate(pred: ScopePredicate, obj: ObjectACLView) -> bool:
    """
    硬闸门 A：公司边界.

    ``private``（以及 NULL 老数据）**与租户无关** —— 这是上游 Rev2 的 I1 不变式：
    「owner=我、但 tenant 不在集合内」的个人库文档必须仍然可见。
    """
    level = _norm_level(obj.access_level)
    if level == ACCESS_PRIVATE:
        return True
    if pred.unrestricted:
        return True
    if pred.tenant_ids is None:      # 无公司上下文 → fail-closed（绝不因 falsy 放行）
        return False
    if not pred.tenant_ids:          # 空集 → fail-closed
        return False
    return obj.tenant_id in pred.tenant_ids


def _norm_level(level: str | None) -> str:
    """``access_level`` 归一化（NULL / 空 / 未知 → private，与 PG 侧同口径）。"""
    key = (level or "").strip().lower()
    if key in (ACCESS_PRIVATE, ACCESS_DEPARTMENT, ACCESS_TENANT):
        return key
    return ACCESS_PRIVATE


def _effective_level(obj: ObjectACLView, pred: ScopePredicate) -> int:
    """
    有效密级（决策 7）：**恒取 max**，缺失按 ``strict`` 取 3 或默认 1.

    为什么物化值也要再取一次 ``max``：派生对象只收紧不放宽是 PRD 3.2 的硬约束，
    "物化的 effective 被写错"也必须由判定侧兜住。
    """
    own = obj.security_level
    par = obj.parent_security_level
    eff = obj.effective_security_level
    if eff is None:
        vals = [v for v in (own, par) if v is not None]
        eff = max(vals) if vals else None
    if eff is None:
        return SECURITY_LEVEL_MAX if pred.strict else DEFAULT_SECURITY_LEVEL
    return max(int(eff), int(own or 0), int(par or 0))


def _not_expired(obj: ObjectACLView, now: datetime) -> bool:
    """``acl_expires_at`` 为空 = 长期有效（但每次命中都要审计）。"""
    if obj.acl_expires_at is None:
        return True
    return obj.acl_expires_at > now


def _need_to_know_ok(pred: ScopePredicate, obj: ObjectACLView) -> bool:
    """need-to-know 例外是否成立（命中 acl_allow 且未过期）。"""
    if not (pred.principals & obj.acl_allow):
        return False
    return _not_expired(obj, pred.now)


def _source_gate(pred: ScopePredicate, obj: ObjectACLView) -> bool:
    """
    范围来源（OR 五分支，决策 4）.

    ``project`` 是**第四个** OR 分支 —— 它只**增加**可见性，不替换 ``access_level``
    三值（已裁决 Q3：不动三值、不引入第四值）。存量文档 ``visibility_mode=tier``
    ⇒ 这一分支根本不参与判定 ⇒ **存量行为零变化**。

    ⚠️ 分支顺序**不可随意调整**，它必须与 :func:`_generic_scope_sql` 逐条同构
    （A2 等价性矩阵会抓任何分叉）：

        owner    —— 任何层级都放行（含 private）
        private  —— **只**认本人 / 自建公司集合；项目与 acl 都**不得**放开个人库
                    （产品红线 + B2/B6）
        tenant   —— 租户内全员
        department —— 同部门或宽口径角色
        project  —— 仅对非 private 生效
        acl      —— 仅对非 private 生效
    """
    level = _norm_level(obj.access_level)

    if obj.owner_id and pred.user_id and str(obj.owner_id) == str(pred.user_id):
        return True                                             # owner_hit
    if level == ACCESS_PRIVATE:
        # 个人库：仅本人（上面已判）+ 自建测试公司集合例外（B2，不对称设计不变）
        return bool(pred.owns_tenant_ids) and obj.tenant_id in pred.owns_tenant_ids
    if level == ACCESS_TENANT:
        return True                                             # tenant_hit
    if level == ACCESS_DEPARTMENT:
        if pred.tenant_wide:
            return True                                         # dept_hit（宽口径角色）
        if (
            obj.department_id is not None
            and pred.department_id is not None
            and obj.department_id == pred.department_id
        ):
            return True                                         # dept_hit（同部门）
    # 以下两支**只对非 private 生效**（与 SQL 的 not_(private) 合取同构）
    if obj.visibility_mode == VISIBILITY_MODE_PROJECT and (obj.project_ids & pred.project_ids):
        return True                                             # project_hit
    return _need_to_know_ok(pred, obj)                          # acl_hit


# ═══════════════════════════════════════════════════════════════════════════════
# 编译器 ①：PG SQL（第 7 环关键词腿 / 第 11 环）
# ═══════════════════════════════════════════════════════════════════════════════


def to_sql(pred: ScopePredicate, model: Any) -> ColumnElement:
    """
    编译成 SQLAlchemy 条件（``and_`` 合取）.

    * ``model`` 是 ``Document`` → 租户/ACL 部分**直接复用**既有的
      ``document_scope_clause``（既有三维一行不改），只在其上**追加**密级 / 项目 /
      deny / excluded 条件。
    * ``model`` 是其它模型（``document_objects`` 或 Table.c）→ 走
      :func:`_generic_scope_sql`，它是 :func:`_tenant_gate` + :func:`_source_gate`
      的 SQL 版（**逐条同构**，由 ``test_scope_filter_equivalence.py`` 守着）。

    Raises:
        ScopeCompileError: 模型缺列或类型不兼容 —— 调用方必须 fail-closed。
    """
    if pred is None:
        raise ScopeCompileError("to_sql: ScopePredicate 为 None（拒绝降级为不过滤）")
    try:
        if getattr(model, "__tablename__", None) == "documents":
            base = document_scope_clause(
                owner_id=_as_uuid(pred.user_id),
                department_id=pred.department_id,
                tenant_ids=pred.tenant_ids,
                owns_tenant_ids=pred.owns_tenant_ids,
                tenant_wide=pred.tenant_wide,
                unrestricted=pred.unrestricted,
            )
        else:
            base = _generic_scope_sql(pred, model)
        return and_(base, _security_sql(pred, model))
    except ScopeCompileError:
        raise
    except Exception as exc:      # noqa: BLE001 — 任何编译失败都不得静默放行
        raise ScopeCompileError(f"to_sql 编译失败: {exc}") from exc


def _generic_scope_sql(pred: ScopePredicate, model: Any) -> ColumnElement:
    """``_tenant_gate`` AND ``_source_gate`` 的 SQL 版（逐条同构）。"""
    _require(model, "tenant_id", "owner_id", "access_level", "department_id")
    level = model.access_level
    private = or_(level == ACCESS_PRIVATE, level.is_(None))

    # ── 租户闸门（与 _tenant_gate 逐条同构）──────────────────────────────────
    if pred.unrestricted:
        tenant_ok: ColumnElement = true()
    elif pred.tenant_ids is None or not pred.tenant_ids:
        tenant_ok = false()          # fail-closed（空集 / None 都全拒）
    else:
        tenant_ok = model.tenant_id.in_(sorted(pred.tenant_ids))

    # ── 来源闸门（与 _source_gate 逐条同构）──────────────────────────────────
    clauses: list[ColumnElement] = []
    if pred.user_id:
        clauses.append(model.owner_id == _coerce(model.owner_id, pred.user_id))
    if pred.owns_tenant_ids:
        clauses.append(
            and_(private, model.tenant_id.in_(sorted(pred.owns_tenant_ids)))
        )
    clauses.append(level == ACCESS_TENANT)
    if pred.tenant_wide:
        clauses.append(level == ACCESS_DEPARTMENT)
    elif pred.department_id:
        clauses.append(
            and_(level == ACCESS_DEPARTMENT,
                 model.department_id == pred.department_id)
        )
    # 项目 / acl 两支**只对非 private 生效** —— 个人库对任何人都不开放（产品红线
    # + B2/B6），need-to-know 也**不能**撬开别人的私库。
    if pred.project_ids:
        clauses.append(
            and_(not_(private),
                 model.visibility_mode == VISIBILITY_MODE_PROJECT,
                 _jsonb_overlaps(model.project_ids, pred.project_ids))
        )
    if pred.principals:
        clauses.append(
            and_(not_(private),
                 _jsonb_overlaps(model.acl_allow, pred.principals),
                 or_(model.acl_expires_at.is_(None),
                     model.acl_expires_at > pred.now))
        )
    source = or_(*clauses) if clauses else false()

    return and_(or_(private, and_(not_(private), tenant_ok)), source)


#: 决定"有效密级"的三个列（**顺序无关**，代码里对它们取 ``max``）。
#: 不同的被判定模型可用列不同：``documents`` 只有 ``security_level``（顶层文档
#: 没有父级、没有物化副本），``document_objects`` 三者齐全。
_LEVEL_COLUMNS: tuple[str, ...] = (
    "effective_security_level",
    "security_level",
    "parent_security_level",
)


def _effective_level_sql(model: Any, pred: ScopePredicate) -> ColumnElement:
    """
    :func:`_effective_level` 的 SQL 版（**逐条同构**）—— 密级闸门的唯一表达式.

    语义（与判定内核一字不差）：取 ``max(物化 effective, 自身, 父级)``，**忽略
    NULL**；三者全为 NULL 时按 ``pred.strict`` 取 ``SECURITY_LEVEL_MAX`` 或
    ``DEFAULT_SECURITY_LEVEL``。

    为什么用 ``greatest`` + ``coalesce`` 而不是直接比较 ``effective_security_level``：
    PG 里 ``NULL <= 1`` 求值为 **NULL（假）**，直接比较会把"存量无密级对象"在 SQL
    腿上整个排除，而 ``allows()`` 判它们可见（默认档位 1）—— 两路立刻分叉，
    且是**把存量文档一夜之间弄丢**的那一类分叉（B4 不可退化基线）。
    ``greatest`` 恰好**忽略 NULL**（全 NULL 才返回 NULL），与内核的
    "``[v for v in (...) if v is not None]`` 再取 max" 完全同构；外层的
    ``coalesce`` 补上"全 NULL ⇒ 默认档位"这一支。

    模型可用列逐个判断（``documents`` 缺 ``effective_security_level`` /
    ``parent_security_level``）：只对**实际存在**的列取 ``max``。一个密级列都没有
    ⇒ 抛 :class:`ScopeCompileError`（调用方 fail-closed），绝不静默放行。
    """
    cols = [c for c in (getattr(model, name, None) for name in _LEVEL_COLUMNS)
            if c is not None]
    if not cols:
        raise ScopeCompileError(
            f"模型 {getattr(model, '__tablename__', model)!r} 没有任何密级列 "
            f"{list(_LEVEL_COLUMNS)} —— 无法编译密级条件（拒绝降级为不过滤）"
        )
    inner: ColumnElement = cols[0] if len(cols) == 1 else sa.func.greatest(*cols)
    return sa.func.coalesce(
        inner,
        sa.literal(SECURITY_LEVEL_MAX if pred.strict else DEFAULT_SECURITY_LEVEL),
    )


def _security_sql(pred: ScopePredicate, model: Any) -> ColumnElement:
    """
    新增四条的 SQL 版：密级 / 剔除 / deny / 项目模式.

    ⚠️ 这里**不复用** ``document_scope_clause``，也不给它加 ``column`` 参数
    （避免改动既有签名）—— 这是设计文档 §5.2 明确认可的一处**受控重复**，
    由等价性测试守住。

    列存在性：``acl_allow`` / ``acl_deny`` / ``acl_expires_at`` /
    ``visibility_mode`` / ``project_ids`` 是这四条**必需**的列（缺失即抛）；
    ``excluded`` 只有对象级模型才有（``documents`` 表没有"剔除"这一列，缺列时
    该条退化为恒真），密级列由 :func:`_effective_level_sql` 逐个判断。
    """
    _require(model, "acl_allow", "acl_deny", "acl_expires_at",
             "visibility_mode", "project_ids")
    principals = sorted(pred.principals)
    level = _effective_level_sql(model, pred)

    conds: list[ColumnElement] = [
        # 密级：有效密级 ≤ clearance，或命中未过期的 need-to-know
        or_(
            level <= pred.clearance,
            and_(
                _jsonb_overlaps(model.acl_allow, principals),
                or_(model.acl_expires_at.is_(None),
                    model.acl_expires_at > pred.now),
            ),
        ),
    ]
    # 剔除（对所有人下线）—— 仅对象级模型有此列
    excluded_column = getattr(model, "excluded", None)
    if excluded_column is not None:
        conds.append(excluded_column.is_(False))
    conds.extend([
        # deny 一票否决（含 owner 本人、含 admin）
        not_(_jsonb_overlaps(model.acl_deny, principals)),
        # 项目模式：visibility_mode=project 时必须命中（tier 模式不受影响 ⇒ 存量零变化）
        or_(
            model.visibility_mode.is_(None),
            model.visibility_mode != VISIBILITY_MODE_PROJECT,
            _jsonb_overlaps(model.project_ids, pred.project_ids),
        ),
    ])
    return and_(*conds)


def _jsonb_overlaps(column: Any, values: Iterable[str]) -> ColumnElement:
    """
    JSONB 数组与给定集合是否相交（PG ``?|`` 操作符 + GIN）.

    空集合 → ``false()``（不是"不过滤"）：没有 principal 就不可能命中
    ``acl_allow``，项目集合为空也不可能命中 project_ids。
    """
    vals = sorted({str(v) for v in values})
    if not vals:
        return false()
    return column.op("?|")(sa.cast(vals, ARRAY(String)))


def _require(model: Any, *names: str) -> None:
    missing = [n for n in names if getattr(model, n, None) is None]
    if missing:
        raise ScopeCompileError(
            f"模型 {getattr(model, '__tablename__', model)!r} 缺少列 {missing} —— "
            "无法编译权限条件（拒绝降级为不过滤）"
        )


def _coerce(column: Any, value: str) -> Any:
    """按列的 Python 类型把字符串转成正确的比较值（UUID 列不能拿 str 比）。"""
    py = getattr(getattr(column, "type", None), "python_type", str)
    if py is uuid.UUID:
        try:
            return uuid.UUID(str(value))
        except (ValueError, AttributeError, TypeError):
            return value
    return str(value)


def _as_uuid(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 编译器 ②：Qdrant Filter（第 7 环向量腿 / 内存 BM25）
# ═══════════════════════════════════════════════════════════════════════════════


def to_qdrant(pred: ScopePredicate, *, strict_prefilter: bool | None = None) -> Any:
    """
    编译成 ``qmodels.Filter``（**deny-list 形态**，直接放进 ``Filter.must_not``）.

    沿用 ``retrieval_service._visibility_conditions`` 的既有形态（Deny-1/2/3
    一行不改地复用），在其上追加三条：

        Deny-4 密级（决策 3：**fail-open** —— 老向量缺字段不排除，交 PG 终判）
        Deny-5 项目：visibility_mode=project 且 project_ids 未命中
        Deny-6 剔除 / deny 一票否决

    Deny-4 为什么要带 need-to-know 例外：否则"clearance=1 但被授予 secret 文档"
    的用户在前置过滤就被排掉了，第 11 环根本没机会看到它。过期判定走
    ``acl_expires_at_ts``（数值）—— Qdrant 无法对 keyword 做时间比较。

    ``strict_prefilter=True``（设置项 ``ACL_SECURITY_PREFILTER_STRICT``）时
    额外排除**字段缺失**的对象 —— 只允许在回填完成、确认无 NULL 之后开启，
    否则存量文档一夜之间全部检索不到（不可退化基线 B4）。

    Raises:
        ScopeCompileError: qdrant 模型不可用或既有条件函数签名变更。
    """
    if pred is None:
        raise ScopeCompileError("to_qdrant: ScopePredicate 为 None（拒绝降级为不过滤）")
    try:
        from qdrant_client.http import models as qmodels
    except Exception as exc:      # noqa: BLE001
        raise ScopeCompileError(f"to_qdrant: qdrant_client 不可用: {exc}") from exc

    def _value(key: str, value: Any):
        return qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))

    def _any(key: str, values: Iterable[str]):
        vals = sorted({str(v) for v in values})
        if not vals:
            return None
        if len(vals) == 1:
            return _value(key, vals[0])
        return qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=vals))

    conds: list[Any] = []

    if pred.unrestricted:
        # 诊断路径：不排除任何东西。用一个**永不存在的 key** 表达"永不命中"，
        # 而不是 must_not=[] —— 空数组的语义在 Qdrant 各版本间并不统一。
        conds.append(_value("__security_unrestricted__", True))
        return qmodels.Filter(must_not=conds)

    # ── 既有 Deny-1/2/3（一行不改地复用，避免第三份实现）─────────────────────
    # 延迟导入：T3 之后 retrieval_service 会 import 本模块，模块级互相导入即成环。
    try:
        from app.services.retrieval_service import _visibility_conditions
    except Exception as exc:      # noqa: BLE001
        raise ScopeCompileError(
            f"to_qdrant: 无法复用 _visibility_conditions（既有三维条件是权威实现，"
            f"不得另写一份）: {exc}"
        ) from exc

    tenant_ids = pred.tenant_ids
    if tenant_ids is None:
        # 非 unrestricted 的 None = fail-closed：按空集下推，与 _tenant_gate 同口径
        # （_visibility_conditions 里 None 表示"诊断不加边界"，语义不同，故在此归一）
        tenant_ids = frozenset()
    conds.extend(
        _visibility_conditions(
            owner_id=pred.user_id,
            department_id=pred.department_id,
            tenant_wide=pred.tenant_wide,
            tenant_ids=tenant_ids,
            owns_tenant_ids=pred.owns_tenant_ids,
        )
    )

    # ── Deny-3（部门库）的 owner / 项目例外 —— 与 _source_gate 对齐 ──────────────
    # ``_visibility_conditions`` 的 Deny-3 只按 ``department_id`` 排除"非本部门"的
    # 部门库文档，**未**豁免「本人所有」与「命中的项目」；而判定内核 ``_source_gate``
    # 在 owner_hit / project_hit 两支里是**放行**的（owner 永远看得见自己创建的文档；
    # 项目成员可以跨部门看到命中项目的文档 —— 这正是项目维度的意义）。
    # 不补例外，就会出现"内核放行、向量前置却把它剪掉"的分叉（等价性矩阵
    # ``u2_employee-dept_*`` / ``project_member_alpha-project_hit``）。
    #
    # 这里**不重写** Deny-3 的判定，只在其 ``must_not`` 上追加两条"例外"：
    #   * ``user_id == 我``            → 本人所有，不排除（私有库的同等例外见 Deny-1）
    #   * ``visibility_mode == project`` → 交给 Deny-5 判项目命中，这里不越权排除
    # 效果是**只减少误排除**（不新增任何一条排除）。被豁免的对象本就是内核判定
    # 放行的，因此相对基准（``allows()``）是"持平"，**不构成放宽**。
    conds = [_patch_department_deny(cond, pred, _value, qmodels) for cond in conds]

    # ── Deny-4 密级（fail-open：字段缺失 ⇒ Range 不匹配 ⇒ 不排除）──────────────
    # 语义：排除「有效密级 > clearance」**且** need-to-know 不成立（未命中
    # acl_allow 或已过期）。有效密级 = max(物化 effective, 自身, 父级)（与
    # ``_effective_level`` 同构）—— 所以这里**必须对三个键都判**："任意一个 >
    # clearance" 即视为超密级。只判 ``effective_security_level`` 会在
    # 「物化的 effective 被写错（低于 max(自身, 父级)）」时放行，而判定内核取
    # ``max`` 会拒绝 —— 两路分叉（等价性矩阵 derived_materialized_wrong）。
    # 拆分成 should 的嵌套 Filter 表达"任一命中"，作为外层 must 的单个元素是合法的。
    over_clearance = qmodels.Filter(
        should=[
            qmodels.FieldCondition(
                key=P_SECURITY_LEVEL, range=qmodels.Range(gt=pred.clearance)),
            qmodels.FieldCondition(
                key=P_PARENT_SECURITY_LEVEL, range=qmodels.Range(gt=pred.clearance)),
            qmodels.FieldCondition(
                key=P_EFFECTIVE_SECURITY_LEVEL, range=qmodels.Range(gt=pred.clearance)),
        ]
    )
    acl_hit = _any(P_ACL_ALLOW, pred.principals)
    if acl_hit is None:
        conds.append(qmodels.Filter(must=[over_clearance]))
    else:
        conds.append(
            qmodels.Filter(
                must=[over_clearance],
                should=[
                    # (a) acl_allow 未命中（含字段缺失）
                    qmodels.Filter(must_not=[acl_hit]),
                    # (b) 已过期（acl_expires_at_ts <= now）
                    qmodels.FieldCondition(
                        key=P_ACL_EXPIRES_AT_TS,
                        range=qmodels.Range(lte=_epoch(pred.now)),
                    ),
                ],
            )
        )

    if strict_prefilter is None:
        strict_prefilter = _setting_bool("ACL_SECURITY_PREFILTER_STRICT", False)
    # Deny-4 是 fail-open 的：三个键**全缺**时都不匹配 ⇒ 不排除。但 ``allows()``
    # 把"三者全缺"的默认档位当作密级 —— 当「默认档位 > 用户 clearance」时
    # （clearance=0 且默认 1；或严格模式下默认 3），这类对象本应被拒。
    # 因此补一条"排除密级三键全缺的对象"，触发条件 ``deny_missing`` 与内核的
    # "全 None ⇒ 默认档位"一跳**严格同构**（**与 Deny-4 同源：三个键都要判**，
    # 且必须是 AND：只有三者全缺才落到默认档位）。
    # ``ACL_SECURITY_PREFILTER_STRICT`` 仍是**独立运维开关**（回填完成后手动开启，
    # 语义是"要求密级字段已回填"）；最终条件 = ``strict_prefilter OR deny_missing``。
    missing_level = SECURITY_LEVEL_MAX if pred.strict else DEFAULT_SECURITY_LEVEL
    deny_missing = missing_level > pred.clearance
    if strict_prefilter or deny_missing:
        conds.append(
            qmodels.Filter(
                must=[
                    qmodels.Filter(
                        should=[
                            qmodels.IsEmptyCondition(
                                is_empty=qmodels.PayloadField(key=key)),
                            qmodels.IsNullCondition(
                                is_null=qmodels.PayloadField(key=key)),
                        ]
                    )
                    for key in _LEVEL_COLUMNS
                ]
            )
        )

    # ── Deny-5 项目：project 模式未命中 ────────────────────────────────────────
    proj_any = _any(P_PROJECT_IDS, pred.project_ids)
    conds.append(
        qmodels.Filter(
            must=[_value(P_VISIBILITY_MODE, VISIBILITY_MODE_PROJECT)],
            # 用户不在任何项目里 → 排除全部 project 模式对象（fail-closed）
            must_not=[proj_any] if proj_any is not None else None,
        )
    )

    # ── Deny-6 剔除 / deny 一票否决 ────────────────────────────────────────────
    deny_any = _any(P_ACL_DENY, pred.principals)
    should = [_value(P_EXCLUDED, True)]
    if deny_any is not None:
        should.append(deny_any)
    conds.append(qmodels.Filter(should=should))

    return qmodels.Filter(must_not=conds)


def _never() -> Any:
    """一个永不命中的条件（字段不存在 ⇒ 不匹配 ⇒ 不排除）。"""
    from qdrant_client.http import models as qmodels

    return qmodels.FieldCondition(
        key="__security_never__", match=qmodels.MatchValue(value=True)
    )


def _is_department_deny(cond: Any) -> bool:
    """
    识别 ``_visibility_conditions`` 产出的**部门库 Deny**（Deny-3）.

    唯一定位形态：一个 ``Filter``，其 ``must`` 恰为
    ``[FieldCondition(key=access_level, match=MatchValue('department'))]``。
    其它 Deny（Deny-1 的 must 是 ``access_level=private``；Deny-2 的 must 为空或
    嵌套 Filter）都不满足，因此不会误伤。
    """
    from qdrant_client.http import models as qmodels

    if not isinstance(cond, qmodels.Filter):
        return False
    must = cond.must or []
    if len(must) != 1:
        return False
    only = must[0]
    if not isinstance(only, qmodels.FieldCondition) or only.key != P_ACCESS_LEVEL:
        return False
    match = only.match
    return isinstance(match, qmodels.MatchValue) and match.value == ACCESS_DEPARTMENT


def _patch_department_deny(cond: Any, pred: ScopePredicate, value_factory: Any,
                           qmodels: Any) -> Any:
    """
    给部门库 Deny-3 追加 ``must_not`` 例外（owner / project），见 :func:`to_qdrant`.

    * ``value_factory`` 是 ``to_qdrant`` 内部的 ``_value(key, value)`` 构造器；
    * ``must_not`` 里的条目"命中即**不**排除"，故追加 owner / project 命中即是例外；
    * 非部门库 Deny 原样返回（零改动）。

    只减少误排除，不新增排除；被豁免的对象都是 ``allows()`` 判定放行的。
    """
    if not _is_department_deny(cond):
        return cond
    exemptions = list(cond.must_not or [])
    if pred.user_id:
        exemptions.append(value_factory(P_USER_ID, str(pred.user_id)))
    exemptions.append(value_factory(P_VISIBILITY_MODE, VISIBILITY_MODE_PROJECT))
    return qmodels.Filter(must=list(cond.must or []), must_not=exemptions)


def qdrant_filter_matches(pred: ScopePredicate, payload: Mapping[str, Any]) -> bool:
    """
    【测试专用】纯 Python 解释 :func:`to_qdrant` 产出的 Filter（含缺失字段的
    fail-open 语义），返回 ``True`` = **未被排除**。

    它的存在是 A2 等价性断言的支点：``allows(view) == qdrant_filter_matches(payload)``
    在 §6.2 的矩阵里逐格相等。没有它，"两路同源"就只能靠人肉比对 Qdrant 的
    过滤语义。
    """
    flt = to_qdrant(pred)
    for cond in (getattr(flt, "must_not", None) or []):
        if _cond_matches(cond, payload):
            return False
    return True


def _cond_matches(node: Any, payload: Mapping[str, Any]) -> bool:
    """递归求值一个 Qdrant 条件（Filter / FieldCondition / IsEmpty / IsNull）。"""
    from qdrant_client.http import models as qmodels

    if isinstance(node, qmodels.Filter):
        if node.must is not None and not all(_cond_matches(c, payload) for c in node.must):
            return False
        if node.must_not is not None and any(_cond_matches(c, payload) for c in node.must_not):
            return False
        if node.should is not None:
            if not node.should:
                return False
            if not any(_cond_matches(c, payload) for c in node.should):
                return False
        return True

    if isinstance(node, qmodels.FieldCondition):
        return _field_matches(node, payload)

    if isinstance(node, qmodels.IsEmptyCondition):
        return node.is_empty.key not in payload

    if isinstance(node, qmodels.IsNullCondition):
        return payload.get(node.is_null.key, None) is None

    if isinstance(node, qmodels.HasIdCondition):  # pragma: no cover - 未使用
        return False

    raise ScopeCompileError(f"qdrant_filter_matches: 未支持的条件类型 {type(node)!r}")


def _field_matches(cond: Any, payload: Mapping[str, Any]) -> bool:
    """FieldCondition 求值：**字段缺失 ⇒ 不匹配**（fail-open 的落点）。"""
    from qdrant_client.http import models as qmodels

    key = cond.key
    if key not in payload:
        return False
    value = payload[key]

    if cond.match is not None:
        match = cond.match
        if isinstance(match, qmodels.MatchValue):
            return _value_matches(value, match.value)
        if isinstance(match, qmodels.MatchAny):
            return any(_value_matches(value, v) for v in (match.any or []))
        if isinstance(match, qmodels.MatchExcept):  # pragma: no cover - 未使用
            return not any(_value_matches(value, v) for v in (match.except_ or []))
        raise ScopeCompileError(f"未支持的 match 类型 {type(match)!r}")

    if cond.range is not None:
        rng = cond.range
        if value is None:
            return False
        try:
            if rng.gt is not None and not (value > rng.gt):
                return False
            if rng.gte is not None and not (value >= rng.gte):
                return False
            if rng.lt is not None and not (value < rng.lt):
                return False
            if rng.lte is not None and not (value <= rng.lte):
                return False
        except TypeError:
            return False
        return True

    return False      # pragma: no cover - geo/values_count 未使用


def _value_matches(actual: Any, expected: Any) -> bool:
    """标量相等；数组字段按"包含"判定（Qdrant 的 keyword array 语义）。"""
    if isinstance(actual, (list, tuple, set, frozenset)):
        return expected in actual or str(expected) in {str(a) for a in actual}
    if isinstance(actual, bool) or isinstance(expected, bool):
        return bool(actual) == bool(expected) and actual == expected or actual is expected
    return actual == expected or str(actual) == str(expected)


# ═══════════════════════════════════════════════════════════════════════════════
# 小工具
# ═══════════════════════════════════════════════════════════════════════════════


def _as_frozenset(values: Any) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, frozenset):
        return frozenset(str(v) for v in values)
    if isinstance(values, (set, list, tuple)):
        return frozenset(str(v) for v in values)
    return frozenset({str(values)})


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    """payload 里的 ISO 字符串 → datetime（无法解析按 None = 长期有效）。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _epoch(value: datetime | None) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _setting_bool(name: str, default: bool) -> bool:
    """
    读取布尔设置，失败回落 ``default``.

    ⚠️ 与 :func:`security_scope._strict_mode` 的处理**故意不同**，不要"统一"：
    本函数的唯一调用点是 ``ACL_SECURITY_PREFILTER_STRICT``（"字段缺失是否前置
    排除"），其 ``default=False`` 是**刻意**的——开启它要求回填完成、确认无 NULL，
    否则存量文档会一夜之间全部检索不到（设计文档标注的不可退化基线 B4）。
    因此这里保持"失败回落 default"的语义不变，**只把静默变成有痕迹**：
    读不到配置本身就是异常状态，必须能在日志里看见（T5 上线前修复）。
    """
    try:
        from app.config import get_settings

        return bool(getattr(get_settings(), name, default))
    except Exception:      # noqa: BLE001 — 语义上回落 default（见 docstring）
        logger.exception(
            "security_policy: 读取设置 %s 失败 —— 回落 default=%s", name, default
        )
        return default


#: 便捷别名（设计文档 §5.1 里写作 ``payload_to_view``）
payload_to_view = ObjectACLView.from_payload


__all__ = [
    "CLEARANCE_ON_FAILURE",
    "DEFAULT_CLEARANCE_BY_ROLE",
    "Decision",
    "ObjectACLView",
    "P_ACL_ALLOW",
    "P_ACL_DENY",
    "P_ACL_EXPIRES_AT",
    "P_ACL_EXPIRES_AT_TS",
    "P_EFFECTIVE_SECURITY_LEVEL",
    "P_EXCLUDED",
    "P_OBJECT_ID",
    "P_OBJECT_TYPE",
    "P_PROJECT_IDS",
    "P_USER_ID",
    "P_VISIBILITY_MODE",
    "ScopeCompileError",
    "ScopePredicate",
    "allows",
    "build_predicate",
    "clearance_for_role",
    "payload_to_view",
    "qdrant_filter_matches",
    "to_qdrant",
    "to_sql",
]
