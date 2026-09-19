"""
Multi-tenant isolation primitives（三层隔离的单一实现点）.

三层模型：

    第一层 Tenant Isolation
        每个用户属于一个租户（``users.tenant_id``，如 ``company_A``）。
        检索的**前置**过滤（Qdrant payload filter + BM25 语料 filter +
        PostgreSQL valid_docs 校验）都必须带租户 —— 跨租户的向量在检索阶段
        就不可见，而不是 rerank 之后才剔除。

        **Rev2 口径（唯一正确性锚点）**：``private / NULL`` 层级只按
        ``owner_id == 我`` 判定、**不参与租户过滤**；``department / tenant``
        层级才受租户集合约束。因此「自己上传、落在任意租户（含 ``default``）
        的个人库文档」永远可见。

        **平台管理员（admin）不再是「全库」**：它的可见范围收敛为
        「自己创建的测试公司集合」（``owns_tenant_ids``，按 ``created_by ==
        当前 admin id`` 锁定）。无自建公司时 fail-closed 返回空，绝不回退全平台。

    第二层 Document ACL
        文档在租户内再按 ``access_level`` 细分可见性：
            private    — 仅上传者本人；**唯一例外**是平台管理员在其
                         自建测试公司内可见他人个人库（``owns_tenant_ids``）
            department — 同 department_id 的成员；知识库管理员/企业管理员/
                         平台管理员另行放宽
            tenant     — 租户内全员（公司库）

    第三层 User/Conversation Isolation
        会话与消息按 conversation_id + tenant_id + user_id 隔离；
        一切缓存键 = 租户集合指纹 + 权限上下文 + 原始键。

本模块只放纯函数与常量，不做 IO —— SQLAlchemy 查询条件的组装也在这里，
保证"谁能看见什么"只有一个地方可以改。唯一的例外是 ``request_scope``：
它需要查注册表拿 admin 的自建公司集合，因此是 async。
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, replace

from sqlalchemy import and_, ColumnElement, false, or_, true

from app.db.models import Document
from app.db.user_models import User
from app.utils.logging import get_logger

logger = get_logger(__name__)

# ── 租户与 ACL 常量 ──────────────────────────────────────────────────────────

# 历史数据（迁移前没有任何租户概念）统一归入该租户；单机/单公司部署下
# 所有用户都在这个租户里，行为与升级前完全一致。
DEFAULT_TENANT_ID = "default"

# 平台管理员（admin）的**展示身份**：它不属于任何一家公司，而是横跨全平台。
PLATFORM_SCOPE_LABEL = "全平台"

ACCESS_PRIVATE = "private"        # 仅上传者        ——「个人知识库」
ACCESS_DEPARTMENT = "department"  # 同部门          ——「部门知识库」
ACCESS_TENANT = "tenant"          # 全租户（公司）  ——「公司知识库」

VALID_ACCESS_LEVELS = {ACCESS_PRIVATE, ACCESS_DEPARTMENT, ACCESS_TENANT}

# 三层知识库的中文标注（前端标签、申请文案、审核意见共用一份，杜绝各写一套）
ACCESS_LABELS: dict[str, str] = {
    ACCESS_PRIVATE: "个人",
    ACCESS_DEPARTMENT: "部门",
    ACCESS_TENANT: "公司",
}

# 层级 → 知识库中文名（用于"发布到部门库 / 公司库"这类动作文案）
ACCESS_SCOPE_NAMES: dict[str, str] = {
    ACCESS_PRIVATE: "个人知识库",
    ACCESS_DEPARTMENT: "部门知识库",
    ACCESS_TENANT: "公司知识库",
}

# 层级 → 中文字典序（列表排序用：个人 → 部门 → 公司）
ACCESS_ORDER: dict[str, int] = {
    ACCESS_PRIVATE: 0,
    ACCESS_DEPARTMENT: 1,
    ACCESS_TENANT: 2,
}


def is_upward_transition(current: str | None, target: str | None) -> bool:
    """*target* 是否**严格高于** *current* 所在层级."""
    return ACCESS_ORDER.get(
        normalize_access_level(target), 0
    ) > ACCESS_ORDER.get(normalize_access_level(current), 0)


def is_downgrade(current: str | None, target: str | None) -> bool:
    """*target* 是否**低于** *current* 所在层级（可见性收缩）."""
    return ACCESS_ORDER.get(
        normalize_access_level(target), 0
    ) < ACCESS_ORDER.get(normalize_access_level(current), 0)


def access_label(level: str | None) -> str:
    """把 access_level 归一化成中文层级标注（未知值按个人库处理）。"""
    value = (level or "").strip()
    if value not in ACCESS_LABELS:
        value = ACCESS_PRIVATE
    return ACCESS_LABELS[value]


def access_scope_name(level: str | None) -> str:
    value = (level or "").strip()
    if value not in ACCESS_SCOPE_NAMES:
        value = ACCESS_PRIVATE
    return ACCESS_SCOPE_NAMES[value]


def normalize_access_level(level: str | None) -> str:
    """把任意 access_level 归一化成三值之一；NULL/未知一律落入 private。"""
    value = (level or "").strip().lower()
    return value if value in VALID_ACCESS_LEVELS else ACCESS_PRIVATE


# 新上传文档的默认可见性：**个人知识库**（仅上传者本人）。
# 非 admin 上传行为不变；admin 上传的默认层级由 API 层在「决策 6」中收敛。
DEFAULT_DOCUMENT_ACCESS_LEVEL = ACCESS_PRIVATE

# tenant_id 允许字符：防目录穿越（它会出现在 uploads/{tenant_id}/ 路径里）
_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def normalize_tenant_id(tenant_id: str | None) -> str:
    """把任意输入归一化成安全的 tenant_id（None/空/非法 → DEFAULT_TENANT_ID）。"""
    value = (tenant_id or "").strip()
    if not value or not _TENANT_ID_RE.match(value):
        return DEFAULT_TENANT_ID
    return value


def company_id_from_name(name: str | None) -> str:
    """
    公司**名称** → 稳定的 tenant_id.

    ⚠️ **仅供一次性回填脚本使用**：业务新代码不再调用（名称→标识的权威源已换成
    注册表查表 ``company_registry.find_by_name``）。保留它是为了兼容历史数据的
    幂等回填，grep 到非脚本调用即视为缺陷。
    """
    raw = (name or "").strip()
    if not raw:
        return DEFAULT_TENANT_ID
    if _TENANT_ID_RE.match(raw):
        return normalize_tenant_id(raw)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"c{digest}"


def department_id_from_name(name: str | None) -> str | None:
    """部门**名称** → 稳定的 department_id（部门不是本轮的一等实体）。"""
    raw = (name or "").strip()
    if not raw:
        return None
    if _TENANT_ID_RE.match(raw):
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"d{digest}"


def company_display_name(user: User | None) -> str:
    """
    用户所属公司的**展示名**（优先 ``users.company_name``，退回 tenant_id）。

    平台管理员（admin）例外：它没有所属公司，显示「全平台」。
    """
    if user is None:
        return DEFAULT_TENANT_ID
    if is_platform_admin(user):
        return PLATFORM_SCOPE_LABEL
    name = (getattr(user, "company_name", None) or "").strip()
    return name or effective_tenant_id(user)


def is_platform_admin(user: User | None) -> bool:
    """是否平台管理员（跨公司身份）。判定入口只有一个：``User.is_admin``。"""
    return bool(user is not None and getattr(user, "is_admin", False))


def department_display_name(user: User | None) -> str | None:
    """用户所属部门的展示名（无部门时返回 None）。"""
    if user is None:
        return None
    name = (getattr(user, "department_name", None) or "").strip()
    return name or effective_department_id(user)


def effective_tenant_id(user: User | None) -> str:
    """用户所属租户；未设置（老账号）时归入 default 租户。"""
    if user is None:
        return DEFAULT_TENANT_ID
    return normalize_tenant_id(getattr(user, "tenant_id", None))


def effective_department_id(user: User | None) -> str | None:
    if user is None:
        return None
    dept = getattr(user, "department_id", None)
    dept = (dept or "").strip() if isinstance(dept, str) else None
    return dept or None


def home_tenant_id(user: User | None) -> str | None:
    """
    用户**所属公司** —— 第三层（会话 / 消息 / 评测记录）的**单一**归属键.

    它与文档可见性的 ``tenant_ids`` **集合**解耦：会话隔离只需要一个键。
    平台管理员（admin）没有归属公司，返回 ``None``（与改造前 ``scope.tenant_id``
    的取值完全一致，保证历史会话/评测记录的归属判定不变）。
    """
    if user is None or is_platform_admin(user):
        return None
    return effective_tenant_id(user)


# ── 第二层：Document ACL ─────────────────────────────────────────────────────

# 在**本租户内**可读全部部门库 / 公司库的角色（企业管理员、知识库管理员）。
# 注意这里**不包含** admin —— 平台管理员的能力由 ``tenant_ids`` +
# ``owns_tenant_ids`` 表达，且它同样看不到别公司/他人的个人库。
TENANT_WIDE_READER_ROLES = frozenset(
    {User.ROLE_COMPANY_ADMIN, User.ROLE_KB_ADMIN}
)


def _default_tenant_ids_for(user: User | None) -> frozenset[str]:
    """
    调用点未显式给 ``tenant_ids`` 时的**保守兜底**（比设计更严的一侧）.

    设计里 ``tenant_ids=None`` 只在 ``unrestricted`` 诊断路径表示「不限制」；
    但 ``can_access_document`` / ``delete_permission_for`` 这类单文档判定没有
    ``unrestricted`` 形参，一旦某个调用点漏传就会退化成「跨公司」。这里把
    ``None`` 当作「按用户身份推导」而不是「不限制」：

        admin   → 空集（fail-closed：不得因漏传而回退全平台）
        普通用户 → {自己租户}
    """
    if user is None:
        return frozenset()
    if is_platform_admin(user):
        return frozenset()
    return frozenset({effective_tenant_id(user)})


@dataclass(frozen=True)
class DocumentScope:
    """
    一个用户的**文档可见范围**（第一层 + 第二层 + 个人库的合体）.

    字段语义：
        owner_id         个人库归属人。**恒为本人 id**（含平台管理员）——
                         为 None 表示"看不到任何个人库"。
        tenant_ids       第一层公司过滤**集合**。``None`` = 不限制（仅
                         ``unrestricted`` 诊断）；空集 = fail-closed（返回空）；
                         非空 = ``tenant_id IN (...)``。普通用户 = `{自己公司}`；
                         平台管理员 = `{自建测试公司}`。
        owns_tenant_ids  「**可见他人 private 文档**」的租户集合。仅当 actor 是
                         该租户的**创建者**时非空 —— **只有平台管理员**，值 =
                         其自建测试公司集合。任何把它写成"角色是 admin 即可"
                         的写法都视为缺陷（P0-8）。
        department_id    第二层部门条件；``tenant_wide`` 为 True 时被忽略。
        tenant_wide      本租户内部门库全通（企业管理员 / 知识库管理员 / 平台管理员）。
    """

    owner_id: uuid.UUID | None
    tenant_ids: frozenset[str] | None
    owns_tenant_ids: frozenset[str] = frozenset()
    department_id: str | None = None
    tenant_wide: bool = False

    @property
    def cross_tenant(self) -> bool:
        """是否跨多个公司（决定"跨公司"展示与缓存细分）。"""
        return self.tenant_ids is not None and len(self.tenant_ids) > 1

    @property
    def label(self) -> str:
        """中文范围标注（界面「可见范围」文案与此同源）。"""
        if not self.tenant_ids:
            return PLATFORM_SCOPE_LABEL
        return "、".join(sorted(self.tenant_ids))

    def acl_kwargs(self) -> dict:
        """统一喂给 list / retrieve / keyword 的 kwargs，杜绝各调用点手拼。"""
        return {
            "owner_id": str(self.owner_id) if self.owner_id else None,
            "tenant_ids": self.tenant_ids,
            "owns_tenant_ids": self.owns_tenant_ids,
            "user_department_id": self.department_id,
            "tenant_wide": self.tenant_wide,
        }


def scope_for(
    user: User | None,
    *,
    owned_tenant_ids: frozenset[str] | None = None,
) -> DocumentScope:
    """
    构造用户的文档可见范围（纯函数）.

        平台管理员 admin  → tenant_ids = owns_tenant_ids = 自建测试公司集合
        企业/知识库管理员  → tenant_ids = {本公司}；本公司部门库全通
        部门负责人/普通成员 → tenant_ids = {本公司}；本部门库 + 公司库 + 自己的个人库

    admin 的自建集合需要查注册表，``scope_for`` 保持纯函数：调用方（``request_scope``）
    先把 ``owned_tenant_ids`` 解析好传进来。
    """
    if user is None:
        # 未登录：什么都不给（fail-closed）。调用方在鉴权层就已经拦下了。
        return DocumentScope(
            owner_id=None, tenant_ids=frozenset(),
            owns_tenant_ids=frozenset(), department_id=None, tenant_wide=False,
        )
    if is_platform_admin(user):
        owned = frozenset(owned_tenant_ids or ())
        return DocumentScope(
            owner_id=user.id,
            tenant_ids=owned,
            owns_tenant_ids=owned,
            department_id=None,
            tenant_wide=True,
        )
    return DocumentScope(
        owner_id=user.id,
        tenant_ids=frozenset({effective_tenant_id(user)}),
        owns_tenant_ids=frozenset(),
        department_id=effective_department_id(user),
        tenant_wide=user.role in TENANT_WIDE_READER_ROLES,
    )


async def request_scope(user: User | None) -> DocumentScope:
    """
    ``scope_for`` 的 async 版本：**平台管理员**额外查注册表拿自建公司集合.

    这是 admin 的 ``tenant_ids`` / ``owns_tenant_ids`` 的**唯一来源**
    （= ``tenant_ids_created_by(admin.id)``）。普通用户直接走 ``scope_for``。
    """
    if user is None:
        return scope_for(None)
    if is_platform_admin(user):
        from app.services.company_registry import tenant_ids_created_by

        owned = await tenant_ids_created_by(getattr(user, "id", None))
        return scope_for(user, owned_tenant_ids=owned)
    return scope_for(user)


# ── 内容消费范围：测试公司隔离（唯一实现点）────────────────────────────────────

async def exclude_test_tenants(
    tenant_ids: frozenset[str] | None,
    owns_tenant_ids: frozenset[str],
) -> tuple[frozenset[str] | None, frozenset[str]]:
    """
    「内容消费排除测试公司」的**唯一**规则实现（供 ``content_scope`` 与
    ``retrieval_service`` 共用，禁止两处各写一遍）.

    规则（用户口径）：**测试公司文档只有平台管理员在列表/管理中能看到，但任何
    人都检索不到、也不进任何内容消费（摘要 / 文档关联 / 对话内文档列表）**。

    触发条件 = 调用者是平台管理员 —— 用 ``owns_tenant_ids`` 非空判定：本系统的
    租户模型里，只有平台管理员才可能拥有非空 ``owns_tenant_ids``（= 其自建测试
    公司集合，见 ``scope_for``），非 admin（含测试公司自己的成员）恒为空集。
    因此**测试公司成员不会被排除**（其 ``tenant_ids`` = {本公司} 予以保留），
    满足「测试账号照旧可用」。

    剔除后 admin 的 ``tenant_ids`` 变空集 —— 不影响其**个人库**（个人库分支只看
    ``owner_id == 我``、与租户无关）。

    **Fail-closed（安全底线）**：注册表查询失败时**绝不原样放行** —— 否则一次 DB
    抖动就能让 admin 检索/摘要到测试公司文档。此情形下对 admin 丢弃**全部公司
    租户**（``tenant_ids`` / ``owns_tenant_ids`` 均置空），只保留其个人库；
    非 admin 因 ``owns_tenant_ids`` 恒为空、根本不会走到这里，故不受影响。
    失败记 ``logger.exception`` 便于观测。
    """
    if not owns_tenant_ids:
        return tenant_ids, owns_tenant_ids
    try:
        from app.services.company_registry import test_tenant_ids

        test_ids = await test_tenant_ids()
    except Exception:      # noqa: BLE001 — 注册表不可用不应让请求整体失败
        # fail-closed：无法判定哪些是测试公司时，宁可对 admin 收窄到"仅个人库"，
        # 也不放行未排除的公司范围（安全 > 可用；集中一处、只影响 admin 身份）。
        logger.exception(
            "exclude_test_tenants: failed to resolve test tenant ids — failing "
            "closed (clearing admin company scope to empty)"
        )
        return frozenset(), frozenset()
    if not test_ids:
        return tenant_ids, owns_tenant_ids
    new_tenants = tenant_ids - test_ids if tenant_ids is not None else None
    return new_tenants, owns_tenant_ids - test_ids


async def content_scope(user: User | None) -> DocumentScope:
    """
    **内容消费**路径的文档可见范围（检索 / 摘要 / 文档关联 / 对话内文档列表）.

    = :func:`request_scope` 之后再按 :func:`exclude_test_tenants` 剔除测试公司
    （仅平台管理员生效）。**唯一实现点** —— 所有内容消费的 scope 都从这里派生，
    杜绝「每个调用点各排除一次」的散落实现。

    与 :func:`request_scope` 的分工：

        content_scope(user)  → 检索 / 摘要 / 关联 / 对话内列表（内容外泄面）
        request_scope(user)  → 管理 / 列表端点（``GET /documents`` 等）：
                               admin **必须仍能看到**测试公司文档以管理它们
    """
    scope = await request_scope(user)
    new_tenants, new_owns = await exclude_test_tenants(
        scope.tenant_ids, scope.owns_tenant_ids
    )
    if new_tenants is scope.tenant_ids and new_owns is scope.owns_tenant_ids:
        return scope
    return replace(scope, tenant_ids=new_tenants, owns_tenant_ids=new_owns)


# ── 第一层：SQL 组装（唯一入口）──────────────────────────────────────────────


def tenant_clause(
    tenant_ids: frozenset[str] | None,
    *,
    column: ColumnElement = Document.tenant_id,
    unrestricted: bool = False,
) -> ColumnElement:
    """
    第一层公司过滤条件（三分支，**绝不写 `if tenant_ids:`** —— frozenset() 是 falsy）.

        tenant_ids is None 且 unrestricted=True  → true()   # 仅诊断脚本「全库」
        tenant_ids is None 且 unrestricted=False → false()  # 无公司上下文：fail-closed
        frozenset()（空集）                       → false()  # fail-closed
        非空 frozenset                           → column.in_(sorted(tenant_ids))
    """
    if tenant_ids is None:
        return true() if unrestricted else false()
    if not tenant_ids:                     # 空集（注意不是 `is None`）
        return false()
    return column.in_(sorted(tenant_ids))


def document_acl_clause(
    *,
    owner_id: uuid.UUID | None,
    department_id: str | None,
    tenant_wide: bool = False,
    owns_tenant_ids: frozenset[str] = frozenset(),
    read_all: bool | None = None,
) -> ColumnElement:
    """
    组装"该用户在同一租户内还受什么 ACL 约束"的 SQLAlchemy 条件.

    规则：
        access_level = private（或 NULL 老数据） → owner_id 本人；
                                                   ``owns_tenant_ids`` 非空时追加
                                                   「tenant ∈ owns」（admin 例外）
        access_level = department                → 同 department_id；
                                                   ``tenant_wide`` 时放开到任意部门
        access_level = tenant                    → 租户内全员

    ``read_all`` 是 ``tenant_wide`` 的旧参数名，保留仅为兼容旧调用点。

    注意：租户过滤（第一层）不在这里 —— 见 :func:`document_scope_clause`。
    """
    if read_all is not None:
        tenant_wide = tenant_wide or read_all

    conds = [Document.access_level == ACCESS_TENANT]
    if tenant_wide:
        # 任意部门库（本租户内，跨公司由租户条件兜住）
        conds.append(Document.access_level == ACCESS_DEPARTMENT)
    elif department_id:
        conds.append(
            and_(
                Document.access_level == ACCESS_DEPARTMENT,
                Document.department_id == department_id,
            )
        )
    if owner_id is not None:
        personal_terms = [Document.owner_id == owner_id]
        if owns_tenant_ids:
            personal_terms.append(
                Document.tenant_id.in_(sorted(owns_tenant_ids))
            )
        conds.append(
            and_(
                or_(
                    Document.access_level == ACCESS_PRIVATE,
                    Document.access_level.is_(None),   # 老数据按 private 处理
                ),
                or_(*personal_terms),
            )
        )
    return or_(*conds)


def document_scope_clause(
    *,
    owner_id: uuid.UUID | None,
    department_id: str | None,
    tenant_ids: frozenset[str] | None,
    owns_tenant_ids: frozenset[str] = frozenset(),
    tenant_wide: bool = False,
    unrestricted: bool = False,
) -> ColumnElement:
    """
    列表 / 检索 / 关键词腿 / 摘要 / DB 兜底校验的**唯一** SQL 组装点.

    可见集 =  ① 公司边界内（tenant ∈ 集合 且 通过 ACL）
             ∪ ② 自己的个人库（owner == 我，**与租户无关**）

    ② 是 Rev2 的核心修复：把「个人库」从 ① 的合取里**拿出来**做 ``or_``，
    否则「owner=我、但 tenant 不在集合内」的个人库文档（如 admin 落在
    ``default`` 的私库）会被第一层直接滤掉。
    """
    # ① 公司边界：租户集合 ∧ ACL
    company_bound = and_(
        tenant_clause(tenant_ids, unrestricted=unrestricted),
        document_acl_clause(
            owner_id=owner_id,
            department_id=department_id,
            tenant_wide=tenant_wide,
            owns_tenant_ids=owns_tenant_ids,
        ),
    )

    clauses: list[ColumnElement] = [company_bound]

    # ② 个人库：private / NULL，只认归属人（+ admin 在自建集合内的例外），不进 tenant_clause
    if owner_id is not None:
        personal_terms = [Document.owner_id == owner_id]
        if owns_tenant_ids:
            personal_terms.append(
                Document.tenant_id.in_(sorted(owns_tenant_ids))
            )
        clauses.append(
            and_(
                or_(
                    Document.access_level == ACCESS_PRIVATE,
                    Document.access_level.is_(None),
                ),
                or_(*personal_terms),
            )
        )

    return or_(*clauses) if len(clauses) > 1 else clauses[0]


# ── 单文档判定 ───────────────────────────────────────────────────────────────


def can_access_document(
    doc: Document,
    user: User,
    *,
    owner_id: uuid.UUID | None = None,
    tenant_ids: frozenset[str] | None = None,
    owns_tenant_ids: frozenset[str] = frozenset(),
) -> bool:
    """
    单文档可见性判定（图片回显 / 下载 / 详情 / 原文预览 / 发布）.

    判定顺序（Rev2）：
        ① 个人库（private / NULL）→ 只认归属人本人；``owns_tenant_ids`` 例外
           （仅 admin 在其自建测试公司内）——**先于公司边界返回**
        ② 公司边界（tenant ∈ 集合）——**只作用于非个人库**
        ③ 层级细粒度（tenant → 全员；department → 同部门 / 宽口径角色）

    ``tenant_ids=None`` 视为「按调用者身份推导」（见 ``_default_tenant_ids_for``），
    绝不因漏传而回退全平台。
    """
    if user is None or doc is None:
        return False

    uid = owner_id if owner_id is not None else user.id
    level = normalize_access_level(getattr(doc, "access_level", None))

    # ── ① 个人库 / NULL：只认归属人（+ 自建集合例外）——与租户无关，最先返回 ──
    if level == ACCESS_PRIVATE:
        if uid is not None and getattr(doc, "owner_id", None) == uid:
            return True
        if owns_tenant_ids and normalize_tenant_id(getattr(doc, "tenant_id", None)) in owns_tenant_ids:
            return True
        return False

    # ── ② 公司边界（只对 department / tenant 层级）──────────────────────────────
    boundary = tenant_ids if tenant_ids is not None else _default_tenant_ids_for(user)
    if normalize_tenant_id(getattr(doc, "tenant_id", None)) not in boundary:
        return False

    # ── ③ 层级细粒度（语义不变）────────────────────────────────────────────────
    if level == ACCESS_TENANT:
        return True
    if level == ACCESS_DEPARTMENT:
        if is_platform_admin(user) or user.role in TENANT_WIDE_READER_ROLES:
            return True
        dept = effective_department_id(user)
        return bool(dept) and dept == (getattr(doc, "department_id", None) or "").strip()
    return False


# ── 三层知识库：发布能力判定 ──────────────────────────────────────────────────

PUBLISH_PERMISSION_BY_LEVEL: dict[str, str | None] = {
    ACCESS_PRIVATE: None,                      # 收回到个人库：永远是本人操作
    ACCESS_DEPARTMENT: "document.publish.department",
    ACCESS_TENANT: "document.publish.company",
}


def publish_requirement(level: str | None) -> str | None:
    """返回发布到该层级所需的权限名（None = 无需额外权限）。"""
    return PUBLISH_PERMISSION_BY_LEVEL.get(normalize_access_level(level))


# ── 删除他人的文档：范围判定 ──────────────────────────────────────────────────

TENANT_WIDE_DELETER_ROLES = TENANT_WIDE_READER_ROLES
DEPARTMENT_WIDE_DELETER_ROLES = frozenset(
    {User.ROLE_DEPT_MANAGER, User.ROLE_MANAGER}
)

# 个人库无权限删除时的统一文案（管理员也不例外）
_PRIVATE_DENY = (
    "个人知识库文档只有归属人本人可以操作，"
    "其他人（包括管理员）都无法查看或删除"
)


def delete_permission_for(
    doc: Document,
    user: User,
    *,
    tenant_ids: frozenset[str] | None = None,
    owns_tenant_ids: frozenset[str] = frozenset(),
) -> tuple[bool, str]:
    """
    判定 *user* 能否删除 *doc*，返回 ``(allowed, 中文原因)``.

    删除权跟着**文档所在层级**走：

        个人知识库(private)     仅文档归属人 —— 平台管理员也**不例外**
                                （team-lead 裁决 10-A：admin 对自建测试公司内
                                 他人私库**可读、不可删**）
        部门知识库(department)  部门负责人本部门 / 知识库管理员 / 企业管理员 /
                                平台管理员（自建测试公司内）
        公司知识库(tenant)      知识库管理员 / 企业管理员（本公司）/
                                平台管理员（自建测试公司内）

    ``private / NULL`` 分支刻意提到**公司边界之前**：否则 admin 连自己
    ``default`` 租户里的 private 都删不了（回归）；但边界信息仍用于
    「跨公司一律按不存在返回」（不泄漏存在性）。
    """
    if user is None or doc is None:
        return False, "文档不存在或无权访问"

    level = normalize_access_level(getattr(doc, "access_level", None))

    # ── ① 个人库：仅归属人（+ 无 owns 例外）——公司边界之前 ──────────────────────
    if level == ACCESS_PRIVATE:
        if getattr(doc, "owner_id", None) is not None and doc.owner_id == user.id:
            return True, ""
        # 非归属人：若是跨公司文档，按「不存在」返回（不泄漏存在性）；
        # 否则给出个人库专属文案（与旧行为一致）。
        boundary = tenant_ids if tenant_ids is not None else _default_tenant_ids_for(user)
        if normalize_tenant_id(getattr(doc, "tenant_id", None)) not in boundary:
            return False, "文档不存在或无权访问"
        return False, _PRIVATE_DENY

    # ── ② 公司边界（不泄漏存在性）──────────────────────────────────────────────
    boundary = tenant_ids if tenant_ids is not None else _default_tenant_ids_for(user)
    if normalize_tenant_id(getattr(doc, "tenant_id", None)) not in boundary:
        return False, "文档不存在或无权访问"

    platform = is_platform_admin(user)

    # ── ③ 层级删除权 ───────────────────────────────────────────────────────────
    if level == ACCESS_TENANT:
        if platform or user.role in TENANT_WIDE_DELETER_ROLES:
            return True, ""
        return (
            False,
            "公司知识库文档由知识库管理员或企业管理员删除。"
            "如确需删除，请在文档上提交「申请删除」，由上级审核",
        )

    # 部门知识库
    if platform or user.role in TENANT_WIDE_DELETER_ROLES:
        return True, ""
    if user.role in DEPARTMENT_WIDE_DELETER_ROLES:
        dept = effective_department_id(user)
        if not dept:
            return False, "你尚未归属任何部门，无法删除部门知识库文档"
        doc_dept = (getattr(doc, "department_id", None) or "").strip()
        if doc_dept and doc_dept == dept:
            return True, ""
        return (
            False,
            "部门负责人仅可删除本部门范围内的文档；该文档属于其他部门",
        )
    return (
        False,
        "部门知识库文档由部门负责人及以上权限删除。"
        "如确需删除，请在文档上提交「申请删除」，由部门负责人审核",
    )


def can_request_delete(
    doc: Document,
    user: User,
    *,
    tenant_ids: frozenset[str] | None = None,
    owns_tenant_ids: frozenset[str] = frozenset(),
) -> bool:
    """
    能否为该文档提交「申请删除」（自己没有删除权、但看得见它）.

    与 ``delete_permission_for`` 互补；个人库永远不需要"申请删除"
    （个人库只有归属人可见、且归属人可直接删）。
    """
    if user is None or doc is None:
        return False
    allowed, _ = delete_permission_for(
        doc, user, tenant_ids=tenant_ids, owns_tenant_ids=owns_tenant_ids
    )
    if allowed:
        return False
    if not can_access_document(
        doc, user, tenant_ids=tenant_ids, owns_tenant_ids=owns_tenant_ids
    ):
        return False
    level = normalize_access_level(getattr(doc, "access_level", None))
    return level in (ACCESS_DEPARTMENT, ACCESS_TENANT)


# ── 第三层：缓存键隔离 ────────────────────────────────────────────────────────


def tenant_scope_fingerprint(tenant_ids: frozenset[str] | None) -> str:
    """
    租户集合的稳定指纹（缓存键的一部分）.

    ``None``（诊断「全库」）/ 空集（fail-closed）/ 非空集合 **三态互异** ——
    这正是「同 owner、不同租户集合」必须产生不同缓存键的落点。
    """
    if tenant_ids is None:
        return "all"
    if not tenant_ids:
        return "none"
    joined = ",".join(sorted(tenant_ids))
    return "s:" + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def permission_context(
    user: User | None,
    *,
    owner_id: uuid.UUID | None = None,
    tenant_ids: frozenset[str] | None = None,
    owns_tenant_ids: frozenset[str] | None = None,
) -> str:
    """
    用户权限上下文的稳定指纹（缓存键的一部分）.

    ``T=`` / ``O=`` 双指纹：租户集合同样参与 —— 「同 owner、异租户集合」必然
    不同键；换租户集合（新建/删除测试公司）→ 指纹变 → 旧缓存自然失效；而
    **改名不改 tenant_ids ⇒ 指纹不变 ⇒ 缓存不失效**（正确）。
    """
    if user is None:
        return "anonymous"
    uid = owner_id if owner_id is not None else user.id
    parts = [
        str(uid),
        user.role or "",
        effective_department_id(user) or "-",
        "T=" + tenant_scope_fingerprint(tenant_ids),
        "O=" + tenant_scope_fingerprint(frozenset(owns_tenant_ids or ())),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def scoped_cache_key(
    tenant_scope: frozenset[str] | None,
    perm_context: str,
    raw_key: str,
) -> str:
    """
    缓存键 = 租户集合指纹 + 权限上下文 + 原始键（第三层隔离的缓存规则）.

    任何查询类缓存（BM25 语料、检索结果、改写结果…）都必须经过它。
    """
    return f"{tenant_scope_fingerprint(tenant_scope)}::{perm_context}::{raw_key}"
