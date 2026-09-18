"""
Multi-tenant isolation primitives（三层隔离的单一实现点）.

三层模型：

    第一层 Tenant Isolation
        每个用户属于一个 tenant（``users.tenant_id``，如 ``company_A``）。
        检索的**前置**过滤（Qdrant payload filter + BM25 语料 filter +
        PostgreSQL valid_docs 校验）都必须带 tenant_id —— 跨租户的向量
        在检索阶段就不可见，而不是 rerank 之后才剔除。

        **平台管理员（admin）是唯一的跨公司身份**：它没有"所属公司"，身份上
        记为「全平台」（``PLATFORM_SCOPE_LABEL``），tenant_id 过滤对它不生效，
        因此可以看到**所有公司**的部门库与公司库文档。但个人库（private）
        对任何人都不开放（包括平台管理员）—— 见第二层。

    第二层 Document ACL
        文档在租户内再按 ``access_level`` 细分可见性：
            private    — 仅上传者本人（**任何人都不例外**，含平台管理员）
            department — 同 department_id 的成员；知识库管理员/企业管理员/
                         平台管理员另行放宽（见下）
            tenant     — 租户内全员（公司库）；平台管理员跨租户可见
        旧数据（迁移前）access_level 为 NULL/'private'，保持"仅本人可见"
        的原语义，不会因为升级而意外共享。

        private 的不可绕过是**产品硬约束**：个人知识库不是"管理员也能看，
        只是普通同事看不到"，而是对所有人都私密。因此管理员失去对他人个人库
        的阅读与合规删除能力 —— 想要他人个人库的内容，只能由本人主动「申请
        共享」发布到部门/公司库。

    第三层 User/Conversation Isolation
        会话与消息按 conversation_id + tenant_id + user_id 隔离；
        图片落盘按 uploads/{tenant_id}/{document_id}/images/ 隔离；
        一切缓存键 = tenant_id + 权限上下文 + 原始键。

本模块只放纯函数与常量，不做 IO —— SQLAlchemy 查询条件的组装也在这里，
保证"谁能看见什么"只有一个地方可以改。
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass

from sqlalchemy import and_, or_

from app.db.models import Document
from app.db.user_models import User

# ── 租户与 ACL 常量 ──────────────────────────────────────────────────────────

# 历史数据（迁移前没有任何租户概念）统一归入该租户；单机/单公司部署下
# 所有用户都在这个租户里，行为与升级前完全一致。
DEFAULT_TENANT_ID = "default"

# 平台管理员（admin）的**展示身份**：它不属于任何一家公司，而是横跨全平台。
# 个人主页 / 成员列表 / 管理后台的「公司」一栏对 admin 都显示这个值 ——
# 显示 "default" 会让人误以为它属于 default 这家"公司"。
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
    """
    *target* 是否**严格高于** *current* 所在层级.

    判定「这一层还能不能申请」的唯一依据。申请共享这条链路的设计意图是
    "自己没有的权限，通过申请向上要"，所以只有向上的目标才成立。

    允许平级/向下申请的后果不是文案错误，而是**静默的可见性收缩**：一份已经
    发布到公司库的文档，申请人可以再提一份"申请共享到部门库"，批准后
    ``set_document_access_level`` 会把它真的降级成部门库 —— 对正在引用它的
    其他部门同事，文档无声消失。同一层级则纯属重复申请（"已在公司知识库中"）。
    """
    return ACCESS_ORDER.get(
        normalize_access_level(target), 0
    ) > ACCESS_ORDER.get(normalize_access_level(current), 0)


def is_downgrade(current: str | None, target: str | None) -> bool:
    """
    *target* 是否**低于** *current* 所在层级（可见性收缩）.

    与 :func:`is_upward_transition` 是一对，但用在不同位置：
    申请链路用前者（只能向上申请），**直接发布**链路用本函数 —— 发布不需要
    审批，所以"向下"在这里更危险：一次 PATCH 就能把公司库文档降成部门库，
    其他部门同事静默失去访问权，审计日志里却只是一条正常的"层级变更"。

    注意 ``private`` 不在拦截范围内：把文档**收回个人库**是归属人的正当操作
    （"已共享的成员将无法再检索到"是明示后果），属于刻意保留的能力，调用方
    需要自行放行。本函数只回答"层级是否变低"，不替调用方决定要不要拦。
    """
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
    value = (level or "").strip().lower()
    return value if value in VALID_ACCESS_LEVELS else ACCESS_PRIVATE


# 新上传文档的默认可见性：**个人知识库**（仅上传者本人）。
# 这正是图上"张三自己的个人知识库"的语义 —— 上传即私有，要共享必须显式
# 发布（有权限的角色）或走"申请共享"（普通员工）。默认私有可以彻底避免
# "新同事一登录就能看到全公司文档"这类越权观感。
# 需要恢复旧行为（上传即是公司库）时把该值改为 tenant 即可，无需改代码。
DEFAULT_DOCUMENT_ACCESS_LEVEL = ACCESS_PRIVATE

# tenant_id 允许字符：防目录穿越（它会出现在 uploads/{tenant_id}/ 路径里）
_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def normalize_tenant_id(tenant_id: str | None) -> str:
    """
    把任意输入归一化成安全的 tenant_id.

    None / 空串 / 非法字符 → DEFAULT_TENANT_ID。tenant_id 会拼进文件
    系统路径与 Qdrant payload filter，这里必须堵住 ``..``、斜杠等穿越字符。
    """
    value = (tenant_id or "").strip()
    if not value or not _TENANT_ID_RE.match(value):
        return DEFAULT_TENANT_ID
    return value


def company_id_from_name(name: str | None) -> str:
    """
    公司**名称** → 稳定的 tenant_id（企业身份验证表单的唯一映射点）.

    产品里用户填的是公司名称，中文、空格、全角符号都合法；而 tenant_id 会拼进
    ``uploads/{tenant_id}/`` 路径与 Qdrant payload filter，必须是安全标识符
    （见 ``normalize_tenant_id`` 的穿越防护）。这里做一层**确定性**映射：

        纯 ASCII 安全名（``company_a`` / ``acme``）  → 原样用作 tenant_id
        其余（``腾讯科技`` / ``ACME 中国``）          → ``c`` + sha1(name)[:12]

    确定性是关键：同一个名称永远得到同一个 ID，公司内成员才能落进同一租户
    互相可见；不同名称不会碰撞到同一租户（跨公司隔离的前提）。
    """
    raw = (name or "").strip()
    if not raw:
        return DEFAULT_TENANT_ID
    if _TENANT_ID_RE.match(raw):
        return normalize_tenant_id(raw)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"c{digest}"


def department_id_from_name(name: str | None) -> str | None:
    """
    部门**名称** → 稳定的 department_id（与 company_id_from_name 同一套规则）.

    ``documents.access_level = department`` 的可见性靠 department_id 相等判定，
    所以同一部门的成员必须映射到同一个 ID —— 规则与公司完全一致，只是前缀用
    ``d`` 以便在日志里一眼区分公司与部门。
    """
    raw = (name or "").strip()
    if not raw:
        return None
    if _TENANT_ID_RE.match(raw):
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"d{digest}"


def company_display_name(user: User | None) -> str:
    """
    用户所属公司的**展示名**.

    优先用身份验证时填写的原始名称（``users.company_name``）；Keycloak 联邦
    账号与老账号没有该字段时退回 tenant_id，保证界面上永远有字可显示。

    平台管理员（admin）例外：它没有所属公司，显示「全平台」——它的权限范围
    是所有公司，把它标成某一家公司（历史实现里是 ``default``）会让界面上
    出现"管理员属于 default 公司"这种自相矛盾的描述。
    """
    if user is None:
        return DEFAULT_TENANT_ID
    if is_platform_admin(user):
        return PLATFORM_SCOPE_LABEL
    name = (getattr(user, "company_name", None) or "").strip()
    return name or effective_tenant_id(user)


def is_platform_admin(user: User | None) -> bool:
    """
    是否平台管理员（跨公司身份）.

    判定入口只有一个：``User.is_admin``（role == "admin"）。做成函数是为了让
    "谁能跨公司"在所有模块里都能被 grep 到，而不是各自写 ``user.is_admin``
    之后再各自解释一遍语义。
    """
    return bool(user is not None and getattr(user, "is_admin", False))


def department_display_name(user: User | None) -> str | None:
    """用户所属部门的展示名（无部门时返回 None，个人主页显示「未设置」）。"""
    if user is None:
        return None
    name = (getattr(user, "department_name", None) or "").strip()
    return name or effective_department_id(user)


def effective_tenant_id(user: User | None) -> str:
    """用户所属租户；未设置（老账号）时归入 default 租户."""
    if user is None:
        return DEFAULT_TENANT_ID
    return normalize_tenant_id(getattr(user, "tenant_id", None))


def effective_department_id(user: User | None) -> str | None:
    if user is None:
        return None
    dept = getattr(user, "department_id", None)
    dept = (dept or "").strip() if isinstance(dept, str) else None
    return dept or None


# ── 第二层：Document ACL ─────────────────────────────────────────────────────

# 在**本租户内**可读全部部门库 / 公司库的角色（企业管理员、知识库管理员）：
# 审核共享申请、执行合规删除需要跨部门视角。
# 注意这里**不包含** admin —— 平台管理员的跨公司能力由 platform_wide 表达，
# 而且他们同样看不到别人的个人库。
# 该集合是权限矩阵 ``document.read.all`` 的唯一事实来源（permissions.py 引用它），
# 避免"能删但看不见"这类矩阵不自洽。
TENANT_WIDE_READER_ROLES = frozenset(
    {User.ROLE_COMPANY_ADMIN, User.ROLE_KB_ADMIN}
)


@dataclass(frozen=True)
class DocumentScope:
    """
    一个用户的**文档可见范围**（第一层 + 第二层 + 第三层个人库的合体）.

    这是"谁能看见什么"的唯一入参形态：所有列表 / 检索 / 预览 / 删除入口
    都先 ``scope_for(user)``，再把 scope 交给下面几个纯函数，不再各自拼
    owner/tenant/department 三件套（历史实现里正是这种散落的拼装导致
    "某条路径忘了带 tenant_id"这类越权）。

    字段语义：
        owner_id       个人库归属人。**恒为本人 id**（含平台管理员）——
                       为 None 表示"看不到任何个人库"。
        tenant_id      第一层过滤值；None = 不限制公司（仅平台管理员）。
        department_id  第二层部门条件；tenant_wide 为 True 时被忽略。
        tenant_wide    本租户内部门库全通（企业管理员 / 知识库管理员 / 平台管理员）。
        platform_wide  跨公司（仅平台管理员）。
    """

    owner_id: uuid.UUID | None
    tenant_id: str | None
    department_id: str | None
    tenant_wide: bool = False
    platform_wide: bool = False

    @property
    def cross_tenant(self) -> bool:
        """是否跨公司（决定是否施加第一层 tenant_id 过滤）。"""
        return self.platform_wide

    @property
    def label(self) -> str:
        """中文范围标注（界面「可见范围」文案与此同源）。"""
        if self.platform_wide:
            return PLATFORM_SCOPE_LABEL
        return self.tenant_id or DEFAULT_TENANT_ID


def scope_for(user: User | None) -> DocumentScope:
    """
    构造用户的文档可见范围（**每个请求只调用一次**，然后层层传递）.

        平台管理员 admin    → 跨公司；所有部门库/公司库；个人库仍只有自己的
        企业/知识库管理员    → 本公司；所有部门库/公司库；个人库只有自己的
        部门负责人/普通成员  → 本公司；本部门库 + 公司库 + 自己的个人库
    """
    if user is None:
        # 未登录：什么都不给（fail-closed）。调用方在鉴权层就已经拦下了。
        return DocumentScope(
            owner_id=None, tenant_id=None, department_id=None,
        )
    platform = is_platform_admin(user)
    return DocumentScope(
        # 个人库永远只属于本人：平台管理员不会因为"跨公司"而获得别人的个人库
        owner_id=user.id,
        tenant_id=None if platform else effective_tenant_id(user),
        department_id=None if platform else effective_department_id(user),
        tenant_wide=platform or user.role in TENANT_WIDE_READER_ROLES,
        platform_wide=platform,
    )


def document_acl_clause(
    *,
    owner_id: uuid.UUID | None,
    department_id: str | None,
    tenant_wide: bool = False,
    platform_wide: bool = False,
    read_all: bool | None = None,
):
    """
    组装"该用户在同一租户内还受什么 ACL 约束"的 SQLAlchemy 条件.

    规则（**个人库对任何人都不开放**）：
        access_level = private（或 NULL 老数据） → 仅 owner_id 本人
        access_level = department                → 同 department_id；
                                                   tenant_wide / platform_wide
                                                   时放开到任意部门
        access_level = tenant                    → 租户内全员

    *tenant_wide*（企业管理员 / 知识库管理员 / 平台管理员）只放宽**部门**维度，
    绝不放宽个人库 —— 这是产品要求"其他人的个人文档看不到"的落点。

    *read_all* 是 ``tenant_wide`` 的旧参数名，保留仅为兼容旧调用点。

    注意：租户过滤（第一层）不在这里 —— 它由调用方按 ``tenant_id`` 施加，
    只有平台管理员（platform_wide）才跳过。因此本函数永远不会跨公司。
    """
    if read_all is not None:
        tenant_wide = tenant_wide or read_all

    conds = [Document.access_level == ACCESS_TENANT]
    if tenant_wide or platform_wide:
        # 任意部门库（本租户内，跨公司由调用方的 tenant 条件兜住）
        conds.append(Document.access_level == ACCESS_DEPARTMENT)
    elif department_id:
        conds.append(
            and_(
                Document.access_level == ACCESS_DEPARTMENT,
                Document.department_id == department_id,
            )
        )
    if owner_id is not None:
        conds.append(
            and_(
                or_(
                    Document.access_level == ACCESS_PRIVATE,
                    Document.access_level.is_(None),  # 老数据按 private 处理
                ),
                Document.owner_id == owner_id,
            )
        )
    return or_(*conds)


def can_access_document(
    doc: Document,
    user: User,
    *,
    owner_id: uuid.UUID | None = None,
) -> bool:
    """
    单文档可见性判定（路由层用：图片回显 / 下载 / 详情 / 原文预览）.

    顺序与 ``document_acl_clause`` 完全一致，只是判定的是"这一份"：

        1. 个人库（private / NULL）→ 只有归属人本人；**平台管理员也不行**
        2. 公司库（tenant）        → 同公司全员；平台管理员跨公司可见
        3. 部门库（department）    → 同部门；知识库管理员/企业管理员/平台管理员全通

    跨公司一律 False（唯一例外是平台管理员的 tenant 级与 department 级）。
    """
    if user is None or doc is None:
        return False

    uid = owner_id if owner_id is not None else user.id
    level = normalize_access_level(getattr(doc, "access_level", None))
    platform = is_platform_admin(user)

    # ── 1. 个人库：仅归属人（管理员没有例外）──────────────────────────────
    if level == ACCESS_PRIVATE:
        doc_owner = getattr(doc, "owner_id", None)
        return doc_owner is not None and doc_owner == uid

    # ── 2/3. 部门库与公司库：先过第一层公司边界 ──────────────────────────
    if not platform:
        if normalize_tenant_id(getattr(doc, "tenant_id", None)) != effective_tenant_id(user):
            return False

    if level == ACCESS_TENANT:
        return True

    if level == ACCESS_DEPARTMENT:
        # 本租户内跨部门的宽口径角色（含平台管理员）
        if platform or user.role in TENANT_WIDE_READER_ROLES:
            return True
        dept = effective_department_id(user)
        return bool(dept) and dept == (getattr(doc, "department_id", None) or "").strip()

    return False


# ── 三层知识库：发布能力判定 ──────────────────────────────────────────────────

# 目标层级 → 需要的权限名（有权限＝可直接发布；无权限＝需要走共享申请）
PUBLISH_PERMISSION_BY_LEVEL: dict[str, str | None] = {
    ACCESS_PRIVATE: None,                      # 收回到个人库：永远是本人操作
    ACCESS_DEPARTMENT: "document.publish.department",
    ACCESS_TENANT: "document.publish.company",
}


def publish_requirement(level: str | None) -> str | None:
    """返回发布到该层级所需的权限名（None = 无需额外权限）。"""
    return PUBLISH_PERMISSION_BY_LEVEL.get(normalize_access_level(level))


# ── 删除他人的文档：范围判定 ──────────────────────────────────────────────────

# 可删除**本租户内**任意部门库 / 公司库文档（不含他人个人库）的角色
TENANT_WIDE_DELETER_ROLES = TENANT_WIDE_READER_ROLES
# 可删除本部门范围内他人文档的角色（部门负责人）
DEPARTMENT_WIDE_DELETER_ROLES = frozenset(
    {User.ROLE_DEPT_MANAGER, User.ROLE_MANAGER}
)


def delete_permission_for(doc: Document, user: User) -> tuple[bool, str]:
    """
    判定 *user* 能否删除 *doc*，返回 ``(allowed, 中文原因)``.

    删除权跟着**文档所在层级**走，而不是"谁上传谁说了算"：

        个人知识库(private)     仅文档归属人 —— 平台管理员也不例外
        部门知识库(department)  部门负责人本部门 / 知识库管理员 / 企业管理员
                                / 平台管理员（跨公司）
        公司知识库(tenant)      知识库管理员 / 企业管理员（本公司）
                                / 平台管理员（跨公司）

    为什么不做"归属人永远能删"：文档一旦发布到部门库/公司库，它已经是
    **组织资产** —— 同事的问答、报告都在引用它。作者离职或误操作一键删掉，
    别人只会在某天提问时突然发现依据没了。所以高层的删除权收归上级，
    归属人若确实要撤，走「申请删除」由上级裁决（与"申请共享"对称）。

    为什么管理员也删不了别人的个人库：写权限跟着**可读性**走。看不见的东西
    不该能被删 —— 否则"个人库绝对私密"就只剩下一句界面文案。

    跨公司一律按"不存在"返回（不泄漏存在性），调用方据此映射 404。
    """
    if user is None or doc is None:
        return False, "文档不存在或无权访问"

    platform = is_platform_admin(user)
    level = normalize_access_level(getattr(doc, "access_level", None))

    # ── 公司边界先行：非平台管理员跨公司一律按"不存在"处理（不泄漏存在性，
    #     个人库也一样 —— 否则"这份 UUID 是别的公司的私人文档"就成了探测信号）
    if not platform:
        if normalize_tenant_id(getattr(doc, "tenant_id", None)) != effective_tenant_id(user):
            return False, "文档不存在或无权访问"

    # ── 个人库：仅归属人（管理员没有例外）────────────────────────────────────
    if level == ACCESS_PRIVATE:
        if getattr(doc, "owner_id", None) is not None and doc.owner_id == user.id:
            return True, ""
        return (
            False,
            "个人知识库文档只有归属人本人可以操作，"
            "其他人（包括管理员）都无法查看或删除",
        )

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


def can_request_delete(doc: Document, user: User) -> bool:
    """
    能否为该文档提交「申请删除」（自己没有删除权、但看得见它）.

    与 ``delete_permission_for`` 互补：能直接删就不用申请；看不见的也不能申请
    （否则成了探测接口）。个人库文档只有归属人可见、且归属人可直接删，
    所以个人库永远不需要"申请删除"；能申请的只有部门库与公司库。
    """
    if user is None or doc is None:
        return False
    allowed, _ = delete_permission_for(doc, user)
    if allowed:
        return False
    if not can_access_document(doc, user):
        return False
    level = normalize_access_level(getattr(doc, "access_level", None))
    return level in (ACCESS_DEPARTMENT, ACCESS_TENANT)


# ── 第三层：缓存键隔离 ────────────────────────────────────────────────────────


def permission_context(user: User | None, *, owner_id: uuid.UUID | None = None) -> str:
    """
    用户权限上下文的稳定指纹（缓存键的一部分）.

    同租户、同部门、同 role 的两个用户权限视图一致，可以共享缓存；
    权限上下文任何一项变化（换部门 / 升降 role）指纹立即变化，旧缓存
    自然失效 —— 不会出现"权限收紧后还能读到旧缓存"的泄漏窗口。

    注意 private 文档按 owner 区分，因此 owner_id 也参与指纹。
    """
    if user is None:
        return "anonymous"
    uid = owner_id if owner_id is not None else user.id
    parts = [
        str(uid),
        user.role or "",
        effective_department_id(user) or "-",
    ]
    if is_platform_admin(user):
        # 平台管理员的可见范围与任何同 role 字符串的账号都不同（跨公司），
        # 显式打标，避免与"某公司里恰好也叫 admin 的角色"共用缓存。
        parts.append(PLATFORM_SCOPE_LABEL)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def scoped_cache_key(tenant_id: str | None, perm_context: str, raw_key: str) -> str:
    """
    缓存键 = tenant_id + 权限上下文 + 原始键（第三层隔离的缓存规则）.

    任何查询类缓存（BM25 语料、检索结果、改写结果…）都必须经过它，
    杜绝"用户 A 的查询结果缓存被用户 B 命中"。
    """
    return f"{normalize_tenant_id(tenant_id)}::{perm_context}::{raw_key}"
