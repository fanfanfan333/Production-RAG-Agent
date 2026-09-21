"""
安全隔离（五维权限模型）的 4 张新表 ORM + ``object_id`` 构造的**唯一决策点**.

本模块是 T1（数据层基础设施）的落点之一，对应
``docs/system_design_security_isolation.md`` §4.1 / §4.3 / §4.4 / §4.5。

四张表
──────
    document_objects  五种对象（doc / text_chunk / table / code / image）的
                      **统一权限视图**。PG 是权威源，Qdrant payload 只是副本。
    projects          项目维度的最小实体（P0，裁决：建最小表）
    project_members   项目成员（带 ``expires_at``，P1-2 临时成员）
    acl_grants        need-to-know 授予的**权威源**（表在 T1 建、功能在 T5）

``object_id`` 的唯一决策点
─────────────────────────
``object_id`` 的形态取决于三个尚未取证的事实：

    (a) ``image_id`` 是否全局唯一
    (b) Qdrant payload 里 ``content_type`` 的实际取值集
    (c) 同一个 ``image_id`` 是否产出多个向量块

**这三项的每一处影响都收敛在 :func:`make_object_id` 一个函数里**（以及它上面的
:data:`OBJECT_ID_MODE` 常量）。QA 取证回来后只需要改这一个常量并重跑
``scripts/backfill_security_level.py``，其余代码零改动：

    OBJECT_ID_MODE = "scoped"  →  ``"{document_id}::{raw}"``（当前默认，最保守：
                                   即使 image_id 只在文档内唯一也不会撞车）
    OBJECT_ID_MODE = "raw"     →  ``"{raw}"``（QA 确认 image_id 全局唯一且
                                   一图一块之后才可切换）

为什么不给 ``documents`` 直接加列就算了：``text_chunk / table / code / image``
这四类对象**只存在于 Qdrant payload**，PG 里装不下它们的权限（见设计文档决策 5）。
而本项目用 ``Base.metadata.create_all`` 建表 —— 它只建不存在的表、不给已存在的
表加列 —— 所以新表走 create_all 即零操作，与 alembic 迁移（IF NOT EXISTS）并存。

⚠️ 命名口径（共享知识 15）：PG 权威列一律叫 ``owner_id``；Qdrant payload 里
所有者的字段名叫 ``user_id``（历史遗留、语义相同）。**不要再发明第三套命名**，
映射只写在 ``security_policy.ObjectACLView.from_payload()`` 一处。
"""

from __future__ import annotations

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.postgres import Base

# ═══════════════════════════════════════════════════════════════════════════════
# 密级 / 可见性模式的字面常量（**全库唯一来源**）
# ═══════════════════════════════════════════════════════════════════════════════
# 这四个常量同时被：ORM 列默认值、alembic 迁移的 server_default、回填脚本、
# security_policy 的判定内核消费。任何一处想写 ``1`` / ``"tier"`` 字面量，
# 都应该 import 这里 —— 否则"改档位"又变成一次全局搜索。

# 密级档位（已裁决 Q2：4 档）
SECURITY_LEVEL_PUBLIC = 0        # 公开
SECURITY_LEVEL_INTERNAL = 1      # 内部（**存量未标注的默认档位**）
SECURITY_LEVEL_CONFIDENTIAL = 2  # 机密
SECURITY_LEVEL_SECRET = 3        # 绝密
SECURITY_LEVEL_MIN = SECURITY_LEVEL_PUBLIC
SECURITY_LEVEL_MAX = SECURITY_LEVEL_SECRET

#: 存量/缺失密级的默认档位（已裁决 Q2 = 1 内部）
DEFAULT_SECURITY_LEVEL = SECURITY_LEVEL_INTERNAL

# 可见性模式（已裁决 Q3：不动 access_level 三值，新增横向维度）
VISIBILITY_MODE_TIER = "tier"        # 按三层知识库（private/department/tenant）
VISIBILITY_MODE_PROJECT = "project"  # 按项目成员
DEFAULT_VISIBILITY_MODE = VISIBILITY_MODE_TIER

# 对象类型
OBJECT_TYPE_DOC = "doc"
OBJECT_TYPE_TEXT_CHUNK = "text_chunk"
OBJECT_TYPE_TABLE = "table"
OBJECT_TYPE_CODE = "code"
OBJECT_TYPE_IMAGE = "image"
VALID_OBJECT_TYPES = frozenset(
    {
        OBJECT_TYPE_DOC,
        OBJECT_TYPE_TEXT_CHUNK,
        OBJECT_TYPE_TABLE,
        OBJECT_TYPE_CODE,
        OBJECT_TYPE_IMAGE,
    }
)

# ACL 同步水位（决策：sync 状态机）
ACL_SYNC_SYNCED = "synced"
ACL_SYNC_PENDING = "pending"
ACL_SYNC_STALE = "stale"

# acl_grants.status
GRANT_STATUS_PENDING = "pending"
GRANT_STATUS_APPROVED = "approved"
GRANT_STATUS_REJECTED = "rejected"
GRANT_STATUS_REVOKED = "revoked"
GRANT_STATUS_EXPIRED = "expired"

GRANT_EFFECT_ALLOW = "allow"
GRANT_EFFECT_DENY = "deny"


# ═══════════════════════════════════════════════════════════════════════════════
# object_id 构造 —— **唯一决策点**
# ═══════════════════════════════════════════════════════════════════════════════

#: 见模块 docstring。(a)(b)(c) 三项取证回来后**只改这一个常量**。
OBJECT_ID_MODE = "scoped"          # "scoped" | "raw"
OBJECT_ID_SEPARATOR = "::"

#: ``object_id`` 列宽上限（与迁移 DDL 的 VARCHAR(128) 对齐，构造后必须不超）
OBJECT_ID_MAX_LEN = 128


def make_object_id(
    document_id: str | uuid.UUID | None,
    raw_object_id: str | None = None,
    *,
    object_type: str = OBJECT_TYPE_DOC,
) -> str:
    """
    构造 ``document_objects.object_id`` —— **全仓库唯一的构造入口**.

        doc         → ``str(document_id)``          （与 PK 等价，便于 join）
        text_chunk  → ``{document_id}::{point_id}`` （point_id 是 uuid5，天然唯一）
        image       → ``{document_id}::{image_id}``
        table/code  → ``{document_id}::{point_id}``

    为什么默认给非 doc 类型加 ``{document_id}::`` 前缀：在 (a) ``image_id`` 是否
    全局唯一、(c) 同一 image_id 是否产出多块**尚未取证**之前，加前缀是唯一能在
    三种答案下都不出错的形态（文档内唯一 ⇒ 加前缀后全局唯一）。取证完成后把
    :data:`OBJECT_ID_MODE` 改成 ``"raw"`` 即可，调用点一行不动。

    Raises:
        ValueError: 非 doc 类型却没给 ``raw_object_id``（宁可炸，也不要静默
            产出一个会互相覆盖的 object_id）。
    """
    doc = "" if document_id is None else str(document_id)
    if object_type == OBJECT_TYPE_DOC:
        if not doc:
            raise ValueError("make_object_id: object_type='doc' 必须有 document_id")
        return doc

    raw = (raw_object_id or "").strip()
    if not raw:
        raise ValueError(
            f"make_object_id: object_type={object_type!r} 需要 raw_object_id "
            "(point_id / image_id)，缺失会导致对象行互相覆盖"
        )
    if OBJECT_ID_MODE == "raw":
        return raw
    if not doc:
        return raw
    return f"{doc}{OBJECT_ID_SEPARATOR}{raw}"


def object_type_from_content_type(content_type: str | None) -> str:
    """
    Qdrant payload 的 ``content_type`` → ``document_objects.object_type``.

    ⚠️ 这里同样受待确认项 (b)（``content_type`` 实际取值集）影响：**新取值一律
    落到 ``text_chunk``**（最宽松、不收紧），确认后再补映射即可，不会漏对象。
    """
    key = (content_type or "").strip().lower()
    if key == "table":
        return OBJECT_TYPE_TABLE
    if key == "code":
        return OBJECT_TYPE_CODE
    if key == "image":
        return OBJECT_TYPE_IMAGE
    return OBJECT_TYPE_TEXT_CHUNK


# ═══════════════════════════════════════════════════════════════════════════════
# document_objects —— 五种对象的统一权限视图
# ═══════════════════════════════════════════════════════════════════════════════


class DocumentObject(Base):
    """
    一个**可检索对象**的权限行（PG 权威源）.

    ``object_type='doc'`` 的行是 ``documents`` 的镜像（供第 11 / 12 环的
    "对象级统一视图"消费，不参与列表 / SQL 主路径 —— 见设计文档决策 6）。
    派生对象（OCR 出的 table / code / text）的 ``parent_object_id`` 指向
    **源图片**而不是文档，有效密级恒取 ``max(父文档, 源图片)``（决策 12）。
    """

    __tablename__ = "document_objects"
    __table_args__ = (
        Index("ix_dobj_document", "document_id"),
        Index("ix_dobj_parent", "parent_object_id"),
        Index("ix_dobj_type_tenant", "object_type", "tenant_id"),
        Index("ix_dobj_eff_level", "effective_security_level"),
        # 部分索引：只索引"未同步"的行 —— synced 是绝大多数，索引它毫无收益
        Index(
            "ix_dobj_sync",
            "acl_sync_state",
            postgresql_where=sa_text("acl_sync_state <> 'synced'"),
        ),
        Index("ix_dobj_projects", "project_ids", postgresql_using="gin"),
        Index("ix_dobj_acl_allow", "acl_allow", postgresql_using="gin"),
        # 同一文档内 chunk_index 唯一（图片对象 chunk_index 为 NULL，不进这条索引）
        Index(
            "uq_dobj_doc_chunk",
            "document_id",
            "chunk_index",
            unique=True,
            postgresql_where=sa_text("chunk_index IS NOT NULL"),
        ),
    )

    # ── 主键与归属 ────────────────────────────────────────────────────────────
    object_id: Mapped[str] = mapped_column(String(OBJECT_ID_MAX_LEN), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    object_type: Mapped[str] = mapped_column(
        String(16), default=OBJECT_TYPE_DOC,
        server_default=OBJECT_TYPE_DOC, nullable=False,
    )
    parent_object_id: Mapped[str | None] = mapped_column(String(OBJECT_ID_MAX_LEN), nullable=True)
    inherited_from: Mapped[str | None] = mapped_column(String(OBJECT_ID_MAX_LEN), nullable=True)
    inherited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ── 硬边界（三层隔离的既有三维，一个不缺）──────────────────────────────────
    tenant_id: Mapped[str] = mapped_column(
        String(64), default="default", server_default="default", nullable=False,
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    department_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ── 层级与可见范围 ────────────────────────────────────────────────────────
    access_level: Mapped[str] = mapped_column(
        String(20), default="private", server_default="private", nullable=False,
    )
    visibility_mode: Mapped[str] = mapped_column(
        String(16), default=DEFAULT_VISIBILITY_MODE,
        server_default=DEFAULT_VISIBILITY_MODE, nullable=False,
    )
    project_ids: Mapped[list] = mapped_column(
        JSONB, default=list, server_default=sa_text("'[]'::jsonb"), nullable=False,
    )
    # 派生只读：self|department|tenant|project|acl（回显用，不参与判定）
    visible_scope: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # ── 密级 ──────────────────────────────────────────────────────────────────
    security_level: Mapped[int] = mapped_column(
        SmallInteger, default=DEFAULT_SECURITY_LEVEL,
        server_default=str(DEFAULT_SECURITY_LEVEL), nullable=False,
    )
    parent_security_level: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    effective_security_level: Mapped[int] = mapped_column(
        SmallInteger, default=DEFAULT_SECURITY_LEVEL,
        server_default=str(DEFAULT_SECURITY_LEVEL), nullable=False,
    )

    # ── ACL 主体（need-to-know 的物化副本；权威源是 acl_grants）─────────────────
    acl_allow: Mapped[list] = mapped_column(
        JSONB, default=list, server_default=sa_text("'[]'::jsonb"), nullable=False,
    )
    acl_deny: Mapped[list] = mapped_column(
        JSONB, default=list, server_default=sa_text("'[]'::jsonb"), nullable=False,
    )
    acl_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ── 同步与共享中间态 ──────────────────────────────────────────────────────
    acl_sync_state: Mapped[str] = mapped_column(
        String(16), default=ACL_SYNC_SYNCED, server_default=ACL_SYNC_SYNCED, nullable=False,
    )
    # 【裁决 Q4-B】剔除（对所有人下线）—— 与 acl_deny（对特定主体拒绝）语义独立
    excluded: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False,
    )
    share_status: Mapped[str] = mapped_column(
        String(16), default="none", server_default="none", nullable=False,
    )
    share_grant_scope: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # ── 定位（回显与引用溯源）──────────────────────────────────────────────────
    chunk_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_id: Mapped[str | None] = mapped_column(String(OBJECT_ID_MAX_LEN), nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"<DocumentObject {self.object_id} type={self.object_type} "
            f"eff={self.effective_security_level} vis={self.visibility_mode}>"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# projects / project_members —— 项目维度（P0 最小集）
# ═══════════════════════════════════════════════════════════════════════════════


class Project(Base):
    """一个项目（横向维度的最小实体；功能在 T5，表在 T1 建）。"""

    __tablename__ = "projects"
    __table_args__ = (Index("ix_projects_tenant", "tenant_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(64), default="default", server_default="default", nullable=False,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<Project {self.id} name={self.name!r} tenant={self.tenant_id}>"


class ProjectMember(Base):
    """项目成员（``expires_at`` 支持 P1-2 的临时成员）。"""

    __tablename__ = "project_members"
    __table_args__ = (
        Index("ix_pmember_user", "user_id"),
        # 复合主键本身就是 (project_id, user_id) 的索引，无需再建
    )

    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    added_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<ProjectMember {self.project_id}:{self.user_id}>"


# ═══════════════════════════════════════════════════════════════════════════════
# acl_grants —— need-to-know 的权威源（表 T1 建、功能 T5）
# ═══════════════════════════════════════════════════════════════════════════════


class AclGrant(Base):
    """
    一条**按主体**的授权（per-subject 一条，带各自的有效期）.

    为什么必须有这张表：``document_objects.acl_allow`` 是数组、``acl_expires_at``
    是单值 —— 它们表达不了"三个主体、各自不同到期时间"。权威源放这里，
    ``acl_allow`` 只是**物化副本**（取最早到期时间），判定侧再取严一次。
    """

    __tablename__ = "acl_grants"
    __table_args__ = (
        Index("ix_aclgrants_object_status", "object_id", "status"),
        Index("ix_aclgrants_subject", "subject", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True,
        server_default=sa_text("gen_random_uuid()"),
    )
    object_id: Mapped[str] = mapped_column(String(OBJECT_ID_MAX_LEN), nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False,
    )
    # "user:<id>" | "dept:<id>" | "role:<r>" | "project:<p>"（共享知识 10）
    subject: Mapped[str] = mapped_column(String(128), nullable=False)
    effect: Mapped[str] = mapped_column(String(8), nullable=False)   # allow | deny
    status: Mapped[str] = mapped_column(
        String(16), default=GRANT_STATUS_PENDING,
        server_default=GRANT_STATUS_PENDING, nullable=False,
    )
    granted_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<AclGrant {self.subject} {self.effect} {self.status} on {self.object_id}>"


__all__ = [
    "ACL_SYNC_PENDING",
    "ACL_SYNC_STALE",
    "ACL_SYNC_SYNCED",
    "DEFAULT_SECURITY_LEVEL",
    "DEFAULT_VISIBILITY_MODE",
    "GRANT_EFFECT_ALLOW",
    "GRANT_EFFECT_DENY",
    "GRANT_STATUS_APPROVED",
    "GRANT_STATUS_EXPIRED",
    "GRANT_STATUS_PENDING",
    "GRANT_STATUS_REJECTED",
    "GRANT_STATUS_REVOKED",
    "OBJECT_ID_MAX_LEN",
    "OBJECT_ID_MODE",
    "OBJECT_ID_SEPARATOR",
    "OBJECT_TYPE_CODE",
    "OBJECT_TYPE_DOC",
    "OBJECT_TYPE_IMAGE",
    "OBJECT_TYPE_TABLE",
    "OBJECT_TYPE_TEXT_CHUNK",
    "SECURITY_LEVEL_CONFIDENTIAL",
    "SECURITY_LEVEL_INTERNAL",
    "SECURITY_LEVEL_MAX",
    "SECURITY_LEVEL_MIN",
    "SECURITY_LEVEL_PUBLIC",
    "SECURITY_LEVEL_SECRET",
    "VALID_OBJECT_TYPES",
    "VISIBILITY_MODE_PROJECT",
    "VISIBILITY_MODE_TIER",
    "AclGrant",
    "DocumentObject",
    "Project",
    "ProjectMember",
    "make_object_id",
    "object_type_from_content_type",
]
