"""
Multi-user auth models (企业落地第一阶段).

Tables:
    users        — local accounts with bcrypt password hashes and roles
    collections  — business-level knowledge-base groupings (per user),
                   distinct from raw Qdrant collections (infra-level)
    audit_logs   — who did what, when, from where (compliance trail)

All models inherit the same Base so create_all picks them up automatically
when this module is imported (main.py imports it for exactly that reason).
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.postgres import Base


class User(Base):
    """
    A user account, either Keycloak-federated or locally registered.

    Business roles mirror the 三层知识库 capability matrix:

        普通员工 employee        上传 / 个人知识库；发布到部门库需申请，
                                发布到公司库无权限，不能删除他人文档
        部门负责人 dept_manager   + 直接发布到部门库、审部门申请、
                                  删除本部门范围内他人文档
        知识库管理员 kb_admin     + 直接发布到公司库、审公司申请、
                                  删除全公司文档
        企业管理员 company_admin  本公司全部权限（含成员与审计）

    Legacy roles (admin/manager/editor/viewer/user) are kept and mapped onto
    the same capability sets so existing accounts keep their behaviour.
    """

    __tablename__ = "users"

    # ── 业务角色（三层知识库权限矩阵）────────────────────────────────────────
    ROLE_COMPANY_ADMIN = "company_admin"
    ROLE_KB_ADMIN = "kb_admin"
    ROLE_DEPT_MANAGER = "dept_manager"
    ROLE_EMPLOYEE = "employee"

    # ── 历史角色（保留，等价映射到上面的能力集）──────────────────────────────
    ROLE_ADMIN = "admin"          # 平台管理员（跨租户）
    ROLE_MANAGER = "manager"      # ~ 部门负责人
    ROLE_EDITOR = "editor"        # ~ 普通员工
    ROLE_USER = "user"            # ~ 普通员工
    ROLE_VIEWER = "viewer"        # 只读

    VALID_ROLES = {
        ROLE_COMPANY_ADMIN, ROLE_KB_ADMIN, ROLE_DEPT_MANAGER, ROLE_EMPLOYEE,
        ROLE_ADMIN, ROLE_MANAGER, ROLE_EDITOR, ROLE_VIEWER, ROLE_USER,
    }

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    username: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )
    # 本地账号用 bcrypt 摘要；Keycloak 联邦账号没有本地密码，该列可为空。
    password_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # 展示名（Keycloak 的 name / preferred_username）
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    role: Mapped[str] = mapped_column(
        String(20), default=ROLE_USER, server_default=ROLE_USER, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )
    # ── Multi-Tenant（第一层：Tenant Isolation）──────────────────────────────
    # 每个用户属于一个租户（如 company_A）；检索在向量/BM25 层就按它过滤。
    # 老账号迁移进 "default" 租户，单机部署行为与升级前一致。
    tenant_id: Mapped[str] = mapped_column(
        String(64), default="default", server_default="default",
        nullable=False, index=True,
    )
    # 第二层 Document ACL 用：access_level=department 的文档仅同部门可见。
    department_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
    )
    # ── 企业身份（身份验证申请通过后由 admin_users staff_service 写入）─────────
    # company_name 是用户在「身份验证」表单里填的**公司名称原文**（中文也合法）；
    # tenant_id 是它映射出的安全 ID（见 tenancy.company_id_from_name）。
    # 两者分离的原因：tenant_id 会进文件路径，不能存中文；但个人主页要显示
    # 用户认得出的名字。
    company_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # 部门名称原文（department_id 同样是映射后的安全 ID）。
    department_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # 部门职责 / 职务（如「嵌入式软件工程师」），区别于 role（权限等级）。
    job_title: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # ── Keycloak 联邦身份 ────────────────────────────────────────────────────
    # keycloak_sub = JWT 的 "sub"（Realm 内唯一且稳定）。首次用 Keycloak 登录
    # 时按它把远端身份落到本地行，之后的鉴权/权限上下文全走本地 users 表 ——
    # 检索链路上的 tenant_id / department / role 只需一处来源。
    keycloak_sub: Mapped[str | None] = mapped_column(
        String(128), unique=True, nullable=True, index=True,
    )
    auth_source: Mapped[str] = mapped_column(
        String(16), default="local", server_default="local", nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    @property
    def is_admin(self) -> bool:
        """平台管理员（跨租户）。业务上的"企业管理员"由权限矩阵表达。"""
        return self.role == self.ROLE_ADMIN

    @property
    def is_keycloak_user(self) -> bool:
        return self.auth_source == "keycloak"

    def __repr__(self) -> str:
        return (
            f"<User id={self.id} username={self.username!r} role={self.role} "
            f"tenant={self.tenant_id!r}>"
        )


class Collection(Base):
    """
    A business knowledge-base collection ("知识库").

    Groups Document rows for scoped Q&A. Vectors carry the collection id in
    their Qdrant payload so retrieval can filter at the vector layer.
    """

    __tablename__ = "collections"
    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uq_collections_owner_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Collection id={self.id} name={self.name!r} owner={self.owner_id}>"


class AuditLog(Base):
    """
    One auditable action. Written best-effort — audit failures must never
    take down the primary operation.
    """

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<AuditLog action={self.action!r} user={self.username!r}>"
