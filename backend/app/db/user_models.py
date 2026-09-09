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
    """A local user account. The first registered account becomes admin."""

    __tablename__ = "users"

    # RBAC roles. ROLE_USER remains a legacy alias and maps to editor-level
    # permissions, so existing accounts keep their current capabilities.
    ROLE_ADMIN = "admin"
    ROLE_MANAGER = "manager"
    ROLE_EDITOR = "editor"
    ROLE_VIEWER = "viewer"
    ROLE_USER = "user"

    VALID_ROLES = {ROLE_ADMIN, ROLE_MANAGER, ROLE_EDITOR, ROLE_VIEWER, ROLE_USER}

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
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(
        String(20), default=ROLE_USER, server_default=ROLE_USER, nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    @property
    def is_admin(self) -> bool:
        return self.role == self.ROLE_ADMIN

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} role={self.role}>"


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
