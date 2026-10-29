"""Persistent coordination task + append-only event log.

A task belongs to exactly one authenticated user. ``(tenant_id, user_id,
idempotency_key)`` is unique, so a timed-out client can retry the *same* submit
without creating a second container/research run. The idempotency scope is the
user, not the tenant: two colleagues may independently use ``"retry-1"``.

``events`` is append-only and ordered by per-task ``seq``; it is the durable
source the UI replays after a restart (SSE alone is ephemeral).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.postgres import Base


class CoordinationTask(Base):
    __tablename__ = "coordination_tasks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_text: Mapped[str] = mapped_column(Text, nullable=False)
    requested_kind: Mapped[str] = mapped_column(String(24), default="auto", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    plan: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    result: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    stop_requested: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "user_id",
            "idempotency_key",
            name="uq_coordination_tasks_owner_idempotency",
        ),
        Index("ix_coordination_tasks_owner_created", "tenant_id", "user_id", "created_at"),
        Index("ix_coordination_tasks_owner_status", "tenant_id", "user_id", "status"),
    )


class CoordinationTaskEvent(Base):
    __tablename__ = "coordination_task_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coordination_tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(48), nullable=False)
    data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("task_id", "seq", name="uq_coordination_task_events_seq"),
    )


class ForgeFlowIdentityLink(Base):
    __tablename__ = "forgeflow_identity_links"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    forgeflow_tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    forgeflow_user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    __table_args__ = (UniqueConstraint("forgeflow_tenant_id", "forgeflow_user_id", name="uq_forgeflow_identity_link"),)
