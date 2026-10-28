"""长期记忆（Agent Memory）ORM.

与 `conversation_models.py` 同样的做法：单独一个文件、共用 `app.db.postgres.Base`，
由 `app/main.py` 导入后随 `Base.metadata.create_all` 建表。

隔离键是 `(tenant_id, user_id)` 两列**同时**等值匹配，没有"管理员视角"分支 ——
个人记忆不属于租户资产，平台管理员审核别人的训练样本，不代表能读别人的长期设定。
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.postgres import Base


class AgentMemory(Base):
    """一条用户级长期记忆.

    只存**用户自己说过、且被显式提取**的内容（"记住：…"），不存模型推测、
    不存检索片段、不存对话原文。`revoked_at` 非空的记录不参与任何读取，
    但保留行本身用于审计追溯。
    """

    __tablename__ = "agent_memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # 第三层隔离键：会话租户 + 用户，两者都必须匹配才可读
    tenant_id: Mapped[str] = mapped_column(
        String(64), default="default", server_default="default", nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), default="note", nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 来源：explicit_user（用户说"记住…"）| manual_api（管理/自助接口写入）
    source: Mapped[str] = mapped_column(String(32), default="explicit_user", nullable=False)
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    # 撤销语义：不删除、只置位；撤销后不参与读取，历史仍可查
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    revoke_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    __table_args__ = (
        Index("ix_agent_memories_scope", "tenant_id", "user_id", "revoked_at"),
    )
