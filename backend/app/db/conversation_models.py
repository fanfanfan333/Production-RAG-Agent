"""
Conversation and Message ORM models (Phase 4).

Kept in a separate file from app.db.models (Phase 2) so Phase 2 code
is never touched.  Both models inherit the same Base so create_all
picks them up automatically when this module is imported.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.postgres import Base


class Conversation(Base):
    """
    A conversation session.  One conversation holds many Message rows.

    A new Conversation is created automatically on the first query when the
    caller does not supply a conversation_id.  Subsequent queries in the same
    session pass back the returned conversation_id to continue the thread.
    """

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,   # legacy conversations created before auth have no owner
        index=True,
    )
    # 第三层隔离：会话归属租户。隔离键 = conversation_id + tenant_id + user_id，
    # 跨租户拿到 conversation_id 也无法续聊/读历史。
    tenant_id: Mapped[str] = mapped_column(
        String(64), default="default", server_default="default",
        nullable=False, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id}>"


class Message(Base):
    """
    A single turn in a conversation.

    role must be one of: "user" | "assistant".
    content is the raw text — no special encoding.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 第三层隔离：消息写入时的真实用户与租户（assistant 消息同属该发起用户）。
    # 老数据为 NULL，校验时按"继承所属 Conversation"处理。
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
    )
    tenant_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    # 回答侧的"依据快照"：sources / citation_check / evidence / output_guard /
    # document / intent。这些数据此前**只存在于前端内存**，一旦重新加载
    # （切页、切窗口、重开标签页）就整批丢失 —— 用户看到的现象是
    # "上一次提问的引用来源不见了"。落库后历史会话可以完整复现当时的依据。
    # 只有 assistant 消息会写；user 消息与老数据为 NULL。
    # 注意：SQLAlchemy 声明式基类占用了 `metadata`，因此字段名取 `meta`。
    meta: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role!r}>"
