"""
Bad Case 回流 ORM 模型（持续监控闭环）.

一张表承载"自动回流 + 人工反馈"两条来源：

    自动回流：引用校验失败 / 证据门控拒答 / 输出净化命中
    人工反馈：用户点👎（/feedback 时同写入本表，reason=feedback_down）

字段设计围绕**可复现**：question + answer + sources_snapshot + detail
四者齐全时，审阅者不需要翻对话历史就能复现这次问答的完整上下文 ——
这也是"回流"能变成"回归测试集"的前提。

独立成模块，避免改动既有的 feedback / conversation 表结构。
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.postgres import Base


class BadCase(Base):
    """一条待审阅的坏答案（自动回流或用户👎）."""

    __tablename__ = "bad_cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # 回流原因：citation_unsupported | evidence_refused | output_guard | feedback_down
    reason: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(
        String(16), nullable=False, default="medium", index=True
    )
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # 可复现快照
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)          # JSON
    sources_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON

    # 审阅状态机：open → triaged → resolved（或 wontfix）
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="open", index=True
    )
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[str | None] = mapped_column(String(256), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<BadCase id={self.id} reason={self.reason!r} status={self.status!r}>"
