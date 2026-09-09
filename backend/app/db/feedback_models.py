"""
Answer feedback ORM model (反馈闭环优化).

Stores a user's 👍/👎 rating on one assistant answer together with a snapshot
of the Q&A pair — the raw material for badcase review and retrieval-quality
iteration.  Kept in its own module so earlier phases are never touched.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.postgres import Base


class AnswerFeedback(Base):
    """One 👍/👎 rating on an assistant answer (badcase 收集)."""

    __tablename__ = "answer_feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Which conversation the rated answer belongs to (nullable: feedback may
    # arrive before the conversation id reaches the client).
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    rating: Mapped[str] = mapped_column(
        String(8), nullable=False,  # "up" | "down"
    )
    # Q&A snapshot — the answer text lives in `messages`, but snapshots keep
    # feedback self-contained even after the conversation is deleted.
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<AnswerFeedback id={self.id} rating={self.rating!r}>"
