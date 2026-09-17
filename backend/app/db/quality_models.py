"""
RAG 运行期质量事件 ORM 模型（持续监控的"记忆"）.

为什么需要这张表
────────────────
监控面板的比率此前只来自**进程内计数器**（monitoring_service._Metrics），
后端一重启就归零 —— 于是面板上方的"证据拒答率 / 引用通过率"长期显示 "—"，
与下方来自数据库的 Bad Case 累计数自相矛盾（截图问题：指标全是 "—"）。

本表把**每一轮问答**的关键质量信号落库一行：拒答、证据门控、引用校验、
输出净化、时延。比率改为按时间窗从库里聚合，重启不再丢历史，还能看趋势。

与 bad_cases 表的分工：
- bad_cases：只有"出问题"的轮次，带全文快照，供人工审阅（重、少）；
- quality_events：**每一轮**都有，只有指标字段，供聚合出比率（轻、多）。

独立成模块，与 badcase_models 同一套风格（不动既有表结构）。
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.postgres import Base


class QualityEvent(Base):
    """一轮问答的质量信号快照（每轮一行，只承载可聚合的指标字段）."""

    __tablename__ = "quality_events"

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
    )

    # 路由意图：knowledge_qa | document_summary | general_chat | doc_relations |
    # list_documents | document_agent
    intent: Mapped[str] = mapped_column(String(32), nullable=False, default="", index=True)

    # 证据门控（仅 knowledge_qa / document_agent 经过；其余分支为 None）
    evidence_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    evidence_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 引用校验（仅 knowledge_qa 经过；overall: verified | partial | unsupported |
    # no_citations | refused_by_model）
    citation_overall: Mapped[str | None] = mapped_column(String(24), nullable=True)
    citation_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    citation_passed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    citation_unsupported: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    citation_hallucinated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    citation_misattributed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    citation_number_mismatch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    citation_date_mismatch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 输出净化 / 拒答
    output_guard_changed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    refusal_source: Mapped[str] = mapped_column(
        String(8), nullable=False, default=""
    )  # gate | model | ""（未拒答）

    # 规模与时延
    sources_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    answer_chars: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return (
            f"<QualityEvent id={self.id} intent={self.intent!r} "
            f"citation={self.citation_overall!r}>"
        )
