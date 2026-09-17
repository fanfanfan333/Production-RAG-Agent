"""add quality_events table (RAG 质量监控持久化)

每轮问答落一行质量信号（意图 / 证据门控 / 引用校验 / 输出净化 / 拒答 / 时延），
质量监控面板的比率按时间窗从这张表聚合 —— 进程内计数器重启归零，
历史比率不能再靠"内存里的计数"（截图问题：重启后指标全是 "—"）。

注：应用启动时 `Base.metadata.create_all` 也会建这张表；本迁移用于**存量库**
（已经跑过 alembic upgrade 的库）补齐。全部用 IF NOT EXISTS，与 create_all
并发/先后执行都安全（幂等），与 f7a1c9d3e5b2 的写法保持一致。

Revision ID: m8g9h0i1j2k3
Revises: f7a1c9d3e5b2
Create Date: 2026-09-17 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "m8g9h0i1j2k3"
down_revision: Union[str, None] = "f7a1c9d3e5b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS quality_events (
            id UUID PRIMARY KEY,
            user_id UUID REFERENCES users(id) ON DELETE SET NULL,
            username VARCHAR(64),
            conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
            intent VARCHAR(32) NOT NULL DEFAULT '',
            evidence_passed BOOLEAN,
            evidence_confidence DOUBLE PRECISION,
            citation_overall VARCHAR(24),
            citation_total INTEGER NOT NULL DEFAULT 0,
            citation_passed INTEGER NOT NULL DEFAULT 0,
            citation_unsupported INTEGER NOT NULL DEFAULT 0,
            citation_hallucinated INTEGER NOT NULL DEFAULT 0,
            citation_misattributed INTEGER NOT NULL DEFAULT 0,
            citation_number_mismatch INTEGER NOT NULL DEFAULT 0,
            citation_date_mismatch INTEGER NOT NULL DEFAULT 0,
            output_guard_changed BOOLEAN NOT NULL DEFAULT FALSE,
            refusal_source VARCHAR(8) NOT NULL DEFAULT '',
            sources_count INTEGER NOT NULL DEFAULT 0,
            answer_chars INTEGER NOT NULL DEFAULT 0,
            latency_ms INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_quality_events_created_at "
        "ON quality_events (created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_quality_events_intent "
        "ON quality_events (intent)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_quality_events_user_id "
        "ON quality_events (user_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS quality_events")
