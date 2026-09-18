"""add eval_runs table (评测基线持久化)

把 ``/eval/history`` 从"进程内 deque"升级为"落库"。原因是评测最常见的用法
恰好会抹掉历史：改配置 → 重启后端 → 再评测，而 deque 在重启时就清了，
于是线上 ``/eval/history`` 长期是 ``[]``、``/eval/metrics.latest_eval``
长期是 ``null`` —— 没有历史就没有基线，"这次改完是变好还是变坏"无从回答。

只存**聚合结果**（每个用例的逐条明细是诊断材料，不该压在业务库里）：
逐条明细随用例数线性增长，聚合值只有几十个数字，适合长期留存做趋势对比。

注：应用启动时 `Base.metadata.create_all` 也会建这张表；本迁移用于**存量库**
（已经跑过 alembic upgrade 的库）补齐。全部用 IF NOT EXISTS，与 create_all
并发/先后执行都安全（幂等），与 m8g9h0i1j2k3 的写法保持一致。

Revision ID: n9h0i1j2k3l4
Revises: m8g9h0i1j2k3
Create Date: 2026-09-18 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "n9h0i1j2k3l4"
down_revision: Union[str, None] = "m8g9h0i1j2k3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS eval_runs (
            id UUID PRIMARY KEY,
            name VARCHAR(128) NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            run_by VARCHAR(128),
            tenant_id VARCHAR(64),
            collection_id VARCHAR(64),
            total_cases INTEGER NOT NULL DEFAULT 0,
            scored_cases INTEGER NOT NULL DEFAULT 0,
            top_k INTEGER NOT NULL DEFAULT 0,
            mrr DOUBLE PRECISION,
            map_score DOUBLE PRECISION,
            recall JSONB NOT NULL DEFAULT '{}'::jsonb,
            precision JSONB NOT NULL DEFAULT '{}'::jsonb,
            ndcg JSONB NOT NULL DEFAULT '{}'::jsonb,
            hit_rate JSONB NOT NULL DEFAULT '{}'::jsonb,
            by_modality JSONB NOT NULL DEFAULT '{}'::jsonb,
            citation JSONB,
            k_values JSONB NOT NULL DEFAULT '[]'::jsonb,
            config JSONB NOT NULL DEFAULT '{}'::jsonb,
            generated_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_eval_runs_generated_at "
        "ON eval_runs (generated_at DESC)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_name ON eval_runs (name)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_eval_runs_created_at ON eval_runs (created_at)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS eval_runs")
