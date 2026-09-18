"""add eval_runs.evidence_slices / gold_ratio (多证据 + 分带余量的先行指标)

补齐一个**观测盲区**，而不是"多存几个字段"。

``eval_runs`` 是跨重启的趋势基线。此前它只存到「整体 recall 掉下来」这一步
才能看见的指标，而相对分带（``RERANK_MIN_SCORE_RATIO``）调过头时的失效顺序是：

    ① 分带余量缩小（min_gold_ratio 逼近 ratio）
    ② 某条**合法次要证据**被带砍掉
    ③ 多证据用例不再"一条不漏"（all_gold_found_rate 下降）
    ④ 整体 recall 才掉

①②③ 在整体指标上完全不可见 —— recall 恒 1.0 时余量可能已经从 1.4 倍掉到
1.02 倍。只存 ④ 的话，回头看历史只能看到"某一轮 recall 突然掉了"，查不到它
前几轮已经在贴着边界跑，等于把基线价值砍掉一半。

这与本次修复的原始动机同源：旧金标集 11 例金标**全是精排第 1 名**（gold == head）
⇒ gold/head ≡ 1.0 ⇒ "提高 ratio 会不会误杀次要证据"在该集上恒测不出。
把同一类盲区从**评测集**消灭之后，不能在**持久化层**又留一个。

两个字段都是 JSONB 且可空默认空对象：
  * 存量行的值为 ``{}``，读侧以 ``or {}`` 兜底 —— "没存过"不会被渲染成"测出 0"；
  * 全部 IF NOT EXISTS，与启动时的 ``Base.metadata.create_all`` 并发/先后执行
    都安全（幂等），与 n9h0i1j2k3l4 / m8g9h0i1j2k3 的写法保持一致。

Revision ID: o0i1j2k3l4m5
Revises: n9h0i1j2k3l4
Create Date: 2026-09-18 17:05:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "o0i1j2k3l4m5"
down_revision: Union[str, None] = "n9h0i1j2k3l4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE eval_runs "
        "ADD COLUMN IF NOT EXISTS evidence_slices JSONB NOT NULL DEFAULT '{}'::jsonb"
    )
    op.execute(
        "ALTER TABLE eval_runs "
        "ADD COLUMN IF NOT EXISTS gold_ratio JSONB NOT NULL DEFAULT '{}'::jsonb"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE eval_runs DROP COLUMN IF EXISTS gold_ratio")
    op.execute("ALTER TABLE eval_runs DROP COLUMN IF EXISTS evidence_slices")
