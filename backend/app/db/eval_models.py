"""
评测运行历史 ORM 模型（把 /eval/history 从"进程内"升级为"跨重启持久化"）.

为什么需要这张表
────────────────
``services/evaluation`` 的运行历史此前只存在 ``deque(maxlen=20)`` 里，
**后端一重启就归零**。后果不是"少了几行" —— 而是评测这件事根本立不住：

  * 换 Embedding / 改切分 / 调阈值之后要对比"变好还是变坏"，而每次改配置
    都要重启后端，历史恰好在重启时被清空。于是线上 ``/eval/history`` 永远
    是 ``[]``，``/eval/metrics`` 的 ``latest_eval`` 永远是 ``null``；
  * 没有历史就没有基线，"回归"无从谈起 —— 只能靠单次跑出来的绝对值拍脑袋。

本表把每一轮评测的**聚合结果**落库一行（不存逐条明细：逐条是诊断材料，
放评测产出目录，不该压在业务库里）。逐条明细的大小随用例数线性增长，
而聚合值只有几十个数字，适合长期留存做趋势对比。

与 quality_events / bad_cases 同一套风格：独立模块，不动既有表结构；
容器启动时 ``Base.metadata.create_all`` 会自动建表，同时提供 alembic
迁移供**存量库**补齐。
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.postgres import Base


class EvalRunRow(Base):
    """一轮评测的聚合结果（每轮一行，供 /eval/history 做跨版本回归对比）."""

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # 金标集标识：name 是稳定代号（如 "golden-v1"），description 写清口径
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # 谁跑的、在哪个权限范围里跑的 —— 同一条金标在不同 scope 下分数不同，
    # 不记 scope 的话两次运行不可比
    run_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tenant_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    collection_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 规模
    total_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scored_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 聚合指标（None = 该指标在本轮无有效样本，与"0 分"是两件事）
    mrr: Mapped[float | None] = mapped_column(Float, nullable=True)
    map_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 分档指标：{"1": 0.95, "3": 1.0, ...}（键统一为字符串，JSON 的键只能是字符串）
    recall: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    precision: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    ndcg: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    hit_rate: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # 明细维度与引用准确率
    by_modality: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    citation: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    k_values: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # ── 多证据分组与分带余量（"召回变差"的**先行指标**）──────────────────────
    #
    # 为什么这两个字段非落库不可（它们不是"多存点细节"，而是补一个观测盲区）：
    #
    # 分带（RERANK_MIN_SCORE_RATIO）调过头时，被砍掉的是**合法性仅次于头名**
    # 的次要证据。它的失效顺序是：
    #     ① 分带余量缩小（min_gold_ratio 靠近 ratio）→ ② 某条次要证据被砍
    #     → ③ 多证据用例不再"一条不漏"（all_gold_found_rate 掉下来）
    #     → ④ 整体 recall 才掉
    #
    # ④ 之前的三步**在整体指标上完全看不见**（recall 恒 1.0 时，余量可能已经从
    # 1.4 倍掉到 1.02 倍）。而 eval_runs 是跨重启的趋势基线 —— 如果只存
    # 「④ 之后才出现的」那些指标，那么"余量侵蚀"这一段永远无法回溯，
    # 本表作为基线的一半价值就落空了：你会看到 recall 突然从 1.0 掉下来，
    # 却查不到它前几轮已经在贴着边界跑。
    #
    # 这正是本次修复要消灭的那类盲区（旧金标集"金标全是第 1 名 ⇒ 分带风险
    # 恒测不出"）在**持久化层**的翻版，所以必须一起补齐。
    evidence_slices: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    gold_ratio: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # 配置快照：没有它，"分数变了"无法归因到"改了什么"
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return (
            f"<EvalRunRow id={self.id} name={self.name!r} "
            f"cases={self.scored_cases}/{self.total_cases} mrr={self.mrr}>"
        )


__all__ = ["EvalRunRow"]
