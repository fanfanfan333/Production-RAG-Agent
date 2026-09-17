"""
RAG 质量监控 API（设置页「RAG 质量监控」面板的数据源）.

    GET /quality/stats?window=session|24h|7d|30d|all

与 /badcases/stats 的分工
────────────────────────
/badcases/stats 回答"有多少坏答案等着人审"（队列视角）；
本端点回答"系统最近答得怎么样"（比率视角）：

    window=session  → 进程内计数器（本次运行，实时但重启归零）
    window=24h/7d/30d/all → quality_events 表聚合（重启不丢，可看趋势）

比率的分母为 0 时返回 null，同时返回 samples（各指标的样本量），
让前端能把"—"解释成"该窗口内还没有这类问答"，而不是假装有数据。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.db.user_models import User
from app.services.monitoring_service import metrics_snapshot, quality_stats
from app.services.permissions import require_permission
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Quality"])

# 时间窗 → 秒数；session / all 是两种特殊窗口
_WINDOWS: dict[str, float | None] = {
    "24h": 24 * 3600,
    "7d": 7 * 24 * 3600,
    "30d": 30 * 24 * 3600,
    "all": None,
}


@router.get(
    "/quality/stats",
    summary="RAG 质量比率（按时间窗聚合）",
    description=(
        "window=session 返回进程内实时计数；24h/7d/30d/all 从 quality_events "
        "表聚合，重启不丢历史。比率分母为 0 时为 null，samples 给出各指标样本量。"
    ),
)
async def get_quality_stats(
    user: Annotated[User, Depends(require_permission("audit.read"))],
    window: str = Query(default="24h", pattern="^(session|24h|7d|30d|all)$"),
) -> dict:
    if window == "session":
        snap = metrics_snapshot()
        return {
            "window": window,
            "source": "in_process",
            "total_queries": snap["counters"].get("rag.queries.total", 0),
            "ratios": snap["ratios"],
            "samples": {
                "evidence_gate": snap["counters"].get("evidence_gate.total", 0),
                "citations_checked": snap["counters"].get("citation.checked", 0),
                "citation_answers": snap["counters"].get("citation.total", 0),
                "queries": snap["counters"].get("rag.queries.total", 0),
            },
            "by_intent": {
                k.removeprefix("rag.intent."): v
                for k, v in snap["counters"].items()
                if k.startswith("rag.intent.")
            },
            "latency_ms": {
                stage: {
                    "avg": lat.get("avg_ms"),
                    "p95": None,  # 进程内只记 sum/max，无分位数
                    "max": lat.get("max_ms"),
                }
                for stage, lat in snap["latencies"].items()
            },
            "citation_failure_reasons": {
                "unsupported": snap["counters"].get("citation.unsupported", 0),
                "hallucinated": 0,  # 进程内未单独计数幻觉引用
                "misattributed": snap["counters"].get("citation.misattributed", 0),
                "number_mismatch": snap["counters"].get("citation.number_mismatch", 0),
                "date_mismatch": snap["counters"].get("citation.date_mismatch", 0),
            },
            "generated_at": snap["generated_at"],
        }

    stats = await quality_stats(_WINDOWS[window])
    return {"window": window, "source": "database", **stats}


__all__ = ["router"]
