"""
RAG 持续监控 + Bad Case 回流（运营闭环）.

两部分职责，故意放在同一个模块里，因为它们是同一个闭环的两端：

    监控（发现）  ──▶  Bad Case 回流（沉淀）  ──▶  评测/回归（改进）
        ▲                                                  │
        └──────────────── 指标变化可验证 ──────────────────┘

一、运行期指标（in-process，零依赖）
────────────────────────────────────
用最朴素的方式记录**能反映 RAG 质量的关键比率**，不引入 Prometheus 客户端：

    证据门控拒答率    evidence_gate.refuse / evidence_gate.total
    引用校验通过率    citation.verified / citation.total
    引用失效率        citation.unsupported / citation.total
    输出净化命中率    output_guard.changed / output_guard.total
    分阶段时延        retrieval / generation / total（sum + count + max）

多进程部署（uvicorn 多 worker）时每个 worker 各持一份计数器，
`snapshot()` 返回的是**本进程**视角 —— 单 worker 是本项目的默认部署
（见 README「容器卫生：单 worker」），因此这已经够用；要跨进程聚合，
把 snapshot() 输出接到 Prometheus / OTLP 即可，接口形状无需改变。

进程内计数器的天然短板是**重启归零**：重启后比率分母为 0，面板显示 "—"。
因此每轮问答还会落一行 quality_events（见 record_quality_event），
`quality_stats(window)` 按时间窗从数据库聚合出同样的比率 ——
历史视角用库、实时视角用进程内计数，两者在 /quality/stats 汇合。

二、Bad Case 回流（自动沉淀）
────────────────────────────────────
不是所有坏答案都会有人点👎。以下四类信号**自动**写入 bad_cases 表，
让人工审阅队列不再依赖"用户有没有心情点踩"：

    citation_unsupported —— 引用校验发现"不被原文支持 / 数字日期不一致"
    evidence_refused     —— 证据门控拒答（可能是召回失败，也可能是问得太偏）
    output_guard         —— 输出净化命中（引用越界 / 提示词泄露 / 工具越权）
    feedback_down        —— 用户点了👎（由 /feedback 触发，见 api/feedback.py）

写入是 best-effort：失败只记日志，绝不影响问答主链路。
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── 指标容器 ─────────────────────────────────────────────────────────────────

class _Metrics:
    """线程安全的极简计数器（counter / gauge / latency）."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._latency_sum: dict[str, float] = {}
        self._latency_count: dict[str, int] = {}
        self._latency_max: dict[str, float] = {}

    def incr(self, key: str, value: int = 1) -> None:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, key: str, seconds: float) -> None:
        with self._lock:
            self._latency_sum[key] = self._latency_sum.get(key, 0.0) + seconds
            self._latency_count[key] = self._latency_count.get(key, 0) + 1
            self._latency_max[key] = max(self._latency_max.get(key, 0.0), seconds)

    def snapshot(self) -> dict:
        with self._lock:
            counters = dict(self._counters)
            latencies = {
                key: {
                    "count": self._latency_count[key],
                    "avg_ms": round(
                        self._latency_sum[key] / self._latency_count[key] * 1000, 1
                    ),
                    "max_ms": round(self._latency_max[key] * 1000, 1),
                }
                for key in self._latency_count
            }
        return {"counters": counters, "latencies": latencies}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latency_sum.clear()
            self._latency_count.clear()
            self._latency_max.clear()


_metrics = _Metrics()


# ── 记录入口（供 master_graph / api 调用）─────────────────────────────────────

def record_query(intent: str) -> None:
    _metrics.incr("rag.queries.total")
    if intent:
        _metrics.incr(f"rag.intent.{intent}")


def record_evidence_gate(passed: bool, confidence: float = 0.0) -> None:
    _metrics.incr("evidence_gate.total")
    _metrics.incr("evidence_gate.pass" if passed else "evidence_gate.refuse")
    # 置信度分档（<0.4 / 0.4–0.7 / ≥0.7），便于看"证据整体变强了还是变弱了"
    if passed:
        bucket = "high" if confidence >= 0.7 else ("mid" if confidence >= 0.4 else "low")
        _metrics.incr(f"evidence_gate.confidence.{bucket}")


def record_citation_check(report) -> None:
    """*report* 为 citation_verifier.VerificationReport."""
    _metrics.incr("citation.total")
    overall = getattr(report, "overall", "no_citations")
    _metrics.incr(f"citation.{overall}")
    _metrics.incr("citation.checked", int(getattr(report, "total", 0) or 0))
    failed = (
        len(set(getattr(report, "unsupported_indices", ()) or ()))
        + len(set(getattr(report, "hallucinated_indices", ()) or ()))
    )
    _metrics.incr("citation.failed", failed)
    if getattr(report, "misattributed_indices", ()):
        _metrics.incr("citation.misattributed", len(report.misattributed_indices))
    if getattr(report, "number_mismatch_indices", ()):
        _metrics.incr("citation.number_mismatch", len(report.number_mismatch_indices))
    if getattr(report, "date_mismatch_indices", ()):
        _metrics.incr("citation.date_mismatch", len(report.date_mismatch_indices))


def record_output_guard(changed: bool, blocked: bool = False) -> None:
    _metrics.incr("output_guard.total")
    if changed:
        _metrics.incr("output_guard.changed")
    if blocked:
        _metrics.incr("output_guard.blocked")


def record_latency(stage: str, seconds: float) -> None:
    _metrics.observe(f"latency.{stage}", seconds)


def record_counter(key: str, value: int = 1) -> None:
    """
    记录一个自定义计数器（供分阶段/结果分类的细分指标使用）.

    与 ``record_*`` 系列的区别：那些是**固定语义**的指标（拒答、引用校验…），
    这里是给"某条支路到底有没有生效"这类**存在性**问题用的。典型场景：
    查询增强层（改写/多查询扩展/HyDE）过去因超时而每轮静默回退，日志里只留一条
    warning，监控面板上完全看不出"配置全开着但一次都没生效"。有了计数器，
    ``rewrite.source.original`` 与 ``rewrite.source.llm`` 的比值就是**有效率**，
    退化不再隐默。
    """
    _metrics.incr(key, value)


def record_badcase(reason: str) -> None:
    _metrics.incr("badcase.captured")
    _metrics.incr(f"badcase.reason.{reason}")


def record_refusal(source: str) -> None:
    """*source*: gate | model —— 区分"门控拒答"与"模型主动拒答"."""
    _metrics.incr("refusal.total")
    _metrics.incr(f"refusal.{source}")


def metrics_snapshot() -> dict:
    """返回当前指标快照 + 派生比率（前端/运维直接可读）."""
    snap = _metrics.snapshot()
    c = snap["counters"]

    def _ratio(num_key: str, den_key: str) -> float | None:
        den = c.get(den_key, 0)
        if not den:
            return None
        return round(c.get(num_key, 0) / den, 4)

    snap["ratios"] = {
        "evidence_refuse_rate": _ratio("evidence_gate.refuse", "evidence_gate.total"),
        "citation_pass_rate": _ratio("citation.verified", "citation.total"),
        "citation_unsupported_rate": _ratio(
            "citation.unsupported", "citation.total"
        ),
        "citation_failed_per_check": _ratio("citation.failed", "citation.checked"),
        "output_guard_change_rate": _ratio(
            "output_guard.changed", "output_guard.total"
        ),
        "refusal_rate": _ratio("refusal.total", "rag.queries.total"),
    }
    snap["generated_at"] = datetime.now(tz=timezone.utc).isoformat()
    return snap


def reset_metrics() -> None:
    """仅供单测使用."""
    _metrics.reset()


# ── Bad Case 回流 ────────────────────────────────────────────────────────────

@dataclass
class BadCaseSignal:
    """一次"值得回流"的信号."""

    reason: str                      # citation_unsupported | evidence_refused | output_guard | feedback_down
    question: str
    answer: str
    intent: str = ""
    severity: str = "medium"         # low | medium | high
    user_id: str | None = None
    username: str | None = None
    conversation_id: str | None = None
    detail: dict | None = None       # 审计信号（JSON 化后落库）
    sources: list[dict] | None = None  # 引用快照（JSON 化后落库）

    def to_row_kwargs(self) -> dict:
        return {
            "id": uuid.uuid4(),
            "reason": self.reason[:32],
            "severity": self.severity[:16],
            "intent": (self.intent or "")[:64] or None,
            "question": (self.question or "")[:8000],
            "answer": (self.answer or "")[:20000],
            "user_id": self.user_id,
            "username": (self.username or "")[:64] or None,
            "conversation_id": self.conversation_id,
            "detail": json.dumps(self.detail or {}, ensure_ascii=False)[:8000],
            "sources_snapshot": json.dumps(
                self.sources or [], ensure_ascii=False
            )[:20000],
            "status": "open",
        }


async def capture_bad_case(signal: BadCaseSignal) -> bool:
    """
    把一条可疑信号写入 bad_cases 表（best-effort，绝不抛异常）.

    自动回流是"持续监控"的落地动作：指标告诉我们"变差了"，Bad Case 队列
    告诉我们"具体差在哪一句、哪一条引用"，两者缺一不可。
    """
    from app.config import get_settings

    settings = get_settings()
    if not settings.BADCASE_AUTO_CAPTURE:
        return False

    record_badcase(signal.reason)

    try:
        from app.db.badcase_models import BadCase
        from app.db.postgres import get_db_session

        kwargs = signal.to_row_kwargs()
        # user_id 需要是 UUID 或 None（避免字符串污染外键）
        uid = kwargs.get("user_id")
        if uid:
            try:
                kwargs["user_id"] = uuid.UUID(str(uid))
            except (ValueError, TypeError):
                kwargs["user_id"] = None
        cid = kwargs.get("conversation_id")
        if cid:
            try:
                kwargs["conversation_id"] = uuid.UUID(str(cid))
            except (ValueError, TypeError):
                kwargs["conversation_id"] = None

        async with get_db_session() as session:
            session.add(BadCase(**kwargs))
        logger.info(
            "badcase captured: reason=%s severity=%s intent=%s q=%r",
            signal.reason, signal.severity, signal.intent, (signal.question or "")[:80],
        )
        return True
    except Exception as exc:      # noqa: BLE001 — 回流失败不能影响主链路
        logger.warning("Failed to capture bad case (%s): %s", signal.reason, exc)
        return False


# ── 质量事件持久化（比率的历史来源，修复"重启后指标全是 —"）──────────────────

@dataclass
class QualityEventSignal:
    """一轮问答的质量信号（只承载可聚合字段，不存答案正文）."""

    intent: str
    evidence_passed: bool | None = None
    evidence_confidence: float | None = None
    citation_overall: str | None = None
    citation_total: int = 0
    citation_passed: int = 0
    citation_unsupported: int = 0
    citation_hallucinated: int = 0
    citation_misattributed: int = 0
    citation_number_mismatch: int = 0
    citation_date_mismatch: int = 0
    output_guard_changed: bool = False
    refusal_source: str = ""               # gate | model | ""
    sources_count: int = 0
    answer_chars: int = 0
    latency_ms: int | None = None
    user_id: str | None = None
    username: str | None = None
    conversation_id: str | None = None


async def record_quality_event(signal: QualityEventSignal) -> bool:
    """
    把一轮问答的质量信号写入 quality_events 表（best-effort，绝不抛异常）.

    与 Bad Case 回流互补：bad_cases 只记"出问题的轮次"（供人审），
    quality_events 记**每一轮**（供机器算比率）—— 面板上的通过率/拒答率
    从这张表按时间窗聚合，后端重启不再丢历史。
    """
    try:
        from app.db.postgres import get_db_session
        from app.db.quality_models import QualityEvent

        def _uuid(value: str | None):
            if not value:
                return None
            try:
                return uuid.UUID(str(value))
            except (ValueError, TypeError):
                return None

        row = QualityEvent(
            id=uuid.uuid4(),
            user_id=_uuid(signal.user_id),
            username=(signal.username or "")[:64] or None,
            conversation_id=_uuid(signal.conversation_id),
            intent=(signal.intent or "")[:32],
            evidence_passed=signal.evidence_passed,
            evidence_confidence=signal.evidence_confidence,
            citation_overall=(signal.citation_overall or None),
            citation_total=int(signal.citation_total or 0),
            citation_passed=int(signal.citation_passed or 0),
            citation_unsupported=int(signal.citation_unsupported or 0),
            citation_hallucinated=int(signal.citation_hallucinated or 0),
            citation_misattributed=int(signal.citation_misattributed or 0),
            citation_number_mismatch=int(signal.citation_number_mismatch or 0),
            citation_date_mismatch=int(signal.citation_date_mismatch or 0),
            output_guard_changed=bool(signal.output_guard_changed),
            refusal_source=(signal.refusal_source or "")[:8],
            sources_count=int(signal.sources_count or 0),
            answer_chars=int(signal.answer_chars or 0),
            latency_ms=signal.latency_ms,
        )
        async with get_db_session() as session:
            session.add(row)
        return True
    except Exception as exc:      # noqa: BLE001 — 落库失败不能影响主链路
        logger.warning("Failed to record quality event: %s", exc)
        return False


async def quality_stats(window_seconds: float | None) -> dict:
    """
    按时间窗聚合 quality_events（面板比率的历史来源）.

    Args:
        window_seconds: 时间窗秒数；None = 全部历史。

    Returns:
        可直接给前端的结构：总量、五项比率（含样本量）、意图分布、
        时延（avg / p95 / max）、引用校验失败原因分布。
        比率的分母为 0 时返回 None（前端显示 "—"），并附 sample 数，
        让"—是因为没数据"这件事在界面上可解释。
    """
    from sqlalchemy import case, func, select

    from app.db.postgres import get_db_session
    from app.db.quality_models import QualityEvent

    stmt_filter = True
    if window_seconds is not None:
        cutoff = datetime.now(tz=timezone.utc).timestamp() - window_seconds
        stmt_filter = QualityEvent.created_at >= datetime.fromtimestamp(
            cutoff, tz=timezone.utc
        )

    def _ratio(num: int | None, den: int | None) -> float | None:
        if not den:
            return None
        return round((num or 0) / den, 4)

    async with get_db_session() as session:
        base = select(
            func.count().label("total"),
            func.count(QualityEvent.evidence_passed).label("gate_total"),
            func.sum(case((QualityEvent.evidence_passed.is_(False), 1), else_=0)).label("gate_refused"),
            func.sum(QualityEvent.citation_total).label("cit_total"),
            func.sum(QualityEvent.citation_passed).label("cit_passed"),
            func.sum(case((QualityEvent.citation_total > 0, 1), else_=0)).label("cit_rows"),
            func.sum(
                case((
                    (QualityEvent.citation_total > 0)
                    & (QualityEvent.citation_overall.in_(("partial", "unsupported"))),
                    1,
                ), else_=0)
            ).label("cit_doubt_rows"),
            func.sum(case((QualityEvent.output_guard_changed.is_(True), 1), else_=0)).label("guard_changed"),
            func.sum(case((QualityEvent.refusal_source != "", 1), else_=0)).label("refused"),
            func.avg(QualityEvent.latency_ms).label("lat_avg"),
            func.max(QualityEvent.latency_ms).label("lat_max"),
            func.percentile_cont(0.95).within_group(QualityEvent.latency_ms).label("lat_p95"),
            func.sum(QualityEvent.citation_unsupported).label("r_unsupported"),
            func.sum(QualityEvent.citation_hallucinated).label("r_hallucinated"),
            func.sum(QualityEvent.citation_misattributed).label("r_misattributed"),
            func.sum(QualityEvent.citation_number_mismatch).label("r_number"),
            func.sum(QualityEvent.citation_date_mismatch).label("r_date"),
        ).where(stmt_filter)
        row = (await session.execute(base)).one()

        intent_rows = (
            await session.execute(
                select(QualityEvent.intent, func.count())
                .where(stmt_filter)
                .group_by(QualityEvent.intent)
                .order_by(func.count().desc())
            )
        ).all()

    total = int(row.total or 0)
    gate_total = int(row.gate_total or 0)
    cit_total = int(row.cit_total or 0)
    cit_rows = int(row.cit_rows or 0)

    return {
        "total_queries": total,
        "ratios": {
            "evidence_refuse_rate": _ratio(int(row.gate_refused or 0), gate_total),
            "citation_pass_rate": _ratio(int(row.cit_passed or 0), cit_total),
            "citation_unsupported_rate": _ratio(int(row.cit_doubt_rows or 0), cit_rows),
            "output_guard_change_rate": _ratio(int(row.guard_changed or 0), total),
            "refusal_rate": _ratio(int(row.refused or 0), total),
        },
        "samples": {
            "evidence_gate": gate_total,
            "citations_checked": cit_total,
            "citation_answers": cit_rows,
            "queries": total,
        },
        "by_intent": {str(k or "unknown"): int(v) for k, v in intent_rows},
        "latency_ms": {
            "avg": round(float(row.lat_avg), 1) if row.lat_avg is not None else None,
            "p95": round(float(row.lat_p95), 1) if row.lat_p95 is not None else None,
            "max": int(row.lat_max) if row.lat_max is not None else None,
        },
        "citation_failure_reasons": {
            "unsupported": int(row.r_unsupported or 0),
            "hallucinated": int(row.r_hallucinated or 0),
            "misattributed": int(row.r_misattributed or 0),
            "number_mismatch": int(row.r_number or 0),
            "date_mismatch": int(row.r_date or 0),
        },
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


__all__ = [
    "BadCaseSignal",
    "QualityEventSignal",
    "capture_bad_case",
    "metrics_snapshot",
    "quality_stats",
    "record_badcase",
    "record_citation_check",
    "record_counter",
    "record_evidence_gate",
    "record_latency",
    "record_output_guard",
    "record_quality_event",
    "record_query",
    "record_refusal",
    "reset_metrics",
]
