"""
Evidence Gate 节点（架构图 Retrieval Grader → Evidence Gate → Generate/Refuse）.

为什么在 Grader 之后还要一道门
────────────────────────────
Grader 是 **LLM 语义判断**："这段证据能不能回答问题"。它能抓住"高分但不对题"，
但它有三个结构性弱点：

1. 它本身也会出错 —— 判断错了就直接把幻觉放行到生成阶段；
2. 它没有"量"的概念 —— 一条 12 字的碎片和一段完整段落，在它眼里都是
   "relevant"；
3. 它在超时/异常时会降级为"放行"（fail-open）。

Evidence Gate 是这道门之后**确定性的、可复现的、fail-closed 的**兜底：
不看语义，只看"证据的客观形态"——条数够不够、最高分够不够、问题里的关键
词在证据里覆盖率够不够、证据本身是不是空壳。任何一条不达标就拒答。

    证据不足 → 拒答（而不是让模型硬答）

它与 Grader 是**互补**关系，不是替代：

    Grader      看语义  → "这段话说的是不是这个问题"     fail-open
    Evidence Gate 看形态 → "这些证据够不够撑起一个回答"     fail-closed

两者都通过才进入生成；任一不通过则走拒答。这样"宁可拒答，也不硬答"不再
只依赖一个会出错的 LLM 判断。

同时本模块给出**统一的拒答文案**，让"模型主动拒答"与"门控拒答"在用户
看来是同一种体验（见 REFUSAL_ANSWER 与 is_refusal()）。

本模块为纯函数、零第三方依赖，可直接单测（tests/test_evidence_gate.py）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── 统一拒答文案 ─────────────────────────────────────────────────────────────
#
# 门控拒答（evidence_gate）与模型主动拒答（prompt 里的规则 9）使用同一段
# 文本：用户看到的拒答不因"是谁拒的"而不同，Bad Case 回流时也只需匹配
# 一个字符串即可识别"这是一次拒答"。
REFUSAL_ANSWER = (
    "抱歉，我在当前知识库中没有找到与这个问题足够相关的信息，"
    "因此无法给出有依据的回答。\n\n"
    "建议：\n"
    "1. 尝试换一种问法，或提供更具体的关键词；\n"
    "2. 确认相关文档已经上传并完成索引；\n"
    "3. 检查是否选择了正确的知识库分组。"
)

# 模型被允许（且被要求）在证据不足时输出这句话 —— 见 rag_graph._SYSTEM_TEMPLATE
# 的规则 9。用一句**短而独特**的锚点句，便于确定性识别"模型自己拒答了"。
MODEL_REFUSAL_SENTINEL = "抱歉，我在当前知识库中没有找到与这个问题足够相关的信息"


def is_refusal(answer: str) -> bool:
    """判断一段答案是否是"拒答"（模型主动拒答或门控拒答）."""
    if not answer:
        return False
    head = answer.strip()[:80]
    return MODEL_REFUSAL_SENTINEL[:20] in head


# ── 判定结果 ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvidenceSignal:
    """单条证据信号（可解释、可审计）."""

    name: str
    passed: bool
    detail: str
    value: float | None = None
    threshold: float | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class EvidenceDecision:
    """Evidence Gate 的判定结果."""

    passed: bool
    reason: str
    confidence: float                       # 0–1：证据强度（不是"答案正确率"）
    signals: tuple[EvidenceSignal, ...] = ()
    evidence_count: int = 0
    top_score: float = 0.0
    coverage: float = 0.0

    def as_audit(self) -> dict:
        """给 SSE / 审计用的紧凑结构."""
        return {
            "passed": self.passed,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count,
            "top_score": round(self.top_score, 4),
            "coverage": round(self.coverage, 4),
            "failed_signals": [
                s.name for s in self.signals if not s.passed
            ],
            "signals": [s.as_dict() for s in self.signals],
        }


# ── 覆盖率计算 ───────────────────────────────────────────────────────────────
#
# 为什么**不**复用 BM25 的字符 bigram：
#
#   bigram 对词的切分边界极其敏感。问题里写「营收」（口语/缩写），文档里写
#   「营业收入」（正式全称）时，两者的 bigram 集合
#       「营收」      → {营收}
#       「营业收入」  → {营业, 业收, 收入}
#   交集为 **0**。于是证据明明就是答案所在，覆盖率却是 0，门控会把一次正确
#   检索判成"跑题"直接拒答 —— 这是比"漏拦"严重得多的错误。
#
# 因此覆盖率改用**内容单字 + ASCII 词**：
#   - CJK 取单字，消除词边界敏感性（营收 / 营业收入 都能命中「营」「收」）；
#   - 剔除高频虚词，避免「的/是/在/和」这类字把不相关文本的覆盖率抬上去
#     （不做剔除时，一段完全无关的中文考勤制度也能和合同问题凑出 0.9 的
#     "覆盖率"，信号就废了）；
#   - ASCII 仍按整词匹配（型号、编号、版本号等必须精确）。

_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_RUN_RE = re.compile(r"[a-z0-9]+")

# 高频虚词/功能字：几乎不承载查询意图，留着只会虚高覆盖率。
# 刻意**不**收录 为/以/对/从/到/上/下/中/大/小/多/少 等兼有实义的字。
_CJK_STOP_CHARS = frozenset(
    "的了是在和与及或也不很都就还而之其所以因为但如果虽然这那们个把被让使"
    "可将会能要且并则若此该我你他她它有无没太更最又再只才已正吗呢吧啊着过"
    "很一些什么怎样如何为何多少"
)


def content_terms(text: str) -> set[str]:
    """
    抽取"内容词"：CJK 单字（去虚词）+ ASCII 整词.

    这是覆盖率信号的比对单位。刻意比 BM25 的 bigram 更粗、更稳 ——
    它只需要回答"证据和问题是不是在说同一件事"，不需要精确到词形。
    """
    if not text:
        return set()
    lowered = text.lower()
    terms: set[str] = set(_ASCII_RUN_RE.findall(lowered))
    terms.update(
        ch for ch in _CJK_CHAR_RE.findall(text) if ch not in _CJK_STOP_CHARS
    )
    return terms


def query_coverage(query: str, evidence_text: str) -> float:
    """
    问题内容词在证据里的覆盖率 ∈ [0, 1].

    问题没有任何内容词（纯标点 / 纯虚词）时返回 1.0 —— 无从判断，
    就不因覆盖率拒答，把决定权交给其它信号。
    """
    q_terms = content_terms(query or "")
    if not q_terms:
        return 1.0
    ev_terms = content_terms(evidence_text or "")
    if not ev_terms:
        return 0.0
    return len(q_terms & ev_terms) / len(q_terms)


# ── 主判定 ───────────────────────────────────────────────────────────────────

def evaluate_evidence(
    query: str,
    chunks: list,
    *,
    min_chunks: int = 1,
    min_top_score: float = 0.25,
    min_coverage: float = 0.20,
    min_evidence_chars: int = 20,
    enabled: bool = True,
) -> EvidenceDecision:
    """
    对检索结果做确定性证据门控.

    Args:
        query:   用户问题（改写后的自包含问题最佳）.
        chunks:  精排后的检索结果（需具备 ``.text`` 与 ``.score`` 属性）.
        min_chunks:           至少 N 条证据.
        min_top_score:        最高精排分下限（与 RERANK_MIN_SCORE 对齐）.
        min_coverage:         问题关键词覆盖率下限.
        min_evidence_chars:   证据正文总长度下限 —— 这是一道**空壳检查**，
                              只拦"检索返回了空/垃圾 payload"这种情况，因此
                              阈值刻意很低（默认 20）。设高会误杀合法的短
                              结构化块（例如一张 40 字的小表格本身就能回答
                              "营收是多少"），而证据质量已由分数与覆盖率把关。
        enabled:              False 时直接放行（返回 passed=True）.

    Returns:
        EvidenceDecision —— passed=False 时调用方应当拒答.
    """
    if not enabled:
        return EvidenceDecision(
            passed=True, reason="evidence_gate_disabled", confidence=1.0
        )

    chunks = chunks or []
    signals: list[EvidenceSignal] = []

    # ── 信号 1：证据条数 ─────────────────────────────────────────────────────
    count = len(chunks)
    signals.append(EvidenceSignal(
        name="has_evidence",
        passed=count >= max(1, min_chunks),
        detail=f"{count} 条证据（要求 ≥ {max(1, min_chunks)}）",
        value=float(count),
        threshold=float(max(1, min_chunks)),
    ))

    if count == 0:
        return EvidenceDecision(
            passed=False,
            reason="no_evidence",
            confidence=0.0,
            signals=tuple(signals),
            evidence_count=0,
            top_score=0.0,
            coverage=0.0,
        )

    texts: list[str] = []
    scores: list[float] = []
    for c in chunks:
        text = getattr(c, "text", "") or ""
        # 图片块没有正文时用它的可检索文本兜底（已在检索层合并），此处不再特殊处理
        texts.append(text)
        try:
            scores.append(float(getattr(c, "score", 0.0) or 0.0))
        except (TypeError, ValueError):
            scores.append(0.0)

    top_score = max(scores) if scores else 0.0
    evidence_text = "\n".join(texts)
    coverage = query_coverage(query, evidence_text)
    total_chars = len(evidence_text.strip())

    # ── 信号 2：最高精排分 ───────────────────────────────────────────────────
    signals.append(EvidenceSignal(
        name="top_score",
        passed=top_score >= min_top_score,
        detail=f"最高精排分 {top_score:.3f}（要求 ≥ {min_top_score:.2f}）",
        value=round(top_score, 4),
        threshold=min_top_score,
    ))

    # ── 信号 3：问题关键词覆盖率 ─────────────────────────────────────────────
    signals.append(EvidenceSignal(
        name="query_coverage",
        passed=coverage >= min_coverage,
        detail=f"问题关键词覆盖率 {coverage:.0%}（要求 ≥ {min_coverage:.0%}）",
        value=round(coverage, 4),
        threshold=min_coverage,
    ))

    # ── 信号 4：证据正文长度（空壳检查）──────────────────────────────────────
    signals.append(EvidenceSignal(
        name="evidence_length",
        passed=total_chars >= max(0, min_evidence_chars),
        detail=f"证据正文 {total_chars} 字（要求 ≥ {max(0, min_evidence_chars)}）",
        value=float(total_chars),
        threshold=float(max(0, min_evidence_chars)),
    ))

    failed = [s for s in signals if not s.passed]

    # 置信度：把通过的信号按"权重"折算成一个 0–1 的强度值。
    # 条数与长度为"资格信号"（不通过直接 0），分数与覆盖率决定强度。
    if not failed:
        confidence = 0.5 * min(1.0, top_score) + 0.5 * coverage
    else:
        confidence = 0.0

    if failed:
        reason = "insufficient_evidence:" + ",".join(s.name for s in failed)
        logger.info(
            "evidence_gate: REFUSE — %s (chunks=%d top=%.3f coverage=%.2f chars=%d)",
            reason, count, top_score, coverage, total_chars,
        )
    else:
        reason = "evidence_sufficient"
        logger.info(
            "evidence_gate: PASS — chunks=%d top=%.3f coverage=%.2f chars=%d confidence=%.2f",
            count, top_score, coverage, total_chars, confidence,
        )

    return EvidenceDecision(
        passed=not failed,
        reason=reason,
        confidence=round(confidence, 4),
        signals=tuple(signals),
        evidence_count=count,
        top_score=round(top_score, 4),
        coverage=round(coverage, 4),
    )


__all__ = [
    "EvidenceDecision",
    "EvidenceSignal",
    "MODEL_REFUSAL_SENTINEL",
    "REFUSAL_ANSWER",
    "evaluate_evidence",
    "is_refusal",
    "query_coverage",
]
