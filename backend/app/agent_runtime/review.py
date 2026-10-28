"""条件化多 Agent：有界证据交叉审阅.

## 结论先说：不把主图改成多 Agent

调研的四家（Microsoft Agent Framework / Google ADK / OpenAI Agents SDK /
LangGraph supervisor）都支持多 Agent，但它们**没有一家**建议把多 Agent 当成
默认拓扑。LangGraph 官方文档的建议最直白：多数场景用「一个 supervisor + 若干
工具」就够，多 Agent 真正换来的是**职责隔离**，代价是调用次数、时延与故障面
同时上升。

本项目现状是「一张图 + 17 个节点」，节点之间已经通过 `MasterState` 隔离了职责。
把 17 个节点重写成 17 个 Agent，只会得到同样的答案、2~3 倍的 LLM 调用和一条
更难调试的链路。所以这里做的是**局部增强**：

* 只对**确定性规则**判定为复杂的请求（跨文档 + 多跳/对比信号）追加一次审阅；
* 审阅者只输出结构化结论，**不重写答案** —— 答案 token 已经流式发到浏览器了，
  事后重写会造成「用户看到的 ≠ 落库的」，与 Citation Verifier 当年踩过的坑一样；
* 结论只可能**追加一句提示**（append-only），不可能删改正文；
* 默认关闭，单次请求至多 1 次额外 LLM 调用，超时/异常一律静默回退。

## 为什么不让它「辩论」

多角色辩论（自洽性 / Multiagent debate）在开放式创作上有效，在**有证据的问答**
上收益很低：答案对不对由检索片段决定，不由投票决定。多辩论两轮只会放大时延，
并且让「哪一轮的引用是真的」变得不可判定。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "MAX_EXTRA_LLM_CALLS",
    "REVIEW_SYSTEM_PROMPT",
    "ReviewDecision",
    "ReviewVerdict",
    "build_evidence_digest",
    "build_review_prompt",
    "distinct_documents",
    "should_review",
    "parse_verdict",
    "apply_verdict",
]

# 硬上界：一次问答最多为审阅花 1 次 LLM 调用。写死成常量而不是配置项，
# 因为「想要 2 次」这个需求本身应该先过实验证明确有收益。
MAX_EXTRA_LLM_CALLS = 1

MAX_DIGEST_DOCS = 3
# 每份文档最多放几段：只放一段是错的 —— 回答引用的常常是同一份文档的第 3 段，
# 而按文档去重只保留第一段，会让审阅"看不到它要判的那句原文"。
MAX_DIGEST_CHUNKS_PER_DOC = 2
MAX_DIGEST_CHARS = 1600
MAX_DIGEST_PER_DOC = 400
MAX_REVIEW_ANSWER_CHARS = 1800

# 这些引用校验结论意味着"答案本身已经不可信"，此时再花一次调用没有意义，
# 且审阅追加的软提示会与 verifier 的硬结论互相打架。
# 取值必须与 `nodes/citation_verifier.py` 的 VerificationReport.overall 一致
# （verified | partial | unsupported | no_citations | refused_by_model）。
_BLOCKING_CITATION_OVERALL = frozenset({"unsupported", "refused_by_model"})

# 跨文档信号（用户明确要比对多份材料）
_CROSS_DOC_MARKERS = (
    "对比", "比较", "差异", "区别", "分别", "各自", "相比", "不同之处",
    "冲突", "不一致", "哪个更", "谁更", "vs", "versus", "compare", "differ",
    "conflict", "两份", "三份", "多份",
)
# 多跳信号（答案需要串起两个以上事实）
_MULTIHOP_MARKERS = (
    "然后", "之后", "先后", "再根据", "基于", "前序", "上游", "下游",
    "继承", "传递", "间接", "第一步", "第二步", "chain", "then", "derived",
)

REVIEW_SYSTEM_PROMPT = (
    "你是证据审阅员，只做一件事：判断给定回答的每条结论是否都能被给定证据片段支持。\n"
    "规则：\n"
    "1. 只依据证据片段判断，不使用你自己的领域知识；\n"
    "2. 证据之间互相矛盾，或关键结论缺少对应片段，就必须指出；\n"
    "3. 不要重写回答，不要补充新事实，不要给出引用编号以外的内容；\n"
    "4. 只输出一个 JSON 对象，不要 Markdown 代码块，不要解释。\n"
    '输出格式：{"status": "supported|needs_caveat|insufficient", "note": "不超过40字的中文说明",'
    ' "unsupported_claims": ["原样摘录不被支持的结论，最多3条"]}\n'
    "status 取值含义：supported=全部结论有据可依；needs_caveat=部分结论证据不足或证据冲突；"
    "insufficient=关键结论缺乏证据。"
)

_VERDICT_STATUSES = ("supported", "needs_caveat", "insufficient")

# 审阅模型的输入里有不可信文档片段，输出可能被引导成携带控制字符/换行的长文本。
# note 与 unsupported_claims 会进 trace 落库并可能被前端展示，落库前一律压平。
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean_verdict_text(text: object, limit: int) -> str:
    cleaned = _CTRL_RE.sub("", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit]

_CAVEAT_TEXT = {
    "needs_caveat": "\n\n> 交叉审阅：部分结论的证据支持较弱，建议以引用原文复核。",
    "insufficient": "\n\n> 交叉审阅：关键结论缺少直接证据，请以引用原文为准并谨慎采用。",
}


@dataclass(frozen=True)
class ReviewDecision:
    """是否触发审阅的确定性判定结果（含理由，用于审计与回放）."""

    fire: bool
    reason: str
    document_count: int
    signals: tuple[str, ...]

    def as_audit(self) -> dict:
        return {
            "fire": self.fire,
            "reason": self.reason,
            "document_count": self.document_count,
            "signals": list(self.signals),
            "max_extra_llm_calls": MAX_EXTRA_LLM_CALLS,
        }


@dataclass(frozen=True)
class ReviewVerdict:
    status: str
    note: str = ""
    unsupported_claims: tuple[str, ...] = ()
    raw_ok: bool = True

    def as_audit(self) -> dict:
        return {
            "status": self.status,
            "note": self.note,
            "unsupported_claims": list(self.unsupported_claims),
            "parsed": self.raw_ok,
        }


def distinct_documents(sources: Sequence[dict]) -> int:
    ids = set()
    for source in sources or ():
        doc_id = source.get("document_id") or source.get("doc_id")
        name = source.get("document_name") or source.get("filename") or source.get("title")
        key = str(doc_id or name or "")
        if key:
            ids.add(key)
    return len(ids)


def should_review(
    *,
    enabled: bool,
    intent: str,
    query: str,
    sources: Sequence[dict],
    evidence_passed: bool,
    citation_overall: str = "",
) -> ReviewDecision:
    """确定性判定是否值得为这一轮多花一次 LLM 调用.

    只有同时满足「跨文档 + 多跳/对比信号 + 证据门控已通过」才触发。三个条件是
    与关系而不是或关系 —— 单文档问题不需要交叉审阅，无信号的问题不需要多跳
    审阅，证据没通过门控的问题已经在走拒答。
    """
    doc_count = distinct_documents(sources)
    if not enabled:
        return ReviewDecision(False, "disabled", doc_count, ())
    if intent != "knowledge_qa":
        return ReviewDecision(False, f"intent={intent} 不参与审阅", doc_count, ())
    if not evidence_passed:
        return ReviewDecision(False, "evidence_gate 未通过", doc_count, ())
    if citation_overall in _BLOCKING_CITATION_OVERALL:
        return ReviewDecision(
            False,
            f"引用校验={citation_overall}，优先由 Citation Verifier 处置",
            doc_count,
            (),
        )
    if doc_count < 2:
        return ReviewDecision(False, f"仅 {doc_count} 份文档，无需交叉审阅", doc_count, ())

    lowered = (query or "").lower()
    signals: list[str] = [f"doc:{m}" for m in _CROSS_DOC_MARKERS if m in lowered]
    signals += [f"hop:{m}" for m in _MULTIHOP_MARKERS if m in lowered]
    if not signals:
        return ReviewDecision(False, "缺少跨文档/多跳信号", doc_count, ())
    return ReviewDecision(True, "跨文档 + 信号命中", doc_count, tuple(signals))


def _source_text(source: dict) -> str:
    """取该来源**可用的最长正文**.

    ``text`` 是 citation_verifier 补进来的完整 chunk 正文；``text_snippet`` 是
    给前端展示的 300 字预览。审阅必须用前者：拿预览当原文会系统性地把
    「数字落在截断之后」判成「不被支持」—— 与引用校验当年踩的是同一个坑
    （见 `master_graph._sources_with_full_text` 的注释）。
    """
    for key in ("text", "text_snippet", "snippet", "content"):
        value = source.get(key)
        if value:
            text = str(value).strip()
            if text:
                return text
    return ""


def build_evidence_digest(sources: Sequence[dict], *, answer: str = "") -> str:
    """把来源压成审阅用的短证据摘要（只用于审阅，不外发）.

    排序规则：**回答里引用到的片段排在最前**。审阅要回答的问题是"这些结论
    有没有据"，先把被引用的证据给它，比按检索分数给更省 token 也更准。

    Args:
        sources: 证据来源（最好已由 ``_sources_with_full_text`` 补上 ``text``）。
        answer: 待审阅答案，用于识别 ``[n]`` 形式的引用编号（1 基，对应 sources 顺序）。
    """
    rows: list[dict] = []
    for order, source in enumerate(sources or (), start=1):
        text = _source_text(source)
        if not text:
            continue
        doc_key = str(
            source.get("document_id") or source.get("document_name") or source.get("title") or ""
        )
        name = str(source.get("document_name") or source.get("title") or doc_key or "未命名文档")
        rows.append(
            {
                "order": order,
                "doc": doc_key or name,
                "name": name,
                "chunk": str(source.get("chunk_index") if source.get("chunk_index") is not None else ""),
                "text": text,
            }
        )
    if not rows:
        return ""

    cited = {int(match) for match in re.findall(r"\[(\d+)\]", answer or "")}
    rows.sort(key=lambda row: (0 if row["order"] in cited else 1, row["order"]))

    blocks: list[str] = []
    used = 0
    seen: set[tuple[str, str]] = set()
    per_doc: dict[str, int] = {}
    for row in rows:
        key = (row["doc"], row["chunk"])
        if key in seen:
            continue
        if per_doc.get(row["doc"], 0) >= MAX_DIGEST_CHUNKS_PER_DOC:
            continue
        # 新文档已超过上限就跳过，但同文档的第二段仍允许补进来
        if row["doc"] not in per_doc and len(per_doc) >= MAX_DIGEST_DOCS:
            continue
        block = f"[{row['name']}] {row['text'][:MAX_DIGEST_PER_DOC]}"
        if used + len(block) > MAX_DIGEST_CHARS:
            break
        seen.add(key)
        per_doc[row["doc"]] = per_doc.get(row["doc"], 0) + 1
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_review_prompt(*, question: str, answer: str, evidence_digest: str) -> str:
    """构造审阅请求正文（单条 user 消息）."""
    return (
        f"问题：{question}\n\n"
        f"待审阅回答：\n{answer[:MAX_REVIEW_ANSWER_CHARS]}\n\n"
        f"证据片段：\n{evidence_digest or '（无）'}\n\n"
        "按系统提示的 JSON 格式输出审阅结论。"
    )


def _extract_json(raw: str) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None
    # 模型偶尔仍然包一层 ```json，剥掉再解析；不追求容错，解析失败就回退。
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_verdict(raw: str) -> ReviewVerdict:
    """解析审阅模型输出；解析不了就当作「未产生可用结论」（status=supported）.

    这里刻意**失败即放行**：审阅是增强项，解析失败不该让用户看到一句
    「交叉审阅失败」的噪声，也不该改变答案。
    """
    parsed = _extract_json(raw)
    if not parsed:
        return ReviewVerdict(status="supported", raw_ok=False)
    status = str(parsed.get("status") or "").strip().lower()
    if status not in _VERDICT_STATUSES:
        return ReviewVerdict(status="supported", raw_ok=False)
    note = _clean_verdict_text(parsed.get("note"), 80)
    claims = parsed.get("unsupported_claims") or []
    if not isinstance(claims, list):
        claims = []
    claims = tuple(
        _clean_verdict_text(item, 120) for item in claims[:3] if _clean_verdict_text(item, 120)
    )
    return ReviewVerdict(status=status, note=note, unsupported_claims=claims)


def apply_verdict(answer: str, verdict: ReviewVerdict) -> tuple[str, bool]:
    """按结论决定是否在答案尾部追加提示；返回 (新答案, 是否追加).

    append-only：不删、不改、不重排正文。已追加过同类提示时不重复追加
    （重试或重复审阅都不该让提示堆叠）。
    """
    if verdict.status == "supported":
        return answer, False
    suffix = _CAVEAT_TEXT.get(verdict.status)
    if not suffix:
        return answer, False
    if suffix.strip() in (answer or ""):
        return answer, False
    return answer + suffix, True
