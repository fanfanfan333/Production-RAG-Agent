"""
Retrieval Grader（架构图 Retrieval Grader 节点）.

职责
────
精排之后、生成之前，用 LLM 判断「这批检索结果到底能不能回答用户的问题」。
这是比"精排分数阈值"更强的证据质量评估：

- 分数守卫（现有 HALLUCINATION_GUARD）只能说"最像的那条有多像"，
  回答不了"它像的是不是同一件事"。典型失败：问题问 A 指标，库里只有
  B 指标，B 与问题字面高度相似 → 分数不低 → 模型照着 B 编出 A 的答案。
- LLM grader 直接判"这条证据是否包含回答问题所需的信息"，能抓住
  上述"高分但不对题"的情况。

判定结果用于条件路由（master_graph）：
    good → generate        证据充分，正常生成
    bad  → rewrite + retry 证据不足，改写查询重来（架构图 Retry 分支）

工程约束（与 query_transform 同一套纪律）
────────────────────────────────────────
- 整个调用带超时（RETRIEVAL_GRADER_TIMEOUT_SECONDS）。
- 超时 / 解析失败 / 模型报错 → 一律降级为"用分数守卫的结论"，
  绝不抛异常阻塞主链路 —— grader 是增益不是依赖。
- 只对 top-N（默认前 6 条）做评估，控制延迟；后面的证据对判定影响很小。
"""

from __future__ import annotations

import asyncio
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.services.retrieval_service import RetrievedChunk
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── Prompt ───────────────────────────────────────────────────────────────────

_GRADER_SYSTEM_PROMPT = """\
你是 RAG 系统的检索质量评估器。判断给定的文档片段是否包含回答用户问题所需的信息。

对每个片段输出一个判断：
- "relevant"：该片段确实包含能回答（或部分回答）这个问题的信息
- "irrelevant"：该片段与问题无关，或只有字面相似但实际答不上来

重要：
1. 严格按"能否据此作答"判断，不要被字面相似度误导。
   例如问题问"营收增长率"，片段只讲了"利润率"→ 判 irrelevant。
2. 片段只要包含部分有用信息就判 relevant（不需要完整回答）。
3. 不要评估片段的真伪，只判断相关性。

只输出一个 JSON 对象，不要任何解释：
{"verdicts": ["relevant", "irrelevant", ...]}

数组长度必须与输入片段数完全一致，顺序一致。"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# 只评估前 N 条 —— 后面的证据基本不改变判定，但会显著增加延迟
_GRADER_MAX_CHUNKS = 6


def _extract_json(text: str) -> dict | None:
    """LLM 输出可能带 ```json 围栏或前后杂文 —— 宽松提取第一个 JSON 对象."""
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _score_fallback_verdict(chunks: list[RetrievedChunk]) -> bool:
    """
    降级判定：grader 不可用时，退回原来的"精排分数守卫"逻辑.

    与 rag_graph._route_after_retrieve 保持一致的语义：
    候选为空 / 最高分低于 RERANK_MIN_SCORE → bad。
    """
    settings = get_settings()
    if not chunks:
        return False
    best = max(c.score for c in chunks)
    return best >= settings.RERANK_MIN_SCORE


async def grade_retrieval(
    query: str,
    chunks: list[RetrievedChunk],
) -> tuple[bool, list[bool], str]:
    """
    评估检索结果能否支撑回答 *query*.

    Args:
        query:  用户问题（用改写后的自包含问题效果最好）
        chunks: 精排后的候选，按分数降序

    Returns:
        (is_good, per_chunk_verdicts, reason)
        - is_good:              True = 证据充分，可进入 generate
        - per_chunk_verdicts:   每条 chunk 的 relevant 布尔值（长度 = len(chunks)）
        - reason:               "llm" | "fallback:score" | "disabled" | "no_chunks"
                                便于日志和前端观察路由原因
    """
    settings = get_settings()

    if not chunks:
        return False, [], "no_chunks"

    if not settings.RETRIEVAL_GRADER_ENABLED:
        good = _score_fallback_verdict(chunks)
        return good, [c.score >= settings.RERANK_MIN_SCORE for c in chunks], "disabled"

    # 只评估前几条，控制延迟
    assessed = chunks[:_GRADER_MAX_CHUNKS]

    try:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.0,          # 判定要确定性
            # 同 query_router：相关性判定是简单分类任务，qwen3 的 thinking
            # 实测会让它慢 10 倍以上（7~13s → 0.7s）且判定结果不变，
            # 因此关闭思考以保住 6s 超时预算。
            reasoning=False,
            num_predict=256,          # 需输出一个 verdicts 数组，比路由长
            num_ctx=min(4096, settings.OLLAMA_NUM_CTX),
        )

        # 每条证据截断，避免长文档把 prompt 撑爆（判定相关性看开头足够）
        evidence_lines: list[str] = []
        for i, c in enumerate(assessed, start=1):
            snippet = c.text[:600].replace("\n", " ")
            evidence_lines.append(f"[{i}] {c.filename} (p{c.page_number}): {snippet}")

        user_content = (
            f"用户问题：{query}\n\n"
            f"文档片段：\n" + "\n".join(evidence_lines)
        )

        result = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=_GRADER_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]),
            timeout=settings.RETRIEVAL_GRADER_TIMEOUT_SECONDS,
        )

        parsed = _extract_json(str(result.content))
        if not parsed:
            logger.warning("Retrieval grader returned unparseable output — falling back to score guard")
            return _score_fallback_verdict(chunks), [], "fallback:score"

        raw_verdicts = parsed.get("verdicts")
        if not isinstance(raw_verdicts, list):
            logger.warning("Retrieval grader missing 'verdicts' — falling back to score guard")
            return _score_fallback_verdict(chunks), [], "fallback:score"

        # 对齐长度：模型可能少输出，缺失的按 irrelevant 处理（保守）
        verdicts: list[bool] = []
        for i in range(len(assessed)):
            v = raw_verdicts[i] if i < len(raw_verdicts) else "irrelevant"
            verdicts.append(str(v).strip().lower() == "relevant")

        # 未被评估的尾部 chunk：沿用分数守卫（保守但不过度惩罚）
        tail = [c.score >= settings.RERANK_MIN_SCORE for c in chunks[len(assessed):]]
        all_verdicts = verdicts + tail

        relevant_count = sum(1 for v in verdicts if v)
        is_good = relevant_count >= settings.RETRIEVAL_GRADER_MIN_RELEVANT

        logger.info(
            "retrieval_grader: query=%r relevant=%d/%d → %s",
            query[:60], relevant_count, len(assessed),
            "good" if is_good else "bad",
        )
        return is_good, all_verdicts, "llm"

    except asyncio.TimeoutError:
        logger.warning(
            "Retrieval grader timed out (%.0fs) — falling back to score guard",
            settings.RETRIEVAL_GRADER_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("Retrieval grader failed — falling back to score guard")

    return _score_fallback_verdict(chunks), [], "fallback:score"
