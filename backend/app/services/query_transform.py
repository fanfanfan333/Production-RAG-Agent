"""
查询改写层（召回优化 / 抗幻觉第一道防线）.

两个功能，一次 LLM 调用完成：

1. 指代消解改写（condense question）
   多轮对话里用户常用"它 / 上面那个 / 第三个"等指代，直接拿去检索会
   丢召回。这里把「历史 + 当前问题」压缩成一个自包含的独立问题再检索。

2. 多查询扩展（multi-query expansion）
   一个问题生成多个表述变体，多路向量召回后用 RRF 融合 —— 不同表述
   命中不同语义邻域，显著降低"换个问法就查不到"的漏召回。

工程约束：
- 整个调用带超时（QUERY_REWRITE_TIMEOUT_SECONDS），超时/失败一律回退
  原查询 —— 改写是增益不是依赖，绝不能拖垮主链路。
- 只在有历史或启用多查询时才调用，空历史的短问题直接跳过省延迟。
"""

from __future__ import annotations

import asyncio
import json
import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_REWRITE_SYSTEM_PROMPT = """\
你是 RAG 检索系统的查询改写器。根据对话历史改写用户的问题，用于文档检索。

规则：
1. 把问题改写成一个不依赖对话历史、自包含的独立问题（消解所有"它/这个/上面/之前"等指代）。
2. 生成 2 个语义等价但表述不同的检索变体（换关键词、换角度），用于提升召回。
3. 改写后的文本必须只保留问题本身，不要回答，不要解释。
4. 如果问题已经自包含，"rewritten" 就原样返回问题。

只输出一个 JSON 对象，格式：
{"rewritten": "改写后的独立问题", "variants": ["变体1", "变体2"]}"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


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


async def rewrite_query_with_history(
    query: str,
    history_messages: list[BaseMessage],
) -> tuple[str, list[str]]:
    """
    改写查询并生成检索变体.

    Returns:
        (rewritten_query, variants) —— 失败/超时/禁用时返回 (query, []).
    """
    settings = get_settings()
    if not settings.QUERY_REWRITE_ENABLED:
        return query, []

    has_history = bool(history_messages)
    need_expansion = settings.MULTI_QUERY_ENABLED and settings.MULTI_QUERY_VARIANTS > 0

    # 空历史且未启用扩展 —— 没有可改写的东西，直接跳过 LLM 调用
    if not has_history and not need_expansion:
        return query, []

    try:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.0,          # 改写要确定性，不要创造性
            num_predict=256,          # 短输出，控延迟
            num_ctx=min(4096, settings.OLLAMA_NUM_CTX),  # 改写 prompt 很短，小窗省内存
        )

        # 只带最近几条历史进改写 prompt —— 指代基本都出现在最近一两轮
        recent = history_messages[-4:] if has_history else []
        transcript = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {str(m.content)[:300]}"
            for m in recent
        ) or "（无历史）"

        messages = [
            SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=f"对话历史：\n{transcript}\n\n当前问题：{query}"),
        ]

        result = await asyncio.wait_for(
            llm.ainvoke(messages),
            timeout=settings.QUERY_REWRITE_TIMEOUT_SECONDS,
        )

        parsed = _extract_json(str(result.content))
        if not parsed:
            logger.warning("Query rewrite returned unparseable output — using original")
            return query, []

        rewritten = str(parsed.get("rewritten", "")).strip()
        raw_variants = parsed.get("variants", [])
        variants = [
            str(v).strip()
            for v in (raw_variants if isinstance(raw_variants, list) else [])
            if str(v).strip() and str(v).strip() != query
        ][: settings.MULTI_QUERY_VARIANTS]

        final = rewritten if len(rewritten) >= 2 else query
        if final != query:
            logger.info("Query rewritten: %r → %r (+%d variants)", query[:60], final[:60], len(variants))
        return final, variants

    except asyncio.TimeoutError:
        logger.warning("Query rewrite timed out (%.0fs) — using original query",
                       settings.QUERY_REWRITE_TIMEOUT_SECONDS)
    except Exception:
        logger.exception("Query rewrite failed — using original query")
    return query, []
