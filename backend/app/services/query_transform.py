"""
查询改写层（召回优化 / 抗幻觉第一道防线）.

一次 LLM 调用同时产出四样东西
──────────────────────────────
1. **指代消解改写（condense）**
   多轮对话里用户常说"它 / 上面那个 / 第三个"，直接检索会丢召回。改写为标准
   的自包含问题再检索。

2. **多查询扩展（multi-query）**
   一个问题生成若干语义等价但表述不同的变体，多路召回后 RRF 融合，降低
   "换个问法就查不到"。

3. **查询分解（decomposition，新）**
   多跳 / 对比类问题（"A 和 B 的差异""先…再…"）在单次检索里往往两头都够不着：
   Top-K 会被其中一个子话题占满。拆成子问题分别召回再融合，让两个子话题
   都进候选池。

4. **HyDE（Hypothetical Document Embeddings，新）**
   先让模型写一段"假如答案存在，它大概长这样"的假设性段落，用**这段的向量**
   去召回。对"用户用口语提问、文档用书面术语陈述"的错配特别有效 —— 假设段落
   自动把口语映射到文档语体，比单纯的同义改写更接近目标分布。
   ⚠️ HyDE 的产物**只用于召回**，绝不进上下文（它可能是编的），所以它对
   幻觉的影响是中性偏正：召回面变宽，证据仍由真实文档提供。

改写为什么必须有"防漂移闸门"
────────────────────────────
改写器是**会出错**的一环：它可能把"2024 年"改成"2023 年"、把「营收」
理解成「利润」、或者对短问题过度发挥写出一段看似合理但主题偏移的检索式。

这类错误有个危险特性：**后续所有防线都拦不住它**。引用校验只能校验"答案是否
被检索到的原文支持"，而检索本身已经沿着错误方向跑偏了 —— 结果是一份引用
完全自洽、但回答的其实是另一个问题的答案。用户无法察觉。

所以这里做确定性的双向校验（``_passes_drift_gate``）：
  · 与原问题的词元重叠率过低 → 判为漂移，丢弃改写，回退原查询；
  · 原问题里的**数字/编号词元**在改写后消失 → 直接丢弃（数字是硬约束，
    丢失它等于换了问题）。
丢弃改写的代价只是"这次没获得召回增益"，而接受了漂移改写的代价是
"静默答错"—— 两者不对称，所以宁可保守。

工程约束：
- 整个调用带超时（QUERY_REWRITE_TIMEOUT_SECONDS），超时/失败一律回退原查询；
- 结果按 (问题, 历史指纹) 做 TTL + LRU 缓存 —— 企业内网里同一批问题被反复问
  是常态（月报、制度类查询），缓存直接省掉一次 LLM 往返；
- 明显是**精确查找**的问题（型号/编号/标准号）走快路径，完全不调 LLM：
  这类查询靠 BM25 精确命中即可，任何改写都只会引入偏差；
- **数据/指令隔离**：历史在进改写 prompt 前先 normalize_text +
  sanitize_document_context，屏蔽其中可能混入的可执行指令片段。改写器只应
  该看到"事实与问题"，不应该被历史里的指令撬动。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.config import get_settings
from app.services.prompt_security import (
    normalize_text,
    sanitize_document_context,
)
from app.utils.logging import get_logger

if TYPE_CHECKING:      # 仅类型标注；运行时不导入（见 _invoke_rewrite_llm 的说明）
    from langchain_core.messages import BaseMessage

logger = get_logger(__name__)

_REWRITE_SYSTEM_PROMPT = """\
你是 RAG 检索系统的查询改写器。你的输出只用于**检索文档**，不用于回答。

严格遵守：
1. 绝不改变问题里的数字、年份、编号、专有名词（它们是硬约束，改了就等于换了问题）。
2. 不要回答问题，不要解释，不要输出与检索无关的内容。

请产出四个字段：

- "rewritten": 把问题改写成一个不依赖对话历史、自包含的独立问题（消解所有
  "它/这个/上面/之前"等指代）。若问题本身已自包含，原样返回。

- "variants": 2 个语义等价但表述不同的检索变体（换关键词、换角度），
  用于多路召回。每个变体都必须是完整可检索的问句或短语。

- "subqueries": 仅当问题包含**多个子话题或需要多步推理**时，拆成 2-3 个
  可独立检索的子问题；否则返回空数组 []。

- "hyde": 一段 100-200 字的**假设性文档片段** —— 设想知识库里回答该问题的
  段落大概会怎么写（用书面语、用文档里可能出现的术语，可以包含数值）。
  只写假设的文档内容，不要写"这个问题涉及…"这类元叙述。

只输出 JSON：
{"rewritten": "...", "variants": ["...", "..."], "subqueries": [], "hyde": "..."}"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# 精确查找类问题：型号 / 标准号 / 编号 / 法规号 / 合同号。这类问题的检索质量
# 几乎完全由精确词匹配决定，改写只会稀释它 —— 而且模型很可能"顺手"把型号
# 补全或改成一个相近型号，那是灾难性的：召回了一份长得像但不同型号的文档，
# 用户从答案上根本看不出来。
#
# 判据需要**两个信号**同时成立，缺一不可，否则会把普通问题误判成编号查找：
#   · _MIXED_CODE_RE（字母与数字混合、允许分隔符）—— "ABX-300"、"ITIL4"
#   · 或 字母词 + 4 位以上数字 的组合 —— "GB/T 19001"、"ISO 9001"
# 而 "2024年营业收入是多少" 只有**裸的年份数字、没有任何字母**，不满足 → 走
# 正常 LLM 改写（对这类问题改写是有价值的）。这是上一版只按"含数字"判断时
# 的误判，已被这条规则修掉。
_MIXED_CODE_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_./]*\d")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]{2,}")
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{4,}(?!\d)")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 数值词元：出现在原问题里却在改写后消失 → 直接判漂移
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class RewriteResult:
    """改写结果（字段全部有安全默认，失败时等价于"什么都没做"）."""

    rewritten: str
    variants: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    hyde: str | None = None
    source: str = "original"      # original | llm | cache | fastpath | drift_rejected
    drift_score: float = 1.0

    def extra_queries(self) -> list[str]:
        """
        除主查询外需要**额外**检索的查询列表.

        顺序即优先级：分解出的子问题排在最前（它们是"必答项"），
        HyDE 段落列最后（它是分布对齐工具，语义上最远）。
        """
        out: list[str] = []
        for item in list(self.subqueries) + list(self.variants):
            if item and item != self.rewritten and item not in out:
                out.append(item)
        if self.hyde and self.hyde not in out:
            out.append(self.hyde)
        return out

    def to_dict(self) -> dict:
        return {
            "rewritten": self.rewritten,
            "variants": list(self.variants),
            "subqueries": list(self.subqueries),
            "hyde_chars": len(self.hyde or ""),
            "source": self.source,
            "drift_score": round(self.drift_score, 3),
        }


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


# ── 缓存（TTL + LRU）─────────────────────────────────────────────────────────
# 企业内网的问题分布高度重复（"今年营收多少""差旅报销标准"），一次改写的成本
# 是 1-3 秒 LLM 往返。缓存命中率通常很高，是性价比最高的一处优化。
# 用 OrderedDict 手写 LRU 而不是 functools.lru_cache：需要 TTL，也需要在
# 多事件循环/线程下可控（lru_cache 会跨测试用例串味）。

_rewrite_cache: "OrderedDict[str, tuple[float, RewriteResult]]" = OrderedDict()


def _cache_get(key: str, ttl: float) -> RewriteResult | None:
    hit = _rewrite_cache.get(key)
    if hit is None:
        return None
    ts, value = hit
    if time.monotonic() - ts > ttl:
        _rewrite_cache.pop(key, None)
        return None
    _rewrite_cache.move_to_end(key)
    return RewriteResult(**{**value.__dict__, "source": "cache"})


def _cache_put(key: str, value: RewriteResult, max_entries: int) -> None:
    _rewrite_cache[key] = (time.monotonic(), value)
    _rewrite_cache.move_to_end(key)
    while len(_rewrite_cache) > max(1, max_entries):
        _rewrite_cache.popitem(last=False)


def clear_rewrite_cache() -> None:
    """清空改写缓存（测试与配置变更后使用）."""
    _rewrite_cache.clear()


def history_fingerprint(history_messages: list[BaseMessage], depth: int = 4) -> str:
    """历史指纹：只有"最近几条"参与，够区分上下文即可，不必求哈希稳定."""
    if not history_messages:
        return "-"
    parts = []
    for m in history_messages[-depth:]:
        parts.append(str(getattr(m, "content", ""))[:120])
    return "|".join(parts)


# ── 防漂移闸门 ────────────────────────────────────────────────────────────────


def _token_set(text: str) -> set[str]:
    from app.services.hybrid_search import tokenize

    return set(tokenize(text or ""))


def _passes_drift_gate(original: str, candidate: str, min_overlap: float) -> tuple[bool, float]:
    """
    判断 *candidate* 相对 *original* 是否算"漂移".

    Returns:
        (是否通过, 重叠分)

    两个条件，任一不满足即拒绝：
      1. 词元重叠率 —— 用 **min 作分母**（而不是原问题的词元数）：改写会消解
         指代、补全省略，长度通常变长；用原问题长度作分母会把正常改写误判成漂移。
         用两边的较小者作分母，衡量的是"较短的那一方被覆盖了多少"。
      2. 数字保全 —— 原问题里的数字必须全部出现在改写里。数字是硬约束
         （年份/金额/指标代码），丢一个就变成另一个问题。
    """
    if not candidate or not candidate.strip():
        return False, 0.0
    cand_tokens = _token_set(candidate)
    orig_tokens = _token_set(original)
    if not orig_tokens:
        return True, 1.0
    if not cand_tokens:
        return False, 0.0

    inter = len(orig_tokens & cand_tokens)
    overlap = inter / max(1, min(len(orig_tokens), len(cand_tokens)))

    if overlap < min_overlap:
        return False, overlap

    missing_numbers = {
        n for n in _NUMBER_RE.findall(original)
        if n not in candidate
    }
    if missing_numbers:
        logger.info(
            "drift gate: rewrite dropped number(s) %s — rejected",
            sorted(missing_numbers)[:5],
        )
        return False, overlap

    return True, overlap


def is_identifier_lookup(query: str, max_chars: int) -> bool:
    """
    是否属于"精确查找"类问题（型号 / 标准号 / 编号）→ 走快路径跳过 LLM.

    三个条件必须同时成立：
      1. **很短**（<= *max_chars*）—— 长问题说明用户在提问，不是在报编号；
      2. **含代码样词元**（字母+数字混合，或字母词 + 4 位以上数字）；
      3. **汉字很少**（<= 6）—— "ABX-300 的参数"是查型号，"ABX-300 的技术参数
         指标分别是多少"是一个完整问题，后者改写是有价值的。
    """
    q = (query or "").strip()
    if not q or len(q) > max_chars:
        return False

    has_code = bool(_MIXED_CODE_RE.search(q)) or (
        bool(_ASCII_LETTER_RE.search(q)) and bool(_LONG_NUMBER_RE.search(q))
    )
    if not has_code:
        return False

    return len(_CJK_RE.findall(q)) <= 6


# ── 主入口 ────────────────────────────────────────────────────────────────────


async def rewrite_query(
    query: str,
    history_messages: list[BaseMessage] | None = None,
) -> RewriteResult:
    """
    改写查询并产出检索变体 / 子问题 / HyDE 段落.

    任何失败路径都返回 ``RewriteResult(rewritten=query)`` —— 改写是增益而非
    依赖，绝不能拖垮主链路。
    """
    settings = get_settings()
    history_messages = history_messages or []

    if not settings.QUERY_REWRITE_ENABLED:
        return RewriteResult(rewritten=query, source="original")

    has_history = bool(history_messages)
    want_expansion = settings.MULTI_QUERY_ENABLED and settings.MULTI_QUERY_VARIANTS > 0
    want_decomp = settings.QUERY_DECOMPOSITION_ENABLED
    want_hyde = settings.QUERY_HYDE_ENABLED

    # ── 快路径 1：没有任何改写工作可做 ────────────────────────────────────────
    if not has_history and not (want_expansion or want_decomp or want_hyde):
        return RewriteResult(rewritten=query, source="fastpath")

    # ── 快路径 2：精确查找（型号/编号）不调 LLM ───────────────────────────────
    if is_identifier_lookup(query, settings.QUERY_REWRITE_FASTPATH_MAX_CHARS):
        logger.info("rewrite fastpath: identifier lookup query=%r — skipping LLM", query[:60])
        return RewriteResult(rewritten=query, source="fastpath")

    cache_key = f"{query}\x00{history_fingerprint(history_messages)}"
    cached = _cache_get(cache_key, settings.QUERY_REWRITE_CACHE_TTL_SECONDS)
    if cached is not None:
        logger.debug("rewrite cache hit for query=%r", query[:60])
        return cached

    try:
        # 延迟导入：langchain_ollama 会拉起 langchain 全家桶（导入成本数百毫秒），
        # 而本模块的纯逻辑（防漂移闸门 / 快路径判定）在单测与诊断脚本里都要用 ——
        # 让它们不必为了调一个纯函数而付出整条 LLM 依赖链的代价。
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.0,          # 改写要确定性，不要创造性
            # 关于 qwen3 thinking 的实测结论（本地 qwen3:8b，真实改写提示词）：
            #   reasoning=True  → 38~44s，且思考 token 会吃光 num_predict
            #   reasoning=False → 10.7s，JSON 合法、字段更完整
            # 改写是"结构化转换"任务（把问题转成 JSON 形式的查询集合），不是
            # 需要多步推理的任务 —— 思考不提升产出质量，只带来 4 倍延迟。
            #
            # ⚠️ 这一条漏了非常久：query_router 与 graders/retrieval_grader 都写了
            # reasoning=False，唯独改写器没写，而 QUERY_REWRITE_TIMEOUT_SECONDS
            # 是 12s —— 于是改写**每轮都超时**，静默回退原查询。整条查询增强层
            # （改写 / 多查询扩展 / 子问题 / HyDE）在生产里从未真正生效过，
            # 表现为"配置全开着、日志里却一条 rewrite 产物都没有"。
            reasoning=False,
            num_predict=768,          # rewritten+variants+subqueries+hyde，比原先长
            num_ctx=min(4096, settings.OLLAMA_NUM_CTX),
            format="json",            # 让 Ollama 保证输出是 JSON，省掉一轮解析失败
        )
        if has_history:
            # 先 normalize_text（Unicode 双向/零宽控制符归一化），再
            # sanitize_document_context 就地屏蔽可执行指令片段。这条清洗与
            # 喂给生成节点的文档清洗走同一套规则，保证「历史 = 数据」。
            recent = history_messages[-4:]
            transcript_parts: list[str] = []
            for m in recent:
                normalized = normalize_text(str(m.content)[:300])
                safe, _masked = sanitize_document_context(normalized)
                role = "用户" if isinstance(m, HumanMessage) else "助手"
                transcript_parts.append(f"{role}: {safe}")
            transcript = "\n".join(transcript_parts)
            user_block = f"对话历史：\n{transcript}\n\n当前问题：{query}"
        else:
            user_block = f"当前问题：{query}"

        result = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=user_block),
            ]),
            timeout=settings.QUERY_REWRITE_TIMEOUT_SECONDS,
        )

        parsed = _extract_json(str(result.content))
        if not parsed:
            logger.warning("Query rewrite returned unparseable output — using original")
            return RewriteResult(rewritten=query, source="original")

        # ── rewritten 过防漂移闸门 ────────────────────────────────────────────
        raw_rewritten = str(parsed.get("rewritten", "")).strip()[:400]
        rewritten = query
        source = "original"
        drift = 1.0
        if len(raw_rewritten) >= 2 and raw_rewritten != query:
            ok, drift = _passes_drift_gate(
                query, raw_rewritten, settings.QUERY_REWRITE_MIN_OVERLAP,
            )
            if ok:
                rewritten = raw_rewritten
                source = "llm"
            else:
                source = "drift_rejected"
                logger.warning(
                    "rewrite rejected by drift gate (overlap=%.2f): %r → %r",
                    drift, query[:60], raw_rewritten[:60],
                )

        def _clean_list(value, cap: int) -> list[str]:
            if not isinstance(value, list):
                return []
            out: list[str] = []
            for item in value:
                text = str(item).strip()[:400]
                if len(text) < 2 or text in (query, rewritten) or text in out:
                    continue
                # 变体/子问题不与原问题**完全脱节**才收（宽松闸门：0.15）
                ok, _score = _passes_drift_gate(
                    query, text, max(0.15, settings.QUERY_REWRITE_MIN_OVERLAP / 2),
                )
                if not ok:
                    logger.debug("dropping off-topic variant %r", text[:60])
                    continue
                out.append(text)
                if len(out) >= cap:
                    break
            return out

        variants = _clean_list(parsed.get("variants"), settings.MULTI_QUERY_VARIANTS) \
            if want_expansion else []
        subqueries = _clean_list(parsed.get("subqueries"), settings.QUERY_DECOMPOSITION_MAX) \
            if want_decomp else []

        hyde: str | None = None
        if want_hyde:
            raw_hyde = str(parsed.get("hyde", "")).strip()
            raw_hyde = raw_hyde[: settings.QUERY_HYDE_MAX_CHARS]
            if len(raw_hyde) >= 40:
                ok, _score = _passes_drift_gate(
                    query, raw_hyde, max(0.15, settings.QUERY_REWRITE_MIN_OVERLAP / 2),
                )
                if ok:
                    hyde = raw_hyde
                else:
                    logger.debug("dropping off-topic HyDE passage (len=%d)", len(raw_hyde))

        result_obj = RewriteResult(
            rewritten=rewritten,
            variants=variants,
            subqueries=subqueries,
            hyde=hyde,
            source=source,
            drift_score=drift,
        )

        if source == "llm" or variants or subqueries or hyde:
            logger.info(
                "rewrite: %r → %r (+%d variants, +%d subqueries, hyde=%d chars, source=%s)",
                query[:60], rewritten[:60], len(variants), len(subqueries),
                len(hyde or ""), source,
            )

        # 只在"真的产出过东西"时缓存 —— 缓存一次空结果没有收益，
        # 还会让后续同样的瞬时故障一直被复用。
        if source in ("llm", "drift_rejected") or variants or subqueries or hyde:
            _cache_put(cache_key, result_obj, settings.QUERY_REWRITE_CACHE_MAX_ENTRIES)
        return result_obj

    except asyncio.TimeoutError:
        logger.warning(
            "Query rewrite timed out (%.0fs) — using original query",
            settings.QUERY_REWRITE_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("Query rewrite failed — using original query")
    return RewriteResult(rewritten=query, source="original")
