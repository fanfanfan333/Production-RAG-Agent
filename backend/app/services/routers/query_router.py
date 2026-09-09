"""
Query Router（架构图 Query Router 节点）.

职责
────
判断用户问题该走哪条处理分支：

    document_summary  → 文档总结分支（架构图 Document Summary）
    knowledge_qa      → 标准 Hybrid RAG 检索问答（默认，最常见）
    general_chat      → 闲聊，不检索知识库（架构图 General Chat）
    doc_relations     → 跨文档关联分析（已有 relation_service）
    list_documents    → 列出知识库文档（已有确定性 DB 列表）

设计取舍 —— 两级路由
────────────────────
1. 确定性前置（本模块内）：
   "知识库里有哪些文档" 这类列表问题用正则就能 100% 判定，而且必须走
   DB 直读才能保证不漏不重 —— 交给 LLM 反而会引入随机性。这部分沿用
   现有的 _detect_mode 规则，先拦截掉。

2. LLM 判定（其余情况）：
   正则判断不了"总结这份文档" vs "这份文档讲了什么" vs 普通提问的
   语义差别，交给 LLM 更准，也能正确处理多轮上下文里的意图漂移。

工程约束（与 query_transform / retrieval_grader 同一套纪律）
────────────────────────────────────────────────────────────
- 整个调用带超时（ROUTER_TIMEOUT_SECONDS）。
- 超时 / 解析失败 / 未知标签 → 一律降级 knowledge_qa（最常见、最安全的
  分支），绝不抛异常阻塞主链路。
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


# ── 合法意图 ─────────────────────────────────────────────────────────────────

VALID_INTENTS = (
    "document_summary",
    "knowledge_qa",
    "general_chat",
    "doc_relations",
    "list_documents",
)

DEFAULT_INTENT = "knowledge_qa"


# ── 确定性前置规则（沿用 api/query.py 已有的正则，保持行为一致）──────────────

_ASKS_RELATION_RE = re.compile(
    r"关联|关系|联系|相关性|关联度|异同|共同点|相同点|相似之处|重叠|互补|主题分布"
)
_REFERENCES_COLLECTION_RE = re.compile(
    r"(这些|这批|这几个|这几份|各个|所有|全部|哪些|库里|库中|库内|知识库|文档库|上传)[^。？?!\n]{0,12}(文档|文件|资料|报告)"
    r"|文档库|知识库"
    r"|(文档|文件|资料|报告)之间"
)
_ASKS_DOC_LIST_RE = re.compile(
    r"(有哪些|有什么|都有哪些|都有什么|多少个?|哪些|列出|列一下|清单|列表|包含哪些)"
    r"[^。？?!\n]{0,4}(文档|文件|资料|pdf)"
    r"|(文档|文件|资料|pdf)(列表|清单)"
    r"|上传了(哪些|什么)(文档|文件|资料)?"
)


def _deterministic_route(query: str) -> str | None:
    """
    确定性前置判定。命中返回意图，未命中返回 None（交给 LLM）。

    只拦截 100% 确定的情况：
    - 列表问题 → list_documents（必须走 DB 直读，LLM 路由会引入随机性）
    - 跨文档关联 → doc_relations（特征明显，且有专门 pipeline）
    """
    if _ASKS_RELATION_RE.search(query) and _REFERENCES_COLLECTION_RE.search(query):
        return "doc_relations"
    if _ASKS_DOC_LIST_RE.search(query):
        return "list_documents"
    return None


# ── LLM 路由 ─────────────────────────────────────────────────────────────────

_ROUTER_SYSTEM_PROMPT = """\
你是 RAG 系统的查询路由器。判断用户的问题应该走哪条处理分支。

可选的分支（只输出其中一个标签）：
- "document_summary"：用户想让总结/概括/概述某个或某些文档的内容。
  例如："总结一下这份报告"、"这份文档主要讲了什么"、"概括一下第三章"。
- "general_chat"：与知识库无关的闲聊、打招呼、问你是谁、让你写诗/写代码/翻译、
  以及纯常识问题。例如："你好"、"你是谁"、"帮我写一首关于春天的诗"。
- "knowledge_qa"：用户想从知识库里查具体事实、数据、条款、流程等。
  例如："合同里约定的付款期限是多久？"、"营收增长率是多少？"。

判断规则：
1. 只要问题涉及知识库里的具体信息，就选 knowledge_qa。
2. 明确要求"总结/概括/概述/主要讲什么/核心观点"文档内容时，选 document_summary。
3. 只有完全不涉及知识库、纯闲聊或纯创作时，才选 general_chat。
4. 拿不准时选 knowledge_qa（最安全）。

只输出一个 JSON 对象，不要任何解释：
{"intent": "knowledge_qa"}"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# 同 query 的路由结果短时缓存（ROUTER_USE_CACHE）—— 用户连续发相同
# 追问、或前端重试时，避免重复打 LLM。
_route_cache: dict[str, str] = {}
_ROUTE_CACHE_MAX = 256


def _extract_json(text: str) -> dict | None:
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


async def route_query(
    query: str,
    history_messages: list[BaseMessage] | None = None,
) -> tuple[str, str]:
    """
    判定 *query* 应走的处理分支.

    Args:
        query:            用户问题
        history_messages: 最近对话（可选，用于判断多轮里的意图漂移）

    Returns:
        (intent, reason)
        - intent: 一定是 VALID_INTENTS 之一
        - reason: "deterministic" | "llm" | "cache" | "disabled" | "fallback"
    """
    settings = get_settings()

    # ── 1. 确定性前置 ────────────────────────────────────────────────────────
    deterministic = _deterministic_route(query)
    if deterministic:
        logger.info("query_router: %r → %s (deterministic)", query[:60], deterministic)
        return deterministic, "deterministic"

    if not settings.ROUTER_ENABLED:
        return DEFAULT_INTENT, "disabled"

    # ── 2. 缓存 ──────────────────────────────────────────────────────────────
    if settings.ROUTER_USE_CACHE and query in _route_cache:
        return _route_cache[query], "cache"

    # ── 3. LLM 判定 ──────────────────────────────────────────────────────────
    try:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.0,
            # 关于 qwen3 thinking 的实测结论（本地 qwen3:8b）：
            #   reasoning=True  → 7~13s，且 num_predict 给小了会被思考吃光
            #                     导致 content 返回空串
            #   reasoning=False → 0.7s，分类结果完全一致
            # 路由是简单分类任务，思考不带来任何准确率提升，只带来 10 倍
            # 以上的延迟，因此显式关闭。（生成类节点仍保留 reasoning=True）
            reasoning=False,
            num_predict=64,
            num_ctx=min(4096, settings.OLLAMA_NUM_CTX),
        )

        # 带最近一轮历史，帮助判断"再总结一遍"这类省略式追问
        recent = (history_messages or [])[-2:]
        transcript = "\n".join(
            f"{'用户' if m.type == 'human' else '助手'}: {str(m.content)[:200]}"
            for m in recent
        ) or "（无历史）"

        result = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=f"对话历史：\n{transcript}\n\n当前问题：{query}"),
            ]),
            timeout=settings.ROUTER_TIMEOUT_SECONDS,
        )

        parsed = _extract_json(str(result.content))
        intent = ""
        if parsed:
            intent = str(parsed.get("intent", "")).strip().strip('"\'')

        if intent not in VALID_INTENTS:
            # 模型偶尔输出 general_chat/doc_summary 之类近义标签 —— 做一次
            # 宽松归一，仍不认识才降级
            normalized = _normalize_intent(intent)
            if normalized:
                intent = normalized
            else:
                logger.warning(
                    "query_router: unknown intent %r from LLM — defaulting to %s",
                    intent, DEFAULT_INTENT,
                )
                return DEFAULT_INTENT, "fallback"

        if settings.ROUTER_USE_CACHE:
            if len(_route_cache) >= _ROUTE_CACHE_MAX:
                _route_cache.clear()
            _route_cache[query] = intent

        logger.info("query_router: %r → %s (llm)", query[:60], intent)
        return intent, "llm"

    except asyncio.TimeoutError:
        logger.warning(
            "query_router: LLM timed out (%.0fs) — defaulting to %s",
            settings.ROUTER_TIMEOUT_SECONDS, DEFAULT_INTENT,
        )
    except Exception:
        logger.exception("query_router: LLM failed — defaulting to %s", DEFAULT_INTENT)

    return DEFAULT_INTENT, "fallback"


def _normalize_intent(raw: str) -> str | None:
    """把模型输出的近义标签归一化到合法意图."""
    if not raw:
        return None
    r = raw.lower()
    # document_summary 的近义写法
    if any(k in r for k in ("summary", "summar", "doc_summary", "总结", "摘要", "概括")):
        return "document_summary"
    # general_chat 的近义写法
    if any(k in r for k in ("chat", "chitchat", "greet", "casual", "闲聊", "打招呼")):
        return "general_chat"
    # knowledge_qa 的近义写法
    if any(k in r for k in ("qa", "question", "retriev", "search", "rag", "问答", "检索")):
        return "knowledge_qa"
    return None


def clear_route_cache() -> None:
    """清空路由缓存（测试或配置变更后调用）."""
    _route_cache.clear()
