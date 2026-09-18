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


# ── 合法意图 & 确定性前置规则 ───────────────────────────────────────────────
#
# 规则本体在 app/services/routers/intent_rules.py —— 零第三方依赖，
# 可脱离 LLM/配置环境单测（backend/tests/test_intent_rules.py）。
# 这里只做再导出，保持历史导入名不变。

from app.services.routers.intent_rules import (  # noqa: E402
    DEFAULT_INTENT,
    VALID_INTENTS,
    deterministic_route as _deterministic_route,
    looks_knowledge_seeking,
)


# ── LLM 路由 ─────────────────────────────────────────────────────────────────

_ROUTER_SYSTEM_PROMPT = """\
你是 RAG 系统的查询路由器。判断用户的问题应该走哪条处理分支。

可选的分支（只输出其中一个标签）：
- "document_summary"：用户想让总结/概括/概述某个或某些文档的内容。
  例如："总结一下这份报告"、"这份文档主要讲了什么"、"概括一下第三章"、
  **"总结所有文档"、"把所有文档总结一下"、"总结一下知识库"**（整库总结）。
- "document_agent"：用户想**生成一份文档/文件**作为交付物（Word 报告、纪要、
  方案等），而不只是要一段回答。例如："根据知识库生成一份调研报告"、
  "把要点整理成 Word 文档"、"帮我写一份项目方案"。
- "general_chat"：与知识库无关的闲聊、打招呼、问你是谁、让你写诗，
  以及**与用户上传资料无关的**独立创作/常识问题。
  例如："你好"、"你是谁"、"帮我写一首关于春天的诗"、"用 Python 写个快速排序"、
  "今天天气怎么样"。
  ⚠️ 只因为它"听起来像通用知识"就选 general_chat 是错的：凡是在问
  **某个系统/机制/流程/工具/术语是怎么工作的**（如"反幻觉机制是怎么工作的？"、
  "检索流程有哪些步骤？"、"这个参数怎么配置？"），一律选 knowledge_qa ——
  用户资料里很可能就有这段说明，走闲聊等于放弃检索、只能凭记忆编。
- "knowledge_qa"：用户想从知识库里查具体事实、数据、条款、流程等。
  例如："合同里约定的付款期限是多久？"、"营收增长率是多少？"。

判断规则：
1. 只要问题的答案**应该来自用户上传的资料**，就选 knowledge_qa。
2. 明确要求"总结/概括/概述/主要讲什么/核心观点"文档内容时，选 document_summary。
   ⚠️ 提到"所有文档 / 全部文档 / 这些文档 / 整个知识库"的总结请求一律选
   document_summary —— 它是**整库总结**，不是从某一两份文档里找事实。
   判成 knowledge_qa 会让整库总结退化成"只总结了召回分最高的那一份文档"。
3. 明确要求"生成/导出/整理成一份文档、报告、Word 文件"时，选 document_agent。
4. 只有**完全不涉及用户资料**的纯闲聊、纯创作或纯常识时，才选 general_chat。
5. 拿不准时选 knowledge_qa（最安全）。

⚠️ 最容易判错的一类：**技术类"怎么做/怎么配置/参数怎么写/命令是什么/报错怎么解"**。
这类问题看起来像"写代码"，但只要它问的是**某个工具、某个库、某段流程的用法**，
而用户的资料里可能就有这段笔记/手册，就必须选 knowledge_qa —— 选了
general_chat 会绕过检索，模型只能凭记忆编，答得看似合理却与用户资料不符。
判别口径是"答案该不该来自资料"，**不是"这问题像不像技术问题"**。

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
            num_ctx=settings.chat_num_ctx,
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

        # ── 纠偏：把"该走检索的知识提问"从闲聊里捞回来 ──────────────────────
        # 实测 BUG：'反幻觉机制是怎么工作的？' 被判成 general_chat → 整条检索链
        # 被绕过 → 模型凭记忆作答、无引用可核验，界面却毫无异常。
        # 纠偏代价非对称：误判成"该检索"最坏是如实拒答（可解释、可发现）；
        # 误判成闲聊则是静默幻觉（不可见、不可核验）。因此这里从严纠偏。
        if intent == "general_chat" and looks_knowledge_seeking(query):
            logger.warning(
                "query_router: LLM 判为 general_chat，但问句在寻求知识 → "
                "纠偏为 knowledge_qa（query=%r）", query[:60],
            )
            intent = "knowledge_qa"

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
    # document_agent 的近义写法（先于 summary 判断：agent 要"交付物"）
    if any(k in r for k in ("agent", "document_agent", "doc_agent", "生成文档", "导出", "生成报告")):
        return "document_agent"
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
