"""
Master RAG Graph（架构图完整流水线的统一入口）.

拓扑
────
                          ┌─ list_documents  → list_docs        ─┐
                          ├─ doc_relations   → digest → analyze  ─┤
    START → router ───────┼─ document_summary→ digest → summarize─┼→ save_history → END
                          ├─ general_chat    → chat              ─┤
                          └─ knowledge_qa    → rewrite → retrieve │
                                                    ↓            │
                                                  grade          │
                                              ↓         ↓        │
                                          generate    (bad)      │
                                              │      ↓     ↓     │
                                              │  rewrite  refuse ┘
                                              │  (retry, ≤N次)

对应架构图各节点
────────────────
- Query Router      → route_query (routers/query_router.py, LLM + 确定性前置)
- Document Summary  → collect_summary_digests + summarize (nodes/document_summary_node.py)
- General Chat      → chat (nodes/general_chat_node.py, 不检索)
- Hybrid RAG        → rewrite → retrieve（复用 query_transform + retrieval_service，
                      内部已是 BM25 + 向量 RRF + cross-encoder 精排，本图不重写）
- Retrieval Grader  → grade (graders/retrieval_grader.py, LLM 证据评估)
- Retry/Rewrite     → grade 判定 bad 时回边到 rewrite，最多 N 次后 refuse
- Context Builder   → nodes/context_builder.py（generate 节点使用）
- Citation/Source   → sources 事件（沿用现有 SSE 协议）

复用原则
────────
rag_graph.py 里已经跑通的流式、历史落库、错误中文化、文档列表，
这里全部 import 复用，不重复实现。本文件只负责"编排"和"新增节点"。
"""

from __future__ import annotations

import uuid
from typing import Any, AsyncGenerator, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.services.conversation_service import save_turn
from app.services.graders.retrieval_grader import grade_retrieval
from app.services.nodes.context_builder import build_context
from app.services.nodes.document_summary_node import (
    build_summary_llm,
    build_summary_messages,
    collect_summary_digests,
)
from app.services.nodes.general_chat_node import (
    build_general_chat_llm,
    build_general_chat_messages,
)
from app.services.query_transform import rewrite_query_with_history
from app.services.rag_graph import (
    _RELATIONS_SYSTEM_TEMPLATE,
    _SYSTEM_TEMPLATE,
    _friendly_error,
    _stream_llm_answer,
    _trim_history,
)
from app.services.relation_service import build_digest_context, collect_document_digests, digest_sources
from app.services.retrieval_service import RetrievedChunk, retrieve_chunks
from app.services.routers.query_router import route_query
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── State ────────────────────────────────────────────────────────────────────

class MasterState(TypedDict):
    """Shared mutable state threaded through every node of the master graph."""

    # 输入
    query: str
    conversation_id: str
    history_messages: list[BaseMessage]
    top_k: int
    owner_id: str | None
    collection_id: str | None
    # 强制模式（前端显式指定时跳过路由）
    forced_mode: str | None

    # 路由
    intent: str
    intent_reason: str

    # Knowledge QA / Retry
    rewritten_query: str
    query_variants: list[str]
    chunks: list[RetrievedChunk]
    sources: list[dict]
    grade_good: bool
    grade_reason: str
    retry_count: int

    # Document Summary / doc_relations 共用
    digests: list[dict]

    # 输出
    answer: str


# ── 节点：路由 ───────────────────────────────────────────────────────────────

async def _route_node(state: MasterState) -> dict:
    """
    Query Router 节点.

    前端显式传 mode 时直接采用（保持向后兼容），否则走 route_query：
    确定性规则优先，其余交给 LLM。
    """
    forced = state.get("forced_mode")
    if forced:
        logger.info("master_route: using forced mode=%s", forced)
        return {"intent": forced, "intent_reason": "forced"}

    if not state.get("intent"):
        intent, reason = await route_query(
            state["query"], state.get("history_messages")
        )
        return {"intent": intent, "intent_reason": reason}

    # 已经路由过（重试回边）→ 保持原意图
    return {}


def _route_by_intent(state: MasterState) -> str:
    """Router 的条件分支."""
    intent = state.get("intent") or "knowledge_qa"
    if intent == "list_documents":
        return "list_docs"
    if intent == "doc_relations":
        return "collect_digests"
    if intent == "document_summary":
        return "summary_digests"
    if intent == "general_chat":
        return "chat"
    return "rewrite"


# ── 节点：Knowledge QA（rewrite → retrieve → grade）────────────────────────────

async def _rewrite_node(state: MasterState) -> dict:
    """
    Query Rewrite 节点（架构图 Query Rewrite）.

    首次进入：指代消解 + 多查询扩展。
    重试进入（grade=bad）：在已有变体基础上再生成新角度的查询，
    打不同的语义邻域，避免同一个问法反复查不到。
    """
    is_retry = state.get("retry_count", 0) > 0

    rewritten, variants = await rewrite_query_with_history(
        query=state["query"],
        history_messages=state["history_messages"],
    )

    if is_retry:
        settings = get_settings()
        # 重试时把上一次的查询也当作一个变体保留，多路召回覆盖面更大
        prev = state.get("rewritten_query")
        if prev and prev not in variants and prev != rewritten:
            variants = [prev] + variants
        if settings.RETRIEVAL_RETRY_ADD_VARIANTS:
            variants = variants[: settings.MULTI_QUERY_VARIANTS + 1]
        logger.info(
            "master_rewrite: retry #%d — rewriting with %d variants",
            state["retry_count"], len(variants),
        )

    return {"rewritten_query": rewritten, "query_variants": variants}


async def _retrieve_node(state: MasterState) -> dict:
    """
    Hybrid RAG 检索节点（架构图 Hybrid Retrieval）.

    内部已是：多路向量 ANN + BM25 → RRF 融合 → cross-encoder 精排。
    这里不重写检索逻辑，只负责调用 + 生成前端 sources。
    """
    chunks = await retrieve_chunks(
        query=state.get("rewritten_query") or state["query"],
        top_k=state["top_k"],
        owner_id=state.get("owner_id"),
        collection_id=state.get("collection_id"),
        extra_queries=state.get("query_variants"),
    )

    sources = [
        {
            "document_id": c.document_id,
            "filename": c.filename,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "text_snippet": c.text[:300],
            "score": round(c.score, 4),
        }
        for c in chunks
    ]

    logger.info(
        "master_retrieve: %d chunks (retry=%d) for query=%r",
        len(chunks), state.get("retry_count", 0),
        (state.get("rewritten_query") or state["query"])[:80],
    )
    return {"chunks": chunks, "sources": sources}


async def _grade_node(state: MasterState) -> dict:
    """
    Retrieval Grader 节点（架构图 Retrieval Grader）.

    用 LLM 判断"这批证据到底能不能回答问题"，比精排分数阈值更能抓住
    "高分但不对题"的情况。失败/超时自动降级为分数守卫。
    """
    settings = get_settings()

    chunks = state.get("chunks") or []
    # 关掉 grader 时，仍保留原有的分数守卫语义（HALLUCINATION_GUARD）
    if not settings.RETRIEVAL_GRADER_ENABLED and not settings.HALLUCINATION_GUARD_ENABLED:
        return {"grade_good": True, "grade_reason": "disabled"}

    good, _verdicts, reason = await grade_retrieval(
        query=state.get("rewritten_query") or state["query"],
        chunks=chunks,
    )

    retry = state.get("retry_count", 0)
    if not good:
        retry += 1
        logger.info(
            "master_grade: bad → retry #%d (%s)", retry, reason
        )

    return {
        "grade_good": good,
        "grade_reason": reason,
        "retry_count": retry,
    }


def _route_after_grade(state: MasterState) -> str:
    """
    Grade 之后的条件路由（架构图 Good / Bad → Retry）.

    good                          → generate
    bad 且还有重试次数            → rewrite（回边，构成 Retry 循环）
    bad 且重试次数用尽            → refuse
    """
    settings = get_settings()
    if state.get("grade_good"):
        return "generate"

    if state.get("retry_count", 0) > settings.RETRIEVAL_MAX_RETRIES:
        logger.info(
            "master_route: retries exhausted (%d > %d) — refusing",
            state.get("retry_count", 0), settings.RETRIEVAL_MAX_RETRIES,
        )
        return "refuse"

    return "rewrite"


async def _generate_node(state: MasterState) -> dict:
    """
    生成节点 —— 用 Context Builder 组装上下文（含 small-to-big 父块回填
    与提示注入清洗），再流式输出。
    """
    settings = get_settings()
    llm = _build_streaming_llm(settings)

    built = build_context(state.get("chunks") or [])

    focused_query = (
        f"[指令：只回答下面这个问题。"
        f"不要重复或以之前轮次的内容作为开头。]\n\n"
        f"{state['query']}"
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=_SYSTEM_TEMPLATE.format(context=built.context)),
        *_trim_history(state["history_messages"]),
        HumanMessage(content=focused_query),
    ]

    answer = await _stream_llm_answer(llm, messages)

    # sources 以 Context Builder 的为准（它带 expanded_to_parent 标记）
    return {"answer": answer, "sources": built.sources or state.get("sources", [])}


async def _refuse_node(state: MasterState) -> dict:
    """重试用尽仍无足够证据 → 明确拒答，不硬编."""
    answer = (
        "抱歉，我在当前知识库中没有找到与这个问题足够相关的信息，"
        "因此无法给出有依据的回答。\n\n"
        "建议：\n"
        "1. 尝试换一种问法，或提供更具体的关键词；\n"
        "2. 确认相关文档已经上传并完成索引；\n"
        "3. 检查是否选择了正确的知识库分组。"
    )
    logger.info(
        "master_refuse: refusing query=%r after %d retries (%s)",
        state["query"][:80], state.get("retry_count", 0),
        state.get("grade_reason", ""),
    )
    return {"answer": answer}


# ── 节点：Document Summary ───────────────────────────────────────────────────

async def _summary_digests_node(state: MasterState) -> dict:
    """拉取全库文档摘要（复用 relation_service 的采样逻辑）."""
    digests = await collect_summary_digests(owner_id=state.get("owner_id"))
    return {"digests": digests}


async def _summarize_node(state: MasterState) -> dict:
    """文档总结生成."""
    llm = build_summary_llm()
    messages = build_summary_messages(
        state["query"], state.get("digests") or [], state["history_messages"]
    )
    answer = await _stream_llm_answer(llm, messages)
    return {"answer": answer}


# ── 节点：doc_relations（沿用原有能力）─────────────────────────────────────────

async def _collect_digests_node(state: MasterState) -> dict:
    digests = await collect_document_digests(owner_id=state.get("owner_id"))
    return {"digests": digest_sources(digests)}


async def _analyze_relations_node(state: MasterState) -> dict:
    settings = get_settings()
    llm = _build_streaming_llm(settings)

    digests = state.get("digests") or []
    context = (
        build_digest_context(digests)
        if digests
        else "The knowledge base currently contains no indexed documents."
    )
    focused_query = (
        f"[指令：分析知识库中这些文档之间的关联，"
        f"并按要求的格式以文档为单位组织回答。]\n\n"
        f"{state['query']}"
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=_RELATIONS_SYSTEM_TEMPLATE.format(context=context)),
        *_trim_history(state["history_messages"]),
        HumanMessage(content=focused_query),
    ]
    answer = await _stream_llm_answer(llm, messages)
    return {"answer": answer}


# ── 节点：General Chat ───────────────────────────────────────────────────────

async def _chat_node(state: MasterState) -> dict:
    """闲聊分支 —— 完全不检索，直接 LLM."""
    llm = build_general_chat_llm()
    messages = build_general_chat_messages(state["query"], state["history_messages"])
    answer = await _stream_llm_answer(llm, messages)
    return {"answer": answer}


# ── 节点：落库 ───────────────────────────────────────────────────────────────

async def _save_history_node(state: MasterState) -> dict:
    await save_turn(
        conversation_id=uuid.UUID(state["conversation_id"]),
        user_message=state["query"],
        assistant_message=state["answer"],
    )
    return {}


def _build_streaming_llm(settings):
    """与 rag_graph._build_llm 一致：qwen3 流式 + 思考流分离."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.1,
        streaming=True,
        reasoning=True,
    )


# ── Graph 编译 ───────────────────────────────────────────────────────────────

def _compile_master_graph():
    graph = StateGraph(MasterState)

    # 路由
    graph.add_node("route", _route_node)
    # Knowledge QA 链
    graph.add_node("rewrite", _rewrite_node)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("grade", _grade_node)
    graph.add_node("generate", _generate_node)
    graph.add_node("refuse", _refuse_node)
    # Document Summary
    graph.add_node("summary_digests", _summary_digests_node)
    graph.add_node("summarize", _summarize_node)
    # doc_relations
    graph.add_node("collect_digests", _collect_digests_node)
    graph.add_node("analyze_relations", _analyze_relations_node)
    # General Chat
    graph.add_node("chat", _chat_node)
    # 收尾
    graph.add_node("save_history", _save_history_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        _route_by_intent,
        {
            "rewrite": "rewrite",
            "summary_digests": "summary_digests",
            "collect_digests": "collect_digests",
            "chat": "chat",
            "list_docs": "save_history",   # list_documents 由 API 层单独处理
        },
    )

    # Knowledge QA：rewrite → retrieve → grade → {generate | rewrite | refuse}
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        _route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",   # ← Retry 回边
            "refuse": "refuse",
        },
    )

    # 其余分支
    graph.add_edge("summary_digests", "summarize")
    graph.add_edge("summarize", "save_history")
    graph.add_edge("collect_digests", "analyze_relations")
    graph.add_edge("analyze_relations", "save_history")
    graph.add_edge("chat", "save_history")
    graph.add_edge("generate", "save_history")
    graph.add_edge("refuse", "save_history")
    graph.add_edge("save_history", END)

    return graph.compile()


_master_graph = None


def get_master_graph():
    global _master_graph
    if _master_graph is None:
        _master_graph = _compile_master_graph()
        logger.info("LangGraph master graph compiled and cached")
    return _master_graph


# ── 流式事件循环 ─────────────────────────────────────────────────────────────

async def stream_master(
    query: str,
    conversation_id: str,
    history_messages: list[BaseMessage],
    top_k: int,
    owner_id: str | None = None,
    collection_id: str | None = None,
    forced_mode: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    执行 master graph 并产出 SSE 事件字典.

    Event types（在现有协议上新增 3 种，旧类型保持不变）：
        {"type": "intent",   "intent": "...", "reason": "..."}   路由结果
        {"type": "sources",  "sources": [...], "retrieved_count": N}
        {"type": "doc_digests", "documents": [...], "retrieved_count": N}
        {"type": "grade",    "good": bool, "retry": N, "reason": "..."}
        {"type": "chunk",    "content": "<token>"}
        {"type": "thinking_delta", "content": "<token>"}
        {"type": "done", "conversation_id": "...", "total_chars": N}
        {"type": "error", "message": "..."}
    """
    initial_state: MasterState = {
        "query": query,
        "conversation_id": conversation_id,
        "history_messages": history_messages,
        "top_k": top_k,
        "owner_id": owner_id,
        "collection_id": collection_id,
        "forced_mode": forced_mode,
        "intent": "",
        "intent_reason": "",
        "rewritten_query": "",
        "query_variants": [],
        "chunks": [],
        "sources": [],
        "grade_good": False,
        "grade_reason": "",
        "retry_count": 0,
        "digests": [],
        "answer": "",
    }

    graph = get_master_graph()
    total_chars = 0
    intent_emitted = False
    sources_emitted = False

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event["event"]
            name = event.get("name", "")

            # ── 路由结果 ─────────────────────────────────────────────────────
            if kind == "on_chain_end" and name == "route" and not intent_emitted:
                output = event["data"].get("output", {}) or {}
                intent = output.get("intent") or "knowledge_qa"
                yield {
                    "type": "intent",
                    "intent": intent,
                    "reason": output.get("intent_reason", ""),
                }
                intent_emitted = True

            # ── 检索结果 ─────────────────────────────────────────────────────
            elif kind == "on_chain_end" and name == "retrieve":
                output = event["data"].get("output", {}) or {}
                sources = output.get("sources", [])
                yield {
                    "type": "sources",
                    "sources": sources,
                    "retrieved_count": len(sources),
                }
                sources_emitted = True

            # ── 文档摘要（summary / relations）──────────────────────────────
            elif kind == "on_chain_end" and name in ("summary_digests", "collect_digests"):
                output = event["data"].get("output", {}) or {}
                digests = output.get("digests", [])
                yield {
                    "type": "doc_digests",
                    "documents": digests,
                    "retrieved_count": len(digests),
                }
                sources_emitted = True

            # ── Grader 判定 ─────────────────────────────────────────────────
            elif kind == "on_chain_end" and name == "grade":
                output = event["data"].get("output", {}) or {}
                good = bool(output.get("grade_good"))
                yield {
                    "type": "grade",
                    "good": good,
                    "retry": output.get("retry_count", 0),
                    "reason": output.get("grade_reason", ""),
                }

            # ── 拒答（refuse 不调 LLM，文本要手动转发）────────────────────
            elif kind == "on_chain_end" and name == "refuse":
                refusal = str((event["data"].get("output", {}) or {}).get("answer", ""))
                if refusal:
                    total_chars += len(refusal)
                    yield {"type": "chunk", "content": refusal}

            # ── LLM token 流 ────────────────────────────────────────────────
            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk is None:
                    continue

                reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
                if reasoning:
                    total_chars += len(reasoning)
                    yield {"type": "thinking_delta", "content": str(reasoning)}

                raw = chunk.content if hasattr(chunk, "content") else ""
                if isinstance(raw, list):
                    token = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in raw
                    )
                else:
                    token = str(raw)

                if token:
                    total_chars += len(token)
                    yield {"type": "chunk", "content": token}

        yield {
            "type": "done",
            "conversation_id": conversation_id,
            "total_chars": total_chars,
        }

    except Exception as e:
        logger.error(f"Error during master graph generation: {e}")
        yield {"type": "error", "message": _friendly_error(e)}
