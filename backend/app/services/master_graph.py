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

生成路径统一增加 Output Guard 节点（问题3+问题4，架构图 Generate → Citation
Check → END）：在所有 LLM 产出答案后做四层合规检查（引用越界 / 系统提示词
泄露 / 闲聊分支幻觉措辞 / 工具调用意图），保证最终答案符合 Prompt 隔离与
Agent 工具权限控制要求。refuse 节点本身的拒答文本已是合规答案，不进入
output_guard。

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
- Citation Check    → nodes/output_guard_node.py（生成后合规校验）
- Output Guard      → nodes/output_guard_node.py（系统词泄露 / 工具权限控制）
- Citation/Source   → sources 事件（沿用现有 SSE 协议）

复用原则
────────
rag_graph.py 里已经跑通的流式、历史落库、错误中文化、文档列表，
这里全部 import 复用，不重复实现。本文件只负责"编排"和"新增节点"。
"""

from __future__ import annotations

import time
import uuid
from typing import Any, AsyncGenerator, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.services.conversation_service import save_turn
from app.services.graders.retrieval_grader import grade_retrieval
from app.services.monitoring_service import (
    BadCaseSignal,
    QualityEventSignal,
    capture_bad_case,
    record_citation_check,
    record_evidence_gate,
    record_latency,
    record_output_guard,
    record_quality_event,
    record_query,
    record_refusal,
)
from app.services.nodes.citation_verifier import verify_citations
from app.services.nodes.context_builder import build_context
from app.services.nodes.document_summary_node import (
    build_overview_messages,
    build_single_doc_messages,
    build_summary_llm,
    build_target_not_found_answer,
    collect_summary_digests,
    resolve_summary_targets,
)
from app.services.nodes.evidence_gate import (
    REFUSAL_ANSWER,
    evaluate_evidence,
    is_refusal,
)
from app.services.nodes.general_chat_node import (
    build_general_chat_llm,
    build_general_chat_messages,
)
from app.services.nodes.multimodal_context_node import (
    build_multimodal_context,
    image_url_for as _image_url,
)
from app.services.nodes.output_guard_node import inspect_output
from app.services.prompt_security import (
    sanitize_document_context,
    sanitize_retrieval_query,
)
from app.services.query_transform import rewrite_query
from app.services.rag_graph import (
    _RELATIONS_SYSTEM_TEMPLATE,
    _SYSTEM_TEMPLATE,
    _friendly_error,
    _stream_llm_answer,
    _trim_history,
)
from app.services.relation_service import (
    build_digest_context,
    collect_document_digests,
    digest_sources,
    list_accessible_documents,
)
from app.services.retrieval_service import RetrievedChunk, retrieve_chunks
from app.services.routers.query_router import route_query
from app.services.stream_channel import bind_sink, emit_text, reset_sink
from app.services.stream_filter import (
    INTERNAL_LLM_NODES,
    STREAMING_LLM_NODES,
    resolve_llm_node,
    should_stream_token,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── LLM 节点分类：谁的输出能给用户看 ────────────────────────────────────────
#
# master graph 里共有 7 个节点会调用 LLM，但产出的东西性质完全不同：
#
#   生成类（_STREAMING_LLM_NODES）—— 输出就是答案正文，必须流式转发给用户；
#   决策类（_INTERNAL_LLM_NODES） —— 输出是内部中间数据，绝不能外泄。
#
# LangGraph 的 astream_events 对**所有** LLM 调用都会发 on_chat_model_stream，
# 包括用 llm.ainvoke 发的决策类调用（LangChain 会把 ainvoke 包装成单个 chunk）。
# 之前 stream_master 不加区分地全量转发，于是 route / rewrite / grade 的 JSON
# 被拼进了答案 —— 截图1 里那句
#     {"rewritten": "你是谁", "variants": [...]}
# 就是 rewrite 节点的输出。判定逻辑与单测见 app/services/stream_filter.py。
_STREAMING_LLM_NODES = STREAMING_LLM_NODES
_INTERNAL_LLM_NODES = INTERNAL_LLM_NODES


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
    # Multi-Tenant 三层隔离：tenant_id 第一层（检索前置过滤），
    # department_id 第二层（Document ACL），user_id 第三层（会话/消息归属）。
    # tenant_id=None 只表示"无公司边界 = 平台管理员"，此时仍受 ACL 约束：
    # tenant_wide 放宽部门维度，platform_wide 跨公司，个人库则永远只看自己的。
    tenant_id: str | None
    department_id: str | None
    tenant_wide: bool
    platform_wide: bool
    user_id: str | None
    # 强制模式（前端显式指定时跳过路由）
    forced_mode: str | None

    # 路由
    intent: str
    intent_reason: str

    # Knowledge QA / Retry
    rewritten_query: str
    query_variants: list[str]     # 语义等价改写变体（进两条腿）
    query_extra: list[str]        # 子问题 + 变体，进**两条**检索腿
    query_hyde: str | None        # 假设答案段落，**只**进向量腿
    chunks: list[RetrievedChunk]
    sources: list[dict]
    grade_good: bool
    grade_reason: str
    retry_count: int

    # Multimodal Context（部分5+6）：图文分流 + Vision 看图后的最终上下文
    multimodal_context: str
    context_blocks: list[dict]
    context_images: list[dict]
    vision_used: int
    vision_available: bool

    # Evidence Gate（确定性证据门控）：passed=False 时直接拒答，不进生成
    evidence: dict
    # Citation Verifier（五项引用校验）：引用存在 / 位置正确 / 原文支持 / 数字 / 日期
    citation_check: dict
    # 拒答来源：gate（门控拒答）| model（模型主动拒答）| ""（未拒答）
    refusal_source: str

    # Document Agent（最终效果：Word 写入 / Word 插入图片）
    document: dict

    # Document Summary / doc_relations 共用
    digests: list[dict]
    # Document Summary 的范围（用户点名了哪份文档 / 是否没对上）
    summary_scope: dict

    # 输出
    answer: str
    # Output Guard（问题3+问题4）审计信号：citations_removed / leaked_phrases /
    # hallucination_phrases / tool_attempt_phrases / changed。
    output_guard: dict


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
    # knowledge_qa 与 document_agent 共用同一条检索链，
    # 在 multimodal_context 之后再分流（见 _route_after_context）。
    return "rewrite"


# ── 节点：Knowledge QA（rewrite → retrieve → grade）────────────────────────────

async def _rewrite_node(state: MasterState) -> dict:
    """
    Query Rewrite 节点（架构图 Query Rewrite）.

    首次进入：指代消解 + 多查询扩展 + 子问题拆解 + HyDE。
    重试进入（grade=bad）：在已有变体基础上再生成新角度的查询，
    打不同的语义邻域，避免同一个问法反复查不到。

    产物分三条通道下发（**不能合并成一条**，见 _retrieve_node）：

      rewritten_query  改写后的自包含查询 —— 既进两条检索腿，也作为精排基准。
      query_extra      子问题 + 变体 —— 进**两条**腿（都是"提问"，BM25 也吃）。
      query_hyde       假设答案段落 —— **只进向量腿**。它是一段"答案"，喂给
                       BM25 会得到一份词项宽泛杂乱的排名，在 RRF 里稀释真正的
                       精确词命中（见 QUERY_HYDE_VECTOR_ONLY）。
    """
    is_retry = state.get("retry_count", 0) > 0

    # 用 rewrite_query 而不是旧的 rewrite_query_with_history 包装：后者只回
    # (rewritten, variants)，把 subqueries 与 hyde 直接丢掉 —— 等于每轮都白花
    # 一次 LLM 生成（QUERY_DECOMPOSITION_ENABLED / QUERY_HYDE_ENABLED 默认开）。
    result = await rewrite_query(
        query=state["query"],
        history_messages=state["history_messages"],
    )
    rewritten = result.rewritten
    variants = list(result.variants)
    settings = get_settings()

    if is_retry:
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

    # 子问题排在最前：它们是"必答项"，变体只是换个说法问同一件事。
    # 总量由 MULTI_QUERY_MAX_EXTRA 统一封顶 —— 子问题上限 + 变体上限各自守规矩，
    # 加起来却可能是 5 路召回，延迟随路数近似线性上升，需要一个总闸门。
    extra: list[str] = []
    for item in list(result.subqueries) + variants:
        if item and item != rewritten and item not in extra:
            extra.append(item)
    extra = extra[: max(1, int(settings.MULTI_QUERY_MAX_EXTRA))]

    if extra or result.hyde:
        logger.info(
            "master_rewrite: %d extra quer(ies) + hyde=%d chars (source=%s)",
            len(extra), len(result.hyde or ""), result.source,
        )

    return {
        "rewritten_query": rewritten,
        "query_variants": variants,
        "query_extra": extra,
        "query_hyde": result.hyde,
    }


async def _retrieve_node(state: MasterState) -> dict:
    """
    Hybrid RAG 检索节点（架构图 Hybrid Retrieval）.

    内部已是：多路向量 ANN + BM25 → RRF 融合 → cross-encoder 精排。
    这里不重写检索逻辑，只负责调用 + 生成前端 sources。

    **数据/指令隔离（问题3 输入防护）**：rewrite 节点产出的
    rewritten_query / query_extra / query_hyde 在喂给向量库之前都要处理：

    1. ``normalize_text`` —— 剥离 Unicode 双向/零宽控制符（隐藏指令），
       防止把视觉上看不见但 token 里存在的指令送进向量检索，污染 BM25
       与 ANN 的关键词与语义召回。
    2. ``inspect_user_query`` —— 与 query_endpoint 入口同款的高/中风险
       模式扫描；命中 high-risk 的变体直接丢弃（避免诱导召回不相关文档
       或撑爆向量索引），命中 medium-risk 的留痕审计但保留。
    3. 空字符串 / 与原 query 完全相同的兜底 —— 与 query_transform 的
       现有语义保持一致，避免空查询进向量库导致静默空检索。

    HyDE 段落同样过这道闸 —— 它也是 LLM 生成的文本，不能因为是"自己人
    生成的"就免检。
    """
    raw_query = state.get("rewritten_query") or state["query"]
    safe_query = sanitize_retrieval_query(raw_query)
    if not safe_query:
        # sanitize 后丢空 —— 把原始 query 当兜底再洗一次，避免 rewritten
        # query 完全失效时整条 RAG 链路塌掉。原始 query 已经被 query_endpoint
        # inspect_user_query 通过（high-risk 在入口已拦截），所以这一步是
        # normalize-only；万一是入口处 medium-risk 放行的变体，这里至少
        # 走一遍隐形字符剥离。
        safe_query = sanitize_retrieval_query(state["query"]) or state["query"]

    raw_variants = state.get("query_extra") or state.get("query_variants") or []
    safe_variants: list[str] = []
    for v in raw_variants:
        kept = sanitize_retrieval_query(v)
        if kept and kept not in safe_variants:
            safe_variants.append(kept)

    # HyDE 段落走单独一条通道，只进向量腿（见 _rewrite_node 的说明）。
    # 同样过一遍 sanitize_retrieval_query：它是 LLM 生成的文本，可能夹带隐形
    # 控制符；high-risk 直接丢，宁可少一路召回也不能把可疑文本喂进向量库。
    raw_hyde = state.get("query_hyde") or ""
    safe_hyde = sanitize_retrieval_query(raw_hyde) if raw_hyde else ""

    if safe_query != raw_query:
        logger.warning(
            "master_retrieve: query was sanitized — %r → %r",
            raw_query[:80], safe_query[:80],
        )

    chunks = await retrieve_chunks(
        query=safe_query,
        top_k=state["top_k"],
        owner_id=state.get("owner_id"),
        collection_id=state.get("collection_id"),
        extra_queries=safe_variants,
        extra_vector_queries=[safe_hyde] if safe_hyde else None,
        # 三层隔离：第一层公司前置过滤 + 第二层部门 ACL（框架图 Permission
        # Filter 在检索之前，而非 rerank 之后）。tenant_id=None（平台管理员）
        # 只跳过公司维度，ACL 仍然生效 —— 别人的个人库永远检索不到。
        tenant_id=state.get("tenant_id"),
        user_department_id=state.get("department_id"),
        tenant_wide=bool(state.get("tenant_wide")),
        platform_wide=bool(state.get("platform_wide")),
    )

    sources = [
        {
            "document_id": c.document_id,
            "filename": c.filename,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            # 引用快照同样是不可信文档数据（问题3 文档防护）：屏蔽注入段落
            # 后再发给前端展示，与喂给 LLM 的上下文保持同一套清洗规则。
            "text_snippet": sanitize_document_context(c.text[:300])[0],
            "score": round(c.score, 4),
            # ── 位置信息（细粒度引用）：行号 + 一句话溯源 ─────────────────────
            "line_start": c.line_start,
            "line_end": c.line_end,
            "location": c.location_label(),
            # ── 图片位置：文档内序号 + 页面边界框 ─────────────────────────────
            "position": c.position,
            "bbox": list(c.bbox) if c.bbox else None,
            # 产出质检 + 双通道融合（图片理解的可验证事实）
            "analyze_quality": dict(c.analyze_quality or {}),
            "analyze_fusion": dict(c.analyze_fusion or {}),
            "quality_score": c.quality_score,
            # ── 部分3：内容类型与图片信息（前端据此渲染图片引用）──────────
            "content_type": c.content_type,
            "image_id": c.image_id,
            "image_path": c.image_path,
            "image_url": _image_url(c.document_id, c.image_path),
            "image_caption": c.image_caption,
            # 图片分类结论（table/formula/code/chart/diagram/screenshot/photo）
            # 前端据此在引用卡片上显示类型徽标
            "image_type": c.image_type,
            # 产出引擎 + 置信度 + 待人工复核（多引擎图片理解管线）
            "analyze_engine": c.analyze_engine,
            "analyze_confidence": c.analyze_confidence,
            "manual_review": c.manual_review,
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


async def _multimodal_context_node(state: MasterState) -> dict:
    """
    Multimodal Context 节点（部分6：架构图 grade → multimodal_context → generate）.

    把精排后的 Top-K 结果按模态分流：

        text / table chunk  → 直接进上下文（含 small-to-big 父块回填与压缩）
        image chunk         → image_path 取原图 → Vision 带着问题看图
                              → 结论写进上下文（原图同时随 sources 回显）

    产出统一编号的上下文文本、图文信息完整的 sources，以及每个块的
    结构化描述（context_blocks），供 generate 组装 prompt 与前端展示。

    关闭 MULTIMODAL_CONTEXT_ENABLED 时退回原 Context Builder 语义，
    保证行为向后兼容。
    """
    settings = get_settings()
    chunks = state.get("chunks") or []
    query = state.get("rewritten_query") or state["query"]

    if not settings.MULTIMODAL_CONTEXT_ENABLED:
        built = build_context(chunks, query=query)
        return {
            "multimodal_context": built.context,
            "sources": built.sources or state.get("sources", []),
            "context_blocks": [],
            "context_images": [],
            "vision_used": 0,
            "vision_available": False,
        }

    result = await build_multimodal_context(chunks, query)

    context_images = [
        {
            "document_id": b.get("document_id"),
            "filename": b.get("filename"),
            "page_number": b.get("page_number"),
            "image_id": b.get("image_id"),
            "image_path": b.get("image_path"),
            "image_url": b.get("image_url"),
            "vision": b.get("vision"),
        }
        for b in result.sources
        if b.get("content_type") == "image"
    ]

    logger.info(
        "master_multimodal_context: %d text/table + %d image, vision=%s(%d)",
        result.text_count, result.image_count,
        result.vision_available, result.vision_used,
    )

    return {
        "multimodal_context": result.context,
        "sources": result.sources or state.get("sources", []),
        "context_blocks": result.blocks_as_dicts(),
        "context_images": context_images,
        "vision_used": result.vision_used,
        "vision_available": result.vision_available,
    }


# ── 节点：Evidence Gate（架构图 Grader → Evidence Gate → Generate/Refuse）────

async def _evidence_gate_node(state: MasterState) -> dict:
    """
    确定性证据门控节点.

    Grader（LLM 语义判断）通过之后，再看一次"证据的客观形态"：条数、最高
    精排分、问题关键词覆盖率、证据正文长度。任何一项不达标 → 拒答。

    为什么放在 multimodal_context **之后**：图片块经 Vision 补充后，上下文
    才最终成型；门控看的是"最终要喂给模型的东西"，位置必须紧贴 generate。

    为什么是 fail-closed 而不是 fail-open：Grader 超时/异常时是放行的，
    这一层必须反向 —— 宁可拒答，也不把明显不够的证据交给模型硬答。
    """
    settings = get_settings()
    chunks = state.get("chunks") or []

    decision = evaluate_evidence(
        query=state.get("rewritten_query") or state["query"],
        chunks=chunks,
        min_chunks=settings.EVIDENCE_GATE_MIN_CHUNKS,
        min_top_score=settings.EVIDENCE_GATE_MIN_TOP_SCORE,
        min_coverage=settings.EVIDENCE_GATE_MIN_COVERAGE,
        min_evidence_chars=settings.EVIDENCE_GATE_MIN_CHARS,
        enabled=settings.EVIDENCE_GATE_ENABLED,
    )
    record_evidence_gate(decision.passed, decision.confidence)
    return {"evidence": decision.as_audit()}


def _route_after_evidence(state: MasterState) -> str:
    """
    Evidence Gate 之后的分流.

    document_agent   → build_document（生成 Word 交付物，不依赖引用校验）
    门控不通过        → refuse
    其余             → generate
    """
    evidence = state.get("evidence") or {}
    if not evidence.get("passed", True):
        return "refuse"
    if (state.get("intent") or "") == "document_agent":
        return "build_document"
    return "generate"


# ── 节点：Citation Verifier（架构图 Generate → Citation Verifier → Output Guard）

def _sources_with_full_text(state: MasterState) -> list[dict]:
    """
    给引用校验补上**完整正文**，只用于校验，不外发.

    问题：sources 里的 ``text_snippet`` 是给前端展示的预览，被截断到几百字
    （见 context_builder / multimodal_context 的 SNIPPET_CHARS）。引用校验若
    拿它当"原文"，会系统性地误判：

      * 数字 / 日期落在截断之后 → 判"与原文不一致"；
      * 支持度分母被截小 → 判"不被原文支持"。

    两者都会**删掉本来正确的引用**。"该引的没引上"比"多引了一条"更伤信任，
    所以这里把 payload 里的完整 chunk 正文覆盖进 sources 的副本。

    用副本而非原地改 sources：全文只服务于校验，不该随 SSE 发给前端 ——
    引用卡片展示的是预览，把整段原文塞进每次事件只会白白放大流量。
    匹配用 (document_id, chunk_index) 而非下标，避免任何中间环节重排导致
    张冠李戴（那会让"位置正确？"这一项彻底失去意义）。
    """
    sources = [dict(s) for s in (state.get("sources") or [])]
    if not sources:
        return sources

    full_texts: dict[tuple[str, int], str] = {}
    for chunk in state.get("chunks") or []:
        text = getattr(chunk, "text", None)
        if not text:
            continue
        full_texts[(str(chunk.document_id), int(chunk.chunk_index))] = text

    if not full_texts:
        return sources

    for src in sources:
        key = (str(src.get("document_id")), int(src.get("chunk_index") or 0))
        full = full_texts.get(key)
        if full:
            src["text"] = full
    return sources


async def _citation_verifier_node(state: MasterState) -> dict:
    """
    引用校验节点（五项：存在 / 位置 / 支持 / 数字 / 日期）.

    纯确定性文本比对，不调 LLM。产出的净化文本会覆盖 answer，交给
    output_guard 做最后一道合规检查。

    模型**主动拒答**时（允许模型在证据不足时说"我不知道"）本节点直接跳过：
    拒答文本里没有引用，也没有需要校验的结论，硬跑只会产生噪声指标。
    """
    settings = get_settings()
    answer = state.get("answer", "") or ""

    # 允许模型拒绝回答：模型自己判断证据不足 → 记为 model 拒答，不做引用校验
    if is_refusal(answer):
        record_refusal("model")
        logger.info("master_citation_verifier: model self-refused (intent=%s)", state.get("intent"))
        return {"refusal_source": "model", "citation_check": {"overall": "refused_by_model"}}

    if not settings.CITATION_VERIFIER_ENABLED:
        return {}

    report = verify_citations(
        answer=answer,
        sources=_sources_with_full_text(state),
        min_support=settings.CITATION_MIN_SUPPORT,
        misattribution_margin=settings.CITATION_MISATTRIBUTION_MARGIN,
        strip_unsupported=settings.CITATION_STRIP_UNSUPPORTED,
        annotate=settings.CITATION_ANNOTATE,
        # 命中句回标：让引用卡片能标出"答案实际用了哪几句"，而不是只给
        # 整块切片的行范围让用户自己读。
        evidence_enabled=settings.CITATION_EVIDENCE_HIGHLIGHT,
        evidence_min_ratio=settings.CITATION_EVIDENCE_MIN_RATIO,
        evidence_max_sentences=settings.CITATION_EVIDENCE_MAX_SENTENCES,
    )
    record_citation_check(report)

    audit = report.as_audit()
    # changed=True 表示净化真的改动了正文（移除了引用标记，或追加了校验脚注）。
    # 这个布尔值随 citation_check 事件下发，stream_master 据此决定是否回传
    # 净化后全文 —— 与 output_guard 同样的道理：token 已经流式发出无法撤回，
    # 只能由前端整段替换，否则"用户看到的引用"与"校验结论"会自相矛盾。
    audit["changed"] = report.clean_text != answer
    # 有问题的引用会随 SSE citation_check 事件下发，供前端展示"已核验 / 存疑"
    return {
        "answer": report.clean_text if report.total else answer,
        "citation_check": audit,
    }


async def _generate_node(state: MasterState) -> dict:
    """
    生成节点 —— 使用 multimodal_context 节点组装好的上下文
    （图文分流 + Vision 结论 + small-to-big 父块回填 + 提示注入清洗 + 压缩），
    再流式输出。

    理论上 multimodal_context 总是先于本节点执行；若因配置关闭而回退，
    这里用 Context Builder 兜底，保证任何拓扑下都能拿到上下文。
    """
    settings = get_settings()
    llm = _build_streaming_llm(settings)

    context_text = state.get("multimodal_context") or ""
    if not context_text:
        built = build_context(
            state.get("chunks") or [],
            query=state.get("rewritten_query") or state["query"],
        )
        context_text = built.context

    focused_query = (
        f"[指令：只回答下面这个问题。"
        f"不要重复或以之前轮次的内容作为开头。]\n\n"
        f"{state['query']}"
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=_SYSTEM_TEMPLATE.format(context=context_text)),
        *_trim_history(state["history_messages"]),
        HumanMessage(content=focused_query),
    ]

    answer = await _stream_llm_answer(llm, messages)

    # sources 以 multimodal_context 节点的为准（它带图文信息与 vision 结论）
    return {"answer": answer}


async def _refuse_node(state: MasterState) -> dict:
    """
    拒答节点（两条路径共用）.

    1. 重试用尽（grade 一直 bad）→ 拒答；
    2. Evidence Gate 判定证据形态不达标 → 拒答。

    使用与"模型主动拒答"完全相同的文案（evidence_gate.REFUSAL_ANSWER）：
    用户看到的拒答不因"是谁拒的"而不同，Bad Case 回流也只需匹配一个字符串。
    """
    record_refusal("gate")
    logger.info(
        "master_refuse: refusing query=%r after %d retries (grade=%s evidence=%s)",
        state["query"][:80], state.get("retry_count", 0),
        state.get("grade_reason", ""),
        (state.get("evidence") or {}).get("reason", ""),
    )
    return {"answer": REFUSAL_ANSWER, "refusal_source": "gate"}


# ── 节点：Document Summary ───────────────────────────────────────────────────

async def _summary_digests_node(state: MasterState) -> dict:
    """
    拉取文档摘要（复用 relation_service 的采样逻辑）.

    先做一次**范围解析**：用户在提问里点名了哪份文档（"总结《X》"）
    就只采样那几份；没点名就是整库总结；点名了但库里没有对得上的，
    不猜也不硬总结 —— 交给 _summarize_node 给出可选文档清单。
    """
    acl_kwargs = {
        "owner_id": state.get("owner_id"),
        "tenant_id": state.get("tenant_id"),
        "user_department_id": state.get("department_id"),
        "tenant_wide": bool(state.get("tenant_wide")),
        "platform_wide": bool(state.get("platform_wide")),
    }

    try:
        documents = await list_accessible_documents(**acl_kwargs)
    except Exception:
        # 列清单失败不该让总结直接不可用：退化为"整库"（下面的采样会再查一次库）
        logger.exception("document_summary: failed to list accessible documents")
        documents = []

    scope = resolve_summary_targets(state["query"], documents)
    # 记录可访问文档总数：整库总结被 DOC_SUMMARY_MAX_DOCUMENTS 截断时，
    # 答案末尾要明示"只覆盖了最近 N 份"（绝不静默漏文档）。
    scope["total_accessible"] = len(documents)

    # 库里只有一份文档时不存在歧义 —— 用户怎么写都总结它
    if scope.get("missing") and len(documents) == 1:
        scope = dict(scope, missing=False, ids=[str(documents[0][0])],
                     names=[str(documents[0][1])], targeted=True)

    if scope.get("missing"):
        # 点名的文档不存在：不采样、不调 LLM（answer 在 _summarize_node 里确定性生成）
        return {"digests": [], "summary_scope": scope}

    digests = await collect_summary_digests(
        document_ids=scope.get("ids") or None,
        **acl_kwargs,
    )
    return {"digests": digests, "summary_scope": scope}


async def _summarize_node(state: MasterState) -> dict:
    """
    文档总结生成（map-reduce，逐文档循环）.

    结构保证"一份不漏"，与模型行为解耦：
    - map：每份文档独立一次 LLM 调用，节标题由**代码**确定性下发
      （stream_channel 旁路 → SSE chunk，流式体验与单次调用一致）；
      某份调用失败时写入兜底小节，绝不静默跳过；
    - reduce：多份文档时追加一次"总体概览"调用（只看各节摘要）；
    - 截断明示：整库总结被 DOC_SUMMARY_MAX_DOCUMENTS 截断时，末尾
      注明"共 N 份、本次覆盖最近 M 份"。

    节标题为什么不用 `get_stream_writer`：langgraph 1.0.1 的
    `astream_events(version="v2")` 会**丢弃** StreamWriter 的 payload
    （实测只有 `astream(stream_mode="custom")` 收得到，而本图的事件循环
    建立在 astream_events 之上），标题因此到不了前端。改用
    :mod:`app.services.stream_channel` 的显式旁路。
    """
    scope = state.get("summary_scope") or {}
    if scope.get("missing"):
        # 点名了文档但没对上：直接把可选文档列出来，不猜、不编
        return {"answer": build_target_not_found_answer(scope)}

    digests = state.get("digests") or []
    if not digests:
        return {"answer": "知识库中暂时没有已完成索引的文档，请上传文档后再试。"}

    def _emit(text: str) -> None:
        """确定性文本（节标题 / 兜底 / 截断说明）走旁路流式通道.

        没有绑定旁路（例如单测直接调用本节点）时静默丢弃 —— 返回的
        ``answer`` 始终是完整的，流式只是展示层。
        """
        emit_text(text)

    # 逐份正文用带思考的 LLM（要判断哪些数字可信）；
    # 总体概览只是把已经写好的各节归纳一次，关掉思考省一次完整思考的等待。
    llm = build_summary_llm()
    overview_llm = build_summary_llm(reasoning=False)
    query = state["query"]
    parts: list[str] = []
    sections: list[tuple[str, str]] = []

    for digest in digests:
        filename = str(digest.get("filename") or "未知文档")
        header = f"### {filename}\n\n"
        _emit(header)
        parts.append(header)
        try:
            body = (await _stream_llm_answer(
                llm, build_single_doc_messages(query, digest)
            )).strip()
        except Exception:
            # 单份失败不拖垮整批：兜底小节占位，其余文档照常总结
            logger.exception("document_summary: per-doc summary failed for %s", filename)
            body = ""
        if not body:
            body = "（这份文档的总结生成失败，可直接针对该文档提问获取具体内容。）"
            _emit(body)
        parts.append(body + "\n\n")
        sections.append((filename, body))

    if len(sections) > 1:
        header = "### 总体概览\n\n"
        _emit(header)
        parts.append(header)
        try:
            overview = (await _stream_llm_answer(
                overview_llm, build_overview_messages(query, sections)
            )).strip()
        except Exception:
            logger.exception("document_summary: overview generation failed")
            overview = ""
        if overview:
            parts.append(overview)
        else:
            # 概览失败不阻塞交付：节标题已流式发出，补一行确定性说明，
            # 保证「用户看到的 == 落库的」。
            fallback = "（总体概览生成失败，各文档的要点见上方小节。）"
            _emit(fallback)
            parts.append(fallback)

    # 截断明示：可访问总数 > 实际总结数（被 DOC_SUMMARY_MAX_DOCUMENTS 截断）
    total_accessible = int(scope.get("total_accessible") or 0)
    if not scope.get("targeted") and total_accessible > len(digests):
        note = (
            f"\n\n> 注：知识库中共有 {total_accessible} 份文档，受单次总结规模限制，"
            f"以上覆盖最近的 {len(digests)} 份；需要其余文档的总结时，"
            f"请点名文档（例如「总结《文件名》」）。"
        )
        _emit(note)
        parts.append(note)

    return {"answer": "".join(parts).strip()}


# ── 节点：doc_relations（沿用原有能力）─────────────────────────────────────────

async def _collect_digests_node(state: MasterState) -> dict:
    digests = await collect_document_digests(
        owner_id=state.get("owner_id"),
        tenant_id=state.get("tenant_id"),
        user_department_id=state.get("department_id"),
        tenant_wide=bool(state.get("tenant_wide")),
        platform_wide=bool(state.get("platform_wide")),
    )
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


# ── 节点：Output Guard（问题3+问题4，架构图 Generate → Citation Check → END）──

async def _output_guard_node(state: MasterState) -> dict:
    """
    生成后合规校验节点（Generate 之后必经，refuse 不进）。

    严格确定性、不调 LLM。做四件事：
    1. Citation Check（问题4）：扫描 [Source N]，把越界引用移除
    2. 系统提示词泄露检测（问题3 输出防护）
    3. 闲聊分支幻觉措辞（问题3 Prompt 隔离）
    4. Agent 工具调用意图（问题3 工具权限控制）

    净化后的 answer 会覆盖回 state，供 save_history 与前端 SSE 展示。
    审计信号写入 state.output_guard 字段，可对接 audit_service 落库。
    """
    settings = get_settings()
    if not settings.OUTPUT_GUARD_ENABLED:
        return {}

    result = inspect_output(
        answer=state.get("answer", ""),
        sources=state.get("sources", []) or [],
        intent=state.get("intent", "") or "",
    )
    record_output_guard(result.changed, blocked=False)
    if not result.changed:
        return {}

    # OUTPUT_GUARD_BLOCK_ON_LEAK=True（默认关）：疑似系统信息泄露不做片段
    # 替换，直接以固定拒答文本替换整段答案，由 sanitized_answer 回传前端。
    if settings.OUTPUT_GUARD_BLOCK_ON_LEAK and result.leaked_phrases:
        record_output_guard(False, blocked=True)
        return {
            "answer": (
                "抱歉，本次回答经输出安全检测未通过（检测到疑似系统信息泄露），"
                "已整段拦截。请调整问法后重试。"
            ),
            "output_guard": {
                "citations_removed": list(result.citations_removed),
                "leaked_phrases": list(result.leaked_phrases),
                "hallucination_phrases": list(result.hallucination_phrases),
                "tool_attempt_phrases": list(result.tool_attempt_phrases),
                "changed": True,
                "blocked": True,
            },
        }

    audit = {
        "citations_removed": list(result.citations_removed),
        "leaked_phrases": list(result.leaked_phrases),
        "hallucination_phrases": list(result.hallucination_phrases),
        "tool_attempt_phrases": list(result.tool_attempt_phrases),
        "changed": True,
    }
    return {
        "answer": result.sanitized_text,
        "output_guard": audit,
    }


async def _build_document_node(state: MasterState) -> dict:
    """
    Document Agent 节点（最终效果：Word 写入 / Word 插入图片）.

    用 multimodal_context 节点产出的 chunks 组装一份可下载的 Word 文档：

        text  chunk → 正文段落（保留 heading 层级）
        table chunk → 真正的 Word 表格
        image chunk → 从 image_path 取**原始图片**插入文档

    文档落盘后返回 ``document`` 元信息，由 stream_master 以 SSE `document`
    事件下发，前端渲染下载入口。生成失败（缺库 / 写盘失败）不中断对话，
    回退为一份带说明的文本答案。
    """
    import asyncio

    from app.services.document_agent_service import (
        build_fallback_summary,
        generate_document,
    )

    settings = get_settings()
    chunks = state.get("chunks") or []
    query = state["query"]
    title = (state.get("rewritten_query") or query).strip()

    if not settings.DOCUMENT_AGENT_ENABLED:
        return {
            "answer": "当前未启用文档生成能力（DOCUMENT_AGENT_ENABLED=false）。",
            "document": {},
        }

    info = await asyncio.to_thread(generate_document, query, chunks, title=title)

    if info.error:
        logger.warning("master_build_document: generation failed — %s", info.error)
        return {
            "answer": build_fallback_summary(query, info),
            "document": info.as_event_payload(),
            "sources": state.get("sources", []),
        }

    parts = [f"已根据知识库生成 Word 文档：《{info.title}》", ""]
    parts.append(f"- 正文小节：{info.section_count} 个")
    parts.append(f"- 表格：{info.table_count} 个")
    parts.append(f"- 插入图片：{info.image_count} 张")
    parts.append(f"- 参考来源：{len(info.sources)} 条")
    parts.append("")
    parts.append("可在下方直接下载 .docx 文件。")

    logger.info(
        "master_build_document: '%s' ready (%d bytes, %d images)",
        info.filename, info.size_bytes, info.image_count,
    )
    return {
        "answer": "\n".join(parts),
        "document": info.as_event_payload(),
    }


def _route_after_context(state: MasterState) -> str:
    """
    multimodal_context 之后的条件分流（保留给旧拓扑使用）.

    document_agent → build_document（生成 Word 交付物）
    其余           → generate（常规作答）

    新拓扑中 multimodal_context 之后先经过 Evidence Gate，由
    ``_route_after_evidence`` 决定去向；本函数仅作为向后兼容的参考实现保留。
    """
    if (state.get("intent") or "") == "document_agent":
        return "build_document"
    return "generate"


# ── 节点：落库 ───────────────────────────────────────────────────────────────

def _build_turn_meta(state: MasterState) -> dict:
    """
    组装一条回答的"依据快照"，随 assistant 消息一起落库.

    这些字段原先只随 SSE 事件发给浏览器、活在内存里：切页 / 切窗口 / 重开
    标签页之后，引用来源、引用校验结论、图文分流统计全部消失 —— 用户看到的
    就是"上一次提问的数据来源不见了"。落库之后重开历史可以完整复现。

    只保留**前端渲染真正用到**的键，不整份 state 落库：``chunks`` 里是全文
    分块（几十 KB），而 ``sources`` 已带展示所需的片段文本。
    """
    context_images = state.get("context_images") or []
    answer = str(state.get("answer") or "")
    from app.services.nodes.evidence_gate import is_refusal

    refused = is_refusal(answer)
    # "以下来源未被采用"只在**确实有来源**时才有意义：无检索管线
    # （document_summary / general_chat / list_documents）拒答时列不出任何
    # 来源，这句提示会变成新的自相矛盾。与 stream_master 的实时事件保持同一
    # 判定，避免"实时流与历史回放各说一套"。
    has_sources = bool(state.get("sources"))
    return {
        # 管线徽章：重新打开历史时仍能显示"这轮走的哪条链路"
        "intent": state.get("intent") or "",
        # 引用来源（含图片对象：image_url / image_type / vision）
        "sources": state.get("sources") or [],
        # 答复性质：拒答时重开历史同样要把来源标成"未采用"，
        # 否则实时流修正好了、看历史又出现"拒答 + 引用"的矛盾画面
        "answer_status": {
            "refused": refused,
            "sources_used": not refused,
            "note": (
                "本次答复为拒答：检索到的片段不足以支撑结论，以下来源未被采用。"
                if refused and has_sources
                else ""
            ),
        },
        # 五项引用校验（前端逐条标"已核验 / 存疑"）
        "citation_check": state.get("citation_check") or {},
        # 证据门控（不足时显示"证据不足"徽标）
        "evidence": state.get("evidence") or {},
        # 输出合规校验（已合规校验 / 已合规净化）
        "output_guard": state.get("output_guard") or {},
        # Document Agent 产物（下载卡片）
        "document": state.get("document") or {},
        # 图文分流统计（N 图 · 视觉 M）
        "multimodal": {
            "image_count": len(context_images),
            "vision_used": int(state.get("vision_used") or 0),
            "vision_available": bool(state.get("vision_available")),
        },
    }


async def _save_history_node(state: MasterState) -> dict:
    # 第三层隔离：消息带上真实用户与租户（conversation_id + tenant_id +
    # user_id 的隔离键），跨用户/跨租户无法续读到这段记忆。
    user_id = state.get("user_id")
    await save_turn(
        conversation_id=uuid.UUID(state["conversation_id"]),
        user_message=state["query"],
        assistant_message=state["answer"],
        user_id=uuid.UUID(user_id) if user_id else None,
        tenant_id=state.get("tenant_id"),
        # 依据快照随回答一起落库，历史会话才能还原当时的引用来源
        assistant_meta=_build_turn_meta(state),
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
        # 不传 num_ctx 时 Ollama 默认 40960 —— 小显存机器必 OOM 的元凶
        num_ctx=settings.chat_num_ctx,
        # 小显存机器可通过 OLLAMA_NUM_GPU=0 强制 CPU / 部分层 offload
        num_gpu=settings.OLLAMA_NUM_GPU,
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
    # Multimodal Context（部分6）：grade → multimodal_context → generate
    graph.add_node("multimodal_context", _multimodal_context_node)
    # Evidence Gate（确定性证据门控）：形态不达标 → refuse，绝不硬答
    graph.add_node("evidence_gate", _evidence_gate_node)
    graph.add_node("generate", _generate_node)
    # Citation Verifier（五项引用校验）：Generate → Citation Verifier → Output Guard
    graph.add_node("citation_verifier", _citation_verifier_node)
    # Document Agent：生成 Word 交付物（最终效果）
    graph.add_node("build_document", _build_document_node)
    graph.add_node("refuse", _refuse_node)
    # Document Summary
    graph.add_node("summary_digests", _summary_digests_node)
    graph.add_node("summarize", _summarize_node)
    # doc_relations
    graph.add_node("collect_digests", _collect_digests_node)
    graph.add_node("analyze_relations", _analyze_relations_node)
    # General Chat
    graph.add_node("chat", _chat_node)
    # Output Guard（问题3+问题4）：生成后合规校验，所有生成路径必经；refuse 不进
    graph.add_node("output_guard", _output_guard_node)
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

    # Knowledge QA / Document Agent：
    #   rewrite → retrieve → grade → multimodal_context → evidence_gate
    #                                                → {generate | build_document | refuse}
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        _route_after_grade,
        {
            "generate": "multimodal_context",   # 先做图文分流 + Vision 看图，再门控
            "rewrite": "rewrite",               # ← Retry 回边
            "refuse": "refuse",
        },
    )
    graph.add_edge("multimodal_context", "evidence_gate")
    graph.add_conditional_edges(
        "evidence_gate",
        _route_after_evidence,
        {
            "generate": "generate",              # 常规问答
            "build_document": "build_document",  # Document Agent（Word 生成）
            "refuse": "refuse",                  # 证据形态不达标 → 拒答
        },
    )

    # 其余分支
    graph.add_edge("summary_digests", "summarize")
    graph.add_edge("summarize", "output_guard")
    graph.add_edge("collect_digests", "analyze_relations")
    graph.add_edge("analyze_relations", "output_guard")
    graph.add_edge("chat", "output_guard")
    # 生成路径：generate → citation_verifier → output_guard
    graph.add_edge("generate", "citation_verifier")
    graph.add_edge("citation_verifier", "output_guard")
    # Document Agent 产物同样过一遍 Output Guard（引用编号 / 合规校验）；
    # 其文本由代码确定性拼装、不含引用，无需走 Citation Verifier
    graph.add_edge("build_document", "output_guard")
    # refuse 不接 output_guard —— 拒答文本本身就是合规答案
    graph.add_edge("refuse", "save_history")
    graph.add_edge("output_guard", "save_history")
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

async def _maybe_capture_bad_case(
    *,
    query: str,
    answer: str,
    intent: str,
    conversation_id: str | None,
    user_id: str | None,
    username: str | None,
    sources: list[dict],
    gate_audit: dict,
    citation_audit: dict,
    guard_audit: dict,
) -> None:
    """
    把本轮"值得回流"的信号自动写入 Bad Case 队列（每轮最多一条）.

    优先级：引用不被原文支持 > 证据门控拒答 > 输出净化命中。
    引用问题排第一，因为它最隐蔽 —— 用户看到的是"有引用"的答案，
    只有校验才能发现引用是错的。每轮只回流一条，避免同一问答在队列里
    刷出三条重复记录把人工审阅淹没。

    回流是 best-effort，任何异常都不影响已经发出的答案。
    """
    reason: str | None = None
    severity = "medium"
    detail: dict = {}

    if citation_audit:
        bad = (
            set(citation_audit.get("unsupported") or [])
            | set(citation_audit.get("hallucinated") or [])
            | set(citation_audit.get("misattributed") or [])
            | set(citation_audit.get("number_mismatch") or [])
            | set(citation_audit.get("date_mismatch") or [])
        )
        if bad:
            reason = "citation_unsupported"
            # 引用了不存在的来源 = 纯幻觉，定级最高
            severity = "high" if citation_audit.get("hallucinated") else "medium"
            detail = {"citation_check": citation_audit}

    if reason is None and gate_audit and gate_audit.get("passed") is False:
        reason = "evidence_refused"
        severity = "medium"
        detail = {"evidence_gate": gate_audit}

    if reason is None and guard_audit and guard_audit.get("changed"):
        reason = "output_guard"
        severity = "medium"
        detail = {"output_guard": guard_audit}

    if reason is None:
        return

    await capture_bad_case(BadCaseSignal(
        reason=reason,
        severity=severity,
        question=query,
        answer=answer,
        intent=intent,
        user_id=user_id,
        username=username,
        conversation_id=conversation_id,
        detail=detail,
        # 只留定位所需字段，避免把整篇原文塞进回流队列
        sources=[
            {
                "document_id": s.get("document_id"),
                "filename": s.get("filename"),
                "page_number": s.get("page_number"),
                "line_start": s.get("line_start"),
                "line_end": s.get("line_end"),
                "location": s.get("location"),
                "score": s.get("score"),
            }
            for s in (sources or [])
        ],
    ))


async def stream_master(
    query: str,
    conversation_id: str,
    history_messages: list[BaseMessage],
    top_k: int,
    owner_id: str | None = None,
    collection_id: str | None = None,
    forced_mode: str | None = None,
    user_id: str | None = None,
    username: str | None = None,
    tenant_id: str | None = None,
    department_id: str | None = None,
    tenant_wide: bool = False,
    platform_wide: bool = False,
) -> AsyncGenerator[dict, None]:
    """
    执行 master graph 并产出 SSE 事件字典.

    Event types（在现有协议上新增 6 种，旧类型保持不变）：
        {"type": "intent",   "intent": "...", "reason": "..."}   路由结果
        {"type": "sources",  "sources": [...], "retrieved_count": N}
        {"type": "doc_digests", "documents": [...], "retrieved_count": N}
        {"type": "grade",    "good": bool, "retry": N, "reason": "..."}
        {"type": "evidence", "passed": bool, "reason": "...", "confidence": f,
                             "failed_signals": [...], "signals": [...]}   # Evidence Gate
        {"type": "citation_check", "overall": "verified|partial|unsupported|no_citations",
                             "total": N, "passed": M, "unsupported": [...],
                             "hallucinated": [...], "misattributed": [...],
                             "number_mismatch": [...], "date_mismatch": [...],
                             "verdicts": [...], "sanitized_answer": str|None}
                             # Citation Verifier
        #   sanitized_answer：净化改动了正文（移除引用标记 / 追加校验脚注）时
        #   回传的完整答案，前端用它替换已流式渲染的内容，保证展示与落库一致。
        {"type": "chunk",    "content": "<token>"}
        {"type": "thinking_delta", "content": "<token>"}
        {"type": "output_guard", "changed": bool, "sanitized_answer": str|None,
                                  "citations_removed": [...],
                                  "leaked_phrases": [...],
                                  "hallucination_phrases": [...],
                                  "tool_attempt_phrases": [...]}   # 问题3+4
        #   sanitized_answer：changed=true 时回传净化后的完整答案，
        #   前端用它替换已流式渲染的内容，保证展示与落库一致。
        {"type": "answer_status", "refused": bool, "sources_used": bool,
                                  "note": str}   # 拒答时把来源标为"未采用"
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
        "tenant_id": tenant_id,
        "department_id": department_id,
        "tenant_wide": tenant_wide,
        "platform_wide": platform_wide,
        "user_id": user_id,
        "forced_mode": forced_mode,
        "intent": "",
        "intent_reason": "",
        "rewritten_query": "",
        "query_variants": [],
        "query_extra": [],
        "query_hyde": None,
        "chunks": [],
        "sources": [],
        "grade_good": False,
        "grade_reason": "",
        "retry_count": 0,
        "digests": [],
        "answer": "",
        "output_guard": {},
        "evidence": {},
        "citation_check": {},
        "refusal_source": "",
    }

    graph = get_master_graph()
    started_at = time.monotonic()
    total_chars = 0
    intent_emitted = False
    sources_emitted = False
    # 当前是否处于某个「生成类节点」/「决策类节点」内部 —— 用于过滤内部
    # LLM 的中间输出。见下方 on_chat_model_stream 分支的说明。
    streaming_node: str | None = None
    internal_node: str | None = None

    # ── 监控 / 回流用的滚动快照 ─────────────────────────────────────────────
    # astream_events 只给节点级输出，这里把关键节点的产出攒起来，循环结束后
    # 一次性写指标与 Bad Case（避免在事件循环里做 IO 影响流式体验）。
    detected_intent = ""
    gate_audit: dict = {}
    citation_audit: dict = {}
    guard_audit: dict = {}
    final_sources: list[dict] = []
    final_answer = ""
    # 本轮是否已经流式发出过答案 token —— 用于判断"确定性答复"（不调 LLM 的
    # 节点产物，如 summarize 的"点名文档没找到"）是否需要手动补发。
    answer_streamed = False

    # ── 节点 → 外层的确定性文本旁路（Document Summary 的节标题等）──────────
    # 节点用 stream_channel.emit_text 追加，这里在**每个事件到达时先冲刷**
    # 再处理事件：节标题因此一定排在该节正文（首个 LLM token）之前。
    deterministic_buf: list[str] = []
    _sink_token = bind_sink(deterministic_buf)

    try:
        async for event in graph.astream_events(initial_state, version="v2"):
            # 冲刷旁路缓冲（顺序敏感：必须在处理本事件之前）
            while deterministic_buf:
                text = deterministic_buf.pop(0)
                final_answer += text
                total_chars += len(text)
                answer_streamed = True
                yield {"type": "chunk", "content": text}

            kind = event["event"]
            name = event.get("name", "")

            # ── 追踪 LLM 节点的进入/退出 ────────────────────────────────────
            # LLM 的 on_chat_model_stream 事件嵌套在节点的 on_chain_start /
            # on_chain_end 之间，借此判断 token 属于哪个节点。这是
            # metadata.langgraph_node 之外的兜底通道（部分 LangGraph 版本
            # 不注入该元数据）。
            if kind == "on_chain_start":
                if name in _STREAMING_LLM_NODES:
                    streaming_node = name
                elif name in _INTERNAL_LLM_NODES:
                    internal_node = name
            elif kind == "on_chain_end":
                if name in _STREAMING_LLM_NODES:
                    streaming_node = None
                elif name in _INTERNAL_LLM_NODES:
                    internal_node = None

            # ── 路由结果 ─────────────────────────────────────────────────────
            if kind == "on_chain_end" and name == "route" and not intent_emitted:
                output = event["data"].get("output", {}) or {}
                intent = output.get("intent") or "knowledge_qa"
                detected_intent = intent
                record_query(intent)
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
                if sources:
                    final_sources = sources
                yield {
                    "type": "sources",
                    "sources": sources,
                    "retrieved_count": len(sources),
                }
                sources_emitted = True

            # ── Multimodal Context（部分5+6）─────────────────────────────────
            # 图文分流 + Vision 看图之后的权威 sources：带 content_type /
            # image_url / vision 结论。前端整体替换早前的 sources 列表。
            elif kind == "on_chain_end" and name == "multimodal_context":
                output = event["data"].get("output", {}) or {}
                sources = output.get("sources", [])
                if sources:
                    final_sources = sources
                    yield {
                        "type": "sources",
                        "sources": sources,
                        "retrieved_count": len(sources),
                    }
                yield {
                    "type": "multimodal",
                    "blocks": output.get("context_blocks", []),
                    "images": output.get("context_images", []),
                    "image_count": len(output.get("context_images", []) or []),
                    "vision_used": output.get("vision_used", 0),
                    "vision_available": output.get("vision_available", False),
                }

            # ── Evidence Gate（确定性证据门控）──────────────────────────────
            # passed=false 时后续会走 refuse；前端据此提示"证据不足"。
            elif kind == "on_chain_end" and name == "evidence_gate":
                output = event["data"].get("output", {}) or {}
                evidence = output.get("evidence") or {}
                if evidence:
                    gate_audit = evidence
                    yield {"type": "evidence", **evidence}

            # ── Citation Verifier（五项引用校验）────────────────────────────
            # 逐条给出 存在/位置/支持/数字/日期 的结论，前端引用卡片据此
            # 显示"已核验"或"存疑"。
            elif kind == "on_chain_end" and name == "citation_verifier":
                output = event["data"].get("output", {}) or {}
                check = output.get("citation_check") or {}
                if check:
                    citation_audit = check
                    # 引用被移除 / 追加校验脚注时，正文已经和先前流出的 token
                    # 不一致了。与 output_guard 同样的处理：把净化后全文一起
                    # 回传，由前端整段替换，保证「用户看到的 == 落库的 == 校验过的」。
                    sanitized = output.get("answer")
                    yield {
                        "type": "citation_check",
                        "overall": check.get("overall", "no_citations"),
                        "total": check.get("total", 0),
                        "passed": check.get("passed", 0),
                        "unsupported": check.get("unsupported", []),
                        "hallucinated": check.get("hallucinated", []),
                        "misattributed": check.get("misattributed", []),
                        "number_mismatch": check.get("number_mismatch", []),
                        "date_mismatch": check.get("date_mismatch", []),
                        "verdicts": check.get("verdicts", []),
                        "sanitized_answer": (
                            sanitized
                            if check.get("changed")
                            and isinstance(sanitized, str)
                            and sanitized
                            else None
                        ),
                    }

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
                    final_answer = refusal
                    total_chars += len(refusal)
                    yield {"type": "chunk", "content": refusal}

            # ── Document Agent 产物（Word 生成，最终效果）───────────────────
            # 本节点不调 LLM，answer 是确定性文本，需手动转发；
            # 同时下发 document 事件，前端据此渲染 .docx 下载入口。
            elif kind == "on_chain_end" and name == "build_document":
                output = event["data"].get("output", {}) or {}
                answer_text = str(output.get("answer", ""))
                if answer_text:
                    # 同 token 分支：确定性产物也要计入 final_answer，
                    # 否则这一轮的 answer_status / bad case 回流拿到空答案。
                    final_answer = answer_text
                    total_chars += len(answer_text)
                    yield {"type": "chunk", "content": answer_text}
                document = output.get("document") or {}
                if document:
                    yield {"type": "document", "document": document}

            # ── Document Summary 的确定性答复（点名文档但没找到）────────────
            # 正常总结的 answer 由 LLM token 流式发出；"没找到你点名的文档，
            # 可选文档如下"这类答复不调 LLM、没有 token，必须在这里补发，
            # 否则前端拿到空答案。
            elif kind == "on_chain_end" and name == "summarize":
                if not answer_streamed:
                    output = event["data"].get("output", {}) or {}
                    answer_text = str(output.get("answer", ""))
                    if answer_text:
                        final_answer = answer_text
                        total_chars += len(answer_text)
                        yield {"type": "chunk", "content": answer_text}

            # ── Output Guard（问题3+问题4）审计信号透传 ──────────────────────
            elif kind == "on_chain_end" and name == "output_guard":
                output = event["data"].get("output", {}) or {}
                audit = output.get("output_guard") or {}
                guard_audit = audit
                sanitized = output.get("answer")
                if isinstance(sanitized, str) and sanitized:
                    final_answer = sanitized
                if not audit:
                    continue
                # changed=true 时回传净化后的完整答案：token 已经流式发给前端，
                # 无法撤回，所以由前端用这份全文替换已渲染的内容，保证
                # 「用户看到的 == 落库的」。
                yield {
                    "type": "output_guard",
                    "changed": bool(audit.get("changed")),
                    "sanitized_answer": (
                        sanitized if isinstance(sanitized, str) and sanitized else None
                    ),
                    "citations_removed": audit.get("citations_removed", []),
                    "leaked_phrases": list(audit.get("leaked_phrases", [])),
                    "hallucination_phrases": list(audit.get("hallucination_phrases", [])),
                    "tool_attempt_phrases": list(audit.get("tool_attempt_phrases", [])),
                }

            # ── 节点内确定性文本流（Document Summary 的节标题/兜底/截断说明）──
            # 不在这里处理 —— 见循环开头对 deterministic_buf 的冲刷。
            # 之所以不走 `on_custom_event`：langgraph 1.0.1 的 astream_events
            # 会丢弃 get_stream_writer 的 payload（实测），该分支永远收不到数据。

            # ── LLM token 流 ────────────────────────────────────────────────
            elif kind == "on_chat_model_stream":
                # 问题1 根因修复（截图1 中 JSON 泄漏到答案的元凶）。
                # 只有生成类节点的输出是答案正文；route / rewrite / grade 产出
                # 的是内部决策 JSON，一律丢弃。分类与理由见文件顶部注释及
                # app/services/stream_filter.py。
                if not should_stream_token(event, streaming_node, internal_node):
                    continue
                node = resolve_llm_node(event, streaming_node, internal_node)
                if node and node not in _STREAMING_LLM_NODES:
                    # 归属不明的 token 保守放行（不因元数据缺失吞掉正常答案），
                    # 但记 debug 日志便于排查。
                    logger.debug(
                        "stream_master: 放行未知节点 %r 的 LLM token", node
                    )

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
                    # 累积"用户实际看到的答案"。final_answer 此前只由 refuse /
                    # output_guard 两个节点赋值，走 LLM 正常生成的路径时它一直是
                    # 空串 —— 于是 is_refusal("") 恒为 False，模型自己说"没有找到
                    # 相关信息"时实时流仍上报 refused=false，界面继续出现
                    # "答不出来 + N 个引用来源"的矛盾画面（问题1 截图1）。
                    # 落库走的是 state["answer"]，所以历史回放一直是对的，
                    # 只有实时流错 —— 这正是"重开历史就正常"的原因。
                    final_answer += token
                    total_chars += len(token)
                    answer_streamed = True
                    yield {"type": "chunk", "content": token}

        # 循环收尾：把旁路缓冲里最后残留的确定性文本补发掉
        # （正常情况已被后续事件冲刷，这里是兜底）。
        while deterministic_buf:
            text = deterministic_buf.pop(0)
            final_answer += text
            total_chars += len(text)
            answer_streamed = True
            yield {"type": "chunk", "content": text}

        # ── 持续监控 + Bad Case 回流（循环结束后一次性结算）────────────────
        # 放在 done 之前：先记录时延与回流，再告诉前端"本轮结束"。
        elapsed = time.monotonic() - started_at
        record_latency("total", elapsed)
        await _maybe_capture_bad_case(
            query=query,
            answer=final_answer,
            intent=detected_intent,
            conversation_id=conversation_id,
            user_id=user_id,
            username=username,
            sources=final_sources,
            gate_audit=gate_audit,
            citation_audit=citation_audit,
            guard_audit=guard_audit,
        )

        # ── 质量事件落库（/quality/stats 的历史来源）────────────────────────
        # 进程内计数器重启归零，面板重启后一片 "—"；每轮在这里落一行
        # quality_events，比率改为按时间窗从库里聚合。best-effort，
        # 失败只记日志，绝不影响已经发出的答案。
        await record_quality_event(QualityEventSignal(
            intent=detected_intent,
            evidence_passed=(
                gate_audit.get("passed") if gate_audit else None
            ),
            evidence_confidence=(
                gate_audit.get("confidence") if gate_audit else None
            ),
            citation_overall=citation_audit.get("overall") or None,
            citation_total=int(citation_audit.get("total") or 0),
            citation_passed=int(citation_audit.get("passed") or 0),
            citation_unsupported=len(citation_audit.get("unsupported") or []),
            citation_hallucinated=len(citation_audit.get("hallucinated") or []),
            citation_misattributed=len(citation_audit.get("misattributed") or []),
            citation_number_mismatch=len(citation_audit.get("number_mismatch") or []),
            citation_date_mismatch=len(citation_audit.get("date_mismatch") or []),
            output_guard_changed=bool(guard_audit.get("changed")),
            refusal_source=(
                "gate" if gate_audit.get("passed") is False
                else ("model" if citation_audit.get("overall") == "refused_by_model" else "")
            ),
            sources_count=len(final_sources),
            answer_chars=len(final_answer),
            latency_ms=int(elapsed * 1000),
            user_id=user_id,
            username=username,
            conversation_id=conversation_id,
        ))

        # ── 答复性质：拒答时把引用来源标成"未采用"──────────────────────────
        # sources 事件先于"是否拒答"发出：检索确实命中了片段，但 Evidence Gate
        # 或模型判定证据不足而拒答。若不显式告知前端，界面就会出现
        # "答不出来"＋"1 个引用来源"并存的矛盾画面（问题1 截图1 的显示 BUG）。
        # 历史回放靠 _persist_meta 里的 answer_status 修正；实时流必须在这里
        # 补发同一个事件，否则"实时看矛盾、重开历史才正常"。
        #
        # 无条件发出（不再用 `if final_sources` 收窄）：_build_turn_meta 落库时
        # 就是无条件写的，实时流若只在有来源时才发，两条路径的语义就不一致；
        # 而且无检索管线（document_summary / general_chat / list_documents）的
        # 前端只能靠"事件缺失 ⇒ 默认正常"这种隐式约定，任何依赖 answerStatus
        # 的新 UI 逻辑都会在这些管线上静默失效。note 文案仍与落库口径一致，
        # 仅在确有来源时才提示"以下来源未被采用"。
        from app.services.nodes.evidence_gate import is_refusal

        refused = is_refusal(final_answer)
        yield {
            "type": "answer_status",
            "refused": refused,
            "sources_used": not refused,
            "note": (
                "本次答复为拒答：检索到的片段不足以支撑结论，以下来源未被采用。"
                if refused and final_sources
                else ""
            ),
        }

        yield {
            "type": "done",
            "conversation_id": conversation_id,
            "total_chars": total_chars,
        }

    except Exception as e:
        logger.error(f"Error during master graph generation: {e}")
        yield {"type": "error", "message": _friendly_error(e)}
    finally:
        # 必须还原：否则同一 task 复用上下文时，后续（其实不会再有的）旁路
        # 文本会写进一份已经没人读的列表。
        reset_sink(_sink_token)
