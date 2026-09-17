"""
RAG query API router (Phase 4 / 问题1).

POST /query — accepts a JSON body, runs the LangGraph RAG pipeline,
              and streams the response as Server-Sent Events (SSE).

Two pipelines (selected by ``mode``, auto-detected when omitted):
    rag            — semantic retrieval QA (default)
    doc_relations  — cross-document relation analysis (问题1): answers the
                     "这些文档有什么关联" question organised BY DOCUMENT

SSE event stream format
────────────────────────
Every event is a JSON object on a ``data:`` line, terminated by
two newlines (per the SSE spec).

    data: {"type":"thinking"}

    data: {"type":"sources","sources":[...],"retrieved_count":5}
      — or, for doc_relations mode:
    data: {"type":"doc_digests","documents":[...],"retrieved_count":5}

    data: {"type":"chunk","content":"The annual report..."}

    data: {"type":"chunk","content":" shows revenue growth..."}

    data: {"type":"done","conversation_id":"<uuid>","total_chars":412}

    data: [DONE]

Error event (only when the pipeline fails mid-stream):

    data: {"type":"error","message":"<description>"}

    data: [DONE]

Client-side usage example (JavaScript):

    const es = new EventSource(undefined);
    const resp = await fetch("/query", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query: "...", conversation_id: null}),
    });
    const reader = resp.body.getReader();
    // read SSE lines and parse JSON from each `data:` line
"""

import json
import uuid
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.api.deps import client_ip, enforce_rate_limit
from app.services.permissions import require_permission
from app.config import get_settings
from app.db.user_models import User
from app.schemas.query import QueryRequest, SSEThinking
from app.services.audit_service import record_audit
from app.services.conversation_service import get_or_create_conversation, load_history
from app.services.prompt_security import inspect_user_query
from app.services.master_graph import stream_master
from app.services.rag_graph import stream_document_list
from app.services.routers.intent_rules import deterministic_route
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Query"])

# SSE helpers ──────────────────────────────────────────────────────────────────

def _sse(payload: dict | str) -> str:
    """Format a single SSE data line."""
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# Deterministic pipeline auto-detection (问题1) ──────────────────────────────
#
# 规则本体在 app/services/routers/intent_rules.py，与 master graph 的
# Query Router 共用同一份定义，避免两边正则漂移。
#
# 注意返回值语义（问题1 的关键）：判定不了时返回 **None**，而不是以前的
# "rag"。返回 "rag" 会让调用方误以为"已确定走知识库问答"，进而绕过 LLM
# 路由 —— 这正是闲聊分支永远触发不了的原因之一。


def _detect_mode(query: str) -> str | None:
    """Return the deterministically selected pipeline, or None if undecided."""
    detected = deterministic_route(query)
    if detected in (
        "doc_relations",
        "list_documents",
        "general_chat",
        "document_agent",
    ):
        logger.info("Query auto-detected as %s: %r", detected, query[:80])
        return detected
    return None


# 旧 mode 名 → master graph intent 名。
# 自动检测（未显式传 mode）时不走这里，交给 master graph 的 LLM router，
# 它比正则更准；只有客户端显式指定 mode 才需要归一化。
_LEGACY_MODE_ALIASES = {
    "rag": "knowledge_qa",
}


# Streaming generator ──────────────────────────────────────────────────────────

async def _generate_sse(
    request: Request,
    query: str,
    conversation_id: uuid.UUID | None,
    top_k: int,
    mode: str,
    user: User,
    collection_id: uuid.UUID | None = None,
    forced_mode: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator that drives the entire RAG pipeline and yields raw SSE
    strings suitable for a StreamingResponse.

    Pipeline:
        1. Resolve / create conversation (PostgreSQL, owner-scoped)
        2. Load conversation history (PostgreSQL)
        3. Yield "thinking" event immediately
        4. Run the master LangGraph astream_events:
             a. After the router node   → yield "intent"   (路由结果)
             b. After retrieval/digests → yield "sources" / "doc_digests"
             c. After the grader        → yield "grade"    (good / bad + retry)
             d. Per Ollama token        → yield "chunk"    (1:1 与模型输出)
             e. After output_guard      → yield "output_guard" (问题3+4 合规审计信号)
             f. After graph end         → yield "done"
        5. Yield "[DONE]" SSE terminator

    *mode* is the auto-detected/legacy pipeline label used for logging and for
    the deterministic ``list_documents`` fast path. *forced_mode* is only set
    when the client explicitly passed ``mode`` in the request body — in that
    case the master graph's LLM router is bypassed.

    Retrieval is restricted to documents owned by *user* (admins see all)
    and, when *collection_id* is set, to one knowledge-base collection.
    """
    # ── 1+2. Conversation setup (owner + tenant scoped) ───────────────────────
    from app.services.tenancy import scope_for

    # 可见范围一处组装：平台管理员 = 全平台（跨公司部门库/公司库，但看不到
    # 别人的个人库）；其他人 = 本公司内 个人 + 本部门 + 公司库。
    scope = scope_for(user)
    # 个人库归属**恒为本人**（含平台管理员）：owner_id 不再是"None = admin 全览"，
    # 因此这里必须是 user.id，否则管理员会连自己的个人库都检索不到。
    owner = scope.owner_id
    tenant = scope.tenant_id
    department = scope.department_id
    try:
        conv_id = await get_or_create_conversation(
            conversation_id, owner_id=owner, tenant_id=tenant,
        )
        history = await load_history(conv_id, owner_id=owner, tenant_id=tenant)
    except Exception as exc:
        logger.exception("Conversation setup failed: %s", exc)
        yield _sse({"type": "error", "message": f"Conversation setup failed: {exc}"})
        yield _sse("[DONE]")
        return

    # ── 3. Immediate "thinking" event ─────────────────────────────────────────
    yield _sse(SSEThinking().model_dump())

    # ── 4. LangGraph streaming ────────────────────────────────────────────────
    if mode == "list_documents":
        # 文档列表走确定性 DB 直读 —— 不进 LLM、不检索，保证不漏不重
        graph_events = stream_document_list(
            query=query,
            conversation_id=str(conv_id),
            owner_id=str(owner) if owner else None,
            collection_id=str(collection_id) if collection_id else None,
            tenant_id=tenant,
            user_department_id=department,
            user_id=str(user.id),
            tenant_wide=scope.tenant_wide,
            platform_wide=scope.platform_wide,
        )
    else:
        # 其余全部交给 master graph：
        #   Query Router → {Document Summary | Knowledge QA(Hybrid RAG
        #   + Retrieval Grader + Retry) | General Chat | doc_relations}
        graph_events = stream_master(
            query=query,
            conversation_id=str(conv_id),
            history_messages=history,
            top_k=top_k,
            owner_id=str(owner) if owner else None,
            collection_id=str(collection_id) if collection_id else None,
            forced_mode=forced_mode,
            # 监控 / Bad Case 回流需要"是谁问的"，owner_id 在管理员视角下为 None，
            # 因此单独传真实用户身份（与检索权限解耦）。
            user_id=str(user.id),
            username=user.username,
            # 三层隔离：第一层公司前置过滤 + 第二层部门 ACL + 第三层会话归属。
            # tenant_id=None 仅表示"跨公司"（平台管理员），仍受 ACL 约束。
            tenant_id=tenant,
            department_id=department,
            tenant_wide=scope.tenant_wide,
            platform_wide=scope.platform_wide,
        )

    try:
        async for graph_event in graph_events:
            # Check for client disconnect on each event to stop early
            if await request.is_disconnected():
                logger.info("Client disconnected mid-stream for conv=%s", conv_id)
                return

            yield _sse(graph_event)

    except Exception as exc:
        logger.exception("RAG graph stream error for conv=%s: %s", conv_id, exc)
        yield _sse({"type": "error", "message": str(exc)})

    # ── 5. SSE stream terminator ──────────────────────────────────────────────
    yield _sse("[DONE]")


# Router endpoint ──────────────────────────────────────────────────────────────

@router.post(
    "/query",
    summary="RAG query with streaming response",
    description=(
        "Submit a natural-language question. The response is a "
        "**Server-Sent Events** stream.\n\n"
        "Set `Accept: text/event-stream` or simply consume the stream — "
        "FastAPI will set `Content-Type: text/event-stream` automatically.\n\n"
        "**Query Router**: when `mode` is omitted, an LLM router classifies the "
        "question and dispatches it to one of five pipelines — `knowledge_qa` "
        "(Hybrid RAG: 多路向量 + BM25 → RRF → cross-encoder 精排), "
        "`document_summary` (per-document content summary), `general_chat` "
        "(no retrieval at all), `doc_relations` (cross-document analysis) and "
        "`list_documents` (deterministic DB listing). Passing `mode` explicitly "
        "bypasses the router.\n\n"
        "**Retrieval Grader + Retry**: in `knowledge_qa`, retrieved evidence is "
        "graded by an LLM (not just a score threshold). If the evidence cannot "
        "answer the question, the query is rewritten and retrieval retried up to "
        "`RETRIEVAL_MAX_RETRIES` times before refusing to answer.\n\n"
        "**Event types** (in order):\n"
        "- `thinking` — pipeline started\n"
        "- `intent`   — router decision (`intent`, `reason`)\n"
        "- `sources` / `doc_digests` — retrieved metadata (before any tokens)\n"
        "- `grade`    — evidence grader result (`good`, `retry`, `reason`)\n"
        "- `chunk`    — one LLM token or small token batch\n"
        "- `done`     — answer complete; includes `conversation_id` for follow-up\n"
        "- `error`    — unrecoverable failure (stream still closes cleanly)\n\n"
        "Pass the returned `conversation_id` in subsequent requests to continue "
        "the conversation with full memory."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE stream of RAG events",
            "content": {"text/event-stream": {}},
        },
        422: {"description": "Validation error — invalid request body"},
    },
)
async def query_endpoint(
    body: QueryRequest,
    request: Request,
    user: Annotated[User, Depends(require_permission("conversation.write"))],
) -> StreamingResponse:
    settings = get_settings()

    # Per-user rate limit — protects the shared Ollama instance
    await enforce_rate_limit(
        request,
        scope="query",
        key=str(user.id),
        limit=settings.QUERY_RATE_LIMIT,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    )

    # Prompt-injection guard runs before query rewrite, retrieval and LLM use.
    # Blocked input never reaches the vector store or model; suspicious-but-
    # legitimate security questions are allowed and marked in the audit trail.
    inspection = inspect_user_query(body.query)
    effective_query = inspection.normalized_text
    if settings.PROMPT_GUARD_ENABLED and inspection.blocked and settings.PROMPT_GUARD_BLOCK_HIGH_RISK:
        await record_audit(
            "security.prompt_injection.blocked",
            user_id=user.id,
            username=user.username,
            resource_type="query",
            resource_id=str(body.collection_id) if body.collection_id else None,
            detail=f"risk={inspection.risk}; signals={len(inspection.reasons)}; q={effective_query[:160]}",
            ip=client_ip(request),
        )
        logger.warning(
            "Blocked suspected prompt injection user=%s signals=%d",
            user.username,
            len(inspection.reasons),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该请求包含疑似试图改变系统行为的指令，已被安全策略拦截。请仅提交与知识库内容相关的问题。",
        )

    detected = _detect_mode(effective_query)
    # "auto" 表示交给 master graph 的 LLM 路由决定（不再是以前的 "rag"）。
    mode = body.mode or detected or "auto"

    # 客户端显式指定 mode 时，跳过 master graph 的 LLM 路由。
    # 旧客户端习惯传 mode="rag"，这里归一化成 master graph 的 intent 名。
    forced_mode = None
    if body.mode:
        forced_mode = _LEGACY_MODE_ALIASES.get(body.mode, body.mode)
    elif detected in ("doc_relations", "general_chat", "document_agent"):
        # 确定性判定命中 → 跳过 LLM 路由，省一次调用，且不受路由超时影响。
        # list_documents 不走这里，它有自己的 DB 直读快路径。
        forced_mode = detected

    await record_audit(
        "query.ask" if inspection.risk == "clean" else "query.ask.suspicious",
        user_id=user.id,
        username=user.username,
        resource_type="query",
        resource_id=str(body.collection_id) if body.collection_id else None,
        detail=f"mode={mode}; risk={inspection.risk}; q={effective_query[:300]}",
        ip=client_ip(request),
    )

    logger.info(
        "POST /query | conv=%s top_k=%d mode=%s user=%s collection=%s risk=%s query=%r",
        body.conversation_id,
        body.top_k,
        mode,
        user.username,
        body.collection_id,
        inspection.risk,
        effective_query[:80],
    )

    return StreamingResponse(
        _generate_sse(
            request=request,
            query=effective_query,
            conversation_id=body.conversation_id,
            top_k=body.top_k,
            mode=mode,
            user=user,
            collection_id=body.collection_id,
            forced_mode=forced_mode,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",       # disable nginx proxy buffering
            "Access-Control-Allow-Origin": "*",
        },
    )
