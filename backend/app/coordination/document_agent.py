"""Document-QA specialist that reuses the existing master graph.

This is the compatibility hinge: ordinary ``POST /query`` is untouched, while a
coordinated document subtask calls the *same* graph with the same user-scope
issuance, conversation ownership and citation pipeline. It does not implement a
second retrieval stack and cannot widen the user's document permissions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.db.user_models import User


@dataclass
class DocumentAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    conversation_id: str = ""
    intent: str = "knowledge_qa"


async def answer_document_question(
    *,
    query: str,
    user: User,
    collection_id: uuid.UUID | None = None,
    top_k: int = 8,
    user_scope=None,
) -> DocumentAnswer:
    from app.config import get_settings
    from app.services.conversation_service import (
        get_or_create_conversation,
        load_history,
        split_recent_history,
    )
    from app.services.master_graph import stream_master
    from app.services.tenancy import home_tenant_id

    settings = get_settings()
    if user_scope is None:
        raise RuntimeError("缺少 API 签发的文档权限范围")
    scope = user_scope.base
    owner = scope.owner_id
    conversation_tenant = home_tenant_id(user)
    conversation_id = await get_or_create_conversation(
        None, owner_id=owner, tenant_id=conversation_tenant
    )
    recent_pairs = max(1, int(getattr(settings, "AGENT_RECENT_PAIRS", 3)))
    load_pairs = (
        max(recent_pairs, int(getattr(settings, "AGENT_HISTORY_WINDOW_PAIRS", 12)))
        if bool(getattr(settings, "AGENT_CONTEXT_LAYERS_ENABLED", False))
        else recent_pairs
    )
    history = await load_history(
        conversation_id,
        max_pairs=load_pairs,
        owner_id=owner,
        tenant_id=conversation_tenant,
    )
    history_messages, history_window = split_recent_history(history, recent_pairs)

    answer_parts: list[str] = []
    sources: list[dict] = []
    intent = "knowledge_qa"
    async for event in stream_master(
        query=query,
        conversation_id=str(conversation_id),
        history_messages=history_messages,
        history_window=history_window,
        top_k=top_k,
        owner_id=str(owner) if owner else None,
        collection_id=str(collection_id) if collection_id else None,
        forced_mode="knowledge_qa",
        user_id=str(user.id),
        username=user.username,
        tenant_ids=scope.tenant_ids,
        department_id=scope.department_id,
        tenant_wide=scope.tenant_wide,
        owns_tenant_ids=scope.owns_tenant_ids,
        conversation_tenant_id=conversation_tenant,
        user_scope=user_scope,
    ):
        kind = event.get("type")
        if kind == "intent":
            intent = str(event.get("intent") or intent)
        elif kind == "sources":
            sources = list(event.get("sources") or [])
        elif kind == "chunk":
            answer_parts.append(str(event.get("content") or ""))
        elif kind in {"output_guard", "citation_check"}:
            sanitized = event.get("sanitized_answer")
            if sanitized:
                answer_parts = [str(sanitized)]
        elif kind == "error":
            raise RuntimeError(str(event.get("message") or "文档问答执行失败"))

    answer = "".join(answer_parts).strip()
    if not answer:
        raise RuntimeError("文档问答未产生任何答案")
    return DocumentAnswer(
        answer=answer,
        sources=sources,
        conversation_id=str(conversation_id),
        intent=intent,
    )
