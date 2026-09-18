"""
Conversation history API router (Phase 4 / 企业落地第一阶段).

GET    /conversations               — list recent conversations (owner-scoped)
GET    /conversations/{id}/messages — full message history of one conversation
DELETE /conversations/{id}          — delete a conversation and its messages

Authentication required; non-admin users only ever see their own threads.
Foreign conversation ids resolve to 404 (no existence leak).
"""

from typing import Annotated

import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_current_user
from app.services.permissions import require_permission
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.services.conversation_service import (
    delete_conversation,
    get_conversation_messages,
    list_conversations,
)
from app.services.tenancy import home_tenant_id
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Conversations"])


@router.get(
    "/conversations",
    summary="List recent conversations (owner-scoped)",
)
async def list_conversations_endpoint(
    user: Annotated[User, Depends(require_permission("conversation.read"))],
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    # 会话是**个人**数据：连平台管理员也只列自己的（管理员的跨公司能力
    # 作用于知识库文档，而不是别人的对话 —— 对话正文里可能引用了他的个人库）。
    # 第三层归属键：owner 恒为本人；租户用单一归属公司（admin → None）。
    conversations = await list_conversations(
        limit=limit,
        owner_id=user.id,
        # 第三层隔离：跨公司会话不出现在列表里
        tenant_id=home_tenant_id(user),
    )
    return {"conversations": conversations, "total": len(conversations)}


@router.get(
    "/conversations/{conversation_id}/messages",
    summary="Get all messages in a conversation",
)
async def get_conversation_messages_endpoint(
    conversation_id: uuid.UUID,
    user: Annotated[User, Depends(require_permission("conversation.read"))],
) -> dict:
    messages = await get_conversation_messages(
        conversation_id,
        owner_id=user.id,
        tenant_id=home_tenant_id(user),
    )
    if messages is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return {
        "conversation_id": str(conversation_id),
        "messages": messages,
        "total": len(messages),
    }


@router.delete(
    "/conversations/{conversation_id}",
    summary="Delete a conversation",
)
async def delete_conversation_endpoint(
    conversation_id: uuid.UUID,
    user: Annotated[User, Depends(require_permission("conversation.delete"))],
) -> dict:
    deleted = await delete_conversation(
        conversation_id,
        owner_id=user.id,
        tenant_id=home_tenant_id(user),
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    await record_audit(
        "conversation.delete",
        user_id=user.id,
        username=user.username,
        resource_type="conversation",
        resource_id=str(conversation_id),
    )
    return {"deleted": True, "id": str(conversation_id)}
