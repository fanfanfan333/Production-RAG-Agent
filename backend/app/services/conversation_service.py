"""
Conversation history CRUD service (Phase 4).

Manages Conversation rows and Message rows in PostgreSQL.
All operations are async and use the existing get_db_session() context manager.

Responsibilities:
  - get_or_create_conversation  — resolve or mint a new session UUID
  - load_history                — fetch last N turns as LangChain messages
  - save_turn                   — persist user + assistant message pair
"""

import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import select

from app.db.conversation_models import Conversation, Message
from app.db.postgres import get_db_session
from app.services.tenancy import DEFAULT_TENANT_ID, normalize_tenant_id
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Maximum number of *pairs* (user+assistant) to load as context window.
# Kept at 3 pairs: enough for coreference resolution, small enough to prevent
# the model from repeating unrelated content from distant turns.
_DEFAULT_HISTORY_PAIRS = 3


async def get_or_create_conversation(
    conversation_id: uuid.UUID | None,
    owner_id: uuid.UUID | None = None,
    tenant_id: str | None = None,
) -> uuid.UUID:
    """
    Return the UUID of the requested conversation, or create a new one.

    If *conversation_id* is provided but does not exist (or belongs to a
    different user **or a different tenant**), a **new** conversation is
    created (defensive: avoids 404 mid-stream, and never leaks another
    user's/tenant's thread).

    第三层隔离键 = conversation_id + tenant_id + user_id：
    用户 A 的会话，同租户的用户 B 拿不到，跨租户更拿不到。

    Returns:
        UUID of the resolved or newly created Conversation.
    """
    tenant_id = normalize_tenant_id(tenant_id) if tenant_id else None
    async with get_db_session() as session:
        if conversation_id is not None:
            existing = await session.get(Conversation, conversation_id)
            if existing is not None:
                if owner_id is not None and existing.owner_id not in (None, owner_id):
                    logger.warning(
                        "conversation_id=%s owned by another user — starting a new conversation",
                        conversation_id,
                    )
                elif (
                    tenant_id is not None
                    and normalize_tenant_id(existing.tenant_id) != tenant_id
                    and existing.tenant_id is not None
                ):
                    logger.warning(
                        "conversation_id=%s belongs to another tenant (%s) — starting a new conversation",
                        conversation_id, existing.tenant_id,
                    )
                else:
                    logger.debug("Resuming conversation id=%s", conversation_id)
                    return existing.id

            logger.warning(
                "conversation_id=%s not found — starting new conversation",
                conversation_id,
            )

        new_conv = Conversation(
            id=uuid.uuid4(),
            owner_id=owner_id,
            tenant_id=tenant_id or DEFAULT_TENANT_ID,
        )
        session.add(new_conv)
        await session.flush()
        conv_id = new_conv.id

    logger.info("Created new conversation id=%s (owner=%s tenant=%s)", conv_id, owner_id, tenant_id)
    return conv_id


async def load_history(
    conversation_id: uuid.UUID,
    max_pairs: int = _DEFAULT_HISTORY_PAIRS,
    *,
    owner_id: uuid.UUID | None = None,
    tenant_id: str | None = None,
) -> list[BaseMessage]:
    """
    Load the most recent *max_pairs* user+assistant exchanges from the DB
    and return them as an ordered list of LangChain BaseMessage objects.

    The list is returned in chronological order (oldest first) so it can
    be directly appended to the LLM prompt.

    第三层隔离：传入 owner_id / tenant_id 时会先校验会话归属，不匹配
    直接返回空历史（绝不把别人/别租户的对话记忆喂进当前会话）。

    Args:
        conversation_id: UUID of the conversation to load.
        max_pairs:       Maximum number of user/assistant turns to include.
        owner_id:        期望的会话属主（None = 不校验）。
        tenant_id:       期望的会话租户（None = 不校验）。

    Returns:
        List of HumanMessage / AIMessage objects, oldest first.
    """
    async with get_db_session() as session:
        if owner_id is not None or tenant_id is not None:
            conv = await session.get(Conversation, conversation_id)
            if conv is None:
                return []
            if owner_id is not None and conv.owner_id not in (None, owner_id):
                logger.warning(
                    "load_history blocked: conv=%s owner mismatch", conversation_id,
                )
                return []
            if (
                tenant_id is not None
                and conv.tenant_id is not None
                and normalize_tenant_id(conv.tenant_id) != normalize_tenant_id(tenant_id)
            ):
                logger.warning(
                    "load_history blocked: conv=%s tenant mismatch (%s != %s)",
                    conversation_id, conv.tenant_id, tenant_id,
                )
                return []

        # Fetch the last (max_pairs * 2) rows to cover complete pairs
        result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(max_pairs * 2)
        )
        rows = list(reversed(result.scalars().all()))

    messages: list[BaseMessage] = []
    for row in rows:
        if row.role == "user":
            messages.append(HumanMessage(content=row.content))
        elif row.role == "assistant":
            messages.append(AIMessage(content=row.content))

    logger.debug(
        "Loaded %d history messages for conversation_id=%s",
        len(messages),
        conversation_id,
    )
    return messages


async def save_turn(
    conversation_id: uuid.UUID,
    user_message: str,
    assistant_message: str,
    *,
    user_id: uuid.UUID | None = None,
    tenant_id: str | None = None,
    assistant_meta: dict | None = None,
) -> None:
    """
    Persist one complete user/assistant turn.

    Both messages are written in the same transaction so they either
    both succeed or both fail — no half-saved turns.

    第三层隔离：user_id / tenant_id 随消息落库（隔离键
    conversation_id + tenant_id + user_id 的存储基础），老调用方
    不传则为 NULL，读取侧按"继承所属 Conversation"处理。

    *assistant_meta*：回答的依据快照（引用来源 / 引用校验 / 证据门控 /
    输出合规 / 生成的文档 / 路由意图）。它决定"重新打开这段历史时能不能
    看到当时的引用来源" —— 不落库的话切窗口/切页就全丢。

    Also bumps Conversation.updated_at so the conversation list stays sorted.
    """
    now = datetime.now(tz=timezone.utc)

    async with get_db_session() as session:
        # Bump updated_at on the parent conversation
        conv_result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = conv_result.scalar_one_or_none()
        if conv is not None:
            conv.updated_at = now
            # 消息租户与会话对齐（调用方未传时继承，保证同会话同租户）
            if tenant_id is None:
                tenant_id = conv.tenant_id

        session.add(Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content=user_message,
            user_id=user_id,
            tenant_id=tenant_id,
            created_at=now,
        ))
        session.add(Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_message,
            user_id=user_id,
            tenant_id=tenant_id,
            created_at=now,
            meta=assistant_meta or None,
        ))

    logger.debug(
        "Saved turn for conversation_id=%s (user=%d chars, assistant=%d chars, meta=%s)",
        conversation_id,
        len(user_message),
        len(assistant_message),
        "yes" if assistant_meta else "no",
    )


async def list_conversations(
    limit: int = 50,
    owner_id: uuid.UUID | None = None,
    tenant_id: str | None = None,
) -> list[dict]:
    """
    Return the most recent conversations, newest activity first.

    Non-admin users only see their own threads (owner filter); *tenant_id*
    进一步把列表约束在当前租户内（第三层隔离：跨租户会话不可见）。
    Conversations with no messages are excluded.
    """
    async with get_db_session() as session:
        base_q = (
            select(
                Conversation.id,
                Conversation.created_at,
                Conversation.updated_at,
            )
            .order_by(Conversation.updated_at.desc())
            .limit(limit * 2)
        )
        if owner_id is not None:
            base_q = base_q.where(Conversation.owner_id == owner_id)
        if tenant_id is not None:
            base_q = base_q.where(
                Conversation.tenant_id == normalize_tenant_id(tenant_id)
            )
        rows = (await session.execute(base_q)).all()

        conversations: list[dict] = []
        for conv_id, created_at, updated_at in rows:
            msgs = await session.execute(
                select(Message.role, Message.content, Message.created_at)
                .where(Message.conversation_id == conv_id)
                .order_by(Message.created_at.asc())
            )
            all_msgs = msgs.all()
            if not all_msgs:
                continue

            first_user = next(
                (m.content for m in all_msgs if m.role == "user"), None
            )
            title = (first_user or all_msgs[0].content or "新对话").strip()
            if len(title) > 40:
                title = title[:40] + "…"

            conversations.append({
                "id": str(conv_id),
                "title": title,
                "message_count": len(all_msgs),
                "created_at": created_at.isoformat() if created_at else None,
                "updated_at": updated_at.isoformat() if updated_at else None,
            })
            if len(conversations) >= limit:
                break

    logger.debug("list_conversations → %d threads", len(conversations))
    return conversations


async def get_conversation_messages(
    conversation_id: uuid.UUID,
    owner_id: uuid.UUID | None = None,
    tenant_id: str | None = None,
) -> list[dict] | None:
    """
    Return all messages of a conversation in chronological order.

    *meta* 是 assistant 消息的依据快照（引用来源 / 引用校验 / 证据门控 /
    输出合规 / 生成的文档 / 路由意图），前端据此在重开历史时把引用来源
    一并还原 —— 过去这里只回 content，历史会话里的"数据来源"因此永久丢失。

    Returns None when the conversation does not exist or belongs to a
    different user **or tenant** (404 semantics — no existence leak).
    """
    async with get_db_session() as session:
        conv = await session.get(Conversation, conversation_id)
        if conv is None:
            return None
        if owner_id is not None and conv.owner_id not in (None, owner_id):
            return None
        if (
            tenant_id is not None
            and conv.tenant_id is not None
            and normalize_tenant_id(conv.tenant_id) != normalize_tenant_id(tenant_id)
        ):
            return None

        result = await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        rows = result.scalars().all()

    return [
        {
            "role": row.role,
            "content": row.content,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "meta": row.meta or None,
        }
        for row in rows
    ]


async def delete_conversation(
    conversation_id: uuid.UUID,
    owner_id: uuid.UUID | None = None,
    tenant_id: str | None = None,
) -> bool:
    """Delete a conversation and (via cascade) all of its messages."""
    async with get_db_session() as session:
        conv = await session.get(Conversation, conversation_id)
        if conv is None:
            return False
        if owner_id is not None and conv.owner_id not in (None, owner_id):
            return False
        if (
            tenant_id is not None
            and conv.tenant_id is not None
            and normalize_tenant_id(conv.tenant_id) != normalize_tenant_id(tenant_id)
        ):
            return False
        await session.delete(conv)
    logger.info("Deleted conversation id=%s", conversation_id)
    return True
