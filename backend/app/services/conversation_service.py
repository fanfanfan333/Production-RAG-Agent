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
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Maximum number of *pairs* (user+assistant) to load as context window.
# Kept at 3 pairs: enough for coreference resolution, small enough to prevent
# the model from repeating unrelated content from distant turns.
_DEFAULT_HISTORY_PAIRS = 3


async def get_or_create_conversation(
    conversation_id: uuid.UUID | None,
    owner_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """
    Return the UUID of the requested conversation, or create a new one.

    If *conversation_id* is provided but does not exist (or belongs to a
    different user), a **new** conversation is created (defensive: avoids 404
    mid-stream, and never leaks another user's thread).

    Returns:
        UUID of the resolved or newly created Conversation.
    """
    async with get_db_session() as session:
        if conversation_id is not None:
            existing = await session.get(Conversation, conversation_id)
            if existing is not None:
                if owner_id is not None and existing.owner_id not in (None, owner_id):
                    logger.warning(
                        "conversation_id=%s owned by another user — starting a new conversation",
                        conversation_id,
                    )
                else:
                    logger.debug("Resuming conversation id=%s", conversation_id)
                    return existing.id

            logger.warning(
                "conversation_id=%s not found — starting new conversation",
                conversation_id,
            )

        new_conv = Conversation(id=uuid.uuid4(), owner_id=owner_id)
        session.add(new_conv)
        await session.flush()
        conv_id = new_conv.id

    logger.info("Created new conversation id=%s (owner=%s)", conv_id, owner_id)
    return conv_id


async def load_history(
    conversation_id: uuid.UUID,
    max_pairs: int = _DEFAULT_HISTORY_PAIRS,
) -> list[BaseMessage]:
    """
    Load the most recent *max_pairs* user+assistant exchanges from the DB
    and return them as an ordered list of LangChain BaseMessage objects.

    The list is returned in chronological order (oldest first) so it can
    be directly appended to the LLM prompt.

    Args:
        conversation_id: UUID of the conversation to load.
        max_pairs:       Maximum number of user/assistant turns to include.

    Returns:
        List of HumanMessage / AIMessage objects, oldest first.
    """
    # Fetch the last (max_pairs * 2) rows to cover complete pairs
    async with get_db_session() as session:
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
) -> None:
    """
    Persist one complete user/assistant turn.

    Both messages are written in the same transaction so they either
    both succeed or both fail — no half-saved turns.

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

        session.add(Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content=user_message,
            created_at=now,
        ))
        session.add(Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_message,
            created_at=now,
        ))

    logger.debug(
        "Saved turn for conversation_id=%s (user=%d chars, assistant=%d chars)",
        conversation_id,
        len(user_message),
        len(assistant_message),
    )


async def list_conversations(limit: int = 50, owner_id: uuid.UUID | None = None) -> list[dict]:
    """
    Return the most recent conversations, newest activity first.

    Non-admin users only see their own threads (owner filter).
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
) -> list[dict] | None:
    """
    Return all messages of a conversation in chronological order.

    Returns None when the conversation does not exist or belongs to a
    different user (404 semantics — no existence leak).
    """
    async with get_db_session() as session:
        conv = await session.get(Conversation, conversation_id)
        if conv is None:
            return None
        if owner_id is not None and conv.owner_id not in (None, owner_id):
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
        }
        for row in rows
    ]


async def delete_conversation(
    conversation_id: uuid.UUID,
    owner_id: uuid.UUID | None = None,
) -> bool:
    """Delete a conversation and (via cascade) all of its messages."""
    async with get_db_session() as session:
        conv = await session.get(Conversation, conversation_id)
        if conv is None:
            return False
        if owner_id is not None and conv.owner_id not in (None, owner_id):
            return False
        await session.delete(conv)
    logger.info("Deleted conversation id=%s", conversation_id)
    return True
