"""
Answer feedback API router (反馈闭环优化).

POST /feedback        — record a 👍/👎 rating on an assistant answer (auth)
GET  /feedback        — recent feedback, admin only (badcase 审阅入口)

Ratings store a full Q&A snapshot so the review side stays self-contained
even after the originating conversation has been deleted.
"""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import aliased
from pydantic import BaseModel, Field

from app.services.permissions import require_permission, require_platform_admin
from app.db.feedback_models import AnswerFeedback
from app.db.conversation_models import Conversation, Message
from app.services.tenancy import home_tenant_id, DEFAULT_TENANT_ID
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.services.prompt_security import normalize_text
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Feedback"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    question: str = Field(min_length=1, max_length=8000)
    answer: str = Field(max_length=20000)
    conversation_id: uuid.UUID | None = None
    message_id: uuid.UUID | None = None
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackItem(BaseModel):
    id: str
    username: str
    rating: str
    question: str
    answer: str
    comment: str | None
    conversation_id: str | None
    created_at: datetime


class FeedbackListResponse(BaseModel):
    feedback: list[FeedbackItem]
    total: int


# ── POST /feedback ────────────────────────────────────────────────────────────

@router.post(
    "/feedback",
    summary="Rate an assistant answer (👍/👎)",
    description=(
        "Record the signed-in user's rating for one assistant answer. "
        "The question and answer text are snapshotted with the rating so "
        "reviewers see the full context without joining conversations."
    ),
)
async def submit_feedback(
    body: FeedbackRequest,
    user: Annotated[User, Depends(require_permission("feedback.write"))],
) -> dict:
    # Queries are normalized on ingestion; keep exact legacy history valid too.
    question_candidates = {body.question, normalize_text(body.question)}
    async with get_db_session() as session:
        conversation = await session.get(Conversation, body.conversation_id) if body.conversation_id else None
        if (conversation is None or conversation.owner_id != user.id
                or conversation.tenant_id != (home_tenant_id(user) or DEFAULT_TENANT_ID)):
            raise HTTPException(404, "回答记录不存在或不可访问")
        statement = select(Message).where(
            Message.conversation_id == conversation.id, Message.role == "assistant",
        )
        if body.message_id:
            statement = statement.where(Message.id == body.message_id)
        else:
            user_message = aliased(Message)
            statement = statement.join(user_message, (user_message.conversation_id == Message.conversation_id)
                & (user_message.created_at == Message.created_at) & (user_message.role == "user")).where(
                    Message.content == body.answer, user_message.content.in_(question_candidates))
        message = await session.scalar(statement.order_by(Message.created_at.desc()).limit(1))
        if message is None or message.content != body.answer:
            raise HTTPException(409, "请刷新会话后，对已完成的真实回答提交反馈")
        question = await session.scalar(select(Message).where(
            Message.conversation_id == conversation.id, Message.role == "user",
            Message.created_at == message.created_at,
        ).limit(1))
        if question is None or question.content not in question_candidates:
            raise HTTPException(409, "反馈问题与回答记录不一致")
        body.question, body.answer = question.content, message.content
        row = AnswerFeedback(
            user_id=user.id,
            conversation_id=body.conversation_id,
            rating=body.rating,
            question=body.question[:8000],
            answer=(body.answer or "")[:20000],
            comment=body.comment,
            message_id=message.id,
        )
        session.add(row)
        await session.commit()

    await record_audit(
        "feedback.submit",
        user_id=user.id,
        username=user.username,
        resource_type="feedback",
        resource_id=None,
        detail=f"rating={body.rating}; q={body.question[:200]}",
    )

    # ── Bad Case 回流：👎 自动进入审阅队列（持续监控闭环的人工端）───────────
    # 用户点👎是最强的质量信号，但它稀疏（多数人不会点）。这里把它自动
    # 沉淀到 bad_cases，与"引用校验失败 / 门控拒答 / 输出净化"三类自动信号
    # 汇入同一队列，管理员只需看一个地方。
    if body.rating == "down":
        from app.services.monitoring_service import BadCaseSignal, capture_bad_case

        await capture_bad_case(BadCaseSignal(
            reason="feedback_down",
            severity="high",
            question=body.question,
            answer=body.answer,
            user_id=str(user.id),
            username=user.username,
            conversation_id=str(body.conversation_id) if body.conversation_id else None,
            detail={"comment": body.comment, "rating": body.rating},
        ))

    logger.info(
        "Feedback recorded: user=%s rating=%s conv=%s",
        user.username, body.rating, body.conversation_id,
    )
    return {"recorded": True, "rating": body.rating, "message_id": str(message.id)}


# ── GET /feedback ─────────────────────────────────────────────────────────────

@router.get(
    "/feedback",
    response_model=FeedbackListResponse,
    summary="List recent answer feedback (admin only)",
    description=(
        "Returns the most recent 👍/👎 ratings across all users, newest first. "
        "Down-voted entries are the primary badcase review queue."
    ),
)
async def list_feedback(
    admin: Annotated[User, Depends(require_platform_admin())],
    limit: int = Query(default=100, ge=1, le=500),
) -> FeedbackListResponse:
    from sqlalchemy import select

    from app.db.user_models import User as UserModel

    async with get_db_session() as session:
        stmt = (
            select(AnswerFeedback, UserModel.username)
            .join(UserModel, AnswerFeedback.user_id == UserModel.id, isouter=True)
            .order_by(AnswerFeedback.created_at.desc())
            .limit(limit)
        )
        rows = (await session.execute(stmt)).all()

    items = [
        FeedbackItem(
            id=str(fb.id),
            username=username or "未知用户",
            rating=fb.rating,
            question=fb.question,
            answer=fb.answer,
            comment=fb.comment,
            conversation_id=str(fb.conversation_id) if fb.conversation_id else None,
            created_at=fb.created_at,
        )
        for fb, username in rows
    ]
    return FeedbackListResponse(feedback=items, total=len(items))
