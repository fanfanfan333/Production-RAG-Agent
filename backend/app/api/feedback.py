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

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.services.permissions import require_permission, require_platform_admin
from app.db.feedback_models import AnswerFeedback
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Feedback"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    question: str = Field(min_length=1, max_length=8000)
    answer: str = Field(max_length=20000)
    conversation_id: uuid.UUID | None = None
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
    async with get_db_session() as session:
        row = AnswerFeedback(
            user_id=user.id,
            conversation_id=body.conversation_id,
            rating=body.rating,
            question=body.question[:8000],
            answer=(body.answer or "")[:20000],
            comment=body.comment,
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

    logger.info(
        "Feedback recorded: user=%s rating=%s conv=%s",
        user.username, body.rating, body.conversation_id,
    )
    return {"recorded": True, "rating": body.rating}


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
    admin: Annotated[User, Depends(require_platform_admin)],
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
