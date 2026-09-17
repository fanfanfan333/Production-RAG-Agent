"""
Bad Case 回流 API（持续监控闭环的"人"这一端）.

    GET   /badcases            — 待审阅队列（可按 reason / status / severity 过滤）
    PATCH /badcases/{id}       — 审阅：改状态、写结论、打标签
    GET   /badcases/stats      — 队列分布（按 reason / status 计数）+ 运行期指标

为什么单独成 API 而不是塞进 /feedback：
- /feedback 是**用户**的动作（点👍👎），本模块是**运营**的动作（审阅、定级、结案）；
- 两者的权限、过滤维度、状态机都不同，混在一个端点会让两边都变扭。

队列里的每一条都带 question + answer + sources 快照，审阅者无需回翻对话即可
复现问题现场 —— 这正是"回流"能沉淀成回归测试集的前提。
"""

import json
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.db.badcase_models import BadCase
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.services.monitoring_service import metrics_snapshot
from app.services.permissions import require_permission
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Bad Cases"])

_REASONS = ("citation_unsupported", "evidence_refused", "output_guard", "feedback_down")
_STATUSES = ("open", "triaged", "resolved", "wontfix")


# ── Schemas ──────────────────────────────────────────────────────────────────

class BadCaseItem(BaseModel):
    id: str
    reason: str
    severity: str
    intent: str | None
    question: str
    answer: str
    status: str
    resolution: str | None
    tags: str | None
    username: str | None
    conversation_id: str | None
    created_at: datetime
    detail: dict | None = None
    sources: list[dict] | None = None


class BadCaseListResponse(BaseModel):
    bad_cases: list[BadCaseItem]
    total: int


class BadCaseUpdateRequest(BaseModel):
    status: str | None = Field(default=None, description="open | triaged | resolved | wontfix")
    resolution: str | None = Field(default=None, max_length=4000)
    tags: str | None = Field(default=None, max_length=256)


class BadCaseStatsResponse(BaseModel):
    total: int
    by_reason: dict[str, int]
    by_status: dict[str, int]
    by_severity: dict[str, int]
    metrics: dict


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_json(raw: str | None, fallback):
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return fallback


def _to_item(row: BadCase) -> BadCaseItem:
    return BadCaseItem(
        id=str(row.id),
        reason=row.reason,
        severity=row.severity,
        intent=row.intent,
        question=row.question,
        answer=row.answer,
        status=row.status,
        resolution=row.resolution,
        tags=row.tags,
        username=row.username,
        conversation_id=str(row.conversation_id) if row.conversation_id else None,
        created_at=row.created_at,
        detail=_safe_json(row.detail, None),
        sources=_safe_json(row.sources_snapshot, None),
    )


# ── GET /badcases ────────────────────────────────────────────────────────────

@router.get(
    "/badcases",
    response_model=BadCaseListResponse,
    summary="List bad cases (审阅队列)",
    description=(
        "Returns the bad-case review queue, newest first. Each entry carries a "
        "self-contained snapshot (question / answer / sources) so reviewers can "
        "reproduce the failure without opening the conversation."
    ),
)
async def list_bad_cases(
    user: Annotated[User, Depends(require_permission("audit.read"))],
    reason: str | None = Query(default=None, description="过滤回流原因"),
    status_filter: str | None = Query(default=None, alias="status", description="过滤状态"),
    severity: str | None = Query(default=None, description="过滤严重级别"),
    limit: int = Query(default=50, ge=1, le=500),
) -> BadCaseListResponse:
    from sqlalchemy import select

    async with get_db_session() as session:
        stmt = select(BadCase).order_by(BadCase.created_at.desc())
        if reason:
            stmt = stmt.where(BadCase.reason == reason)
        if status_filter:
            stmt = stmt.where(BadCase.status == status_filter)
        if severity:
            stmt = stmt.where(BadCase.severity == severity)
        rows = (await session.execute(stmt.limit(limit))).scalars().all()

    items = [_to_item(r) for r in rows]
    return BadCaseListResponse(bad_cases=items, total=len(items))


# ── PATCH /badcases/{id} ─────────────────────────────────────────────────────

@router.patch(
    "/badcases/{badcase_id}",
    response_model=BadCaseItem,
    summary="Triage a bad case (状态 / 结论 / 标签)",
    description=(
        "Update the review state of one bad case. `status` moves it along the "
        "open → triaged → resolved (or wontfix) state machine; `resolution` "
        "records the root cause and fix; `tags` is a free-form label "
        "(e.g. `召回缺失,阈值偏高`) used to cluster failures into themes."
    ),
)
async def update_bad_case(
    badcase_id: uuid.UUID,
    body: BadCaseUpdateRequest,
    user: Annotated[User, Depends(require_permission("audit.read"))],
) -> BadCaseItem:
    if body.status is not None and body.status not in _STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"status 必须是 {_STATUSES} 之一",
        )

    from sqlalchemy import select

    async with get_db_session() as session:
        row = (
            await session.execute(select(BadCase).where(BadCase.id == badcase_id))
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="Bad case 不存在")

        if body.status is not None:
            row.status = body.status
        if body.resolution is not None:
            row.resolution = body.resolution
        if body.tags is not None:
            row.tags = body.tags[:256]

    logger.info(
        "badcase triaged: id=%s status=%s by=%s",
        badcase_id, body.status, user.username,
    )
    async with get_db_session() as session:
        row = (
            await session.execute(select(BadCase).where(BadCase.id == badcase_id))
        ).scalar_one()
    return _to_item(row)


# ── GET /badcases/stats ──────────────────────────────────────────────────────

@router.get(
    "/badcases/stats",
    response_model=BadCaseStatsResponse,
    summary="Bad Case 分布 + 运行期 RAG 指标",
    description=(
        "Aggregated bad-case distribution (by reason / status / severity) plus "
        "the in-process RAG quality metrics (evidence-gate refusal rate, "
        "citation pass rate, output-guard hit rate, stage latencies)."
    ),
)
async def bad_case_stats(
    user: Annotated[User, Depends(require_permission("audit.read"))],
) -> BadCaseStatsResponse:
    from sqlalchemy import func, select

    async with get_db_session() as session:
        total = (
            await session.execute(select(func.count()).select_from(BadCase))
        ).scalar_one()

        async def _group(field):
            stmt = select(field, func.count()).select_from(BadCase).group_by(field)
            return {
                str(k): int(v) for k, v in (await session.execute(stmt)).all()
            }

        by_reason = await _group(BadCase.reason)
        by_status = await _group(BadCase.status)
        by_severity = await _group(BadCase.severity)

    return BadCaseStatsResponse(
        total=int(total),
        by_reason=by_reason,
        by_status=by_status,
        by_severity=by_severity,
        metrics=metrics_snapshot(),
    )
