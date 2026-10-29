"""Authenticated coordination task API.

Endpoints never accept a tenant/user id from the caller: ownership comes from
the verified ``User``. A task id belonging to someone else (even in the same
tenant) is reported as 404, exactly like a missing id.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.coordination.forgeflow_client import ForgeFlowRejected, ForgeFlowUnavailable
from app.coordination.service import CoordinationConflict, CoordinationService
from app.db.user_models import User
from app.services.permissions import require_permission
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/agent/tasks", tags=["Coordination"])


class TaskSubmitRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20000)
    idempotency_key: str = Field(min_length=8, max_length=128)
    kind: str = Field(default="auto", pattern="^(auto|document|rag|doc|knowledge|code|research|web|search)$")
    collection_id: uuid.UUID | None = None


class TaskApprovalRequest(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
    note: str = Field(default="", max_length=2000)


@lru_cache(maxsize=1)
def get_coordination_service() -> CoordinationService:
    from app.coordination.repository import SqlTaskRepository

    return CoordinationService(SqlTaskRepository())


def _token(
    x_forgeflow_token: Annotated[str | None, Header(alias="X-ForgeFlow-Token")] = None,
) -> str:
    # The credential is per request and is never persisted or logged.
    return (x_forgeflow_token or "").strip()


def _public(task: dict) -> dict:
    """Strip storage-only fields and never echo a credential."""
    return {key: value for key, value in task.items() if key not in {"idempotency_key"}}


async def _run_in_background(
    service: CoordinationService, task_id: uuid.UUID, user: User, token: str, user_scope
) -> None:
    try:
        await service.run(task_id, user=user, forgeflow_token=token, user_scope=user_scope)
    except Exception:  # noqa: BLE001 - background failure is persisted by service
        logger.exception("coordination worker crashed for task %s", task_id)


@router.post("", status_code=202)
async def submit_task(
    body: TaskSubmitRequest,
    background: BackgroundTasks,
    user: Annotated[User, Depends(require_permission("conversation.write"))],
    service: CoordinationService = Depends(get_coordination_service),
    token: str = Depends(_token),
):
    from app.services.security_scope import request_security_scope

    user_scope = await request_security_scope(user)
    try:
        task, created = await service.create(
            user=user,
            query=body.query,
            idempotency_key=body.idempotency_key,
            kind=body.kind,
            collection_id=body.collection_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CoordinationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if created:
        background.add_task(_run_in_background, service, uuid.UUID(task["id"]), user, token, user_scope)
    return {
        "task": _public(task),
        "accepted": created,
        "idempotent_replay": not created,
        "note": "任务已受理，尚未完成；请以状态查询结果为准",
    }


@router.get("")
async def list_tasks(
    user: Annotated[User, Depends(require_permission("conversation.read"))],
    service: CoordinationService = Depends(get_coordination_service),
    limit: int = Query(30, ge=1, le=100),
):
    rows = await service.list_owned(user, limit=limit)
    return {"total": len(rows), "items": [_public(row) for row in rows]}


@router.get("/identity")
async def get_identity_link(
    user: Annotated[User, Depends(require_permission("conversation.read"))],
    service: CoordinationService = Depends(get_coordination_service),
):
    return {"identity": await service.repository.get_link(str(user.tenant_id), user.id)}


@router.put("/identity")
async def link_identity(
    user: Annotated[User, Depends(require_permission("conversation.write"))],
    service: CoordinationService = Depends(get_coordination_service),
    token: str = Depends(_token),
):
    try:
        return {"identity": await service.link_forgeflow(user, token=token)}
    except CoordinationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ForgeFlowUnavailable, ForgeFlowRejected) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/identity", status_code=204)
async def unlink_identity(
    user: Annotated[User, Depends(require_permission("conversation.write"))],
    service: CoordinationService = Depends(get_coordination_service),
):
    await service.unlink_forgeflow(user)


@router.get("/{task_id}")
async def get_task(
    task_id: uuid.UUID,
    user: Annotated[User, Depends(require_permission("conversation.read"))],
    service: CoordinationService = Depends(get_coordination_service),
    token: str = Depends(_token),
):
    try:
        task = await service.get_owned(task_id, user)
        if token and (task.get("execution_id") or any(item.get("kind") == "code" for item in (task.get("plan") or {}).get("subtasks", []))) and task["status"] in {
            "queued",
            "running",
            "awaiting_approval",
            "interrupted",
            "unavailable",
            "failed",
            "partial_failure",
        }:
            task = await service.refresh(task_id, user, token=token)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    return _public(task)


@router.get("/{task_id}/events")
async def get_task_events(
    task_id: uuid.UUID,
    user: Annotated[User, Depends(require_permission("conversation.read"))],
    service: CoordinationService = Depends(get_coordination_service),
    after_seq: int = Query(0, ge=0),
):
    try:
        events = await service.events(task_id, user, after_seq=after_seq)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    return {"task_id": str(task_id), "events": events}


@router.post("/{task_id}/stop", status_code=202)
async def stop_task(
    task_id: uuid.UUID,
    user: Annotated[User, Depends(require_permission("conversation.write"))],
    service: CoordinationService = Depends(get_coordination_service),
    token: str = Depends(_token),
):
    try:
        task = await service.stop(task_id, user, token=token)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    except CoordinationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "task": _public(task),
        "note": "停止请求已记录；执行状态仍以实际状态查询为准",
    }


@router.post("/{task_id}/approval")
async def decide_task(
    task_id: uuid.UUID,
    body: TaskApprovalRequest,
    user: Annotated[User, Depends(require_permission("conversation.write"))],
    service: CoordinationService = Depends(get_coordination_service),
    token: str = Depends(_token),
):
    try:
        task = await service.decide_code(
            task_id, user, decision=body.decision, note=body.note, token=token
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Task not found") from exc
    except CoordinationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ForgeFlowUnavailable, ForgeFlowRejected) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"task": _public(task), "note": "审批已转发到 ForgeFlow，结果以运行状态为准"}
