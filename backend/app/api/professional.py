"""Opt-in graph indexing and explicitly reviewed, non-private training data."""
import uuid
from typing import Annotated, Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from app.db.postgres import get_db_session
from app.db.professional_models import GraphIndex, TrainingExample
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.services.graph_retrieval import GraphPermissionError, graph_candidates, index_step, visible_document
from app.services.permissions import require_permission, require_platform_admin
from app.services.security_scope import request_security_scope
from app.services.training_data import TrainingDraft, export_examples

router = APIRouter(tags=["Graph and training preparation"])


@router.post("/documents/{document_id}/graph-index")
async def graph_index_step(document_id: uuid.UUID, user: Annotated[User, Depends(require_permission("document.write"))], restart: bool = False):
    scope = await request_security_scope(user)
    try:
        result = await index_step(document_id, scope, restart=restart)
    except GraphPermissionError:
        raise HTTPException(404, "文档或证据不存在或不可访问") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    await record_audit("graph.index", user_id=user.id, username=user.username, resource_type="document",
                       resource_id=str(document_id), detail=f"status={result['status']}; cursor={result['cursor']}")
    return result


@router.get("/documents/{document_id}/graph-index")
async def graph_index_status(document_id: uuid.UUID, user: Annotated[User, Depends(require_permission("document.read"))]):
    scope = await request_security_scope(user)
    async with get_db_session() as session:
        if await visible_document(session, document_id, scope) is None:
            raise HTTPException(404, "文档不存在或不可访问")
        job = await session.get(GraphIndex, document_id)
        return {"status": job.status if job else "not_indexed", "cursor": job.cursor if job else 0,
                "total": job.total if job else 0, "error": job.error if job else None}


@router.post("/documents/{document_id}/cancel-ingestion")
async def cancel_document_ingestion(document_id: uuid.UUID, user: Annotated[User, Depends(require_permission("document.write"))]):
    scope = await request_security_scope(user)
    async with get_db_session() as session:
        document = await visible_document(session, document_id, scope)
        if document is None or document.owner_id != user.id:
            raise HTTPException(404, "文档不存在或不可访问")
    from app.services.document_service import cancel_ingestion
    if not cancel_ingestion(document_id):
        raise HTTPException(409, "文档当前没有正在运行的入库任务")
    await record_audit("document.cancel_ingestion", user_id=user.id, username=user.username,
                       resource_type="document", resource_id=str(document_id))
    return {"status": "cancelling", "message": "当前解析阶段完成后取消；不会停止其他文档或项目"}


@router.get("/documents/{document_id}/graph-evidence")
async def graph_evidence(document_id: uuid.UUID, user: Annotated[User, Depends(require_permission("document.read"))],
                         q: str = Query(min_length=1, max_length=1000)):
    scope = await request_security_scope(user)
    async with get_db_session() as session:
        if await visible_document(session, document_id, scope) is None:
            raise HTTPException(404, "文档不存在或不可访问")
    try:
        _, paths = await graph_candidates(q, scope, document_id=document_id)
    except GraphPermissionError:
        raise HTTPException(404, "文档或证据不存在或不可访问") from None
    return {"paths": paths}


@router.post("/training/examples", status_code=201)
async def create_training_example(body: TrainingDraft, admin: Annotated[User, Depends(require_platform_admin())]):
    async with get_db_session() as session:
        row = TrainingExample(**body.model_dump(), created_by=admin.id, status="pending")
        session.add(row)
        await session.flush()
        identifier = str(row.id)
    await record_audit("training.create", user_id=admin.id, username=admin.username, resource_type="training", resource_id=identifier)
    return {"id": identifier, "status": "pending"}


@router.get("/training/examples")
async def list_training_examples(admin: Annotated[User, Depends(require_platform_admin())], limit: int = Query(50, ge=1, le=100)):
    async with get_db_session() as session:
        rows = (await session.scalars(select(TrainingExample).order_by(TrainingExample.created_at.desc()).limit(limit))).all()
        return [{"id": str(row.id), "status": row.status, **{key: getattr(row, key) for key in TrainingDraft.model_fields}} for row in rows]


@router.post("/training/examples/{example_id}/review")
async def review_training_example(example_id: uuid.UUID, admin: Annotated[User, Depends(require_platform_admin())],
                                  decision: Literal["approved", "rejected"]):
    async with get_db_session() as session:
        row = await session.get(TrainingExample, example_id)
        if row is None:
            raise HTTPException(404, "样本不存在")
        if decision == "approved":
            TrainingDraft.model_validate({key: getattr(row, key) for key in TrainingDraft.model_fields})
        row.status, row.reviewed_by = decision, admin.id
    await record_audit("training.review", user_id=admin.id, username=admin.username, resource_type="training",
                       resource_id=str(example_id), detail=f"decision={decision}")
    return {"id": str(example_id), "status": decision}


@router.post("/training/export")
async def export_training_data(admin: Annotated[User, Depends(require_platform_admin())], kind: Literal["sft", "dpo"]):
    async with get_db_session() as session:
        rows = (await session.scalars(select(TrainingExample).where(TrainingExample.status == "approved").order_by(TrainingExample.id).limit(10000))).all()
        try:
            result = export_examples(rows, kind)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
    await record_audit("training.export", user_id=admin.id, username=admin.username, resource_type="training",
                       resource_id=result["dataset_version"], detail=f"format={kind}; count={len(result['manifest'])}")
    return result
