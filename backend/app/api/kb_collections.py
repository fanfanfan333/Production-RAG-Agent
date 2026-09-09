"""
Knowledge-base collection API router (企业落地第一阶段).

GET    /kb/collections              — list the user's collections + doc counts
POST   /kb/collections              — create a collection
DELETE /kb/collections/{id}         — delete a collection (documents survive,
                                      they become unassigned)
POST   /documents/{id}/collection   — assign/unassign a document
                                      (kept next to kb collections in spirit;
                                      actually lives in document_management)
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, Field

from app.services.permissions import require_permission
from app.db.user_models import User
from app.services import kb_collection_service
from app.services.audit_service import record_audit
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/kb", tags=["Knowledge-Base Collections"])


class CollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str | None = Field(None, max_length=512)


class CollectionOut(BaseModel):
    id: str
    name: str
    description: str | None = None
    document_count: int
    created_at: str | None = None


class CollectionListOut(BaseModel):
    collections: list[CollectionOut]
    total: int


@router.get(
    "/collections",
    response_model=CollectionListOut,
    summary="List the current user's knowledge-base collections",
)
async def list_collections_endpoint(
    user: User = Depends(require_permission("knowledge.read")),
) -> CollectionListOut:
    items = await kb_collection_service.list_collections(user.id)
    out = [CollectionOut(**item) for item in items]
    return CollectionListOut(collections=out, total=len(out))


@router.post(
    "/collections",
    response_model=CollectionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a knowledge-base collection",
)
async def create_collection_endpoint(
    body: CollectionCreateRequest,
    user: User = Depends(require_permission("knowledge.write")),
) -> CollectionOut:
    try:
        created = await kb_collection_service.create_collection(
            user.id, body.name, body.description
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    return CollectionOut(**created)


@router.delete(
    "/collections/{collection_id}",
    summary="Delete a knowledge-base collection",
    description=(
        "Documents in the collection are NOT deleted — they become unassigned "
        "and remain searchable outside any collection filter."
    ),
)
async def delete_collection_endpoint(
    collection_id: uuid.UUID = Path(...),
    user: User = Depends(require_permission("knowledge.delete")),
) -> dict:
    deleted = await kb_collection_service.delete_collection(user.id, collection_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="集合不存在"
        )
    await record_audit(
        "kb_collection.delete",
        user_id=user.id,
        username=user.username,
        resource_type="collection",
        resource_id=str(collection_id),
    )
    return {"deleted": True, "id": str(collection_id)}
