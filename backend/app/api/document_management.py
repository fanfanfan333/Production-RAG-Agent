"""
Document management API router (Phase 3 / 企业落地第一阶段).

Endpoints:
  GET    /documents                    — paginated document list (owner-scoped)
  PATCH  /documents/{id}/collection    — assign/unassign a KB collection
  DELETE /documents/{id}               — delete document + vectors (owner-scoped)

All endpoints require authentication; non-admin users only ever see and
touch their own documents.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.services.permissions import require_permission
from app.db.models import DocumentStatus
from app.db.user_models import User
from app.schemas.document_management import (
    DocumentDeleteResponse,
    DocumentListResponse,
)
from app.services.audit_service import record_audit
from app.services.document_query_service import (
    delete_document,
    get_document_chunks,
    list_documents,
)
from app.services.kb_collection_service import assign_document
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Document Management"])


# ── GET /documents ─────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List uploaded documents (owner-scoped)",
    description=(
        "Returns a paginated list of documents, sorted by upload date "
        "descending. Optionally filter by processing status or knowledge-base "
        "collection. Non-admin users only see their own documents."
    ),
)
async def list_documents_endpoint(
    user: Annotated[User, Depends(require_permission("document.read"))],
    page: Annotated[
        int,
        Query(ge=1, description="Page number (1-indexed)."),
    ] = 1,
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="Items per page (max 100)."),
    ] = 20,
    status: Annotated[
        DocumentStatus | None,
        Query(description="Filter by document status."),
    ] = None,
    collection_id: Annotated[
        uuid.UUID | None,
        Query(description="Filter by knowledge-base collection."),
    ] = None,
) -> DocumentListResponse:
    return await list_documents(
        page=page,
        limit=limit,
        status=status,
        owner_id=None if user.is_admin else user.id,
        collection_id=collection_id,
    )


# ── PATCH /documents/{id}/collection ──────────────────────────────────────────

class CollectionAssignRequest(BaseModel):
    collection_id: uuid.UUID | None = None   # null = unassign


@router.patch(
    "/documents/{document_id}/collection",
    summary="Assign or unassign a document to a knowledge-base collection",
)
async def assign_collection_endpoint(
    document_id: Annotated[uuid.UUID, Path(description="UUID of the document.")],
    body: CollectionAssignRequest,
    user: Annotated[User, Depends(require_permission("document.write"))],
) -> dict:
    try:
        await assign_document(user.id, document_id, body.collection_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    await record_audit(
        "document.assign_collection",
        user_id=user.id,
        username=user.username,
        resource_type="document",
        resource_id=str(document_id),
        detail=f"collection_id={body.collection_id}",
    )
    return {
        "document_id": str(document_id),
        "collection_id": body.collection_id and str(body.collection_id),
    }


# ── DELETE /documents/{id} ────────────────────────────────────────────────────

@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a document and all its vectors",
    description=(
        "Permanently removes the document record from PostgreSQL and all "
        "associated vector embeddings from Qdrant. This operation is "
        "**irreversible**. Qdrant vectors are deleted first so the PostgreSQL "
        "record is preserved on vector-store failure, enabling safe retries. "
        "Non-admin users can only delete their own documents."
    ),
)
async def delete_document_endpoint(
    document_id: Annotated[
        uuid.UUID,
        Path(description="UUID of the document to delete."),
    ],
    user: Annotated[User, Depends(require_permission("document.delete"))],
) -> DocumentDeleteResponse:
    try:
        result = await delete_document(
            document_id,
            owner_id=None if user.is_admin else user.id,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    await record_audit(
        "document.delete",
        user_id=user.id,
        username=user.username,
        resource_type="document",
        resource_id=str(document_id),
        detail=f"filename={result.filename}",
    )
    return result


# ── GET /documents/{id}/chunks ────────────────────────────────────────────────

@router.get(
    "/documents/{document_id}/chunks",
    summary="Get all chunks of a document (原文预览)",
    description=(
        "Returns the document's chunks in reading order, each with its page "
        "number and full text.  Used by the citation '定位原文' feature to "
        "show the exact source passage behind an answer.  Non-admin users "
        "can only preview their own documents."
    ),
)
async def get_document_chunks_endpoint(
    document_id: uuid.UUID,
    user: Annotated[User, Depends(require_permission("document.read"))],
) -> dict:
    try:
        return await get_document_chunks(
            document_id,
            owner_id=None if user.is_admin else user.id,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
