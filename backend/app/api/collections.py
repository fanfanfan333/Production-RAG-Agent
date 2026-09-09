"""
Collections CRUD API router (Phase 3 / 企业落地第一阶段).

Low-level Qdrant vector-collection administration — INFRASTRUCTURE, not the
business "知识库" grouping (that lives under /kb/collections). These
endpoints are restricted to admins.

Endpoints:
  GET    /collections           — list all collections with stats   (admin)
  POST   /collections           — create a new collection           (admin)
  GET    /collections/{name}    — get single collection details     (admin)
  DELETE /collections/{name}    — delete a collection               (admin)
"""

from fastapi import APIRouter, Depends, HTTPException, Path, status
from typing import Annotated

from app.services.permissions import require_platform_admin
from app.db.user_models import User

from app.schemas.collection import (
    CollectionCreate,
    CollectionDeleteResponse,
    CollectionInfo,
    CollectionListResponse,
)
from app.services.collection_service import (
    create_collection,
    delete_collection,
    get_collection,
    list_collections,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/collections", tags=["Collections"])


# ── GET /collections ──────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=CollectionListResponse,
    summary="List all Qdrant collections",
    description=(
        "Returns all collections present in the Qdrant instance, each with "
        "a lightweight stats snapshot (vector count, points count, status)."
    ),
)
async def list_collections_endpoint(
    user: Annotated[User, Depends(require_platform_admin)],
) -> CollectionListResponse:
    return await list_collections()


# ── POST /collections ─────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=CollectionInfo,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new Qdrant collection",
    description=(
        "Creates a new vector collection with the specified configuration. "
        "A `document_id` keyword payload index is created automatically. "
        "Returns the full collection details as stored in Qdrant."
    ),
)
async def create_collection_endpoint(
    payload: CollectionCreate,
    user: Annotated[User, Depends(require_platform_admin)],
) -> CollectionInfo:
    try:
        return await create_collection(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


# ── GET /collections/{name} ───────────────────────────────────────────────────

@router.get(
    "/{name}",
    response_model=CollectionInfo,
    summary="Get collection details",
    description=(
        "Returns full configuration and runtime statistics for the named "
        "collection (vector count, index status, segment count, vector config)."
    ),
)
async def get_collection_endpoint(
    name: Annotated[str, Path(description="Collection name.")],
    user: Annotated[User, Depends(require_platform_admin)],
) -> CollectionInfo:
    try:
        return await get_collection(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


# ── DELETE /collections/{name} ────────────────────────────────────────────────

@router.delete(
    "/{name}",
    response_model=CollectionDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a Qdrant collection",
    description=(
        "Permanently deletes the named collection and **all** its vectors. "
        "This operation is **irreversible**.\n\n"
        "> **Note**: The primary RAG collection (configured via "
        "`QDRANT_COLLECTION`) is protected and cannot be deleted through this "
        "endpoint. Use `DELETE /documents/{id}` to remove individual documents."
    ),
)
async def delete_collection_endpoint(
    name: Annotated[str, Path(description="Collection name to delete.")],
    user: Annotated[User, Depends(require_platform_admin)],
) -> CollectionDeleteResponse:
    try:
        return await delete_collection(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        # Primary collection protection or other business-rule violations
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
