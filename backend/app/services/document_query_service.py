"""
Document query & management service (Phase 3).

Provides:
  - Paginated listing of Document rows from PostgreSQL
  - Hard-delete: removes the PostgreSQL row AND all associated Qdrant vectors

This module is intentionally separate from document_service.py (Phase 2 upload
pipeline) so that Phase 2 code is never touched.
"""

import math
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, func, select

from app.db.models import Document, DocumentStatus
from app.db.postgres import get_db_session
from app.schemas.document_management import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentSummary,
)
from app.services.vector_service import delete_by_document_id
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def list_documents(
    page: int = 1,
    limit: int = 20,
    status: DocumentStatus | None = None,
    owner_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
) -> DocumentListResponse:
    """
    Return a paginated list of Document rows, optionally filtered by status,
    owner (multi-user isolation) and knowledge-base collection.

    Args:
        page:   1-indexed page number.
        limit:  Rows per page (1–100).
        status: Optional filter on DocumentStatus.
        owner_id:   Restrict to documents owned by this user (None = admin/all).
        collection_id: Restrict to one KB collection (None = all documents).
    """
    offset = (page - 1) * limit

    async with get_db_session() as session:
        # ── Base query ────────────────────────────────────────────────────────
        base_q = select(Document)
        count_q = select(func.count()).select_from(Document)

        if status is not None:
            base_q = base_q.where(Document.status == status)
            count_q = count_q.where(Document.status == status)
        if owner_id is not None:
            base_q = base_q.where(Document.owner_id == owner_id)
            count_q = count_q.where(Document.owner_id == owner_id)
        if collection_id is not None:
            base_q = base_q.where(Document.collection_id == collection_id)
            count_q = count_q.where(Document.collection_id == collection_id)

        # ── Total count ───────────────────────────────────────────────────────
        total: int = (await session.execute(count_q)).scalar_one()

        # ── Paginated fetch ───────────────────────────────────────────────────
        rows_result = await session.execute(
            base_q
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        rows: list[Document] = list(rows_result.scalars().all())

    pages = max(1, math.ceil(total / limit))

    summaries = [
        DocumentSummary(
            document_id=row.id,
            filename=row.filename,
            status=row.status,
            page_count=row.page_count,
            chunk_count=row.chunk_count,
            file_size_bytes=row.file_size,
            error=row.error_message,
            created_at=row.created_at,
            updated_at=row.updated_at,
            collection_id=row.collection_id,
        )
        for row in rows
    ]

    logger.info(
        "list_documents → total=%d page=%d/%d limit=%d status=%s",
        total, page, pages, limit, status,
    )

    return DocumentListResponse(
        total=total,
        page=page,
        limit=limit,
        pages=pages,
        documents=summaries,
    )


async def delete_document(
    document_id: uuid.UUID,
    owner_id: uuid.UUID | None = None,
) -> DocumentDeleteResponse:
    """
    Hard-delete a document: remove its Qdrant vectors first, then the PG row.

    Qdrant vectors are deleted before the PG row so that a partial failure
    (Qdrant down) leaves the PG record intact and the operation can be retried.

    Args:
        document_id: UUID of the document to delete.
        owner_id:    Restrict deletion to documents owned by this user
                     (None = admin). Raises KeyError on foreign documents
                     so callers can map it to 404 without an existence leak.

    Returns:
        DocumentDeleteResponse on success.

    Raises:
        KeyError: If no accessible document with the given ID exists.
    """
    # ── 1. Fetch document metadata (owner-scoped) ─────────────────────────────
    async with get_db_session() as session:
        query = select(Document).where(Document.id == document_id)
        if owner_id is not None:
            query = query.where(Document.owner_id == owner_id)
        doc: Document | None = (await session.execute(query)).scalar_one_or_none()

    if doc is None:
        raise KeyError(f"Document '{document_id}' not found.")

    filename = doc.filename
    doc_id_str = str(document_id)

    logger.info("Deleting document id=%s filename='%s'", doc_id_str, filename)

    # ── 2. Delete vectors from Qdrant (fail-safe first) ───────────────────────
    try:
        await delete_by_document_id(doc_id_str)
        logger.info("Qdrant vectors deleted for document_id=%s", doc_id_str)
    except Exception as exc:
        # Log and re-raise; PG row intentionally NOT deleted on Qdrant failure
        logger.error(
            "Failed to delete Qdrant vectors for document_id=%s: %s — aborting delete",
            doc_id_str,
            exc,
        )
        raise RuntimeError(
            f"Vector deletion failed ({exc}). PostgreSQL record preserved for retry."
        ) from exc

    # ── 3. Delete PostgreSQL row ──────────────────────────────────────────────
    async with get_db_session() as session:
        await session.execute(
            delete(Document).where(Document.id == document_id)
        )

    logger.info("Document id=%s deleted successfully", doc_id_str)

    return DocumentDeleteResponse(
        document_id=document_id,
        filename=filename,
    )


async def get_document_chunks(
    document_id: uuid.UUID,
    owner_id: uuid.UUID | None = None,
) -> dict:
    """
    Fetch the full ordered chunk list of one document for原文预览.

    Metadata (filename, page_count) comes from PostgreSQL; chunk text is
    scrolled from the Qdrant payload and sorted by chunk_index.

    Args:
        document_id: UUID of the document.
        owner_id:    Restrict access to documents owned by this user
                     (None = admin). Raises KeyError on foreign documents.

    Returns:
        {"document_id", "filename", "page_count", "total", "chunks": [...]}
        where each chunk is {"chunk_index", "page_number", "text"}.
    """
    # ── 1. Metadata (owner-scoped, 404 without existence leak) ────────────────
    async with get_db_session() as session:
        query = select(Document).where(Document.id == document_id)
        if owner_id is not None:
            query = query.where(Document.owner_id == owner_id)
        doc: Document | None = (await session.execute(query)).scalar_one_or_none()

    if doc is None:
        raise KeyError(f"Document '{document_id}' not found.")

    # ── 2. Scroll all chunks from Qdrant ──────────────────────────────────────
    from qdrant_client.http import models as qmodels

    from app.config import get_settings
    from app.db.qdrant import get_qdrant_client

    settings = get_settings()
    client = get_qdrant_client()
    scroll_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="document_id",
                match=qmodels.MatchValue(value=str(document_id)),
            )
        ]
    )

    raw: list[tuple[int, int, str]] = []  # (chunk_index, page_number, text)
    offset = None
    while True:
        points, next_offset = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            try:
                chunk_index = int(payload.get("chunk_index") or 0)
            except (TypeError, ValueError):
                chunk_index = 0
            try:
                page_number = int(payload.get("page_number") or 1)
            except (TypeError, ValueError):
                page_number = 1
            raw.append((chunk_index, page_number, str(payload.get("text", ""))))
        if next_offset is None:
            break
        offset = next_offset

    raw.sort(key=lambda t: t[0])
    chunks = [
        {"chunk_index": ci, "page_number": pn, "text": text}
        for ci, pn, text in raw
    ]

    logger.info(
        "get_document_chunks id=%s filename='%s' → %d chunks",
        document_id, doc.filename, len(chunks),
    )

    return {
        "document_id": str(document_id),
        "filename": doc.filename,
        "page_count": doc.page_count,
        "total": len(chunks),
        "chunks": chunks,
    }
