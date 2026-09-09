"""
Document upload API router (Phase 2 / 企业落地第一阶段).

POST /upload  — Accept 1–N files, run the ingestion pipeline,
                return per-file status.

Authentication required. Uploaded documents are attributed to the current
user and, when provided, grouped into one of their knowledge-base collections.
"""

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.services.permissions import require_permission
from app.config import get_settings
from app.db.user_models import User
from app.schemas.document import UploadResponse
from app.services.audit_service import record_audit
from app.services.document_service import process_uploads
from app.utils.file_utils import validate_document_upload
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and ingest documents",
    description=(
        "Accept one or more files. Each file is:\n"
        "1. Validated (size limit and supported format)\n"
        "2. Text-extracted with the corresponding parser (embedded images are "
        "OCR'd and merged into the index)\n"
        "3. Recursively chunked\n"
        "4. Embedded via the local BGE model\n"
        "5. Stored in Qdrant (vectors) and PostgreSQL (metadata)\n\n"
        "Optionally pass `collection_id` (form field) to group the upload into "
        "one of the caller's knowledge-base collections.\n\n"
        "A per-file status is returned regardless of individual failures."
    ),
)
async def upload_documents(
    files: list[UploadFile] = File(
        ...,
        description="One or more files (max 50 MB each).",
    ),
    collection_id: uuid.UUID | None = Form(
        None,
        description="Optional knowledge-base collection to group the upload into.",
    ),
    user: User = Depends(require_permission("document.write")),
) -> UploadResponse:
    settings = get_settings()

    # ── Guard: file count ─────────────────────────────────────────────────────
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file must be provided.",
        )

    if len(files) > settings.MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Too many files. Maximum {settings.MAX_FILES_PER_UPLOAD} "
                f"files per request, received {len(files)}."
            ),
        )

    # ── Guard: collection belongs to the caller ───────────────────────────────
    if collection_id is not None:
        from app.db.postgres import get_db_session
        from app.db.user_models import Collection

        async with get_db_session() as session:
            col = await session.get(Collection, collection_id)
        if col is None or col.owner_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="集合不存在或不属于当前用户",
            )

    # ── Validate and buffer all files before starting the pipeline ────────────
    # This ensures we reject invalid uploads immediately, before any DB writes.
    file_payloads: list[tuple[str, bytes]] = []

    for upload in files:
        raw_filename = upload.filename or "unknown.ext"
        # Sanitize against path traversal (handles both / and \ regardless of OS)
        filename = raw_filename.replace("\\", "/").split("/")[-1]
        logger.info("Received upload: '%s' (content-type=%s)", filename, upload.content_type)

        try:
            content = await validate_document_upload(upload, settings)
        except HTTPException:
            raise  # propagate validation errors as-is
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not read file '{filename}': {exc}",
            ) from exc

        file_payloads.append((filename, content))

    # ── Run the ingestion pipeline ─────────────────────────────────────────────
    logger.info(
        "Starting ingestion pipeline for %d file(s) user=%s collection=%s",
        len(file_payloads), user.username, collection_id,
    )
    response = await process_uploads(
        file_payloads,
        owner_id=user.id,
        collection_id=collection_id,
    )

    logger.info(
        "Upload complete — total=%d succeeded=%d failed=%d",
        response.total,
        response.succeeded,
        response.failed,
    )

    await record_audit(
        "document.upload",
        user_id=user.id,
        username=user.username,
        resource_type="document",
        resource_id=",".join(str(r.document_id) for r in response.documents)[:2000],
        detail=(
            f"files={[f for f, _ in file_payloads]}; "
            f"succeeded={response.succeeded}; failed={response.failed}"
        ),
    )

    return response
