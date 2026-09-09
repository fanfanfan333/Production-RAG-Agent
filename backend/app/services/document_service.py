"""
Document upload pipeline orchestrator.

This service wires together every step of the ingestion pipeline:

    ┌──────────┐   bytes   ┌─────────────┐   pages   ┌──────────┐
    │  Raw file│ ────────▶ │  Parser     │ ─────────▶ │  Chunker │
    └──────────┘           └─────────────┘            └──────────┘
                                                            │ chunks
                                                            ▼
    ┌──────────┐  vectors  ┌──────────────────┐  batch  ┌───────────────┐
    │  Qdrant  │ ◀──────── │ embedding_service │ ◀────── │ this service  │
    └──────────┘   upsert  └──────────────────┘         └───────────────┘
         ▲                                                      │
         │ checkpoint                                           │
    ┌──────────┐                                                │
    │PostgreSQL│ ◀──────────────────────────────────────────────┘
    └──────────┘

Features: Chunk-level checkpointing, idempotent resume, dynamic batch fallback.
"""

import asyncio
import hashlib
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update

from app.config import get_settings
from app.db.models import Document, DocumentStatus
from app.db.postgres import get_db_session
from app.schemas.document import DocumentResult, UploadResponse
from app.services.chunker import build_chunks
from app.services.embedding_service import embed_batch_with_retry
from app.services.parsers import get_parser_for_file
from app.services.vector_service import VectorPoint, upsert_vectors, generate_point_id, get_existing_point_ids
from app.utils.logging import get_logger

logger = get_logger(__name__)


# Global concurrency limiter for embeddings
_embedding_semaphore = None

def get_embedding_semaphore():
    global _embedding_semaphore
    if _embedding_semaphore is None:
        _embedding_semaphore = asyncio.Semaphore(get_settings().MAX_CONCURRENT_EMBEDDINGS)
    return _embedding_semaphore


async def _create_document_record(
    filename: str,
    file_size: int,
    file_hash: str,
    owner_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
) -> Document:
    from sqlalchemy.exc import IntegrityError

    try:
        async with get_db_session() as session:
            doc = Document(
                id=uuid.uuid4(),
                filename=filename,
                file_size=file_size,
                file_hash=file_hash,
                status=DocumentStatus.PENDING,
                current_stage="pending",
                owner_id=owner_id,
                collection_id=collection_id,
            )
            session.add(doc)
            await session.flush()
            await session.refresh(doc)
            return doc
    except IntegrityError:
        # 复合唯一约束 (owner_id, file_hash) 冲突：同一用户并发上传同一文件。
        # 回退到按"该用户 + 哈希"查询（不同用户的同内容文件互不影响）。
        async with get_db_session() as session:
            from sqlalchemy import select
            query = select(Document).where(Document.file_hash == file_hash)
            if owner_id is not None:
                query = query.where(Document.owner_id == owner_id)
            doc = await session.scalar(query.limit(1))
            if not doc:
                raise RuntimeError("IntegrityError caught but document not found on fallback.")
            return doc


async def _update_document(
    doc_id: uuid.UUID,
    **kwargs,
) -> None:
    """Patch a document row with arbitrary fields + bump updated_at."""
    async with get_db_session() as session:
        kwargs["updated_at"] = datetime.now(tz=timezone.utc)
        await session.execute(
            update(Document).where(Document.id == doc_id).values(**kwargs)
        )


async def _process_single_file(
    filename: str,
    content: bytes,
    settings,
    owner_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
) -> DocumentResult:
    """
    Run the full ingestion pipeline for one file with state checkpoints and resume.
    """
    file_size = len(content)
    file_hash = hashlib.sha256(content).hexdigest()
    doc: Document | None = None
    is_resume = False

    try:
        # ── 0. 判重（按用户范围）────────────────────────────────────────────
        # 只与"当前用户自己"的文档判重：不同用户上传同一文件应各自独立索引，
        # 旧的无主数据（owner 为 NULL）不应阻塞任何用户的首次上传。
        async with get_db_session() as session:
            dedup_query = select(Document).where(Document.file_hash == file_hash)
            if owner_id is not None:
                dedup_query = dedup_query.where(Document.owner_id == owner_id)
            existing_doc = await session.scalar(dedup_query.limit(1))

            if existing_doc is not None:
                if existing_doc.status == DocumentStatus.COMPLETED:
                    logger.info("Duplicate document detected: filename='%s' owner=%s", filename, owner_id)
                    return DocumentResult(
                        document_id=existing_doc.id,
                        filename=filename,
                        status=DocumentStatus.ALREADY_EXISTS,
                        message="该文档你已上传并索引过，无需重复上传。",
                        existing_document_id=existing_doc.id,
                        uploaded_at=existing_doc.created_at,
                        page_count=existing_doc.page_count,
                        chunk_count=existing_doc.chunk_count,
                        file_size_bytes=existing_doc.file_size,
                        created_at=existing_doc.created_at,
                    )
                else:
                    logger.info("Resuming partial document id=%s from state %s", existing_doc.id, existing_doc.status.value)
                    doc = existing_doc
                    is_resume = True

        # ── 1. Persist metadata record (if not resuming) ──────────────────────
        if not is_resume:
            doc = await _create_document_record(filename, file_size, file_hash, owner_id, collection_id)
        
        doc_id_str = str(doc.id)

        # ── 2. Mark PARSING ───────────────────────────────────────────────────
        await _update_document(doc.id, status=DocumentStatus.PARSING, current_stage="parsing")
        parser = get_parser_for_file(filename)
        extraction = parser.parse(content, filename=filename)

        # ── 3. Mark CHUNKING ──────────────────────────────────────────────────
        await _update_document(
            doc.id, 
            status=DocumentStatus.CHUNKING, 
            current_stage="chunking",
            file_type=extraction.file_type,
            parser_used=extraction.parser_used,
            ocr_used=extraction.ocr_used,
            ocr_engine=extraction.ocr_engine,
            extraction_method=extraction.extraction_method,
            page_count=extraction.page_count,
        )
        
        chunks = build_chunks(
            text=extraction.full_text,
            min_chunk_size=settings.MIN_CHUNK_SIZE,
            max_chunk_size=settings.MAX_CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            page_resolver=extraction.page_for_offset,
            # small-to-big / Hierarchical RAG：为每个子块生成 parent 元数据，
            # 命中子块后可回填父块完整上下文（见 retrieval_service）。
            document_id=doc_id_str,
            enable_small_to_big=settings.HIERARCHICAL_RAG_ENABLED,
        )

        if not chunks:
            raise ValueError("Text extraction produced zero chunks — document may be empty.")

        await _update_document(doc.id, total_chunks=len(chunks), chunk_count=len(chunks))

        # ── 4. Deterministic Idempotency Check ────────────────────────────────
        await _update_document(doc.id, status=DocumentStatus.EMBEDDING, current_stage="checking_existing_vectors")
        
        chunk_map = {}
        for c in chunks:
            pid = generate_point_id(
                doc_id_str, filename, c.page_number, c.section, c.heading, c.chunk_index, c.text
            )
            chunk_map[pid] = c
            
        existing_ids = await get_existing_point_ids(list(chunk_map.keys()))
        missing_ids = set(chunk_map.keys()) - existing_ids
        missing_chunks = [chunk_map[pid] for pid in missing_ids]
        
        embedded_count = len(existing_ids)
        await _update_document(doc.id, embedded_chunks=embedded_count)
        
        logger.info(
            "id=%s → %d total chunks. %d already exist in Qdrant, %d missing.",
            doc_id_str, len(chunks), embedded_count, len(missing_chunks)
        )

        # ── 5. Embed and Index Missing Chunks (Batch Streaming) ────────────────
        await _update_document(doc.id, current_stage="embedding_and_indexing")
        
        # Dynamic batch sizing loop
        current_batch_size = settings.EMBEDDING_BATCH_SIZE
        chunk_idx = 0
        failed_chunks_count = 0
        
        semaphore = get_embedding_semaphore()
        
        while chunk_idx < len(missing_chunks):
            batch = missing_chunks[chunk_idx : chunk_idx + current_batch_size]
            batch_texts = [c.text for c in batch]
            
            try:
                # Concurrency limit applies specifically to the embedding API call
                async with semaphore:
                    vectors = await embed_batch_with_retry(batch_texts, task_type="RETRIEVAL_DOCUMENT")
                
                # Checkpointing Qdrant + Postgres atomically per batch
                points = [
                    VectorPoint(
                        vector=vectors[i],
                        document_id=doc_id_str,
                        filename=filename,
                        chunk_index=c.chunk_index,
                        page_number=c.page_number,
                        text=c.text,
                        heading=c.heading,
                        section=c.section,
                        collection_id=str(collection_id) if collection_id else None,
                        parent_id=c.parent_id,
                        parent_text=c.parent_text,
                        parent_char_start=c.parent_char_start,
                        parent_char_end=c.parent_char_end,
                    )
                    for i, c in enumerate(batch)
                ]
                await upsert_vectors(points)
                
                # Checkpoint progress
                embedded_count += len(batch)
                await _update_document(doc.id, embedded_chunks=embedded_count)
                
                chunk_idx += len(batch)
                
                # Slowly recover batch size if we had previously shrunk it
                if current_batch_size < settings.EMBEDDING_BATCH_SIZE:
                    current_batch_size = min(settings.EMBEDDING_BATCH_SIZE, current_batch_size + 4)
                    
            except Exception as batch_exc:
                # Dynamic fallback: if embedding failed (e.g. 429 too big), shrink batch and try again
                logger.warning("Batch of %d failed: %s", len(batch), batch_exc)
                if current_batch_size > settings.EMBEDDING_BATCH_SIZE_FLOOR:
                    current_batch_size = max(settings.EMBEDDING_BATCH_SIZE_FLOOR, current_batch_size // 2)
                    logger.info("Falling back to smaller batch size: %d", current_batch_size)
                    # Do not increment chunk_idx, loop will retry with smaller batch
                else:
                    logger.error("Batch failed at minimum size of 1. Chunk is unrecoverable.")
                    failed_chunks_count += 1
                    chunk_idx += 1
                    await _update_document(doc.id, failed_chunks=failed_chunks_count)
                    # We continue to the next chunk so one bad chunk doesn't poison the whole doc
        
        if failed_chunks_count > 0:
            raise RuntimeError(f"{failed_chunks_count} chunks failed to embed permanently.")

        # ── 6. Mark COMPLETED ─────────────────────────────────────────────────
        await _update_document(
            doc.id,
            status=DocumentStatus.COMPLETED,
            current_stage="completed"
        )

        return DocumentResult(
            document_id=doc.id,
            filename=filename,
            status=DocumentStatus.COMPLETED,
            page_count=extraction.page_count,
            chunk_count=len(chunks),
            file_size_bytes=file_size,
            created_at=doc.created_at,
        )

    except Exception as exc:
        logger.exception("Failed to process '%s': %s", filename, exc)

        if doc is not None:
            await _update_document(
                doc.id,
                status=DocumentStatus.FAILED,
                current_stage="failed",
                error_message=str(exc)[:2000],
            )
            return DocumentResult(
                document_id=doc.id,
                filename=filename,
                status=DocumentStatus.FAILED,
                file_size_bytes=file_size,
                error=str(exc),
                created_at=doc.created_at,
            )

        return DocumentResult(
            document_id=uuid.uuid4(),
            filename=filename,
            status=DocumentStatus.FAILED,
            file_size_bytes=file_size,
            error=f"Database error during record creation: {exc}",
        )


async def process_uploads(
    files: list[tuple[str, bytes]],
    owner_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
) -> UploadResponse:
    """
    Process multiple uploads sequentially, attributing them to *owner_id*
    and grouping them into *collection_id* when provided.
    """
    settings = get_settings()
    results: list[DocumentResult] = []

    for filename, content in files:
        result = await _process_single_file(
            filename, content, settings, owner_id=owner_id, collection_id=collection_id
        )
        results.append(result)

    succeeded = sum(1 for r in results if r.status == DocumentStatus.COMPLETED)
    failed = sum(1 for r in results if r.status == DocumentStatus.FAILED)

    return UploadResponse(
        total=len(results),
        succeeded=succeeded,
        failed=failed,
        documents=results,
    )


async def recover_stuck_documents() -> None:
    """
    On startup, find any documents that were left in a processing state due
    to a server crash. Mark them as FAILED so they can be resumed cleanly
    by the idempotent resume logic on next upload.
    """
    async with get_db_session() as session:
        result = await session.execute(
            update(Document)
            .where(Document.status.in_([
                DocumentStatus.PENDING,
                DocumentStatus.PARSING,
                DocumentStatus.CHUNKING,
                DocumentStatus.EMBEDDING,
                DocumentStatus.INDEXING,
            ]))
            .values(
                status=DocumentStatus.FAILED,
                current_stage="stuck_recovered",
                error_message="Server restarted during processing. Re-upload the identical file to resume."
            )
        )
        if result.rowcount > 0:
            logger.info("Recovered %d stuck documents from previous crash. Marked as FAILED to allow resume.", result.rowcount)

