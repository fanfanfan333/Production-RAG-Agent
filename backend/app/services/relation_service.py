"""
Cross-document relation analysis service (问题1).

Powers the "这些文档有什么关联" feature.  Instead of a similarity search for
the user's question, this service builds a *per-document* digest:

    PostgreSQL                     Qdrant
    ──────────                     ──────
    completed documents  ───────▶  scroll sample chunks per document_id
            │                              │
            └──────────┬───────────────────┘
                       ▼
        list[DocumentDigest]  →  relation graph prompt

Each digest is a short, ordered sample of a document's indexed chunks —
enough for the LLM to summarise what the document is about and how it
relates to the others, without pulling entire documents into context.
"""

import uuid
from dataclasses import dataclass, field

from qdrant_client.http import models as qmodels
from sqlalchemy import select

from app.config import get_settings
from app.db.models import Document, DocumentStatus
from app.db.postgres import get_db_session
from app.db.qdrant import get_qdrant_client
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentDigest:
    """A per-document content sample used for relation analysis."""

    document_id: str
    filename: str
    digest: str
    page_count: int = 0
    chunk_count: int = 0
    sampled_chunks: int = 0
    index: int = 0                    # 1-based position in the analysis batch
    warnings: list[str] = field(default_factory=list)


async def collect_document_digests(
    max_documents: int | None = None,
    chunks_per_doc: int | None = None,
    digest_chars: int | None = None,
    owner_id: str | None = None,
) -> list[DocumentDigest]:
    """
    Build a content digest for every accessible completed document.

    Args:
        max_documents:   Cap on documents analysed (most recent N win).
        chunks_per_doc:  Chunks sampled per document for its digest.
        digest_chars:    Max characters of digest text per document.
        owner_id:        Restrict analysis to documents owned by this user
                         (None = unrestricted / admin).

    Returns:
        List of DocumentDigest sorted by upload time (oldest first).
        Documents with no vectors in Qdrant still appear, with a warning.
    """
    settings = get_settings()
    max_documents = max_documents or settings.RELATION_MAX_DOCUMENTS
    chunks_per_doc = chunks_per_doc or settings.RELATION_CHUNKS_PER_DOC
    digest_chars = digest_chars or settings.RELATION_DIGEST_CHARS

    # ── 1. Completed documents from PostgreSQL (most recent N) ────────────────
    async with get_db_session() as session:
        query = (
            select(
                Document.id,
                Document.filename,
                Document.page_count,
                Document.chunk_count,
                Document.created_at,
            )
            .where(Document.status == DocumentStatus.COMPLETED)
            .order_by(Document.created_at.desc())
            .limit(max_documents)
        )
        if owner_id:
            query = query.where(Document.owner_id == uuid.UUID(owner_id))
        rows = (await session.execute(query)).all()

    if not rows:
        logger.info("collect_document_digests: no completed documents found")
        return []

    rows = list(reversed(rows))  # chronological order for stable prompt layout

    # ── 2. Sample chunks per document from Qdrant ─────────────────────────────
    client = get_qdrant_client()
    collection = settings.QDRANT_COLLECTION
    digests: list[DocumentDigest] = []

    for position, (doc_id, filename, page_count, chunk_count, _created) in enumerate(
        rows, start=1
    ):
        doc_id_str = str(doc_id)
        warning: list[str] = []

        sample: list[tuple[int, str]] = []
        try:
            points, _next_offset = await client.scroll(
                collection_name=collection,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=doc_id_str),
                        )
                    ]
                ),
                limit=256,                     # generous; sorted + sampled below
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                sample.append(
                    (
                        int(payload.get("chunk_index", 0)),
                        str(payload.get("text", "")),
                    )
                )
            sample.sort(key=lambda item: item[0])  # reading order
        except Exception as exc:
            logger.warning(
                "Qdrant scroll failed for document_id=%s: %s", doc_id_str, exc
            )
            warning.append("向量采样失败")

        if not sample:
            warning.append("未采样到文本内容")

        # Take a spread of chunks (beginning matters most) and clip each one
        # so a single verbose chunk cannot eat the whole digest budget.
        picked = sample[: max(chunks_per_doc, 4)]
        per_chunk = max(digest_chars // max(len(picked), 1), 120)
        parts = [text.strip()[:per_chunk] for _, text in picked if text.strip()]
        digest = "\n…\n".join(parts)[:digest_chars]

        digests.append(
            DocumentDigest(
                document_id=doc_id_str,
                filename=filename,
                digest=digest or "（无可用文本内容）",
                page_count=page_count or 0,
                chunk_count=chunk_count or 0,
                sampled_chunks=len(parts),
                index=position,
                warnings=warning,
            )
        )

    logger.info(
        "collect_document_digests: %d documents (max=%d), sampled %s",
        len(digests),
        max_documents,
        [d.sampled_chunks for d in digests],
    )
    return digests


def build_digest_context(digest_dicts: list[dict]) -> str:
    """
    Render serialised digests (as emitted by ``digest_sources``) as the
    numbered context block embedded in the LLM prompt.
    """
    sections: list[str] = []
    for d in digest_dicts:
        meta_bits = [f"{d.get('page_count', 0)} 页", f"{d.get('chunk_count', 0)} 个文本块"]
        warnings = d.get("warnings") or []
        meta = "，".join(meta_bits + warnings)
        sections.append(
            f"[文档 {d.get('index', 0)}] 文件名: {d.get('filename', 'unknown')}（{meta}）\n"
            f"内容采样:\n{d.get('digest', '')}"
        )
    return "\n\n---\n\n".join(sections)


def digest_sources(digests: list[DocumentDigest]) -> list[dict]:
    """
    Serialise digests for the SSE ``doc_digests`` event (frontend display).
    """
    return [
        {
            "document_id": d.document_id,
            "filename": d.filename,
            "digest": d.digest[:600],
            "page_count": d.page_count,
            "chunk_count": d.chunk_count,
            "sampled_chunks": d.sampled_chunks,
            "warnings": d.warnings,
            "index": d.index,
        }
        for d in digests
    ]


def digest_context_for_log(conversation_id: uuid.UUID | str, count: int) -> str:  # pragma: no cover
    """Tiny logging helper kept separate to avoid leaking digest text to logs."""
    return f"conv={conversation_id} digests={count}"
