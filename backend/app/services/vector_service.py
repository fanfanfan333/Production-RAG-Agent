"""
Qdrant vector store operations.

Responsibilities:
  - Ensure the collection exists with the correct vector configuration.
  - Upsert document chunk vectors with rich payload for retrieval.
  - Provide a clean delete-by-document-id helper for future use.
"""

import uuid
from dataclasses import dataclass

from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.db.qdrant import get_qdrant_client
from app.utils.logging import get_logger

logger = get_logger(__name__)

NAMESPACE_RAG = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

def generate_point_id(
    document_id: str, filename: str, page_number: int, section: str | None, heading: str | None, chunk_index: int, text: str
) -> str:
    """
    Generate a deterministic UUID for a chunk based on its content and metadata.

    document_id participates in the key: the same file uploaded by different
    users (different Document rows) gets different vectors — otherwise the
    second upload would OVERWRITE the first user's points and steal their
    vectors (isolation bug).
    """
    stable_name = (
        f"{document_id}::{filename}::{page_number}::{section}::{heading}::{chunk_index}::{text}"
    )
    return str(uuid.uuid5(NAMESPACE_RAG, stable_name))



@dataclass
class VectorPoint:
    """A single vector + payload to be upserted into Qdrant."""

    vector: list[float]
    document_id: str           # UUID string of the parent Document row
    filename: str
    chunk_index: int
    page_number: int
    text: str
    heading: str | None = None
    section: str | None = None
    collection_id: str | None = None   # KB collection UUID (business grouping)

    # ── small-to-big / Hierarchical RAG metadata（全部 optional 向后兼容） ──
    parent_id: str | None = None
    parent_text: str | None = None
    parent_char_start: int | None = None
    parent_char_end: int | None = None


async def ensure_collection() -> None:
    """
    Create the Qdrant collection if it does not already exist.
    Safe to call on every startup — idempotent.
    """
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    existing = await client.get_collections()
    names = {c.name for c in existing.collections}

    if collection_name in names:
        logger.debug("Qdrant collection '%s' already exists", collection_name)
        await _ensure_payload_indexes(collection_name)
        return

    await client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=settings.EMBEDDING_DIMENSION,
            distance=qmodels.Distance.COSINE,
        ),
        # Optimiser settings tuned for read-heavy RAG workloads
        optimizers_config=qmodels.OptimizersConfigDiff(
            indexing_threshold=20_000,
        ),
        # Payload index for fast filtered retrieval by document_id
        on_disk_payload=True,
    )

    await _ensure_payload_indexes(collection_name)

    logger.info(
        "Created Qdrant collection '%s' (dim=%d, distance=COSINE)",
        collection_name,
        settings.EMBEDDING_DIMENSION,
    )


async def _ensure_payload_indexes(collection_name: str) -> None:
    """Idempotently create the payload indexes used by filtered retrieval."""
    client = get_qdrant_client()

    indexes = [
        ("document_id", qmodels.PayloadSchemaType.KEYWORD),
        ("collection_id", qmodels.PayloadSchemaType.KEYWORD),  # KB filtering (第一阶段)
    ]
    for field_name, schema in indexes:
        try:
            await client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception as exc:
            # Usually "index already exists" — safe to ignore on steady state
            logger.debug("Payload index on '%s': %s", field_name, exc)


async def upsert_vectors(points: list[VectorPoint]) -> int:
    """
    Upsert a batch of vectors into the Qdrant collection.

    Each point receives a random UUID as its Qdrant point ID.
    The ``document_id`` field in the payload is the PostgreSQL row UUID
    and is the join key for cross-store lookups.

    Returns:
        Number of points upserted.
    """
    if not points:
        return 0

    settings = get_settings()
    client = get_qdrant_client()

    qdrant_points = []

    for p in points:
        deterministic_id = generate_point_id(
            p.document_id, p.filename, p.page_number, p.section, p.heading, p.chunk_index, p.text
        )

        qdrant_points.append(
            qmodels.PointStruct(
                id=deterministic_id,
                vector=p.vector,
                payload={
                    "document_id": p.document_id,
                    "filename": p.filename,
                    "chunk_index": p.chunk_index,
                    "page_number": p.page_number,
                    "text": p.text,
                    "char_count": len(p.text),
                    "heading": p.heading,
                    "section": p.section,
                    "collection_id": p.collection_id,
                    # Hierarchical RAG metadata — 仅当 chunker 实际生成了 parent
                    # 字段时才会有值；旧 chunk 留空，不影响检索行为。
                    "parent_id": p.parent_id,
                    "parent_text": p.parent_text,
                    "parent_char_start": p.parent_char_start,
                    "parent_char_end": p.parent_char_end,
                },
            )
        )

    await client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=qdrant_points,
        wait=True,          # wait for WAL flush — guarantees durability
    )

    logger.info(
        "Upserted %d vectors into '%s'",
        len(qdrant_points),
        settings.QDRANT_COLLECTION,
    )
    return len(qdrant_points)


async def delete_by_document_id(document_id: str) -> None:
    """
    Remove all vectors whose payload.document_id matches the given UUID.
    Useful for re-processing or deleting a document.
    """
    settings = get_settings()
    client = get_qdrant_client()

    await client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )
        ),
    )
    logger.info("Deleted vectors for document_id=%s", document_id)


async def get_existing_point_ids(point_ids: list[str]) -> set[str]:
    """
    Given a list of Qdrant point IDs, returns a set of the ones that already exist
    in the collection. Useful for idempotent resumes to avoid re-embedding.
    """
    if not point_ids:
        return set()

    settings = get_settings()
    client = get_qdrant_client()

    # retrieve only the IDs without payload/vectors to be fast
    response = await client.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=point_ids,
        with_payload=False,
        with_vectors=False
    )
    
    return {str(point.id) for point in response}

