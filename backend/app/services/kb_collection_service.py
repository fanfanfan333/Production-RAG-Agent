"""
Knowledge-base collection service (企业落地第一阶段).

Business-level document grouping — what the UI calls "知识库/集合". Rows live
in PostgreSQL (per-owner), and every chunk vector carries the collection id
in its Qdrant payload so retrieval can filter at the vector layer.

Distinct from collection_service.py, which manages raw Qdrant collections
(infrastructure-level, admin-only).
"""

import uuid

from sqlalchemy import func, select, update

from app.db.models import Document
from app.db.postgres import get_db_session
from app.db.user_models import Collection
from app.utils.logging import get_logger

logger = get_logger(__name__)

MAX_COLLECTIONS_PER_USER = 50
MAX_NAME_LENGTH = 128


async def list_collections(owner_id: uuid.UUID) -> list[dict]:
    """List the user's collections with live document counts."""
    async with get_db_session() as session:
        cols = (
            await session.execute(
                select(Collection)
                .where(Collection.owner_id == owner_id)
                .order_by(Collection.created_at.asc())
            )
        ).scalars().all()

        counts: dict[uuid.UUID, int] = {}
        if cols:
            rows = await session.execute(
                select(Document.collection_id, func.count())
                .where(Document.collection_id.in_([c.id for c in cols]))
                .group_by(Document.collection_id)
            )
            counts = {cid: n for cid, n in rows.all()}

    return [
        {
            "id": str(c.id),
            "name": c.name,
            "description": c.description,
            "document_count": int(counts.get(c.id, 0)),
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in cols
    ]


async def create_collection(
    owner_id: uuid.UUID, name: str, description: str | None = None
) -> dict:
    """Create a collection for the user. Duplicate names are rejected."""
    name = name.strip()
    if not name:
        raise ValueError("集合名称不能为空")
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"集合名称最长 {MAX_NAME_LENGTH} 个字符")

    async with get_db_session() as session:
        existing = await session.scalar(
            select(Collection).where(
                Collection.owner_id == owner_id, Collection.name == name
            )
        )
        if existing is not None:
            raise ValueError(f"集合「{name}」已存在")

        total = (
            await session.execute(
                select(func.count()).select_from(Collection).where(
                    Collection.owner_id == owner_id
                )
            )
        ).scalar_one()
        if total >= MAX_COLLECTIONS_PER_USER:
            raise ValueError(f"每个用户最多创建 {MAX_COLLECTIONS_PER_USER} 个集合")

        col = Collection(
            id=uuid.uuid4(),
            name=name,
            description=(description or None),
            owner_id=owner_id,
        )
        session.add(col)
        await session.flush()

        logger.info("Created collection '%s' (id=%s, owner=%s)", name, col.id, owner_id)
        return {
            "id": str(col.id),
            "name": col.name,
            "description": col.description,
            "document_count": 0,
            "created_at": col.created_at.isoformat() if col.created_at else None,
        }


async def delete_collection(owner_id: uuid.UUID, collection_id: uuid.UUID) -> bool:
    """
    Delete a collection. Documents keep existing but become 未分配
    (collection_id is nulled via ON DELETE SET NULL, mirrored explicitly here
    so the Qdrant payload is not the source of truth for membership).
    """
    async with get_db_session() as session:
        col = await session.get(Collection, collection_id)
        if col is None or col.owner_id != owner_id:
            return False

        await session.execute(
            update(Document)
            .where(Document.collection_id == collection_id)
            .values(collection_id=None)
        )
        await session.delete(col)

    logger.info("Deleted collection id=%s (owner=%s)", collection_id, owner_id)
    return True


async def assign_document(
    owner_id: uuid.UUID,
    document_id: uuid.UUID,
    collection_id: uuid.UUID | None,
) -> None:
    """
    Assign (or unassign, when *collection_id* is None) a document to one of
    the owner's collections.

    Raises:
        KeyError:   document does not exist or is not owned by the user
        ValueError: collection does not exist or is not owned by the user
    """
    async with get_db_session() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            raise KeyError(f"文档 {document_id} 不存在")
        if doc.owner_id != owner_id:
            raise KeyError(f"文档 {document_id} 不存在")

        if collection_id is not None:
            col = await session.get(Collection, collection_id)
            if col is None or col.owner_id != owner_id:
                raise ValueError(f"集合 {collection_id} 不存在")

        doc.collection_id = collection_id

    logger.info(
        "Document %s assigned to collection=%s (owner=%s)",
        document_id, collection_id, owner_id,
    )
