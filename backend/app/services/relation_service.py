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


def _spread_pick(items: list, count: int) -> list:
    """
    在一份文档的 chunk 序列上**等距**取 count 个（含首尾）.

    为什么不能只取开头：分块是按阅读顺序排的，``items[:8]`` 对一份 60 块的
    文档等于"只读前 13%"。整库总结时这会直接表现为"这份文档只总结了开篇"。
    等距采样让采样点覆盖全文（首尾必取），摘要才代表整份文档。

    取不满 count 个（步长取整后重复）时不补齐 —— 少一个采样点比重复同一块好。
    """
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[i] for i in sorted({round(i * step) for i in range(count)})]


async def list_accessible_documents(
    *,
    limit: int | None = None,
    owner_id: str | None = None,
    tenant_id: str | None = None,
    user_department_id: str | None = None,
    tenant_wide: bool = False,
    platform_wide: bool = False,
    document_ids: list[str] | None = None,
) -> list[tuple]:
    """
    列出**当前用户有权访问**的已完成文档（(id, filename, page_count,
    chunk_count, created_at)，按上传时间倒序）.

    单独抽出来的理由：文档总结需要"用户点名了哪份文档"的完整候选清单，
    但那一步只要 id + 文件名 —— 为此跑一遍 Qdrant 采样（collect_document_digests）
    是纯浪费。抽成轻量函数后，候选解析与摘要采样各取所需、共用同一套 ACL。
    """
    from app.services.tenancy import document_acl_clause, normalize_tenant_id

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
    )
    if document_ids:
        # 点名了文档 → 必须按 id 过滤后再限量，否则"被点名的老文档"会被
        # 最近 N 份的上限挤掉（用户明明点了它，却总结不到它）。
        query = query.where(Document.id.in_([uuid.UUID(str(d)) for d in document_ids]))
        query = query.limit(max(len(document_ids), 1))
    elif limit:
        query = query.limit(limit)

    if tenant_id and not platform_wide:
        query = query.where(Document.tenant_id == normalize_tenant_id(tenant_id))
    elif not platform_wide and owner_id:
        # 无公司上下文的历史调用：退化为"仅本人"，不跨公司放量
        query = query.where(Document.owner_id == uuid.UUID(owner_id))
    if platform_wide or tenant_id or owner_id:
        query = query.where(
            document_acl_clause(
                owner_id=uuid.UUID(owner_id) if owner_id else None,
                department_id=user_department_id,
                tenant_wide=tenant_wide,
                platform_wide=platform_wide,
            )
        )

    async with get_db_session() as session:
        rows = (await session.execute(query)).all()
    return list(rows)


async def collect_document_digests(
    max_documents: int | None = None,
    chunks_per_doc: int | None = None,
    digest_chars: int | None = None,
    owner_id: str | None = None,
    tenant_id: str | None = None,
    user_department_id: str | None = None,
    tenant_wide: bool = False,
    platform_wide: bool = False,
    document_ids: list[str] | None = None,
) -> list[DocumentDigest]:
    """
    Build a content digest for every accessible completed document.

    Args:
        max_documents:   Cap on documents analysed (most recent N win).
        chunks_per_doc:  Chunks sampled per document for its digest.
        digest_chars:    Max characters of digest text per document.
        owner_id:        个人库归属人（恒为本人 id）；None = 不含任何个人库。
        tenant_id:       第一层隔离 —— 摘要采样只覆盖本公司文档；
                         None 仅当 platform_wide（平台管理员）时为真。
        user_department_id: 第二层 Document ACL 的部门条件。
        tenant_wide:     企业/知识库管理员 —— 本公司的部门库全通。
        platform_wide:   平台管理员 —— 跨公司（个人库仍然只有自己的）。
        document_ids:    只采样这些文档（用户点名总结某几份时用）；
                         None = 全库（受 max_documents 与 ACL 约束）。

    Returns:
        List of DocumentDigest sorted by upload time (oldest first).
        Documents with no vectors in Qdrant still appear, with a warning.
    """
    settings = get_settings()
    max_documents = max_documents or settings.RELATION_MAX_DOCUMENTS
    chunks_per_doc = chunks_per_doc or settings.RELATION_CHUNKS_PER_DOC
    digest_chars = digest_chars or settings.RELATION_DIGEST_CHARS

    # ── 1. Completed documents from PostgreSQL (most recent N) ────────────────
    rows = await list_accessible_documents(
        limit=max_documents,
        owner_id=owner_id,
        tenant_id=tenant_id,
        user_department_id=user_department_id,
        tenant_wide=tenant_wide,
        platform_wide=platform_wide,
        document_ids=document_ids,
    )

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

        # Take an even spread of chunks (first & last always included) and clip
        # each one so a single verbose chunk cannot eat the whole digest budget.
        picked = _spread_pick(sample, max(chunks_per_doc, 4))
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
