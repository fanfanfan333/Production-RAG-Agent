"""
Semantic retrieval service (Phase 4 / 混合检索优化).

Embeds the query, performs an ANN search against Qdrant, and — when hybrid
search is enabled — fuses the vector ranking with a BM25 keyword ranking via
Reciprocal Rank Fusion (RRF).  The keyword leg catches exact-term matches
(product codes, proper nouns) that pure semantic similarity tends to miss.

Reuses embed_texts() from Phase 2 — task_type is the only difference between
document ingestion and query embedding.
"""

import time
from dataclasses import dataclass

from app.config import get_settings
from app.db.qdrant import get_qdrant_client
from app.services.embedding_service import embed_batch_with_retry
from app.services.hybrid_search import BM25Index, rrf_fuse
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A single chunk returned by the vector search."""

    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float   # cosine similarity (0 – 1)

    # ── small-to-big / Hierarchical RAG metadata（全部默认 None，向后兼容）──
    parent_id: str | None = None        # document_id + ":p:" + parent_index
    parent_text: str | None = None      # 命中后回填的父块完整文本
    parent_char_start: int | None = None
    parent_char_end: int | None = None
    heading: str | None = None          # 复制 chunker 写出的 heading
    section: str | None = None          # 复制 chunker 写出的 section


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, int]:
    """Stable identity of a chunk across the vector and keyword legs."""
    return (chunk.document_id, chunk.chunk_index)


# ── BM25 corpus cache ─────────────────────────────────────────────────────────
# Keyed by (kb_collection_id or "__all__", owner_id or "__admin__") so each
# visibility scope builds/holds its own index.  Rebuilt lazily after the TTL —
# new uploads become searchable to BM25 within one TTL window.

_bm25_cache: dict[tuple[str, str], tuple[float, list[RetrievedChunk], BM25Index]] = {}


async def _scroll_corpus(
    client,
    collection_name: str,
    collection_id: str | None,
    max_points: int,
) -> list[RetrievedChunk]:
    """Scroll the whole Qdrant collection (payload only, no vectors)."""
    from qdrant_client.http import models as qmodels

    scroll_filter = None
    if collection_id:
        scroll_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="collection_id",
                    match=qmodels.MatchValue(value=collection_id),
                )
            ]
        )

    corpus: list[RetrievedChunk] = []
    offset = None
    while len(corpus) < max_points:
        points, next_offset = await client.scroll(
            collection_name=collection_name,
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
                page_number = int(payload.get("page_number") or 1)
            except (TypeError, ValueError):
                page_number = 1
            try:
                chunk_index = int(payload.get("chunk_index") or 0)
            except (TypeError, ValueError):
                chunk_index = 0
            corpus.append(
                RetrievedChunk(
                    document_id=str(payload.get("document_id", "")),
                    filename=str(payload.get("filename", "unknown")),
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=str(payload.get("text", "")),
                    score=0.0,
                )
            )
        if next_offset is None:
            break
        offset = next_offset

    return corpus[:max_points]


async def _bm25_candidates(
    query: str,
    top_k: int,
    valid_docs: set[str],
    collection_name: str,
    collection_id: str | None,
    owner_id: str | None,
    client,
) -> list[RetrievedChunk]:
    """
    Keyword (BM25) leg of the hybrid search.

    Returns chunks ranked best-first, restricted to *valid_docs* (documents
    that exist in PostgreSQL, are COMPLETED, and pass the owner filter).
    """
    settings = get_settings()
    cache_key = (collection_id or "__all__", owner_id or "__admin__")

    now = time.monotonic()
    cached = _bm25_cache.get(cache_key)
    if cached is not None and now - cached[0] < settings.HYBRID_CACHE_TTL_SECONDS:
        _, visible, index = cached
    else:
        corpus = await _scroll_corpus(
            client,
            collection_name,
            collection_id,
            settings.HYBRID_MAX_CORPUS_POINTS,
        )
        visible = [c for c in corpus if c.document_id in valid_docs]
        index = BM25Index([c.text for c in visible])
        _bm25_cache[cache_key] = (now, visible, index)
        logger.info(
            "BM25 index built: scope=%s corpus=%d visible=%d",
            cache_key,
            len(corpus),
            len(visible),
        )

    if len(visible) < 2:
        return []

    hits = index.search(query, top_n=max(top_k * 3, top_k))
    return [visible[h.index] for h in hits]


def _apply_score_filters(
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Pure-vector filtering (original Phase 4 semantics):
    absolute RETRIEVAL_MIN_SCORE floor + relative RETRIEVAL_MAX_GAP from the
    top score.  *candidates* must already be sorted by score descending.
    """
    settings = get_settings()
    if not candidates:
        return []

    top_score = candidates[0].score
    gap_cutoff = top_score - settings.RETRIEVAL_MAX_GAP

    kept: list[RetrievedChunk] = []
    for c in candidates:
        if c.score < settings.RETRIEVAL_MIN_SCORE:
            logger.info(
                "Filtering out chunk filename='%s' index=%d (score=%.4f < floor=%.2f)",
                c.filename, c.chunk_index, c.score, settings.RETRIEVAL_MIN_SCORE,
            )
            continue
        if c.score < gap_cutoff:
            logger.info(
                "Filtering out chunk filename='%s' index=%d (score=%.4f < gap_cutoff=%.4f)",
                c.filename, c.chunk_index, c.score, gap_cutoff,
            )
            continue
        kept.append(c)
    return kept[:top_k]


async def retrieve_chunks(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    collection_name: str | None = None,
    owner_id: str | None = None,
    collection_id: str | None = None,
    extra_queries: list[str] | None = None,
    enable_hierarchical: bool | None = None,
) -> list[RetrievedChunk]:
    """
    两阶段检索：粗排（召回+融合）→ 精排（cross-encoder rerank）.

    粗排阶段（HYBRID_SEARCH_ENABLED 时）：
      1. 向量腿 —— 原查询 + *extra_queries*（多查询扩展变体）逐路 ANN，
         同一 chunk 取最高向量分，各路排名进入 RRF；
      2. 关键词腿 —— BM25 对每路查询检索，排名同样进入 RRF；
      3. RRF 融合排名 + 向量 floor/gap 过滤 → 粗排候选池。

    精排阶段（RERANKER_ENABLED 时）：
      对粗排候选池（截断到 RERANKER_MAX_CANDIDATES 条）逐对 cross-encoder
      打分重排，取 top_k。chunk.score 被替换为归一化精排分（[0,1]）。

    Args:
        query:            Natural-language question from the user.
        top_k:            Maximum number of chunks to return.
        score_threshold:  Unused placeholder (kept for API compat).
        collection_name:  Override the default collection from settings.
        owner_id:         Restrict results to documents owned by this user
                          (None = unrestricted / admin).
        collection_id:    Restrict results to one knowledge-base collection
                          via the Qdrant payload filter (None = all).
        extra_queries:    检索变体（multi-query expansion），每路独立向量
                          + BM25 召回后 RRF 融合，提升召回率。

    Returns:
        List of RetrievedChunk, best first.
    """
    settings = get_settings()
    client = get_qdrant_client()
    coll = collection_name or settings.QDRANT_COLLECTION

    # 检索查询集合：原查询（改写后）+ 去重变体
    queries = [query]
    for q in extra_queries or []:
        if q and q not in queries:
            queries.append(q)

    # Embed all queries with RETRIEVAL_QUERY task type (one batch call)
    vectors = await embed_batch_with_retry(queries, task_type="RETRIEVAL_QUERY")

    # Knowledge-base collection filter (vector-layer isolation, 企业落地第一阶段)
    search_filter = None
    if collection_id:
        from qdrant_client.http import models as qmodels

        search_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="collection_id",
                    match=qmodels.MatchValue(value=collection_id),
                )
            ]
        )

    # Wider vector candidate pool so RRF fusion has material to work with
    candidate_pool = max(top_k * 4, top_k + 8, settings.RERANKER_MAX_CANDIDATES)

    # ── 向量腿：每路查询独立 ANN，结果按 chunk 去重合并 ─────────────────────
    all_results: dict[tuple[str, int], tuple[float, dict]] = {}
    vector_rank_lists: list[list[tuple[str, int]]] = []
    for qv in vectors:
        results = await client.search(
            collection_name=coll,
            query_vector=qv,
            limit=candidate_pool,
            with_payload=True,
            query_filter=search_filter,
        )
        rank_list: list[tuple[str, int]] = []
        for rank, hit in enumerate(results):
            payload = hit.payload or {}
            key = (
                str(payload.get("document_id", "")),
                int(payload.get("chunk_index", 0) or 0),
            )
            if key not in all_results or hit.score > all_results[key][0]:
                all_results[key] = (float(hit.score), payload)
            rank_list.append(key)
        vector_rank_lists.append(rank_list)

    search_results_count = len(all_results)
    if search_results_count == 0:
        logger.info("No raw candidates returned from Qdrant search.")
        return []

    chunks: list[RetrievedChunk] = []

    import uuid
    from sqlalchemy import select
    from app.db.postgres import get_db_session
    from app.db.models import Document

    doc_ids = set()
    for _, payload in all_results.values():
        doc_id_str = payload.get("document_id")
        if doc_id_str:
            try:
                doc_ids.add(uuid.UUID(doc_id_str))
            except ValueError:
                pass

    valid_docs = set()
    if doc_ids:
        async with get_db_session() as session:
            from app.db.models import DocumentStatus
            q = select(Document.id).where(
                Document.id.in_(doc_ids),
                Document.status == DocumentStatus.COMPLETED,
            )
            # Owner isolation (企业落地第一阶段): non-admin users can only
            # retrieve chunks from documents they uploaded.
            if owner_id:
                q = q.where(Document.owner_id == uuid.UUID(owner_id))
            res = await session.execute(q)
            valid_docs = {str(r[0]) for r in res}

    if not valid_docs:
        logger.warning("No valid (COMPLETED + owner-visible) documents in candidates.")
        return []

    # First gather all valid chunks (excluding orphans), keyed for fusion
    merged: dict[tuple[str, int], RetrievedChunk] = {}
    for (doc_id_str, _ci), (score, payload) in all_results.items():
        if doc_id_str not in valid_docs:
            logger.warning("Found orphan vector referencing non-existent document_id=%s. Ignoring.", doc_id_str)
            continue
        key = (doc_id_str, _ci)
        merged[key] = RetrievedChunk(
                document_id=doc_id_str,
                filename=payload.get("filename", "unknown"),
                page_number=int(payload.get("page_number", 1)),
                chunk_index=int(payload.get("chunk_index", 0)),
                text=payload.get("text", ""),
                score=score,
                parent_id=payload.get("parent_id"),
                parent_text=payload.get("parent_text"),
                parent_char_start=payload.get("parent_char_start"),
                parent_char_end=payload.get("parent_char_end"),
                heading=payload.get("heading"),
                section=payload.get("section"),
            )

    # Ensure descending order by best vector score
    valid_candidates = sorted(merged.values(), key=lambda c: c.score, reverse=True)
    # Rebuild the primary (original-query) rank list restricted to merged keys
    primary_keys = set(merged.keys())
    vector_rank_lists = [
        [k for k in rl if k in primary_keys] for rl in vector_rank_lists
    ]

    # ── Keyword (BM25) legs: one per query variant ─────────────────────────
    bm25_rank_lists: list[list[tuple[str, int]]] = []
    if settings.HYBRID_SEARCH_ENABLED:
        for q in queries:
            try:
                bm25_ranked = await _bm25_candidates(
                    query=q,
                    top_k=top_k,
                    valid_docs=valid_docs,
                    collection_name=coll,
                    collection_id=collection_id,
                    owner_id=owner_id,
                    client=client,
                )
                bm25_rank_lists.append([_chunk_key(c) for c in bm25_ranked])
                # BM25-only hits are not in the vector candidate list: their
                # vector score is unknown.  Give them the floor score as a
                # neutral, honest placeholder (they earned their place
                # through exact term match).
                floor = settings.RETRIEVAL_MIN_SCORE
                for c in bm25_ranked:
                    key = _chunk_key(c)
                    if key not in merged:
                        merged[key] = RetrievedChunk(
                            document_id=c.document_id,
                            filename=c.filename,
                            page_number=c.page_number,
                            chunk_index=c.chunk_index,
                            text=c.text,
                            score=max(floor, 0.30),
                            parent_id=c.parent_id,
                            parent_text=c.parent_text,
                            parent_char_start=c.parent_char_start,
                            parent_char_end=c.parent_char_end,
                            heading=c.heading,
                            section=c.section,
                        )
            except Exception:
                logger.exception(
                    "Hybrid BM25 leg failed — falling back to vector-only retrieval"
                )

    # ── 粗排融合（RRF over vector legs + BM25 legs） ────────────────────────
    rank_lists = vector_rank_lists + bm25_rank_lists
    strong_bm25 = set()
    for rl in bm25_rank_lists:
        strong_bm25.update(rl[:top_k])

    if not rank_lists or not any(rank_lists):
        # No ranking material at all — pure-vector floor/gap filtering
        candidates = _apply_score_filters(valid_candidates, top_k)
        if settings.RERANKER_ENABLED and len(candidates) > 1:
            from app.services.reranker import rerank_chunks

            chunks = await rerank_chunks(query, candidates, top_k)
            logger.info(
                "Retrieved %d chunks (vector-only fallback, rerank=%s top_k=%d)",
                len(chunks), settings.RERANKER_ENABLED, top_k,
            )
            return chunks
        logger.info(
            "Retrieved %d chunks (vector-only fallback, top_k=%d)",
            len(candidates), top_k,
        )
        return candidates
    else:
        fused_scores = rrf_fuse(rank_lists, k=settings.HYBRID_RRF_K)

        # Filter policy: vector floor+gap OR strong keyword hit
        top_vector_score = valid_candidates[0].score if valid_candidates else 0.0
        gap_cutoff = top_vector_score - settings.RETRIEVAL_MAX_GAP

        survivors = [
            (key, c)
            for key, c in merged.items()
            if (c.score >= settings.RETRIEVAL_MIN_SCORE and c.score >= gap_cutoff)
            or key in strong_bm25
        ]
        survivors.sort(
            key=lambda kc: fused_scores.get(kc[0], 0.0), reverse=True
        )

        # 粗排候选池：截断到精排上限，控制 cross-encoder 推理量
        candidates = [c for _, c in survivors[: settings.RERANKER_MAX_CANDIDATES]]

    # ── 精排（cross-encoder rerank）：用原始（改写后）查询打分 ─────────────
    if settings.RERANKER_ENABLED and len(candidates) > 1:
        from app.services.reranker import rerank_chunks

        chunks = await rerank_chunks(query, candidates, top_k)
    else:
        chunks = candidates[:top_k]

    # ── Hierarchical RAG / small-to-big 钩子（架构图 Hierarchical RAG） ─────
    # 若 document_service 在写入 Qdrant 时把 parent_id/parent_text 写进 payload
    # （enable_small_to_big=True 时才会写），这里把父块 metadata 复制到
    # chunk 上。缺失时降级为 None——调用方可以安全使用，旧流程完全不受影响。
    use_hier = (
        settings.HIERARCHICAL_RAG_ENABLED
        if enable_hierarchical is None
        else enable_hierarchical
    )
    if use_hier and chunks:
        # 真要 small-to-big，回填父块；从 PG 与 Qdrant 任一来源取都 OK，
        # 此处优先用 Qdrant payload（已索引好的 parent_text），缺失再回退
        # 到 document_service 调用方传给我们的 parent_text 字段（若有）。
        for c in chunks:
            # 当前实现：父块 metadata 需要调用方通过 document_service 写入
            # Qdrant payload；如果尚未写，保持 None 不影响主流程。
            # 此处为钩子示例——可直接 payload 读：
            par_id = c.parent_id  # 已经从 dataclass 写入
            par_text = c.parent_text
            if par_id and par_text:
                logger.debug("hierarchical: child chunk (%s, idx=%d) → parent=%s len=%d",
                             c.document_id, c.chunk_index, par_id, len(par_text))

    logger.info(
        "Retrieved %d chunks (floor=%.3f gap=%.3f hybrid=%s rerank=%s queries=%d top_k=%d hierarchical=%s)",
        len(chunks),
        settings.RETRIEVAL_MIN_SCORE,
        settings.RETRIEVAL_MAX_GAP,
        settings.HYBRID_SEARCH_ENABLED,
        settings.RERANKER_ENABLED,
        len(queries),
        top_k,
        use_hier,
    )
    return chunks
