"""诊断：导出某个文档在 Qdrant 中的 chunk 概况（容器内运行）.

    docker exec -e HOME=/tmp rag_backend python /app/scripts/diag_chunks.py <filename|doc_id>

用于回答"这份文档到底被切成什么样了"：块数、每块类型、页号、文本长度、
前若干字符。图片块（image_type/vision）也会列出，便于确认图文分流是否生效。
"""
import asyncio
import sys

TARGET = sys.argv[1]


async def main() -> None:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from app.config import get_settings

    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)

    flt = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=TARGET))])
    records, _ = await client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=flt,
        limit=500,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        # 退一步：按文件名过滤
        flt = Filter(must=[FieldCondition(key="filename", match=MatchValue(value=TARGET))])
        records, _ = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=flt,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )

    print(f"records={len(records)}")
    rows = []
    for r in records:
        p = r.payload or {}
        rows.append(
            (
                p.get("chunk_index", -1),
                p.get("content_type") or p.get("type") or "text",
                p.get("page_number"),
                p.get("image_type") or "",
                len(p.get("text") or p.get("searchable_text") or ""),
                (p.get("text") or p.get("searchable_text") or "")[:180].replace("\n", " "),
            )
        )
    rows.sort(key=lambda x: (x[1] != "text", x[0]))
    for idx, ctype, page, itype, ln, head in rows:
        print(f"[{idx:>3}] type={ctype:<6} page={page} img={itype:<11} len={ln:<5} | {head}")

    # 类型分布
    from collections import Counter

    print("type dist:", Counter(r[1] for r in rows))
    print("image type dist:", Counter(r[3] for r in rows if r[1] != "text"))


asyncio.run(main())
