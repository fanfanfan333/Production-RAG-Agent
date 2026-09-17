"""诊断：导出文档中图片块的质量与检索文本（容器内运行）.

    docker exec -e HOME=/tmp rag_backend python /tmp/dump_img.py <doc_id> [doc_id...]

回答"图片块到底是靠什么文本被检索到的"：OCR 文本是不是乱码、
Vision 描述有没有、门控结果与质检报告是什么。
"""
import asyncio
import json
import sys


async def main() -> None:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from app.config import get_settings

    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)

    for doc in sys.argv[1:]:
        records, _ = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=doc))]
            ),
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            p = r.payload or {}
            if (p.get("content_type") or "") != "image":
                continue
            print("=" * 70)
            print("payload keys:", sorted(p.keys()))
            for key in (
                "image_type", "analyze_engine", "classify_engine",
                "analyze_confidence", "analyze_decision", "manual_review",
            ):
                print(f"  {key} = {p.get(key)}")
            print("  ocr_text_at_all:", repr(p.get("ocr_text"))[:200])
            print("  vision:", repr(p.get("image_caption") or p.get("vision"))[:200])
            print("  searchable:", (p.get("searchable_text") or "")[:280].replace("\n", " | "))
            quality = p.get("analyze_quality") or {}
            if quality:
                print("  quality:", json.dumps(quality, ensure_ascii=False)[:400])


asyncio.run(main())
