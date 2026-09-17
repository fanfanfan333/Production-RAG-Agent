"""验证图片块 payload 与检索命中（容器内运行）."""
import asyncio

from qdrant_client.http import models as qmodels

DOC = "6d1d929c-62c7-4096-8f23-023950aec94d"


async def main() -> None:
    from app.config import get_settings
    from app.db.qdrant import get_qdrant_client
    from app.services.retrieval_service import retrieve_chunks

    c = get_qdrant_client()
    pts, _ = await c.scroll(
        collection_name=get_settings().QDRANT_COLLECTION,
        scroll_filter=qmodels.Filter(must=[qmodels.FieldCondition(
            key="document_id", match=qmodels.MatchValue(value=DOC))]),
        limit=10, with_payload=True,
    )
    for p in pts:
        pl = p.payload or {}
        print(f"--- content_type={pl.get('content_type')} image_type={pl.get('image_type')} "
              f"engine={pl.get('analyze_engine')} review={pl.get('manual_review')}")
        print("    image_path:", pl.get("image_path"), "| tenant:", pl.get("tenant_id"),
              "| access:", pl.get("access_level"))
        print("    text:", (pl.get("text") or "").replace("\n", " ")[:220])

    for q in ("数据流图 input Fan output 节点关系", "风扇系统"):
        hits = await retrieve_chunks(query=q, top_k=3, tenant_id="default")
        print(f"\nquery={q!r} -> {len(hits)} hit(s)")
        for h in hits:
            print(f"   {h.content_type:6s} {h.filename} score={h.score:.3f} :: "
                  f"{h.text[:80].replace(chr(10), ' ')}")


asyncio.run(main())
