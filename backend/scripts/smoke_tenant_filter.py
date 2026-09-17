"""冒烟 3：用 chunk 原文做查询，隔离变量（临时脚本，容器内运行）."""

import asyncio


async def main():
    from app.config import get_settings
    from app.db.qdrant import get_qdrant_client
    from app.services.retrieval_service import retrieve_chunks

    c = get_qdrant_client()
    pts, _ = await c.scroll(
        collection_name=get_settings().QDRANT_COLLECTION, limit=1, with_payload=True,
    )
    text = (pts[0].payload or {}).get("text", "")
    query = text[:20]
    print("query =", repr(query))

    r1 = await retrieve_chunks(query=query, top_k=3)  # admin，无任何过滤
    print(f"admin no-filter            -> {len(r1)}")
    r2 = await retrieve_chunks(query=query, top_k=3, tenant_id="default",
                               owner_id="fe929bd6-1559-4a31-bd7d-01b415c5e118")
    print(f"owner + tenant=default     -> {len(r2)}")
    r3 = await retrieve_chunks(query=query, top_k=3, tenant_id="default")  # 无 owner：private 不可见
    print(f"tenant=default (no owner)  -> {len(r3)} (expect 0: private ACL)")
    r4 = await retrieve_chunks(query=query, top_k=3, tenant_id="company_A",
                               owner_id="fe929bd6-1559-4a31-bd7d-01b415c5e118")
    print(f"owner + tenant=company_A   -> {len(r4)} (expect 0: cross-tenant)")


asyncio.run(main())
