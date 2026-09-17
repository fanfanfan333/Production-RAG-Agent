"""只看**召回**：对给定 query 直接跑检索层，打印命中的 chunk_index / 类型 / 分数.

用途：区分"检索没找到"和"找到了但模型不肯答"。

    docker exec -e HOME=/tmp rag_backend python /tmp/ragscripts/diag_retrieval.py "GDB 条件断点"
"""
from __future__ import annotations

import asyncio
import sys

QUERY = sys.argv[1] if len(sys.argv) > 1 else "GDB 条件断点"
TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 8


async def main() -> None:
    from app.services.retrieval_service import retrieve_chunks

    hits = await retrieve_chunks(QUERY, top_k=TOP_K)
    print(f"query={QUERY!r} → {len(hits)} 命中")
    for i, h in enumerate(hits):
        md = getattr(h, "metadata", None) or {}
        text = (getattr(h, "text", "") or "").replace("\n", " ")
        print(
            f"[{i}] idx={md.get('chunk_index')} type={md.get('content_type')} "
            f"score={getattr(h, 'score', None)} "
            f"fn={str(md.get('filename'))[:26]}\n     {text[:180]}"
        )


asyncio.run(main())
