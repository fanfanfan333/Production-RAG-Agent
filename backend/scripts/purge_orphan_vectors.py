"""回收 Qdrant 里的**孤儿向量**：payload.document_id 在 PostgreSQL ``documents`` 里已不存在.

为什么需要它
────────────
向量库与关系库是两套存储，删除必须两边都做。本仓库的历史清理脚本（e2e 收尾等）
只删了 PG 行 → 向量留下。实测生产库：PG 9 份文档，Qdrant 214 点 / 186 个
document_id，**约 95% 是孤儿**。

孤儿为什么"要治"而不是"忍一忍"：
  * 检索是"先 ANN 取候选、再按 PG 可见性过滤"。孤儿点会**真实占用候选名额** ——
    实测平台管理员（不做租户过滤）ANN top-20 里只有 4 条属于现存文档，80% 名额
    被吃掉，用户自己的分片因此进不了候选池；
  * ``POST /eval/run`` 以调用者身份走同一条检索链路，于是**评测指标被脏数据压低**，
    把排查方向带偏到"检索算法不行"；
  * 集合的点数/段数/磁盘占用按脏数据算，容量规划跟着失真。

默认 dry-run，只报告不删除。真正的删除要显式 ``--delete``：
破坏性动作不该是"忘了加参数"的默认结果。

    docker cp scripts/purge_orphan_vectors.py rag_backend:/tmp/
    docker exec rag_backend python /tmp/purge_orphan_vectors.py            # 只看
    docker exec rag_backend python /tmp/purge_orphan_vectors.py --delete   # 真删
"""
from __future__ import annotations

import asyncio
import sys
from collections import Counter


async def collect_qdrant_ids() -> Counter:
    """扫全集合，返回 {document_id: 点数}。"""
    import json
    import urllib.request

    from app.config import get_settings

    settings = get_settings()
    base = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
    counts: Counter = Counter()
    offset = None
    while True:
        body: dict = {"limit": 512, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(
            f"{base}/collections/{settings.QDRANT_COLLECTION}/points/scroll",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            page = json.loads(resp.read())["result"]
        points = page.get("points", [])
        if not points:
            break
        for p in points:
            did = (p.get("payload") or {}).get("document_id")
            counts[str(did) if did else "__missing_document_id__"] += 1
        offset = page.get("next_page_offset")
        if offset is None:
            break
    return counts


async def main(do_delete: bool) -> int:
    from sqlalchemy import text

    from app.db.postgres import get_db_session

    q_counts = await collect_qdrant_ids()
    async with get_db_session() as session:
        live = {r[0] for r in (await session.execute(text("select id::text from documents"))).all()}

    # payload 里连 document_id 都没有的点单列：它们**无法**被 PG 侧任何校验救回，
    # 也永远进不了 valid_docs，属于纯垃圾。先摘出来，避免它污染"孤儿"的口径。
    missing = q_counts.pop("__missing_document_id__", 0)
    total_points = sum(q_counts.values()) + missing

    orphans = {d: c for d, c in q_counts.items() if d not in live}
    orphan_points = sum(orphans.values())

    print(f"collection points      : {total_points}")
    print(f"distinct document_ids  : {len(q_counts)}")
    print(f"live documents (PG)    : {len(live)}")
    print(f"orphan document_ids    : {len(orphans)}")
    print(f"orphan points          : {orphan_points} "
          f"({orphan_points / max(1, total_points):.1%} of collection)")
    if missing:
        print(f"points without document_id: {missing} (unmatchable by design)")

    if not orphans:
        print("\n没有孤儿向量，无需清理。")
        return 0

    print("\nsample orphan document_ids (up to 10):")
    for did, cnt in list(sorted(orphans.items(), key=lambda kv: -kv[1]))[:10]:
        print(f"  {did}  {cnt} point(s)")

    if not do_delete:
        print("\ndry-run: 什么都没删。确认无误后再加 --delete。")
        return 0

    from app.services.vector_service import delete_by_document_ids

    ids = list(orphans.keys())
    await delete_by_document_ids(ids)

    after = await collect_qdrant_ids()
    after_orphan = sum(c for d, c in after.items() if d not in live)
    print(f"\ndeleted {len(ids)} orphan document_id(s).")
    print(f"collection points now  : {sum(after.values())} (was {total_points})")
    print(f"orphan points now      : {after_orphan} (was {orphan_points})")
    return 0


if __name__ == "__main__":
    asyncio.run(main(do_delete="--delete" in sys.argv))
