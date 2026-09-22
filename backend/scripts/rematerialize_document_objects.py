"""
对象级权限行的**重跑入口**（发现问题 #5：物化失败曾经只写日志、文档照常 COMPLETED）.

背景
----
入库收尾会把这份文档的五类对象（doc / text_chunk / table / code / image）写进
``document_objects``，作为对象级判定（第 11 / 12 环的 ``allows()``、``GET
/documents/{id}/chunks`` 的逐块复核）的权威视图。

这一步失败时**文档仍然标记 COMPLETED**（这是有意的：PG 仍是权威源，权限副本可
异步追平，不该让整份文档判失败）。但 ``filter_chunks_by_acl`` 对"从未物化"的文档
按"缺行回退允许"处理 —— 于是这份文档的对象级保护**永久失效**，而界面、检索、
监控全部正常。修复后这种失败会显式打上 ``acl_sync_state='stale'``，本脚本负责重跑。

怎么找受影响的文档
------------------
    SELECT id, filename, acl_sync_state FROM documents WHERE acl_sync_state <> 'synced';

（``ix_documents_acl_sync_state`` 是该查询的索引。）

本脚本以 **PostgreSQL 的 documents 行为权威**，从 Qdrant 取回该文档的向量点，
重建全部对象行；成功后把 ``acl_sync_state`` 复位为 ``synced``。

用法（容器内）
--------------
    # 1) 先看报告，不写任何东西（默认 dry-run）
    docker exec -e HOME=/tmp rag_backend sh -c "cd /app && python scripts/rematerialize_document_objects.py"

    # 2) 单份排查
    docker exec -e HOME=/tmp rag_backend sh -c "cd /app && python scripts/rematerialize_document_objects.py --doc <uuid>"

    # 3) 确认无误后真正写入
    docker exec -e HOME=/tmp rag_backend sh -c "cd /app && python scripts/rematerialize_document_objects.py --apply"
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def _points_for(client, collection: str, doc_id: str) -> list[dict]:
    """滚动取回某文档的全部向量点（``materialize_document_objects`` 需要的形状）."""
    from qdrant_client.http import models as qmodels

    out: list[dict] = []
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id", match=qmodels.MatchValue(value=doc_id)
                    )
                ]
            ),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points or []:
            out.append({"id": str(p.id), "payload": dict(p.payload or {})})
        if offset is None or not points:
            break
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正写入修复（默认只出报告，不修改任何数据）",
    )
    parser.add_argument("--doc", default=None, help="只处理指定 document_id（排查用）")
    args = parser.parse_args()

    from sqlalchemy import select, update

    from app.config import get_settings
    from app.db.models import ChunkParent, Document
    from app.db.postgres import get_db_session
    from app.db.qdrant import get_qdrant_client

    settings = get_settings()
    client = get_qdrant_client()

    # ── 1. 找出需要重跑的文档 ────────────────────────────────────────────────
    def _base_stmt():
        return select(
            Document.id, Document.filename, Document.access_level,
            Document.department_id, Document.acl_sync_state,
        )

    if args.doc:
        import uuid as _uuid

        stmt = _base_stmt().where(Document.id == _uuid.UUID(str(args.doc)))
    else:
        stmt = _base_stmt().where(Document.acl_sync_state != "synced")
    async with get_db_session() as session:
        rows = (await session.execute(stmt)).all()

    print(f"[objremat] 待处理文档 = {len(rows)}（acl_sync_state != 'synced'）")
    if not rows:
        print(
            "\n[objremat] 没有待处理文档。\n"
            "           若用户仍在报『原文预览为空 / 该文档检索不到』，"
            "请改跑\n"
            "           scripts/repair_object_access_level.py（PG 内部两行分叉）"
            "或\n"
            "           scripts/repair_acl_payload.py（PG ↔ Qdrant 分叉）。"
        )
        return

    for doc_id, filename, level, dept, state in rows[:40]:
        print(
            f"    - {filename or '-'}  {doc_id}\n"
            f"        access_level={level} dept={dept} acl_sync_state={state}"
        )
    if len(rows) > 40:
        print(f"    ... 其余 {len(rows) - 40} 份省略")

    if not args.apply:
        print(
            "\n[objremat] DRY-RUN — 未修改任何数据。"
            "确认上面列出的文档之后再追加 --apply。"
        )
        return

    # ── 2. 逐份重建（以 PG 的 documents 行为权威）───────────────────────────
    from app.services.security_cascade import materialize_document_objects

    fixed = failed = 0
    for doc_id, filename, _level, _dept, _state in rows:
        doc_key = str(doc_id)
        try:
            points = await _points_for(client, settings.QDRANT_COLLECTION, doc_key)
            async with get_db_session() as session:
                doc = (
                    await session.execute(select(Document).where(Document.id == doc_id))
                ).scalar_one_or_none()
                parents = (
                    await session.execute(
                        select(ChunkParent).where(ChunkParent.document_id == doc_id)
                    )
                ).scalars().all() if doc is not None else []
            if doc is None:
                print(f"    !! {filename} ({doc_key}) 文档已不存在，跳过")
                failed += 1
                continue
            if not points:
                # 没有向量点 = 入库没有产出任何块 → 对象行无从重建，保持 stale
                print(f"    !! {filename} ({doc_key}) 向量库中无该文档的点，保持 stale")
                failed += 1
                continue
            stats = await materialize_document_objects(
                doc, points, parents=parents, push_payload=False
            )
            async with get_db_session() as session:
                await session.execute(
                    update(Document)
                    .where(Document.id == doc_id)
                    .values(acl_sync_state="synced")
                )
            fixed += 1
            print(f"    ok {filename} ({doc_key}) rows={stats}")
        except Exception as exc:      # noqa: BLE001 — 单份失败不影响其余
            failed += 1
            print(f"    !! {filename} ({doc_key}) 重跑失败: {exc}")

    print(f"\n[objremat] done: fixed={fixed} failed={failed} / total={len(rows)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
