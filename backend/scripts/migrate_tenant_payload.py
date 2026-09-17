"""
存量 Qdrant 向量的租户 payload 回填脚本（三层隔离迁移）.

背景：三层隔离上线后，检索的第一层过滤是 Qdrant payload 的
``tenant_id``（向量 ANN 与 BM25 语料 scroll 都带此前置条件）。
升级前写入的向量 payload 没有该字段 —— 不回填的话，存量文档对
所有非管理员用户直接"消失"（filter 匹配不到缺失字段）。

本脚本以 PostgreSQL 的 Document 行为权威来源，把
``tenant_id / user_id / access_level / department_id / source``
回填到对应向量的 payload 里。幂等：已有 tenant_id 的点会跳过。

用法（容器内）：
    docker exec -e HOME=/tmp rag_backend sh -c "cd /app && python scripts/migrate_tenant_payload.py"
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select


async def main() -> None:
    from app.config import get_settings
    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.db.qdrant import get_qdrant_client
    from qdrant_client.http import models as qmodels

    settings = get_settings()
    client = get_qdrant_client()

    # ── 1. PG 权威映射：document_id → 隔离载荷 ───────────────────────────────
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(
                    Document.id,
                    Document.tenant_id,
                    Document.owner_id,
                    Document.access_level,
                    Document.department_id,
                    Document.filename,
                )
            )
        ).all()
    doc_map = {
        str(doc_id): {
            "tenant_id": tenant or "default",
            "user_id": str(owner_id) if owner_id else None,
            "access_level": access or "private",
            "department_id": dept,
            "source": filename,
        }
        for doc_id, tenant, owner_id, access, dept, filename in rows
    }
    print(f"[migrate] {len(doc_map)} document rows loaded from PostgreSQL")

    # ── 2. 滚动全量向量，收集缺 tenant_id 的点 ───────────────────────────────
    updated = skipped = orphan = 0
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        pending: list[tuple[str, dict]] = []
        for p in points:
            payload = p.payload or {}
            if payload.get("tenant_id"):
                skipped += 1
                continue
            doc_id = str(payload.get("document_id") or "")
            patch = doc_map.get(doc_id)
            if patch is None:
                orphan += 1
                # 孤儿向量（PG 无对应文档）：归入 default 私有，避免消失
                patch = {
                    "tenant_id": "default",
                    "user_id": None,
                    "access_level": "private",
                    "department_id": None,
                    "source": payload.get("filename"),
                }
            pending.append((p.id, patch))

        # ── 3. 批量回填 ───────────────────────────────────────────────────────
        for point_id, patch in pending:
            await client.set_payload(
                collection_name=settings.QDRANT_COLLECTION,
                payload=patch,
                points=[point_id],
                wait=True,
            )
        updated += len(pending)
        print(f"[migrate] batch: +{len(pending)} patched (total={updated})")

        if offset is None:
            break

    print(
        f"[migrate] done: patched={updated} already_ok={skipped} orphan={orphan}"
    )


if __name__ == "__main__":
    asyncio.run(main())
