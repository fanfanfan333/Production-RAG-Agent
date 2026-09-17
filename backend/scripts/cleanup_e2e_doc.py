"""
清理 e2e 脚本产生的测试文档（容器内运行）.

e2e_image_chain.py 每次运行都会造一份同名 DOCX 并上传。由于这些上传的
owner_id 为 NULL，而 Postgres 的 UNIQUE(owner_id, file_hash) 对 NULL 不生效
（NULL 互不相等），判重不会拦住它们 —— 于是知识库里会堆积内容完全相同的
"e2e_image_chain.docx"，干扰真实检索与评测。

本脚本按文件名删除这类文档：PostgreSQL 行 + Qdrant 向量 + 落盘图片。

运行：
    docker cp scripts/cleanup_e2e_doc.py rag_backend:/tmp/
    docker exec rag_backend python /tmp/cleanup_e2e_doc.py [--dry-run]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

TARGET_FILENAME = "e2e_image_chain.docx"


async def main(dry_run: bool = True) -> int:
    from sqlalchemy import delete, select

    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.services.storage import delete_document_images
    from app.services.vector_service import delete_by_document_id

    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(Document).where(Document.filename == TARGET_FILENAME)
            )
        ).scalars().all()
        doc_ids = [str(d.id) for d in rows]
        print(f"found {len(doc_ids)} test document(s): {doc_ids}")
        if dry_run:
            print("dry-run: nothing deleted. Re-run with --delete to remove.")
            return 0

        for doc_id in doc_ids:
            try:
                await delete_by_document_id(doc_id)
                print(f"  qdrant vectors deleted: {doc_id}")
            except Exception as exc:      # noqa: BLE001
                print(f"  qdrant delete failed ({doc_id}): {exc}")
            try:
                delete_document_images(doc_id)
                print(f"  images deleted: {doc_id}")
            except Exception as exc:      # noqa: BLE001
                print(f"  image delete failed ({doc_id}): {exc}")

        await session.execute(delete(Document).where(Document.filename == TARGET_FILENAME))
        await session.commit()
        print(f"deleted {len(doc_ids)} postgres row(s)")

    print("cleanup done")
    return 0


if __name__ == "__main__":
    asyncio.run(main(dry_run="--delete" not in sys.argv))
