"""删除旧索引并重新入库 实战.docx（容器内运行）."""
import asyncio
import sys
import uuid


async def main(docx_path: str) -> None:
    from sqlalchemy import select

    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.db.user_models import User
    from app.services.document_query_service import delete_document
    from app.services.document_service import process_uploads

    async with get_db_session() as s:
        rows = (await s.execute(
            select(Document.id, Document.filename, Document.owner_id)
            .where(Document.filename == "实战.docx")
        )).all()
    print("existing rows:", rows)

    async with get_db_session() as s:
        admin = (await s.execute(select(User).where(User.role == "admin").limit(1))).scalar_one()

    for doc_id, filename, owner_id in rows:
        try:
            await delete_document(doc_id, owner_id=None)  # admin: 不限 owner
            print("deleted:", filename, doc_id)
        except Exception as exc:      # noqa: BLE001
            print("delete failed:", doc_id, exc)

    with open(docx_path, "rb") as fh:
        content = fh.read()

    resp = await process_uploads(
        [("实战.docx", content)],
        owner_id=admin.id,
        tenant_id=admin.tenant_id or "default",
        department_id=admin.department_id,
        access_level="tenant",
    )
    for r in resp.documents:
        print("ingest:", r.status, r.filename,
              "chunks:", getattr(r, "chunk_count", None),
              "images:", getattr(r, "image_count", None),
              "image_objects:", getattr(r, "image_object_count", None),
              "doc_id:", r.document_id,
              "msg:", getattr(r, "message", None))


asyncio.run(main(sys.argv[1]))
