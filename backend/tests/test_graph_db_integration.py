"""Opt-in real PostgreSQL/Qdrant checks on a disposable database copy, without models."""
import asyncio
import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_GRAPH_DB_TESTS") != "1", reason="requires disposable PostgreSQL database copy")


def test_real_graph_retry_acl_stale_source_and_delete(monkeypatch):
    from sqlalchemy import delete, func, select, update
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from app.config import get_settings
    from app.db.models import Document, DocumentChunkTerm, DocumentStatus
    from app.db.professional_models import GraphEntity, GraphIndex, GraphRelation
    from app.db.security_models import DocumentObject
    from app.db.postgres import get_db_session
    from app.db.user_models import User
    from app.db.qdrant import get_qdrant_client
    from app.services.security_cascade import materialize_document_objects
    from app.services.security_scope import request_security_scope
    from app.services import graph_retrieval as graph

    settings = get_settings()
    assert settings.POSTGRES_DB.endswith("_verify"), "refuse to mutate the business database"
    collection = "graph_test_" + uuid.uuid4().hex
    monkeypatch.setattr(settings, "QDRANT_COLLECTION", collection)

    async def run():
        doc_id, point_id = uuid.uuid4(), str(uuid.uuid4())
        client = get_qdrant_client()
        await client.create_collection(collection, vectors_config=VectorParams(size=1, distance=Distance.COSINE))
        try:
            async with get_db_session() as session:
                admin = await session.scalar(select(User).where(User.username == "admin"))
                outsider = await session.scalar(select(User).where(User.id != admin.id).order_by(User.id).limit(1))
                assert admin and outsider
                document = Document(id=doc_id, filename="graph-integration.md", file_size=12, file_hash=uuid.uuid4().hex,
                    owner_id=admin.id, tenant_id="default", access_level="private", status=DocumentStatus.COMPLETED,
                    chunk_count=1, security_level=0, visibility_mode="tier", acl_sync_state="synced")
                session.add(document)
                await session.flush()
                session.add(DocumentChunkTerm(document_id=doc_id, tenant_id="default", access_level="private",
                    chunk_index=0, point_id=point_id, terms="甲 投资 乙"))
            payload = dict(document_id=str(doc_id), filename="graph-integration.md", chunk_index=0, page_number=1,
                           text="甲投资乙", content_type="text", owner_id=str(admin.id), tenant_id="default", access_level="private")
            await client.upsert(collection, [PointStruct(id=point_id, vector=[1.0], payload=payload)])
            await materialize_document_objects(document, [{"id": point_id, "payload": payload}], push_payload=False)
            scope = await request_security_scope(admin)
            foreign_scope = await request_security_scope(outsider)
            async def fail(text): raise ValueError("interrupted extractor")
            monkeypatch.setattr(graph, "extract", fail)
            assert (await graph.index_step(doc_id, scope))["status"] == "failed"
            async def extract(text):
                return graph.validated_relations([dict(subject="甲", subject_type="organization", predicate="投资",
                    object="乙", object_type="organization", evidence=text)], text)
            monkeypatch.setattr(graph, "extract", extract)
            result = await graph.index_step(doc_id, scope)
            assert result["status"] == "completed" and result["cursor"] == 1
            assert len((await graph.graph_candidates("甲", scope, document_id=doc_id))[1]) == 1
            assert await graph.graph_candidates("甲", foreign_scope, document_id=doc_id) == ([], [])
            async with get_db_session() as session:
                await session.execute(update(DocumentObject).where(DocumentObject.document_id == doc_id)
                                      .values(acl_deny=[f"user:{admin.id}"]))
            assert await graph.graph_candidates("甲", scope, document_id=doc_id) == ([], [])
            async with get_db_session() as session:
                await session.execute(update(DocumentObject).where(DocumentObject.document_id == doc_id).values(acl_deny=[]))
            await client.set_payload(collection, {"text": "甲未投资乙"}, points=[point_id])
            assert await graph.graph_candidates("甲", scope, document_id=doc_id) == ([], [])
            assert (await graph.index_step(doc_id, scope, restart=True))["status"] == "completed"
            assert await graph.graph_candidates("甲", scope, document_id=doc_id) == ([], [])
            async with get_db_session() as session:
                await session.execute(delete(Document).where(Document.id == doc_id))
            async with get_db_session() as session:
                for model in (GraphIndex, GraphEntity, GraphRelation):
                    assert await session.scalar(select(func.count()).select_from(model).where(model.document_id == doc_id)) == 0
        finally:
            try:
                async with get_db_session() as session:
                    await session.execute(delete(Document).where(Document.id == doc_id))
            finally:
                await client.delete_collection(collection)
    asyncio.run(run())
