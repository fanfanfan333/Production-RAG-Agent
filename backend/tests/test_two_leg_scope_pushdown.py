"""双路同源下推（第 7 环，决策 6 / 10-①；设计 §13 T3 验收要点 1、2）.

  * **A2 / 验收 1**：同一请求内向量腿与关键词腿拿到的 Filter / SQL 由**同一个**
    ``ScopePredicate`` 实例编译 —— 断言 ``id()`` 集合大小为 1（不是 ``==``）。
  * **验收 2 fail-closed**：``retrieve_chunks_scoped`` 无 scope → 空结果；
    既有 ``retrieve_chunks`` 无任何权限上下文 → 空结果（不降级为不过滤）。

用假的 Qdrant / DB / embeddings 驱动真实的 ``retrieve_chunks_scoped``，全程不碰
真实数据库与向量库。
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from pathlib import Path

from app.services.retrieval_service import _visibility_conditions
from app.services.security_policy import to_qdrant
from app.services.security_scope import UserScope
from app.services.tenancy import DocumentScope

TENANT_A = "c309a7cb9f496"
DEPT_D1 = "d001"
DOC = str(uuid.uuid4())
UID = str(uuid.uuid4())


def _scope(
    *,
    user_id: str = UID,
    tenant_ids=frozenset({TENANT_A}),
    clearance: int = 1,
    department_id: str | None = DEPT_D1,
) -> UserScope:
    base = DocumentScope(
        owner_id=uuid.UUID(user_id),
        tenant_ids=tenant_ids,
        owns_tenant_ids=frozenset(),
        department_id=department_id,
        tenant_wide=False,
    )
    return UserScope(
        base=base, user_id=user_id, role="employee", clearance=clearance,
        project_ids=frozenset(), principals=frozenset(), strict=False,
    )


def _allowed_payload() -> dict:
    return {
        "object_id": "objA",
        "document_id": DOC,
        "chunk_index": 0,
        "filename": "差旅标准.pdf",
        "page_number": 1,
        "text": "本年度差旅报销标准……",
        "tenant_id": TENANT_A,
        "access_level": "tenant",
        "user_id": UID,
        "security_level": 1,
    }


class _Hit:
    def __init__(self, score: float, payload: dict) -> None:
        self.score = score
        self.payload = payload


class _FakeQdrant:
    def __init__(self, hits: list) -> None:
        self._hits = hits
        self.last_filter = None

    async def search(self, *, collection_name, query_vector, limit, with_payload, query_filter):
        self.last_filter = query_filter
        return self._hits

    async def retrieve(self, **_kw):
        return []

    async def scroll(self, **_kw):
        return ([], None)


class _FakeResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    async def execute(self, _stmt):
        return _FakeResult([(DOC,)])


class _FakeSessionCM:
    async def __aenter__(self):
        return _FakeSession()

    async def __aexit__(self, *_exc):
        return False


def _fake_get_db_session():
    return _FakeSessionCM()


def _wire(monkeypatch, rs, *, hits):
    """把 retrieve_chunks 的外部依赖换成离线替身。"""
    fake = _FakeQdrant(hits)
    monkeypatch.setattr(rs, "get_qdrant_client", lambda: fake)

    async def _embed(texts, task_type=None):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(rs, "embed_batch_with_retry", _embed)

    async def _passthrough(tenant_ids, owns_tenant_ids):
        return tenant_ids, owns_tenant_ids

    monkeypatch.setattr(rs, "exclude_test_tenants", _passthrough)

    import app.db.postgres as pg

    monkeypatch.setattr(pg, "get_db_session", _fake_get_db_session)

    # 简化路径：单腿、不精排、不层级、不可信度加权（避免额外依赖）
    settings = rs.get_settings()
    for name, value in (
        ("HYBRID_SEARCH_ENABLED", True),
        ("HYBRID_KEYWORD_BACKEND", "postgres"),
        ("RERANKER_ENABLED", False),
        ("HIERARCHICAL_RAG_ENABLED", False),
        ("EVIDENCE_TRUST_ENABLED", False),
        ("RETRIEVAL_CONTENT_DEDUP_ENABLED", False),
        ("ANN_OVERFETCH_FACTOR", 1),
    ):
        monkeypatch.setattr(settings, name, value, raising=False)
    async def _audit(*args, **kwargs):
        pass

    monkeypatch.setattr(rs, "_record_acl_drops", _audit)
    return fake


def test_both_legs_compile_from_one_predicate(monkeypatch):
    """A2：向量腿 to_qdrant(pred) 与 PG 腿 keyword_search(pred=...) 用同一个实例."""
    import app.services.retrieval_service as rs

    fake = _wire(monkeypatch, rs, hits=[_Hit(0.9, _allowed_payload())])

    seen_qdrant: list = []
    seen_pg: list = []
    real_to_qdrant = rs.to_qdrant

    def spy_qdrant(pred, **kw):
        seen_qdrant.append(pred)          # 保留引用，防止 id() 复用
        return real_to_qdrant(pred, **kw)

    async def fake_pg(**kw):
        seen_pg.append(kw.get("pred"))    # 入口编译好的同一个 ScopePredicate
        return []

    monkeypatch.setattr(rs, "to_qdrant", spy_qdrant)
    monkeypatch.setattr(rs, "_pg_keyword_candidates", fake_pg)

    chunks = asyncio.run(rs.retrieve_chunks_scoped("我们公司差旅标准", scope=_scope(), top_k=5))

    assert [c.document_id for c in chunks] == [DOC], "向量腿应召回可见对象"
    assert seen_pg, "PG 关键词腿未运行（HYBRID_KEYWORD_BACKEND 应为 postgres）"
    assert seen_qdrant and seen_pg
    # 同一 ScopePredicate **实例**（is / id()，不是 ==）
    assert seen_pg[0] is not None, "PG 腿必须收到已编译的 pred（而不是 None/重新现编）"
    assert len({id(p) for p in seen_qdrant + seen_pg}) == 1, "两腿必须共用同一个 ScopePredicate 实例"
    assert seen_qdrant[0] is seen_pg[0]


def test_vector_leg_pushes_down_scope_filter(monkeypatch):
    """向量腿的 ANN 过滤器 == to_qdrant(pred)（过滤发生在检索侧，不是召回后）."""
    import app.services.retrieval_service as rs

    fake = _wire(monkeypatch, rs, hits=[_Hit(0.9, _allowed_payload())])
    scope = _scope()
    asyncio.run(rs.retrieve_chunks_scoped("q", scope=scope, top_k=5))

    predicate = scope.predicate()
    expected = to_qdrant(predicate)
    assert fake.last_filter is not None
    assert list(fake.last_filter.must_not or []) == list(expected.must_not or [])


def test_scoped_entry_fail_closed_without_scope(monkeypatch):
    import app.services.retrieval_service as rs

    monkeypatch.setattr(rs, "get_qdrant_client", lambda: _FakeQdrant([]))
    assert asyncio.run(rs.retrieve_chunks_scoped("q", scope=None)) == []


def test_legacy_entry_fail_closed_without_context(monkeypatch):
    """兼容入口的既有 fail-closed 必须继续生效：漏传权限 = 空结果，不是全库."""
    import app.services.retrieval_service as rs

    monkeypatch.setattr(rs, "get_qdrant_client", lambda: _FakeQdrant([]))
    assert asyncio.run(rs.retrieve_chunks("q", top_k=5)) == []


def test_visibility_conditions_still_reachable_for_legacy_paths():
    """旧三维函数仍在（兼容），且 to_qdrant 复用它（不是另写一份）."""
    conds = _visibility_conditions(
        owner_id=UID, department_id=DEPT_D1, tenant_wide=False,
        tenant_ids=frozenset({TENANT_A}), owns_tenant_ids=frozenset(),
    )
    assert conds, "既有三维条件不应为空"


if __name__ == "__main__":      # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))


@pytest.mark.parametrize("backend", ["postgres", "memory"])
@pytest.mark.parametrize("ann_hit", [False, True])
def test_keyword_only_document_survives_independent_recall(monkeypatch, backend, ann_hit):
    import app.services.retrieval_service as rs
    import app.db.postgres as pg

    other = str(uuid.uuid4())
    payload = {**_allowed_payload(), "document_id": other, "text": "ZX9001 设备维修标准"}
    _wire(monkeypatch, rs, hits=[_Hit(0.9, _allowed_payload())] if ann_hit else [])
    monkeypatch.setattr(rs.get_settings(), "HYBRID_KEYWORD_BACKEND", backend)
    statements = []

    class Session(_FakeSession):
        async def execute(self, statement):
            statements.append(statement)
            return _FakeResult([(DOC,), (other,)])

    class CM(_FakeSessionCM):
        async def __aenter__(self):
            return Session()

    monkeypatch.setattr(pg, "get_db_session", CM)

    async def keywords(**kwargs):
        return [(other, 0, "point-b"), (other, 0, "point-b")]

    async def payloads(*args):
        return {"point-b": payload}

    async def corpus(*args, **kwargs):
        return [rs.RetrievedChunk(document_id=other, filename="b.pdf", page_number=1,
                                 chunk_index=0, text=payload["text"], score=0,
                                 _acl_payload=payload)]

    monkeypatch.setattr(rs, "_pg_keyword_candidates", keywords)
    monkeypatch.setattr(rs, "_fetch_by_point_ids", payloads)
    monkeypatch.setattr(rs, "_scroll_corpus", corpus)
    rs._bm25_cache.clear()
    try:
        chunks = asyncio.run(rs.retrieve_chunks_scoped("ZX9001", scope=_scope(), top_k=5))
        assert sum(c.document_id == other for c in chunks) == 1
        params = statements[0].compile().params
        assert any(isinstance(v, (list, set)) and uuid.UUID(other) in v for v in params.values())
    finally:
        rs._bm25_cache.clear()


@pytest.mark.parametrize("backend", ["postgres", "memory"])
@pytest.mark.parametrize("blocked", ["authority", "object_acl", "metadata", "hybrid_disabled"])
def test_independent_keyword_candidates_keep_existing_gates(monkeypatch, backend, blocked):
    import app.services.retrieval_service as rs
    import app.db.postgres as pg
    from app.services.metadata import MetadataFilter

    _wire(monkeypatch, rs, hits=[])
    settings = rs.get_settings()
    monkeypatch.setattr(settings, "HYBRID_KEYWORD_BACKEND", backend)
    monkeypatch.setattr(settings, "HYBRID_SEARCH_ENABLED", blocked != "hybrid_disabled")
    payload = {**_allowed_payload(), "doc_type": "制度"}
    if blocked == "object_acl":
        payload["security_level"] = 5
    calls = []

    async def keywords(**kwargs):
        calls.append(True)
        return [(DOC, 0, "point-a")]

    async def payloads(*args):
        return {"point-a": payload}

    async def corpus(*args, **kwargs):
        calls.append(True)
        return [rs.RetrievedChunk(document_id=DOC, filename="a.pdf", page_number=1,
                                 chunk_index=0, text="ZX9001", score=0, doc_type="制度", _acl_payload=payload)]

    class Session(_FakeSession):
        async def execute(self, statement):
            sql = str(statement)
            assert "status" in sql and "collection_id" in sql
            return _FakeResult([])

    class CM(_FakeSessionCM):
        async def __aenter__(self):
            return Session()

    if blocked == "authority":
        monkeypatch.setattr(pg, "get_db_session", CM)
    monkeypatch.setattr(rs, "_pg_keyword_candidates", keywords)
    monkeypatch.setattr(rs, "_fetch_by_point_ids", payloads)
    monkeypatch.setattr(rs, "_scroll_corpus", corpus)
    rs._bm25_cache.clear()
    try:
        kwargs = {"collection_id": str(uuid.uuid4())}
        if blocked == "metadata":
            kwargs["metadata_filter"] = MetadataFilter(doc_types=["合同"])
        chunks = asyncio.run(rs.retrieve_chunks_scoped("ZX9001", scope=_scope(), top_k=5, **kwargs))
        assert chunks == []
        if blocked == "hybrid_disabled":
            assert not calls
    finally:
        rs._bm25_cache.clear()
