"""缓存分区（决策 9）：不同 Scope 不共用权限相关缓存.

  * ``cache_key_for_scope`` 对不同 clearance / tenant / project / principals 产出
    不同键；同一 Scope 稳定（哈希性质，不是约定）。
  * ``_bm25_candidates`` 的语料缓存键经 ``cache_key_for_scope`` 分区 —— 换
    clearance 即换键，语料索引不串味。
"""

from __future__ import annotations

import asyncio
import uuid

from app.services.retrieval_service import RetrievedChunk
from app.services.security_scope import UserScope, cache_key_for_scope
from app.services.tenancy import DocumentScope

TENANT_A = "c309a7cb9f496"
TENANT_B = "cf33b1db5679d"
DEPT_D1 = "d001"
DOC = str(uuid.uuid4())
UID = str(uuid.uuid4())


def _scope(
    *,
    user_id: str = UID,
    tenant_ids=frozenset({TENANT_A}),
    clearance: int = 1,
    department_id: str | None = DEPT_D1,
    project_ids=frozenset(),
    principals=frozenset(),
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
        project_ids=project_ids, principals=principals, strict=False,
    )


def test_cache_key_partitions_by_each_dimension():
    base = _scope()
    variants = {
        "clearance": _scope(clearance=2),
        "project": _scope(project_ids=frozenset({"p1"})),
        "principals": _scope(principals=frozenset({"role:employee"})),
        "tenant": _scope(tenant_ids=frozenset({TENANT_B})),
        "department": _scope(department_id="d999"),
        "user": _scope(user_id=str(uuid.uuid4())),
    }
    base_key = cache_key_for_scope(base, "k")
    for name, scope in variants.items():
        assert cache_key_for_scope(scope, "k") != base_key, f"{name} 维度未参与缓存分区"


def test_cache_key_stable_for_same_scope_shape():
    a = _scope(clearance=2, project_ids=frozenset({"p1", "p2"}))
    b = _scope(clearance=2, project_ids=frozenset({"p2", "p1"}))   # 集合顺序无关
    assert cache_key_for_scope(a, "k") == cache_key_for_scope(b, "k")


def test_cache_key_none_is_isolated_partition():
    assert cache_key_for_scope(None, "k").startswith("anon::")
    assert cache_key_for_scope(None, "k") != cache_key_for_scope(_scope(), "k")


def test_bm25_corpus_cache_is_partitioned_by_scope(monkeypatch):
    """换 clearance 即换 BM25 语料缓存键（决策 9：不串缓存）."""
    import app.services.retrieval_service as rs

    rs._bm25_cache.clear()
    chunks = [
        RetrievedChunk(
            document_id=DOC, filename="f.pdf", page_number=1,
            chunk_index=i, text=f"差旅 报销 标准 #{i}", score=0.0,
        )
        for i in range(3)
    ]

    async def _corpus(client, name, cid, mx, tenant_ids=None, pred=None):
        return chunks

    monkeypatch.setattr(rs, "_scroll_corpus", _corpus)

    s1 = _scope(clearance=1)
    s2 = _scope(clearance=2)
    for scope in (s1, s2):
        asyncio.run(rs._bm25_candidates(
            query="差旅报销标准", fetch_n=5, valid_docs={DOC},
            collection_name="docs", collection_id=None, owner_id=None,
            client=None, tenant_ids=None, scope=scope,
        ))

    keys = list(rs._bm25_cache.keys())
    assert any(k.startswith(s1.scope_fingerprint) for k in keys), keys
    assert any(k.startswith(s2.scope_fingerprint) for k in keys), keys
    assert s1.scope_fingerprint != s2.scope_fingerprint
    rs._bm25_cache.clear()


if __name__ == "__main__":      # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
