"""决策 15：Query Rewrite / 子查询的 Scope 继承（契约 + 注入风险 + 阳性对照）.

覆盖设计文档 §13 T3 验收要点 8/9/10/11：

  8.  改写器签名不含任何 Scope 参数；``query_transform.py`` 源码不含权限模块符号；
      ``ScopedQuery.with_text()`` 后 ``scope is`` 原实例。
  9.  改写器注入风险：monkeypatch 改写器产出"其他公司名"的 query，断言检索仍被
      原 Scope 约束（tenant + 密级两层）；并配**阳性对照**（放宽 Scope 后必须能
      召回），否则该测试会因链路本就坏了而假通过。
  10. 同一请求内 main / variant / subquery / hyde 送进检索器时消费的
      ``ScopePredicate`` 是**同一个对象**（``id()`` 集合大小为 1，不是 ``==``）。
  11. 改写缓存 key **不带**指纹；``RewriteResult`` 不含任何权限字段。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

from app.services.query_transform import RewriteResult
from app.services.retrieval_service import ScopedQuery
from app.services.security_policy import qdrant_filter_matches, to_qdrant
from app.services.security_scope import UserScope
from app.services.tenancy import DocumentScope

TENANT_A = "c309a7cb9f496"
TENANT_B = "cf33b1db5679d"
DEPT_D1 = "d001"

_SRC = Path(__file__).resolve().parents[1] / "app" / "services" / "query_transform.py"


def _scope(
    *,
    user_id: str | None = None,
    tenant_ids=frozenset({TENANT_A}),
    clearance: int = 1,
    owns_tenant_ids=frozenset(),
    department_id: str | None = DEPT_D1,
    project_ids=frozenset(),
    principals=frozenset(),
    strict: bool = False,
) -> UserScope:
    uid = user_id or str(uuid.uuid4())
    base = DocumentScope(
        owner_id=uuid.UUID(uid),
        tenant_ids=tenant_ids,
        owns_tenant_ids=owns_tenant_ids,
        department_id=department_id,
        tenant_wide=False,
    )
    return UserScope(
        base=base,
        user_id=uid,
        role="employee",
        clearance=clearance,
        project_ids=project_ids,
        principals=principals,
        strict=strict,
    )


# ── 8-a. ScopedQuery.with_text 不换 scope ─────────────────────────────────────

def test_with_text_keeps_same_scope_instance():
    scope = _scope(clearance=1, project_ids=frozenset({"p_alpha"}))
    sq = ScopedQuery(text="我们部门今年的差旅标准", scope=scope)

    rewritten = sq.with_text("A公司 研发部 2025 差旅报销标准")   # 改写器可能补出别的公司/部门

    # ① 同一实例（is，不是 ==）
    assert rewritten.scope is scope
    # ② 权限属性原样（改写只改文本）
    pred = rewritten.scope.predicate()
    assert pred.clearance == 1
    assert pred.project_ids == frozenset({"p_alpha"})
    assert pred.tenant_ids == frozenset({TENANT_A})
    # ③ text 确实换了、原对象未被改动（frozen）
    assert rewritten.text == "A公司 研发部 2025 差旅报销标准"
    assert sq.text == "我们部门今年的差旅标准"


def test_with_text_has_no_scope_parameter():
    """``with_text`` 的参数表里**没有** scope —— 想换也换不了（结构保证）。"""
    import inspect

    params = set(inspect.signature(ScopedQuery.with_text).parameters)
    assert params == {"self", "new_text"}, params


# ── 8-b. 改写产物镜像：main / variant / subquery / hyde 共用同一 scope ────────

def test_rewrite_node_mirrors_scoped_queries():
    import app.services.master_graph as mg

    scope = _scope()

    async def fake_rewrite(query, history_messages=None):
        return RewriteResult(
            rewritten="改写后的问题",
            variants=["变体A", "变体B"],
            subqueries=["子问题1"],
            hyde="假设答案段落" * 4,
            source="llm",
        )

    state = {"query": "原问题", "history_messages": [], "user_scope": scope}
    with patch.object(mg, "rewrite_query", fake_rewrite):
        out = asyncio.run(mg._rewrite_node(state))

    # 旧的三条 str 通道保留不动（兼容既有拓扑 / 日志）
    assert out["rewritten_query"] == "改写后的问题"
    assert out["query_extra"], out["query_extra"]

    scq = out["scoped_queries"]
    # 主查询 + 每条 extra + HyDE
    assert scq[0].text == "改写后的问题" and scq[0].kind == "main"
    assert scq[-1].text == "假设答案段落" * 4 and scq[-1].kind == "hyde"
    assert scq[0].text not in {q.text for q in scq[1:]}


def test_all_derived_queries_share_one_scope_instance():
    """决策 15-5：派生 query 与父 query 共享**同一个** ScopePredicate 实例."""
    import app.services.master_graph as mg

    scope = _scope()

    async def fake_rewrite(query, history_messages=None):
        return RewriteResult(
            rewritten="改写后的问题",
            variants=["变体A"],
            subqueries=["子问题1"],
            hyde="假设答案段落" * 4,
            source="llm",
        )

    state = {"query": "原问题", "history_messages": [], "user_scope": scope}
    with patch.object(mg, "rewrite_query", fake_rewrite):
        out = asyncio.run(mg._rewrite_node(state))

    scq = out["scoped_queries"]
    kinds = {q.kind for q in scq}
    assert "main" in kinds and "hyde" in kinds        # 至少覆盖两类派生
    # 用 id() 集合大小断言"同一实例"，而不是 ==（== 会被值相等蒙混过关）
    scope_ids = {id(q.scope) for q in scq}
    assert scope_ids == {id(scope)}, "派生 query 必须共享同一个 scope 实例"


# ── 8-c. 改写器拿不到 Scope（源码级断言，防反向泄密）─────────────────────────

def test_query_transform_has_no_scope_dependency():
    src = _SRC.read_text(encoding="utf-8")
    for forbidden in ("security_scope", "security_policy", "UserScope", "clearance"):
        assert forbidden not in src, f"query_transform.py 不得出现 {forbidden!r}（防反向泄密）"


def test_rewrite_result_has_no_permission_fields():
    """决策 15-6 前提：RewriteResult 只含字符串、不含任何权限字段."""
    names = {f.name for f in fields(RewriteResult)}
    assert names == {"rewritten", "variants", "subqueries", "hyde", "source", "drift_score"}, names


def test_rewrite_cache_key_keeps_no_fingerprint():
    """决策 15-6：改写缓存 key 刻意**不带** scope 指纹（否则命中率≈0）."""
    src = _SRC.read_text(encoding="utf-8")
    assert "history_fingerprint(history_messages)" in src
    assert "scope_fingerprint" not in src


# ── 9. 注入风险：改写器补出"别的公司名"也不越权（含阳性对照）─────────────────

class _FakeQdrant:
    """只记录 filter、返回空命中的 Qdrant 替身（向量腿在 ANN 后即早退，无需 DB）."""

    def __init__(self) -> None:
        self.filters: list = []

    async def search(self, *, collection_name, query_vector, limit, with_payload, query_filter):
        self.filters.append(query_filter)
        return []

    async def retrieve(self, **_kw):
        return []

    async def scroll(self, **_kw):
        return ([], None)


def _patch_retrieval(monkeypatch, fake: _FakeQdrant, seen_preds: list):
    import app.services.retrieval_service as rs

    monkeypatch.setattr(rs, "get_qdrant_client", lambda: fake)

    async def _embed(texts, task_type=None):
        return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(rs, "embed_batch_with_retry", _embed)

    async def _passthrough(tenant_ids, owns_tenant_ids):
        return tenant_ids, owns_tenant_ids

    monkeypatch.setattr(rs, "exclude_test_tenants", _passthrough)

    real_to_qdrant = rs.to_qdrant

    def spy_to_qdrant(pred, **kw):
        seen_preds.append(pred)
        return real_to_qdrant(pred, **kw)

    monkeypatch.setattr(rs, "to_qdrant", spy_to_qdrant)


def test_rewritten_query_cannot_escape_scope(monkeypatch):
    """改写器被诱导产出含"B公司"的 query —— 召回仍被原 Scope 约束."""
    import app.services.retrieval_service as rs

    fake = _FakeQdrant()
    seen: list = []
    _patch_retrieval(monkeypatch, fake, seen)

    narrow = _scope(tenant_ids=frozenset({TENANT_A}), clearance=1)
    # 注入：检索用的文本明确含"B 公司 / 薪酬方案"（模拟改写器补出别的公司名）
    asyncio.run(rs.retrieve_chunks_scoped(
        "B公司 2025 年度薪酬方案实施细则", scope=narrow, top_k=20,
    ))

    assert seen, "检索未经过 to_qdrant —— 前置下推缺失（过滤发生在检索侧才安全）"
    pred = seen[-1]
    # ① 过滤用的 predicate 与原 scope 绑定，**与查询文本无关**
    assert pred.clearance == 1
    assert pred.tenant_ids == frozenset({TENANT_A})

    # ② B 公司的对象被排除（跨租户）；超密级对象被排除（clearance 闸门）
    payload_b = {
        "object_id": "b1", "tenant_id": TENANT_B, "access_level": "tenant",
        "user_id": "someone-else",
    }
    payload_hi = {
        "object_id": "h1", "tenant_id": TENANT_A, "access_level": "tenant",
        "effective_security_level": 3,
    }
    assert qdrant_filter_matches(pred, payload_b) is False, "B 公司对象未被排除"
    assert qdrant_filter_matches(pred, payload_hi) is False, "超密级对象未被排除"

    # ③ 阳性对照：放宽 Scope（加 B 公司 + clearance=3）后，同一载荷必须"能"被召回。
    #    没有这一步，"断言全为 False"的测试会因为过滤器恒拒绝而假通过。
    seen.clear()
    wide = _scope(tenant_ids=frozenset({TENANT_A, TENANT_B}), clearance=3)
    asyncio.run(rs.retrieve_chunks_scoped(
        "B公司 2025 年度薪酬方案实施细则", scope=wide, top_k=20,
    ))
    pred_w = seen[-1]
    assert qdrant_filter_matches(pred_w, payload_b) is True, "阳性对照失败：放宽后仍召回不到 B 公司对象"
    assert qdrant_filter_matches(pred_w, payload_hi) is True, "阳性对照失败：放宽密级后仍召回不到"
    # 两次检索用的过滤器必须不同（证明"过滤确实随 Scope 变化"）
    assert to_qdrant(pred) != to_qdrant(pred_w)


if __name__ == "__main__":      # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
