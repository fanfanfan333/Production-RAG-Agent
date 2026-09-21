"""第 12 环（assemble 时对象级 ACL 复核 + 引用快照）在**live 路径**上的接线验证.

设计 §1 环节 12 / 决策 16：一个 chunk 从检索候选走到"真正进入 LLM prompt"之间还隔着
精排 / 父块回填 / Vision / 压缩等多步。第 7 环（检索前下推）与第 11 环（检索后 PG
复核）保护的是**候选集**，拦不住 assemble 阶段从别处取来的正文。这里验证的是
**master_graph 的 assemble 调用点确实把 `pred/view_index/materialized` 传了下去** ——
即"守卫已接线"，而不只是 `build_context` 单个函数能过滤。

测试要点（全部离线，不碰 DB / Qdrant / LLM）：

  * **阳性对照（live 路径）**：驱动真实的 ``master_graph._multimodal_context_node``，
    state 带一个真实 ``UserScope``（clearance 低于某 chunk 的密级）⇒ 该 chunk 必须
    从返回的 context / sources 里消失，而放行的 chunk 存活。
  * **阴性对照**：同样的输入、去掉 ``user_scope`` ⇒ 两个 chunk 都在，且上下文与
    改动前 ``build_context(chunks, query=...)`` 的输出**逐字相同**（行为零变化）。
  * ``build_context`` 直接单测：``materialized`` 接受 ``bool``（广播）与
    ``{document_id: bool}``（逐文档），且 Mapping 中缺失的文档按 ``False`` 处理。
  * ``load_view_indexes`` 的批量键规则与单文档版逐条一致（用假 session，不发 SQL）。
"""

from __future__ import annotations

import asyncio
import uuid

from app.services.master_graph import _multimodal_context_node
from app.services.nodes.context_builder import build_context
from app.services.security_cascade import load_view_indexes
from app.services.security_policy import ObjectACLView
from app.services.security_scope import UserScope
from app.services.tenancy import DocumentScope
from app.services.retrieval_service import RetrievedChunk

TENANT_A = "c309a7cb9f496"
DEPT_D1 = "d001"
DOC = str(uuid.uuid4())
DOC_EMPTY = str(uuid.uuid4())
UID = str(uuid.uuid4())


# ── 夹具 ─────────────────────────────────────────────────────────────────────

def _scope(*, clearance: int = 1) -> UserScope:
    base = DocumentScope(
        owner_id=uuid.UUID(UID),
        tenant_ids=frozenset({TENANT_A}),
        owns_tenant_ids=frozenset(),
        department_id=DEPT_D1,
        tenant_wide=False,
    )
    return UserScope(
        base=base, user_id=UID, role="employee", clearance=clearance,
        project_ids=frozenset(), principals=frozenset({f"user:{UID}"}), strict=False,
    )


def _chunk(chunk_index: int, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=DOC, filename="差旅标准.pdf", page_number=1,
        chunk_index=chunk_index, text=text, score=score, content_type="text",
    )


def _view(object_id: str, level: int) -> ObjectACLView:
    return ObjectACLView(
        object_id=object_id, object_type="text_chunk", document_id=DOC,
        tenant_id=TENANT_A, owner_id=None, access_level="tenant",
        visibility_mode="tier", security_level=level,
        effective_security_level=level,
    )


def _settings(monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    for name, value in (
        ("MULTIMODAL_CONTEXT_ENABLED", False),
        ("HIERARCHICAL_RAG_ENABLED", False),
        ("CONTEXT_COMPRESSION_ENABLED", False),
    ):
        monkeypatch.setattr(settings, name, value, raising=False)
    return settings


# view_index：chunk 0 = 低密级（可见）；chunk 1 = 高密级（clearance=1 必须被剔除）
VIEW_INDEX = {DOC: {"ci:0": _view("objA", 1), "ci:1": _view("objB", 3)}}
MATERIALIZED = {DOC: True}


# ── 阳性对照：live 路径必须真的丢弃低于密级的 chunk ──────────────────────────

def test_live_assemble_path_drops_over_clearance_chunk(monkeypatch):
    import app.services.security_cascade as cascade

    _settings(monkeypatch)

    async def fake_load(document_ids, *, session=None):
        return dict(VIEW_INDEX), dict(MATERIALIZED)

    monkeypatch.setattr(cascade, "load_view_indexes", fake_load)

    chunk_a = _chunk(0, "本年度差旅报销标准：市内交通 80 元/天。", 0.9)
    chunk_b = _chunk(1, "绝密：并购底稿与谈判价格底线。", 0.8)

    state = {
        "chunks": [chunk_a, chunk_b],
        "query": "差旅标准",
        "top_k": 5,
        "user_scope": _scope(clearance=1),
    }
    result = asyncio.run(_multimodal_context_node(state))

    assert len(result["sources"]) == 1, (
        "clearance 低于密级的 chunk 必须被第 12 环剔除（live 路径未接线？）"
    )
    assert result["sources"][0]["chunk_index"] == 0
    assert "并购" not in result["multimodal_context"], "被剔除 chunk 的正文不得进入上下文"
    assert "差旅报销标准" in result["multimodal_context"]


def test_live_assemble_path_widening_scope_keeps_both(monkeypatch):
    """阳性对照的**反向**：把 clearance 抬到足够高 ⇒ 两个 chunk 都保留.

    这证明上一条的"消失"确实由**该 scope 的密级**驱动，而不是别的原因（例如
    view / materialized 解析失败导致的误伤）。
    """
    import app.services.security_cascade as cascade

    _settings(monkeypatch)

    async def fake_load(document_ids, *, session=None):
        return dict(VIEW_INDEX), dict(MATERIALIZED)

    monkeypatch.setattr(cascade, "load_view_indexes", fake_load)

    chunk_a = _chunk(0, "本年度差旅报销标准：市内交通 80 元/天。", 0.9)
    chunk_b = _chunk(1, "绝密：并购底稿与谈判价格底线。", 0.8)

    state = {
        "chunks": [chunk_a, chunk_b],
        "query": "差旅标准",
        "top_k": 5,
        "user_scope": _scope(clearance=3),
    }
    result = asyncio.run(_multimodal_context_node(state))
    assert len(result["sources"]) == 2


# ── 阴性对照：scope 缺失 ⇒ 行为逐字不变 ─────────────────────────────────────

def test_live_assemble_path_without_scope_is_byte_identical(monkeypatch):
    import app.services.security_cascade as cascade

    _settings(monkeypatch)

    called = {"n": 0}

    async def fake_load(document_ids, *, session=None):      # pragma: no cover - 不应被调用
        called["n"] += 1
        return {}, {}

    monkeypatch.setattr(cascade, "load_view_indexes", fake_load)

    chunk_a = _chunk(0, "本年度差旅报销标准：市内交通 80 元/天。", 0.9)
    chunk_b = _chunk(1, "绝密：并购底稿与谈判价格底线。", 0.8)

    state = {"chunks": [chunk_a, chunk_b], "query": "差旅标准", "top_k": 5}
    result = asyncio.run(_multimodal_context_node(state))

    baseline = build_context([chunk_a, chunk_b], query="差旅标准")
    assert len(result["sources"]) == 2, "无 scope 时不得剔除任何 chunk"
    assert result["multimodal_context"] == baseline.context, "无 scope 时必须与改动前逐字一致"
    assert called["n"] == 0, "无 scope 时不应触发任何视图解析（零额外查询）"


# ── build_context：materialized 的 bool vs Mapping 语义 ──────────────────────

def test_build_context_materialized_mapping_vs_bool(monkeypatch):
    _settings(monkeypatch)

    # 该 chunk 的视图键在 view_index 里**缺失** → 判定完全取决于 materialized。
    chunk = _chunk(5, "只有正文，无对象行。", 0.9)
    pred = _scope().predicate()
    empty_index = {DOC: {}}

    # bool True（默认）：缺失对象行 ⇒ fail-closed 丢弃
    out_bool_true = build_context(
        [chunk], pred=pred, view_index=empty_index, materialized=True
    )
    assert out_bool_true.dropped_count == 1 and out_bool_true.sources == []

    # Mapping {DOC: False}：该文档未物化 ⇒ 缺行回退允许
    out_map_false = build_context(
        [chunk], pred=pred, view_index=empty_index, materialized={DOC: False}
    )
    assert out_map_false.dropped_count == 0 and len(out_map_false.sources) == 1

    # Mapping {}：缺失的文档按 False 处理 ⇒ 回退允许
    out_map_missing = build_context(
        [chunk], pred=pred, view_index=empty_index, materialized={}
    )
    assert out_map_missing.dropped_count == 0 and len(out_map_missing.sources) == 1

    # Mapping {DOC: True}：权威且完整 ⇒ 缺行 fail-closed
    out_map_true = build_context(
        [chunk], pred=pred, view_index=empty_index, materialized={DOC: True}
    )
    assert out_map_true.dropped_count == 1 and out_map_true.sources == []

    # pred=None ⇒ 整段守卫跳过，materialized 取值不影响（行为不变）
    out_no_pred = build_context(
        [chunk], pred=None, view_index=empty_index, materialized={DOC: True}
    )
    assert out_no_pred.dropped_count == 0 and len(out_no_pred.sources) == 1


# ── load_view_indexes：批量键规则与单文档版逐条一致 ──────────────────────────

class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeScalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _FakeScalars(self._rows)


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _stmt):
        return _FakeResult(self._rows)


def test_load_view_indexes_batch_key_rules_and_materialized():
    rows = [
        # 文档镜像行（object_id == document_id）
        _Row(object_id=DOC, object_type="doc", document_id=DOC,
             chunk_index=None, image_id=None, tenant_id=TENANT_A),
        # 文本块行 → ci:0
        _Row(object_id="objA", object_type="text_chunk", document_id=DOC,
             chunk_index=0, image_id=None, tenant_id=TENANT_A),
        # 图片行 → img:img1（chunk_index 为 NULL）
        _Row(object_id="objImg", object_type="image", document_id=DOC,
             chunk_index=None, image_id="img1", tenant_id=TENANT_A),
    ]
    session = _FakeSession(rows)

    view_index, materialized = asyncio.run(
        load_view_indexes([DOC, DOC_EMPTY], session=session)
    )

    # 键规则与 load_document_view_index 一致
    assert set(view_index[DOC].keys()) == {"doc", "ci:0", "img:img1"}
    # 每个被请求的文档都有条目：有行 ⇒ True，无行 ⇒ False
    assert materialized == {DOC: True, DOC_EMPTY: False}
    assert view_index[DOC_EMPTY] == {}


def test_load_view_indexes_empty_input():
    view_index, materialized = asyncio.run(load_view_indexes([]))
    assert view_index == {} and materialized == {}


if __name__ == "__main__":      # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
