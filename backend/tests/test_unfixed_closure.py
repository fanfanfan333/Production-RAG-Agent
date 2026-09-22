"""
上线前报告「未修项」的收口回归门禁（三轮 · 2026-09-21）.

覆盖本轮闭环的五项（此前报告里状态为 ⚠️ 未修）：

    #5   对象级物化失败 → 只写日志、照常 COMPLETED（对象级保护永久失效且无入口）
    #13  内存 BM25 腿从不写 ``object_payloads`` → 该腿整条**跳过** ``allows()``
    #14  small-to-big 父块正文绕过对象级判定（子块通过 ⇒ 父块全文进 LLM）
    #15  ``user_scope=None`` 的 fail-closed 在用户侧表现为"知识库没有相关内容"
    #16  剔除审计逐条 ``await`` → 一次检索几十次串行写库（全在请求路径上）

另四项（#17 逐份摘要超时 / #18 TXT·MD 编码回退 / #19 PPTX 表格·图表 / #20 XLSX
空列错位）在代码中**已经是修好的**（上一轮工人产出已入库），本轮只做状态确认，
其回归门禁在 ``tests/test_prelaunch_parser_fixes.py``（不要再写第二份）。

本文件刻意分成"行为断言"（证明缺陷真被堵住）与"源码断言"（证明接线没漏）两类：
对象级/装配类的缺陷有一个共同特征 —— **忘了调用**在 DB 层与"本来就没这条数据"
无法区分，只有源码断言能精确抓到，这与 ``test_access_level_object_sync.py`` C 组
同一套纪律。
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from pathlib import Path

import pytest

DOC_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")
OTHER_UID = "22222222-2222-2222-2222-222222222222"


def _pred():
    """部门同事的五维 Scope（密级 1 —— 见 test_access_level_object_sync 的口径说明）."""
    from app.services.security_policy import ScopePredicate

    return ScopePredicate(
        user_id=OTHER_UID,
        tenant_ids=frozenset({"c_a"}),
        department_id="d_tech",
        clearance=1,
    )


def _view(*, excluded: bool = False):
    from app.services.security_policy import ObjectACLView

    return ObjectACLView(
        object_id=f"{DOC_ID}::pc",
        object_type="parent_chunk",
        document_id=str(DOC_ID),
        tenant_id="c_a",
        owner_id="99999999-9999-9999-9999-999999999999",
        access_level="tenant",
        department_id=None,
        security_level=1,
        effective_security_level=1,
        excluded=excluded,
    )


def _chunk(parent_id: str | None = "doc:p:0"):
    from app.services.retrieval_service import RetrievedChunk

    return RetrievedChunk(
        document_id=str(DOC_ID), filename="f.pdf", page_number=1,
        chunk_index=0, text="child text", score=0.5, parent_id=parent_id,
    )


@pytest.fixture(autouse=True)
def _silence_audit(monkeypatch):
    """
    默认把**最内层**的落库点换成空实现（本文件只关心"谁被调用了几次"）.

    ⚠️ 只拦 ``record_audit`` / ``record_audit_many`` 这两个真正写库的函数，
    **不要**拦 ``record_acl_drop(s)`` —— 它们是被测对象本身，拦掉就等于测试
    什么都没验（本轮真踩过：D 组因为夹具把 ``record_acl_drops`` 也换成 no-op
    而恒失败）。
    """
    import app.services.audit_service as audit

    async def _noop(*_a, **_kw):
        return None

    monkeypatch.setattr(audit, "record_audit", _noop, raising=False)
    monkeypatch.setattr(audit, "record_audit_many", _noop, raising=False)


# ═══════════════════════════════════════════════════════════════════════════════
# A. #13 —— 内存 BM25 腿的对象级复核
# ═══════════════════════════════════════════════════════════════════════════════


def test_object_level_filter_drops_missing_view_in_strict_mode(monkeypatch) -> None:
    """
    ``SECURITY_STRICT_MODE=true`` 时，**缺对象视图**的候选必须 fail-closed.

    这是 #13 的兜底：内存腿此前不写 ``object_payloads``，所有候选都会落进
    "缺 payload ⇒ 放行"这一支，于是严格模式形同虚设。
    """
    import app.services.retrieval_service as rs

    monkeypatch.setattr(rs, "_security_strict_mode", lambda: True)
    kept, dropped = asyncio.run(rs._object_level_filter(
        [_chunk("doc:p:0")], {}, _pred(),
        stage="postcheck", scope_fingerprint=None,
    ))
    assert kept == [] and dropped == 1


def test_object_level_filter_keeps_missing_view_when_not_strict() -> None:
    """阳性对照：默认（非严格）模式下缺视图**保留** —— 不因缺视图凭空丢证据."""
    import app.services.retrieval_service as rs

    kept, dropped = asyncio.run(rs._object_level_filter(
        [_chunk("doc:p:0")], {}, _pred(),
        stage="postcheck", scope_fingerprint=None,
    ))
    assert [c.document_id for c in kept] == [str(DOC_ID)] and dropped == 0


def test_scroll_pages_attaches_raw_payload_for_object_level_check() -> None:
    """
    语料 scroll 必须把原始 payload 带在 chunk 上.

    不带的话，内存 BM25 腿的候选在收尾复核处**拿不到对象视图**：它只能放行，
    等价于这条腿整体跳过 ``allows()``（且剔除不进审计）。向量腿 / PG 腿手里都有
    payload 字典，只有内存腿依赖这个字段，因此它是"三条腿同源"的必要条件。
    """
    import app.services.retrieval_service as rs

    src = inspect.getsource(rs._scroll_pages)
    assert "_acl_payload=payload" in src, "语料 scroll 必须把 payload 附到 RetrievedChunk"
    assert "_acl_payload" in inspect.getsource(rs.RetrievedChunk)


def test_memory_bm25_leg_registers_object_payloads() -> None:
    """内存 BM25 腿必须把 payload 注册进 ``object_payloads``（否则收尾复核无视图）."""
    import app.services.retrieval_service as rs

    src = inspect.getsource(rs.retrieve_chunks)
    leg = src.index('backend == "postgres"')
    memory_branch = src[src.index("内存 BM25 腿", leg):]
    assert "object_payloads.setdefault(key, c._acl_payload)" in memory_branch


# ═══════════════════════════════════════════════════════════════════════════════
# B. #14 —— small-to-big 父块正文的对象级判定
# ═══════════════════════════════════════════════════════════════════════════════


def test_parent_block_allows_denies_escalated_parent() -> None:
    """父块被单独剔除/提级 ⇒ **不**展开（正文不再进 LLM）."""
    from app.services.nodes.final_check_node import parent_block_allows

    view_index = {str(DOC_ID): {"pc:doc:p:0": _view(excluded=True)}}
    assert parent_block_allows(_chunk(), _pred(), view_index, {str(DOC_ID): True}) is False


def test_parent_block_allows_keeps_normal_parent() -> None:
    """阳性对照：正常父块照常展开（判定不是恒拒）."""
    from app.services.nodes.final_check_node import parent_block_allows

    view_index = {str(DOC_ID): {"pc:doc:p:0": _view()}}
    assert parent_block_allows(_chunk(), _pred(), view_index, {str(DOC_ID): True}) is True


def test_parent_block_allows_falls_back_for_legacy_documents() -> None:
    """
    存量文档（**一个父块行都没有**）必须回退允许，而不是集体失效.

    父块对象行是后来才加的；若按"已物化文档缺行即拒"，small-to-big 会对整个
    存量语料静默失效 —— 那是把一个安全修复做成了功能退化。
    """
    from app.services.nodes.final_check_node import parent_block_allows

    # 文档已物化（有 ci: 行）但没有任何 pc: 行 —— 正是存量形态
    view_index = {str(DOC_ID): {"ci:0": _view()}}
    assert parent_block_allows(_chunk(), _pred(), view_index, {str(DOC_ID): True}) is True


def test_parent_block_allows_is_noop_without_pred() -> None:
    """未给 pred（旧调用点）时行为零变化：照常展开."""
    from app.services.nodes.final_check_node import parent_block_allows

    assert parent_block_allows(_chunk(), None, {str(DOC_ID): {"pc:doc:p:0": _view(excluded=True)}}, {}) is True


def test_both_assemblers_guard_parent_expansion() -> None:
    """
    两个上下文组装入口**都要**过父块判定.

    只改一个入口是本类缺陷的经典复发形态：另一个入口在某个开关（多模态）下
    才被走到，测出来"修好了"，线上却仍漏。
    """
    from app.services.nodes import context_builder, multimodal_context_node

    for mod in (context_builder, multimodal_context_node):
        assert "parent_block_allows" in inspect.getsource(mod), (
            f"{mod.__name__} 未对父块正文做对象级复核"
        )


def test_ingestion_materializes_parent_object_rows() -> None:
    """
    入库时必须把 ``chunk_parents`` 一并物化成 ``parent_chunk`` 对象行.

    否则设计 §19.2-A 整条是死代码：父块没有权限行 ⇒ 既无法被单独提级/剔除，
    也永远进不了第 12 环判定范围（B 组其余用例会因为"没有 pc: 行"而自动放行）。
    """
    from app.services import document_service

    src = inspect.getsource(document_service)
    idx = src.index("materialize_document_objects(")
    call = src[idx: idx + 1200]
    assert "parents=" in call, "materialize_document_objects 未传 parents ⇒ 父块行不会生成"


# ═══════════════════════════════════════════════════════════════════════════════
# C. #5 —— 物化失败不再静默
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeSession:
    """按顺序返回 rowcount 的假 session（够 ``mark_materialization_failed`` 用）."""

    def __init__(self, counts: list[int]) -> None:
        self._counts = list(counts)
        self.stmts: list = []

    async def execute(self, stmt, *_a, **_kw):
        self.stmts.append(stmt)
        return _FakeResult(self._counts.pop(0) if self._counts else 0)


def test_mark_materialization_failed_marks_doc_and_object_rows() -> None:
    """
    失败标记必须落在 ``documents`` **和** 已存在的对象行上，且取值为 ``stale``.

    只标其中一处都会漏：``documents`` 那一处是"一句 SQL 能捞出全部受影响文档"的
    入口（有索引），对象行那一处是 ``ix_dobj_sync`` 部分索引（``<> 'synced'``）的
    覆盖面。
    """
    from app.services.security_cascade import mark_materialization_failed

    sess = _FakeSession([1, 7])
    counts = asyncio.run(mark_materialization_failed(DOC_ID, session=sess))

    assert counts == {"document_rows": 1, "object_rows": 7}
    assert len(sess.stmts) == 2
    for stmt in sess.stmts:
        text = str(stmt)
        assert "acl_sync_state" in text
        assert "stale" in stmt.compile().params.values(), "标记值必须是 stale"


def test_mark_materialization_failed_never_raises() -> None:
    """标记是旁路：写失败只记日志，绝不能再制造一个新失败点（best-effort 红线）."""
    from app.services.security_cascade import mark_materialization_failed

    class _Boom:
        async def execute(self, *_a, **_kw):
            raise RuntimeError("db down")

    counts = asyncio.run(mark_materialization_failed(DOC_ID, session=_Boom()))
    assert counts == {"document_rows": 0, "object_rows": 0}


def test_ingestion_marks_stale_when_materialization_fails() -> None:
    """入库收尾的物化失败分支必须调用标记（源码断言 —— 漏调用就是静默复发）."""
    from app.services import document_service

    src = inspect.getsource(document_service)
    idx = src.index("materialization failed")
    tail = src[idx: idx + 1200]
    assert "mark_materialization_failed" in tail, (
        "物化失败后没有留可查标记 ⇒ 文档照常 COMPLETED 且对象级保护静默失效"
    )


def test_rematerialize_script_defaults_to_dry_run() -> None:
    """重跑入口必须存在，且默认**只出报告不写入**（与既有修复脚本同约定）."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "rematerialize_document_objects.py"
    assert script.exists(), "缺少对象级物化的重跑入口"
    src = script.read_text(encoding="utf-8")
    assert "--apply" in src and "action=\"store_true\"" in src
    assert "if not args.apply:" in src, "必须默认 dry-run"
    assert "materialize_document_objects" in src, "重跑必须复用既有物化实现，不得另写一份"


# ═══════════════════════════════════════════════════════════════════════════════
# D. #16 —— 剔除审计：一条事务写一批
# ═══════════════════════════════════════════════════════════════════════════════


def test_object_level_filter_writes_audit_in_one_batch(monkeypatch) -> None:
    """
    两个对象被剔除 ⇒ **一次**批量写库，其中**两条**记录（每条仍是独立审计行）.

    这条测试同时是"不能为了省写库而合并成一条记录"的守门员：P0-10 要求可追溯到
    具体对象，``resource_id`` 必须逐个对象。
    """
    import app.services.audit_service as audit
    import app.services.retrieval_service as rs

    calls: list[list[dict]] = []

    async def _capture(stage, drops, **kw):
        calls.append([{"stage": stage, **d} for d in drops])

    monkeypatch.setattr(audit, "record_acl_drops", _capture)

    pred = _pred()
    ch_a, ch_b = _chunk("doc:p:0"), _chunk("doc:p:1")
    ch_b.chunk_index = 1
    payloads = {
        (str(DOC_ID), 0): {"object_id": "obj_a", "document_id": str(DOC_ID),
                           "tenant_id": "c_a", "access_level": "tenant",
                           "user_id": OTHER_UID, "excluded": True},
        (str(DOC_ID), 1): {"object_id": "obj_b", "document_id": str(DOC_ID),
                           "tenant_id": "c_a", "access_level": "tenant",
                           "user_id": OTHER_UID, "effective_security_level": 3},
    }
    kept, dropped = asyncio.run(rs._object_level_filter(
        [ch_a, ch_b], payloads, pred, stage="postcheck", scope_fingerprint="fp",
    ))

    assert kept == [] and dropped == 2
    assert len(calls) == 1, f"必须一次批量写库，实际 {len(calls)} 次"
    assert [d["object_id"] for d in calls[0]] == ["obj_a", "obj_b"]
    assert all(d["stage"] == "postcheck" for d in calls[0])


def test_audit_service_bulk_writer_uses_single_transaction(monkeypatch) -> None:
    """``record_acl_drops`` 内部只调用**一次** ``record_audit_many``（写库次数与剔除数解耦）."""
    import app.services.audit_service as audit

    seen: list[list[dict]] = []

    async def _capture(entries):
        seen.append(list(entries))

    monkeypatch.setattr(audit, "record_audit_many", _capture)
    asyncio.run(audit.record_acl_drops(
        "postcheck",
        [{"object_id": f"o{i}", "document_id": "d", "reason": "r", "gate": "g"}
         for i in range(20)],
        scope_fingerprint="fp",
    ))

    assert len(seen) == 1, "20 个剔除必须只写一次库"
    entries = seen[0]
    assert len(entries) == 20, "每条剔除仍单独成行（可逐对象追溯）"
    assert [e["resource_id"] for e in entries] == [f"o{i}" for i in range(20)]
    assert all(e["detail"].startswith("stage=postcheck") for e in entries)


def test_single_acl_drop_shares_detail_format_with_bulk() -> None:
    """单条与批量共用同一 detail 格式（两处各写一份迟早分叉）."""
    import app.services.audit_service as audit

    one = audit._acl_drop_detail("postcheck", document_id="d1", reason="r", gate="g")
    assert one is not None and one.startswith("stage=postcheck")
    assert "document_id=d1" in one and "reason=r" in one and "gate=g" in one


# ═══════════════════════════════════════════════════════════════════════════════
# E. #15 —— 无 Scope 的 fail-closed 必须留痕
# ═══════════════════════════════════════════════════════════════════════════════


def test_unscoped_retrieval_is_refused_and_audited(monkeypatch) -> None:
    """
    ``scope=None`` ⇒ 返回空（fail-closed）**且**写一条审计.

    行为一直是 fail-closed 的，问题在"静默"：用户看到的是"知识库没有相关内容"，
    与"真的没有"逐字相同，而日志里那条 error 没有请求上下文。审计让"谁在什么时候
    没带 Scope 进来"事后可查。
    """
    import app.services.audit_service as audit
    import app.services.retrieval_service as rs

    recorded: list[tuple[str, dict]] = []

    async def _capture(action, **kw):
        recorded.append((action, kw))

    monkeypatch.setattr(audit, "record_audit", _capture)

    out = asyncio.run(rs.retrieve_chunks_scoped("季度营收", scope=None, top_k=3))
    assert out == [], "无 Scope 必须 fail-closed 返回空"
    assert [a for a, _ in recorded] == ["retrieval.unscoped_refused"]
    assert "季度营收" in recorded[0][1].get("detail", "")


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
