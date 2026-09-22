"""第 11 环：对象级复核 + 越权剔除审计（决策 10 / PRD P0-6；验收要点 3）.

  * ``_object_level_filter`` 对每个候选跑 ``allows(pred, obj)``；被剔除的写审计
    （``action="acl.drop.postcheck"``、``resource_type="document_object"``、
    ``resource_id=object_id``、``detail`` 含 stage / reason / gate / document_id）。
  * **阳性对照**：可见对象必须被保留，且**不**产生审计（证明判定不是恒拒）。
  * 取不到 payload 的候选保留（不因缺视图凭空丢证据）。
  * 【#16 收口】同一批剔除走**一次**批量写库（``record_audit_many``）——
    "一条事务"与"每对象一行"必须同时成立，因此这里验的是最终 entry 列表：
    合并成一条记录同样是回归。

⚠️ 拦截点选在最内层的 ``record_audit_many``：这样 ① detail 的构造
（``_acl_drop_detail``）也在被测范围内；② 顺带证明整批只写一次库。
"""

from __future__ import annotations

import asyncio
import uuid

from app.services.retrieval_service import RetrievedChunk, _object_level_filter
from app.services.security_scope import UserScope
from app.services.tenancy import DocumentScope

TENANT_A = "c309a7cb9f496"
TENANT_B = "cf33b1db5679d"
DEPT_D1 = "d001"
UID = str(uuid.uuid4())
DOC_VISIBLE = str(uuid.uuid4())
DOC_FORBIDDEN = str(uuid.uuid4())


def _scope() -> UserScope:
    base = DocumentScope(
        owner_id=uuid.UUID(UID),
        tenant_ids=frozenset({TENANT_A}),
        owns_tenant_ids=frozenset(),
        department_id=DEPT_D1,
        tenant_wide=False,
    )
    return UserScope(
        base=base, user_id=UID, role="employee", clearance=1,
        project_ids=frozenset(), principals=frozenset(), strict=False,
    )


def _chunk(document_id: str, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=document_id, filename="f.pdf", page_number=1,
        chunk_index=index, text="t", score=0.9,
    )


def _capture_audit(monkeypatch):
    """把审计写库换成"记录最终 entries"，返回 captures 列表（每批一项）"""
    import app.services.audit_service as audit

    batches: list[list[dict]] = []

    async def fake_record_many(entries):
        batches.append([dict(e) for e in entries])

    monkeypatch.setattr(audit, "record_audit_many", fake_record_many)
    return batches


def test_object_level_filter_drops_and_audits(monkeypatch):
    batches = _capture_audit(monkeypatch)

    pred = _scope().predicate()
    chunks = [_chunk(DOC_VISIBLE), _chunk(DOC_FORBIDDEN)]
    payloads = {
        (DOC_VISIBLE, 0): {
            "object_id": "obj_ok", "document_id": DOC_VISIBLE,
            "tenant_id": TENANT_A, "access_level": "tenant",
            "user_id": UID, "security_level": 1,
        },
        (DOC_FORBIDDEN, 0): {
            "object_id": "obj_bad", "document_id": DOC_FORBIDDEN,
            "tenant_id": TENANT_B, "access_level": "tenant",
            "user_id": "someone-else", "security_level": 1,
        },
    }

    kept, dropped = asyncio.run(_object_level_filter(
        chunks, payloads, pred, stage="postcheck", scope_fingerprint="fp123",
    ))

    assert [c.document_id for c in kept] == [DOC_VISIBLE]
    assert dropped == 1
    assert len(batches) == 1, "整批剔除应只写一次库"
    assert len(batches[0]) == 1
    entry = batches[0][0]
    assert entry["action"] == "acl.drop.postcheck"
    assert entry["resource_type"] == "document_object"
    assert entry["resource_id"] == "obj_bad"
    assert "stage=postcheck" in entry["detail"]
    assert f"document_id={DOC_FORBIDDEN}" in entry["detail"]
    assert "gate=" in entry["detail"] and "reason=" in entry["detail"]


def test_object_level_filter_keeps_visible_and_no_audit(monkeypatch):
    """阳性对照：可见对象保留且**不**产生审计（判定不是恒拒）."""
    batches = _capture_audit(monkeypatch)

    pred = _scope().predicate()
    payloads = {
        (DOC_VISIBLE, 0): {
            "object_id": "obj_ok", "document_id": DOC_VISIBLE,
            "tenant_id": TENANT_A, "access_level": "tenant",
            "user_id": UID, "security_level": 1,
        },
    }
    kept, dropped = asyncio.run(_object_level_filter(
        [_chunk(DOC_VISIBLE)], payloads, pred, stage="postcheck", scope_fingerprint=None,
    ))
    assert [c.document_id for c in kept] == [DOC_VISIBLE]
    assert dropped == 0 and batches == []


def test_object_level_filter_keeps_chunks_without_payload(monkeypatch):
    """取不到 payload 的候选保留（交由文档级 valid_docs 兜底，不凭空丢证据）."""
    batches = _capture_audit(monkeypatch)

    kept, dropped = asyncio.run(_object_level_filter(
        [_chunk(DOC_VISIBLE)], {}, _scope().predicate(),
        stage="postcheck", scope_fingerprint=None,
    ))
    assert [c.document_id for c in kept] == [DOC_VISIBLE]
    assert dropped == 0 and batches == []


def test_object_level_filter_drops_excluded_and_over_clearance(monkeypatch):
    batches = _capture_audit(monkeypatch)

    pred = _scope().predicate()
    d_excl = str(uuid.uuid4())
    d_hi = str(uuid.uuid4())
    payloads = {
        (d_excl, 0): {
            "object_id": "e1", "document_id": d_excl, "tenant_id": TENANT_A,
            "access_level": "tenant", "user_id": UID, "excluded": True,
        },
        (d_hi, 0): {
            "object_id": "h1", "document_id": d_hi, "tenant_id": TENANT_A,
            "access_level": "tenant", "user_id": UID, "effective_security_level": 3,
        },
    }
    kept, dropped = asyncio.run(_object_level_filter(
        [_chunk(d_excl), _chunk(d_hi)], payloads, pred,
        stage="postcheck", scope_fingerprint=None,
    ))
    assert kept == [] and dropped == 2
    # 两条剔除 ⇒ 一次写库、两行记录（不是合并成一行）
    assert len(batches) == 1 and len(batches[0]) == 2
    assert [e["action"] for e in batches[0]] == [
        "acl.drop.postcheck", "acl.drop.postcheck",
    ]
    assert [e["resource_id"] for e in batches[0]] == ["e1", "h1"]


if __name__ == "__main__":      # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
