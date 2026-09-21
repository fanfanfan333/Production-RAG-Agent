"""
T4 —— 权限继承与级联 + OCR 取严 + excluded / 图片提级的三腿一致性.

覆盖设计文档 §7 与决策 8 / 11 / 12 的**红线**：

    - 派生对象只收紧不放宽（``effective_security_level >= 父文档``）
    - OCR 派生块（``image_id`` 非空）追溯**源图片对象**，有效密级 ``>= 源图``
    - 源图 ``excluded`` / 提级后，派生块同步取严（不可检索 / 不可回源）
    - ``excluded`` 与图片提级在**三腿**（``allows`` / ``to_sql`` / ``to_qdrant``）
      上判定一致

三腿一致性的 SQL 求值器**复用** ``test_scope_filter_equivalence`` 的替身模型
（刻意不复用 ``allows()`` 的判定函数，否则就是永真断言）。
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.db.security_models import (
    ACL_SYNC_PENDING,
    DEFAULT_SECURITY_LEVEL,
    OBJECT_TYPE_DOC,
    OBJECT_TYPE_IMAGE,
    OBJECT_TYPE_TABLE,
    make_object_id,
)
from app.services.retrieval_service import RetrievedChunk
from app.services.security_cascade import (
    build_object_rows,
    derive_child_fields,
)
from app.services.nodes.final_check_node import (
    NO_EVIDENCE_ANSWER,
    PARTIAL_NOTICE,
    STATUS_ALL_DROPPED,
    STATUS_CLEAN,
    STATUS_PARTIAL,
    build_permission_snapshot,
    filter_chunks_by_acl,
)
from app.services.security_policy import (
    ObjectACLView,
    allows,
    qdrant_filter_matches,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_scope_filter_equivalence import sql_clause_matches  # noqa: E402

FIXED_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)

DOC_ID = str(uuid.uuid4())
TENANT = "company_a"
DEPT = "d001"
U_OWNER = str(uuid.uuid4())


# ── 替身 ──────────────────────────────────────────────────────────────────────


class _Doc:
    """文档权限快照（对齐 build_object_rows 读取的字段）."""

    def __init__(self, **kw):
        self.id = kw.get("id", DOC_ID)
        self.tenant_id = kw.get("tenant_id", TENANT)
        self.owner_id = kw.get("owner_id", U_OWNER)
        self.department_id = kw.get("department_id", DEPT)
        self.access_level = kw.get("access_level", "private")
        self.security_level = kw.get("security_level", DEFAULT_SECURITY_LEVEL)
        self.visibility_mode = kw.get("visibility_mode", "tier")
        self.project_ids = kw.get("project_ids", [])
        self.acl_allow = kw.get("acl_allow", [])
        self.acl_deny = kw.get("acl_deny", [])
        self.acl_expires_at = kw.get("acl_expires_at", None)
        self.share_status = kw.get("share_status", "none")
        self.share_grant_scope = kw.get("share_grant_scope", None)


def _point(pid: str, **payload) -> dict:
    return {"id": pid, "payload": payload}


def _chunk(content_type="text", image_id=None, chunk_index=0, text="正文") -> RetrievedChunk:
    return RetrievedChunk(
        document_id=DOC_ID,
        filename="doc.pdf",
        page_number=1,
        chunk_index=chunk_index,
        text=text,
        score=0.9,
        content_type=content_type,
        image_id=image_id,
    )


def _pred(clearance=1, **kw):
    from app.services.security_policy import ScopePredicate

    return ScopePredicate(
        user_id=U_OWNER,
        tenant_ids=frozenset({TENANT}),
        owns_tenant_ids=frozenset(),
        department_id=DEPT,
        tenant_wide=False,
        clearance=clearance,
        principals=frozenset({f"user:{U_OWNER}"}),
        now=FIXED_NOW,
        **kw,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ① 行构造：doc 镜像 / 继承 / OCR 取严
# ═══════════════════════════════════════════════════════════════════════════════


def test_doc_mirror_row_and_chunk_inheritance():
    doc = _Doc(security_level=1)
    points = [
        _point("p1", content_type="text", chunk_index=0),
        _point("p2", content_type="table", chunk_index=1),
    ]
    rows, stats = build_object_rows(doc, points, now=FIXED_NOW)
    by_id = {r["object_id"]: r for r in rows}

    # doc 镜像行
    assert DOC_ID in by_id
    assert by_id[DOC_ID]["object_type"] == OBJECT_TYPE_DOC
    assert by_id[DOC_ID]["effective_security_level"] == 1

    # 普通分块继承文档：父 = 文档，effective = max(doc, self) = 1
    p1 = by_id[make_object_id(DOC_ID, "p1", object_type="text_chunk")]
    assert p1["parent_object_id"] == DOC_ID
    assert p1["inherited_from"] == DOC_ID
    assert p1["effective_security_level"] == 1
    assert stats["chunk"] == 2


def test_derived_effective_never_below_doc():
    """派生对象只收紧不放宽：doc=2 时，派生块 effective 必须 >= 2."""
    doc = _Doc(security_level=2)
    points = [_point("p1", content_type="text", chunk_index=0)]
    rows, _ = build_object_rows(doc, points, now=FIXED_NOW)
    p1 = rows[1]
    assert p1["effective_security_level"] >= 2
    assert p1["effective_security_level"] == max(2, int(p1["security_level"]))


def test_ocr_derived_parent_points_to_source_image():
    """OCR 派生块（image_id 非空、content_type != image）父 = 源图片对象."""
    doc = _Doc(security_level=1)
    points = [
        _point("imgpoint", content_type="image", image_id="img_1", image_path="images/a.png"),
        _point("tblpoint", content_type="table", image_id="img_1", chunk_index=3,
               image_path="images/a.png"),
    ]
    rows, stats = build_object_rows(doc, points, now=FIXED_NOW)
    src_image_id = make_object_id(DOC_ID, "img_1", object_type=OBJECT_TYPE_IMAGE)
    derived = next(
        r for r in rows if r["object_type"] == OBJECT_TYPE_TABLE
    )
    assert derived["parent_object_id"] == src_image_id
    assert derived["inherited_from"] == src_image_id
    assert stats["image"] == 1
    assert stats["derived"] == 1


def test_ocr_derived_effective_ge_source_image():
    """源图提级到 3 后，其 OCR 派生块 effective 必须 >= 3（图看不了字也搜不到）."""
    doc = _Doc(security_level=1)
    # 模拟源图已被提级：build 时通过 payload 无法表达，这里直接验证 derive_child_fields
    src = {"security_level": 3, "effective_security_level": 3, "acl_deny": [], "excluded": False}
    child = {"security_level": 1, "effective_security_level": 1, "acl_deny": [], "excluded": False}
    fields = derive_child_fields(child, src)
    assert fields["effective_security_level"] >= 3
    assert fields["parent_security_level"] == 3


def test_ocr_derived_with_escalated_source_image_in_row_build():
    """行构造阶段：源图先建、派生取 max(文档, 源图) —— 用 doc=1 但构造断言 max 语义."""
    doc = _Doc(security_level=1)
    points = [
        _point("imgpoint", content_type="image", image_id="img_1"),
        _point("tblpoint", content_type="table", image_id="img_1", chunk_index=1),
    ]
    rows, _ = build_object_rows(doc, points, now=FIXED_NOW)
    derived = next(r for r in rows if r["object_type"] == OBJECT_TYPE_TABLE)
    src = next(r for r in rows if r["object_type"] == OBJECT_TYPE_IMAGE)
    assert derived["effective_security_level"] >= int(src["effective_security_level"])
    assert derived["effective_security_level"] >= 1


def test_derived_acl_allow_is_empty():
    """派生对象不得通过 acl_allow 获得父之外的可见性（PRD 3.2 / 共享知识 9）."""
    doc = _Doc(security_level=1, acl_allow=["user:someone", "role:employee"])
    points = [
        _point("imgpoint", content_type="image", image_id="img_1"),
        _point("tblpoint", content_type="table", image_id="img_1", chunk_index=1),
    ]
    rows, _ = build_object_rows(doc, points, now=FIXED_NOW)
    derived = next(r for r in rows if r["object_type"] == OBJECT_TYPE_TABLE)
    assert derived["acl_allow"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# ② derive_child_fields：只收紧 / 剔除与 deny 向下传染 / 同步水位
# ═══════════════════════════════════════════════════════════════════════════════


def test_derive_child_fields_never_loosens():
    """即使源图被降级，子自身密级也不被拉低（只收紧不放宽）."""
    src = {"security_level": 0, "effective_security_level": 0, "acl_deny": [], "excluded": False}
    child = {"security_level": 2, "effective_security_level": 2, "acl_deny": [], "excluded": False}
    fields = derive_child_fields(child, src)
    assert fields["effective_security_level"] == 2


def test_derive_child_fields_propagates_excluded_and_deny_and_pending():
    src = {
        "security_level": 3, "effective_security_level": 3,
        "acl_deny": ["role:employee"], "excluded": True,
    }
    child = {"security_level": 1, "effective_security_level": 1, "acl_deny": [], "excluded": False}
    fields = derive_child_fields(child, src)
    assert fields["excluded"] is True                 # 剔除向下传染
    assert "role:employee" in fields["acl_deny"]      # deny 向下传染
    assert fields["acl_sync_state"] == ACL_SYNC_PENDING  # 同步水位
    assert fields["acl_allow"] == []


# ═══════════════════════════════════════════════════════════════════════════════
# ③ 三腿一致性：excluded / 图片提级（allows / to_sql / to_qdrant）
# ═══════════════════════════════════════════════════════════════════════════════


def _view(**kw) -> ObjectACLView:
    return ObjectACLView(
        object_id=kw.get("object_id", "obj"),
        document_id=DOC_ID,
        tenant_id=kw.get("tenant_id", TENANT),
        owner_id=kw.get("owner_id", U_OWNER),
        access_level=kw.get("access_level", "private"),
        effective_security_level=kw.get("effective_security_level", 1),
        security_level=kw.get("security_level", 1),
        excluded=kw.get("excluded", False),
        acl_deny=frozenset(kw.get("acl_deny", [])),
    )


def test_excluded_object_denied_on_all_three_legs():
    pred = _pred(clearance=3)
    obj = _view(excluded=True)
    assert allows(pred, obj).allowed is False
    assert allows(pred, obj).reason == "object_excluded"
    assert qdrant_filter_matches(pred, obj.to_payload()) is False
    assert sql_clause_matches(pred, obj) is False


def test_escalated_image_denies_low_clearance_on_all_three_legs():
    pred_low = _pred(clearance=1)
    obj = _view(effective_security_level=3, security_level=3)
    assert allows(pred_low, obj).allowed is False
    assert qdrant_filter_matches(pred_low, obj.to_payload()) is False
    assert sql_clause_matches(pred_low, obj) is False

    # 阳性对照：clearance=3 必须放行（否则"拦住"可能只是整条链路坏了）
    pred_high = _pred(clearance=3)
    assert allows(pred_high, obj).allowed is True
    assert qdrant_filter_matches(pred_high, obj.to_payload()) is True
    assert sql_clause_matches(pred_high, obj) is True


# ═══════════════════════════════════════════════════════════════════════════════
# ④ 第 12 环过滤：派生块被剔除 / 无占位符 / 三段降级
# ═══════════════════════════════════════════════════════════════════════════════


def _index_from_views(mapping: dict[str, ObjectACLView]) -> dict:
    return {DOC_ID: mapping}


def test_filter_drops_derived_when_source_image_excluded():
    """图片被 excluded → 其 OCR 派生 table 块不可检索（通过 allows 的 excluded 分支）."""
    pred = _pred(clearance=3)
    keep = _chunk(content_type="text", chunk_index=0)
    derived = _chunk(content_type="table", image_id="img_1", chunk_index=1)
    view_index = _index_from_views({
        "ci:0": _view(object_id="ci0", excluded=False),
        "ci:1": _view(object_id="ci1", excluded=True),   # 派生块已同步 excluded
    })
    outcome = filter_chunks_by_acl([keep, derived], pred, view_index, materialized={DOC_ID: True})
    assert keep in outcome.allowed
    assert derived not in outcome.allowed
    assert outcome.dropped_count == 1
    assert outcome.status == STATUS_PARTIAL


def test_filter_no_placeholder_markers():
    """被剔除的块**不出现在返回列表里**，也不返回任何占位标记."""
    pred = _pred(clearance=3)
    keep = _chunk(content_type="text", chunk_index=0, text="保留的正文字")
    hidden = _chunk(content_type="table", image_id="img_2", chunk_index=1, text="机密表格")
    view_index = _index_from_views({
        "ci:0": _view(object_id="ci0", excluded=False),
        "ci:1": _view(object_id="ci1", excluded=True),
    })
    outcome = filter_chunks_by_acl([keep, hidden], pred, view_index, materialized={DOC_ID: True})
    texts = [c.text for c in outcome.allowed]
    assert "机密表格" not in texts
    assert all("占位" not in t and "已屏蔽" not in t for t in texts)


def test_filter_all_dropped_status_and_uniform_answer():
    pred = _pred(clearance=1)
    a = _chunk(content_type="table", image_id="i1", chunk_index=0)
    b = _chunk(content_type="table", image_id="i2", chunk_index=1)
    vi = _index_from_views({
        "ci:0": _view(object_id="a", effective_security_level=3, security_level=3),
        "ci:1": _view(object_id="b", effective_security_level=3, security_level=3),
    })
    outcome = filter_chunks_by_acl([a, b], pred, vi, materialized={DOC_ID: True})
    assert outcome.status == STATUS_ALL_DROPPED
    assert outcome.allowed == []
    # 全剔除文案与"证据不足"逐字一致（不泄露原因）
    from app.services.nodes.evidence_gate import REFUSAL_ANSWER

    assert NO_EVIDENCE_ANSWER == REFUSAL_ANSWER


def test_filter_materialized_doc_missing_row_is_fail_closed():
    """已物化文档缺少对象行 → fail-closed 丢弃."""
    pred = _pred(clearance=3)
    orphan = _chunk(content_type="text", chunk_index=7)
    outcome = filter_chunks_by_acl([orphan], pred, _index_from_views({}), materialized={DOC_ID: True})
    assert outcome.allowed == []
    assert outcome.status == STATUS_ALL_DROPPED


def test_filter_unmaterialized_doc_falls_back_allow():
    """未物化文档（回填未覆盖）→ 缺行回退允许，避免存量文档集体不可见."""
    pred = _pred(clearance=3)
    orphan = _chunk(content_type="text", chunk_index=7)
    outcome = filter_chunks_by_acl([orphan], pred, _index_from_views({}), materialized={DOC_ID: False})
    assert outcome.allowed == [orphan]
    assert outcome.status == STATUS_CLEAN


def test_filter_positive_control_legit_user_keeps_everything():
    """阳性对照：合法用户（高密级、同租户）必须拿到全部块，防止"全拦"假通过."""
    pred = _pred(clearance=3)
    chunks = [_chunk(chunk_index=i) for i in range(3)]
    vi = _index_from_views({
        f"ci:{i}": _view(object_id=f"c{i}", excluded=False, effective_security_level=1)
        for i in range(3)
    })
    outcome = filter_chunks_by_acl(chunks, pred, vi, materialized={DOC_ID: True})
    assert outcome.allowed_count == 3
    assert outcome.status == STATUS_CLEAN


# ═══════════════════════════════════════════════════════════════════════════════
# ⑤ 引用快照：只存指纹，不存明文
# ═══════════════════════════════════════════════════════════════════════════════


def test_permission_snapshot_has_fingerprint_but_no_plaintext():
    pred = _pred(clearance=2)
    # 给 ScopePredicate 加明文敏感集合
    from dataclasses import replace

    pred = replace(pred, project_ids=frozenset({"p_alpha"}), department_id="secret_dept")
    view = _view(effective_security_level=2)
    snap = build_permission_snapshot(
        view, pred, object_id="obj1", object_type="table", parent_object_id="img1",
    )
    assert snap["scope_fingerprint"]
    assert snap["object_id"] == "obj1"
    assert snap["parent_object_id"] == "img1"
    assert snap["effective_security_level"] == 2
    # 明文权限属性一律不得出现
    dumped = str(snap)
    assert "p_alpha" not in dumped
    assert "secret_dept" not in dumped
    assert "clearance" not in dumped


def test_partial_notice_only_when_explicitly_enabled():
    """默认不追加"因权限未包含"提示（存在性泄露）；显式开启才追加."""
    from app.services.nodes.final_check_node import FinalCheckResult, FilterOutcome

    outcome = FilterOutcome(allowed=[1], dropped=[(2, None)], status=STATUS_PARTIAL, total=2)
    res = FinalCheckResult(outcome=outcome)
    assert res.append_partial_notice("答案") == "答案"
    assert PARTIAL_NOTICE in res.append_partial_notice("答案", announce_partial=True)
