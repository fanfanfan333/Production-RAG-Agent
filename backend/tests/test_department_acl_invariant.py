"""
不变量门禁：``access_level == 'department'`` ⇒ ``department_id`` 非空.

背景（缺陷 B）
──────────────
历史上有两个写点会在"目标层级是部门库、但解析不到部门"时**静默写入 NULL**：

    set_document_access_level（knowledge_tier_service.py:319–323）
    upload_documents（api/documents.py:243–245）

产出的 ``department + department_id IS NULL`` 退化态，对 **owner 与部门同事都
检索不到**（``tenancy.document_scope_clause`` 的 department 分支要求 department_id
相等，``can_access_document`` 的 department 分支也没有 owner 快放）。修复把不变量
收到这两处：解析不到部门 → 拒绝（raise / 4xx），绝不静默落库。

红-绿：注释掉 ``set_document_access_level`` 里新加的那段 ``if … resolved_department
is None: raise TierError`` —— ``test_department_without_department_raises`` 必红。

不变量本身**不依赖真实 DB**：行为断言走替身会话；接线断言走源码（``inspect``）。
"""

from __future__ import annotations

import asyncio
import inspect
import types
import uuid

import pytest
from fastapi import HTTPException

from app.services import knowledge_tier_service as kt

DOC_UUID = uuid.uuid4()
UOWN = uuid.uuid4()
DEPT = "dd6104e61cf30"


# ── 替身 ──────────────────────────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, row) -> None:
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class _FakeSession:
    def __init__(self, row) -> None:
        self._row = row

    async def execute(self, _stmt):
        return _FakeResult(self._row)

    async def flush(self) -> None:
        return None

    async def refresh(self, _obj) -> None:
        return None


class _FakeSessionCM:
    def __init__(self, row) -> None:
        self._row = row

    async def __aenter__(self) -> _FakeSession:
        return _FakeSession(self._row)

    async def __aexit__(self, *_exc) -> bool:
        return False


@pytest.fixture()
def no_write_point(monkeypatch):
    """把三处写副作用全部替换成计数器：证明"该拦的时候一处都没写"."""
    from app.services import security_cascade as sc
    from app.services import vector_service as vs

    calls = {"sync": 0, "payload": 0, "audit": 0}

    async def _sync(*_a, **_k):
        calls["sync"] += 1
        return 1

    async def _payload(*_a, **_k):
        calls["payload"] += 1
        return 1

    async def _audit(*_a, **_k):
        calls["audit"] += 1
        return None

    monkeypatch.setattr(sc, "sync_access_level", _sync)
    monkeypatch.setattr(vs, "update_document_access_payload", _payload)
    monkeypatch.setattr(kt, "record_audit", _audit)
    return calls


def _bind_doc(monkeypatch, **fields) -> types.SimpleNamespace:
    doc = types.SimpleNamespace(
        id=DOC_UUID,
        access_level=fields.pop("access_level", "private"),
        department_id=fields.pop("department_id", None),
        **fields,
    )
    monkeypatch.setattr(kt, "get_db_session", lambda: _FakeSessionCM(doc))
    return doc


# ═══════════════════════════════════════════════════════════════════════════════
# ① 唯一写点：department + 空部门 → raise（红-绿门禁）
# ═══════════════════════════════════════════════════════════════════════════════


def test_department_without_department_raises(monkeypatch, no_write_point):
    """department 层级解析不到部门时必须 raise TierError(400)，绝不静默写 NULL."""
    _bind_doc(monkeypatch, access_level="private", department_id=None)

    with pytest.raises(kt.TierError) as excinfo:
        asyncio.run(
            kt.set_document_access_level(
                DOC_UUID, level="department", department_id=None
            )
        )
    assert excinfo.value.status_code == 400
    # 拒绝发生在任何一处事实源被改动之前
    assert no_write_point == {"sync": 0, "payload": 0, "audit": 0}


def test_department_with_department_still_passes(monkeypatch, no_write_point):
    """正向对照：department + 有部门必须正常走完（闸门不能误伤合法调用）."""
    doc = _bind_doc(monkeypatch, access_level="private", department_id=None)

    out = asyncio.run(
        kt.set_document_access_level(
            DOC_UUID, level="department", department_id=DEPT
        )
    )
    assert out.access_level == "department"
    assert out.department_id == DEPT
    assert no_write_point["sync"] == 1
    assert no_write_point["payload"] == 1


def test_non_department_level_clears_department(monkeypatch, no_write_point):
    """公司库/个人库一律清空部门归属（不变量只约束 department 层级）."""
    _bind_doc(monkeypatch, access_level="department", department_id=DEPT)

    out = asyncio.run(
        kt.set_document_access_level(DOC_UUID, level="tenant", department_id=DEPT)
    )
    assert out.access_level == "tenant"
    assert out.department_id is None


# ═══════════════════════════════════════════════════════════════════════════════
# ② 上传路径：department 目标 + 无部门用户 → 4xx（不是 500，也不是落库）
# ═══════════════════════════════════════════════════════════════════════════════


def test_upload_department_without_department_user_is_4xx(monkeypatch):
    """上传到部门库但操作者无部门 → 4xx；前置拦截，不会创建任何行."""
    from app.api import documents as docs_mod

    monkeypatch.setattr(
        docs_mod, "get_settings",
        lambda: types.SimpleNamespace(MAX_FILES_PER_UPLOAD=10),
    )

    async def _fake_validate(_upload, _settings):
        return b"data"

    monkeypatch.setattr(docs_mod, "validate_document_upload", _fake_validate)

    user = types.SimpleNamespace(
        id=UOWN, role="dept_manager", department_id=None, is_admin=False,
        tenant_id="t1", username="dm-no-dept",
    )
    upload = types.SimpleNamespace(filename="a.pdf", content_type="application/pdf")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            docs_mod.upload_documents(
                files=[upload],
                collection_id=None,
                access_level="department",
                company_id=None,
                user=user,
            )
        )
    assert excinfo.value.status_code == 400
    assert excinfo.value.status_code < 500


# ═══════════════════════════════════════════════════════════════════════════════
# ③ 回填脚本：报告谓词 = 不变量；默认只读
# ═══════════════════════════════════════════════════════════════════════════════


def _load_dept_script():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "repair_department_acl.py"
    )
    spec = importlib.util.spec_from_file_location("_ops_deptfix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_script_predicate_is_the_invariant() -> None:
    """回填脚本的扫描条件必须恰好是 ``department AND department_id IS NULL``."""
    ops = _load_dept_script()
    src = inspect.getsource(ops._collect_targets)
    assert 'access_level == "department"' in src
    assert "department_id.is_(None)" in src


def test_repair_script_is_dry_run_by_default() -> None:
    """默认不写：``--apply`` 是唯一写入开关，且 dry-run 早退在写调用之前."""
    ops = _load_dept_script()
    src = inspect.getsource(ops.main)
    assert "--apply" in src
    guard = src.index("if not args.apply:")
    tail = src[guard:]
    assert "return" in tail, "缺少 dry-run 早退"
    assert tail.index("return") < tail.index("set_document_access_level"), (
        "dry-run 分支必须在写入调用之前返回"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ④ 已知且文档化的不对称：对象级 owner 恒放行，文档级 department 分支不放行
# ═══════════════════════════════════════════════════════════════════════════════


def test_department_level_owner_asymmetry_is_known_and_documented() -> None:
    """
    记录一处**故意不修**的不对称（留作单独决策）：

        · 对象级 ``_source_gate`` 对 owner **恒放行**（排在 department 分支之前）；
        · 文档级 ``can_access_document`` 的 department 分支**没有** owner 快放。

    不修的理由：文档一旦"交到部门"，owner 未必仍在该部门内；给 owner 加文档级
    快放是一次**可见性放宽**，有独立的设计含义，应当单独评审，而不是顺手改。
    本用例把这个现状钉住——谁哪天加了快放，这条会红，提醒那是需要决策的改动。
    """
    from app.services.security_policy import ObjectACLView, ScopePredicate, _source_gate
    from app.services.tenancy import can_access_document

    # 对象级：owner 命中 → 放行（即便 department_id 为空）
    pred = ScopePredicate(
        user_id=str(UOWN),
        tenant_ids=frozenset({"t1"}),
        department_id=None,
        principals=frozenset({f"user:{UOWN}"}),
    )
    obj = ObjectACLView(
        object_id="o1", tenant_id="t1", owner_id=str(UOWN),
        access_level="department", department_id=None,
    )
    assert _source_gate(pred, obj) is True

    # 文档级：department + dept=NULL 的 owner 被拒（当前行为，已知）
    doc = types.SimpleNamespace(
        access_level="department", department_id=None, owner_id=UOWN, tenant_id="t1"
    )
    user = types.SimpleNamespace(
        id=UOWN, role="employee", department_id=None, is_admin=False, tenant_id="t1"
    )
    assert can_access_document(
        doc, user, owner_id=UOWN, tenant_ids=frozenset({"t1"})
    ) is False
