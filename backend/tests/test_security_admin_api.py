"""
T5 —— 密级 / 可见性 / 项目维度管理面 API（P0）验收.

    设置密级 → 触发 T4 取严级联（派生对象 effective 取 max）
    图片提级/剔除 → OCR 派生块同步取严（cascade_image_derived）
    无权角色调用管理面 → 403
    全局开关自检（strict / prefilter + 前置条件）

不连数据库：级联与向量推送用替身，判定仍走真身代码。
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

try:
    import pytest
    from fastapi import FastAPI, HTTPException, status
    from fastapi.testclient import TestClient

    from app.api import deps as api_deps
    from app.api.security import router as security_router
    from app.config import get_settings
    from app.db.security_models import OBJECT_TYPE_IMAGE
    from app.services import security_cascade
    from app.services.grant_service import (
        GrantError,
        escalate_object,
        set_document_security,
    )
except ImportError as exc:      # 宿主机缺依赖 → 整份跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")

FIXED_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
FUTURE = FIXED_NOW + timedelta(days=1)
TENANT = "cleanA"
DOC_ID = str(uuid.uuid4())
IMAGE_OBJECT_ID = f"{DOC_ID}::img1"


class _StubUser:
    def __init__(self, *, role: str = "kb_admin", is_admin: bool = False,
                 tenant_id: str = TENANT):
        self.id = uuid.uuid4()
        self.username = f"u_{uuid.uuid4().hex[:6]}"
        self.tenant_id = tenant_id
        self.department_id = "d001"
        self.role = role
        self._is_admin = is_admin
        self.is_active = True

    @property
    def is_admin(self) -> bool:
        return self._is_admin


def _key(ident):
    if isinstance(ident, tuple):
        return tuple(str(x) for x in ident)
    return str(ident)


class _FakeScalars:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self, *, by_entity=None, get_map=None):
        self.by_entity = by_entity or {}
        self.get_map = get_map or {}

    async def get(self, model, ident):
        return self.get_map.get((model.__tablename__, _key(ident)))

    async def scalars(self, stmt):
        entity = stmt.column_descriptions[0].get("entity")
        return _FakeScalars(self.by_entity.get(entity, []))

    def add(self, obj):
        return None

    async def flush(self) -> None:
        return None

    async def refresh(self, _obj) -> None:
        return None


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _no_push(monkeypatch):
    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr("app.services.grant_service._push_payload", _noop)

    async def _no_audit(*_a, **_k):
        return None

    monkeypatch.setattr("app.services.grant_service.record_audit", _no_audit)


# ═══════════════════════════════════════════════════════════════════════════════
# ① 设置密级触发级联 + 项目维度同步 + need-to-know 重物化
# ═══════════════════════════════════════════════════════════════════════════════


def _doc_row():
    return SimpleNamespace(
        id=uuid.UUID(DOC_ID), tenant_id=TENANT, security_level=1,
        visibility_mode="tier", project_ids=[], acl_allow=[],
        acl_expires_at=None, acl_sync_state="synced",
    )


def _object_rows():
    rows = [
        SimpleNamespace(object_id=DOC_ID, object_type="doc", image_id=None),
        SimpleNamespace(object_id=f"{DOC_ID}::c0", object_type="text_chunk", image_id=None),
        SimpleNamespace(object_id=IMAGE_OBJECT_ID, object_type="image", image_id="img1"),
        SimpleNamespace(object_id=f"{DOC_ID}::d0", object_type="text_chunk", image_id="img1"),
    ]
    for r in rows:
        r.visibility_mode = "tier"
        r.project_ids = []
        r.acl_allow = []
        r.acl_expires_at = None
        r.acl_sync_state = "synced"
        r.document_id = uuid.UUID(DOC_ID)
    return rows


def test_set_document_security_cascades_and_propagates(monkeypatch) -> None:
    """密级变更：① 调 T4 级联 ② 项目维度同步到全部对象行 ③ 重物化 need-to-know。"""
    from app.db.security_models import AclGrant, DocumentObject

    doc = _doc_row()
    rows = _object_rows()
    calls: list[dict] = []

    async def _fake_sync(document_id, *, session=None, security_level=None,
                        visibility_mode=None, project_ids=None, **_kw):
        calls.append({
            "security_level": security_level,
            "visibility_mode": visibility_mode,
            "project_ids": project_ids,
        })
        # 模拟 T4：把文档真值写进 doc 行
        if security_level is not None:
            doc.security_level = security_level
        if visibility_mode is not None:
            doc.visibility_mode = visibility_mode
        if project_ids is not None:
            doc.project_ids = list(project_ids)
        return {"document_rows": 1, "object_rows": len(rows)}

    monkeypatch.setattr(security_cascade, "sync_doc_row", _fake_sync)
    monkeypatch.setattr("app.services.security_cascade.sync_doc_row", _fake_sync, raising=False)

    session = _FakeSession(
        by_entity={DocumentObject: rows, AclGrant: []},
        get_map={("documents", DOC_ID): doc},
    )

    actor = _StubUser()
    result = _run(set_document_security(
        actor, DOC_ID, security_level=3, visibility_mode="project",
        project_ids=["p_alpha"], session=session,
    ))

    assert calls == [{"security_level": 3, "visibility_mode": "project",
                      "project_ids": ["p_alpha"]}]
    assert result["security_level"] == 3
    assert result["visibility_mode"] == "project"
    assert result["project_ids"] == ["p_alpha"]
    # 项目维度同步到**全部**对象行（含派生块）
    assert all(r.visibility_mode == "project" for r in rows)
    assert all(r.project_ids == ["p_alpha"] for r in rows)


def test_set_document_security_rejects_bad_level(monkeypatch) -> None:
    async def _fake_sync(*_a, **_k):
        raise AssertionError("非法密级不该触发级联")

    monkeypatch.setattr("app.services.security_cascade.sync_doc_row", _fake_sync, raising=False)
    with pytest.raises(GrantError):
        _run(set_document_security(_StubUser(), DOC_ID, security_level=9,
                                   session=_FakeSession()))


# ═══════════════════════════════════════════════════════════════════════════════
# ② 图片提级/剔除 → OCR 派生取严级联
# ═══════════════════════════════════════════════════════════════════════════════


def test_image_escalation_triggers_ocr_cascade(monkeypatch) -> None:
    from app.db.security_models import AclGrant, DocumentObject

    doc = _doc_row()
    image_row = SimpleNamespace(
        object_id=IMAGE_OBJECT_ID, object_type=OBJECT_TYPE_IMAGE,
        image_id="img1", document_id=uuid.UUID(DOC_ID),
        security_level=1, effective_security_level=1, excluded=False,
        acl_allow=[], acl_expires_at=None, acl_sync_state="synced",
    )
    seen: list[dict] = []

    async def _fake_cascade(document_id, image_object_id, *, session=None,
                            src_level=None, src_excluded=None):
        seen.append({"doc": str(document_id), "image": image_object_id,
                     "src_level": src_level, "src_excluded": src_excluded})
        return {"derived_rows": 1, "image_rows": 1}

    monkeypatch.setattr("app.services.security_cascade.cascade_image_derived", _fake_cascade, raising=False)

    session = _FakeSession(
        by_entity={DocumentObject: [image_row], AclGrant: []},
        get_map={
            ("document_objects", IMAGE_OBJECT_ID): image_row,
            ("documents", DOC_ID): doc,
        },
    )

    result = _run(escalate_object(
        _StubUser(), DOC_ID, IMAGE_OBJECT_ID, security_level=3, session=session,
    ))

    assert seen and seen[0]["src_level"] == 3
    assert seen[0]["image"] == IMAGE_OBJECT_ID
    assert result["object_type"] == OBJECT_TYPE_IMAGE


def test_escalate_requires_a_target() -> None:
    with pytest.raises(GrantError):
        _run(escalate_object(_StubUser(), DOC_ID, IMAGE_OBJECT_ID, session=_FakeSession()))


# ═══════════════════════════════════════════════════════════════════════════════
# ③ API 权限门禁（无权角色 → 403）
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(get_settings(), "REQUIRE_IDENTITY_VERIFICATION", False, raising=False)
    employee = _StubUser(role="employee")
    admin = _StubUser(role="admin", is_admin=True)
    tokens = {"tok-emp": employee, "tok-admin": admin}

    async def _fake_resolve_user(token: str | None):
        user = tokens.get(token or "")
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="no")
        return user

    monkeypatch.setattr(api_deps, "_resolve_user", _fake_resolve_user)
    app = FastAPI()
    app.include_router(security_router)
    tc = TestClient(app)
    return tc


def _hdr(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_employee_cannot_set_document_security(client) -> None:
    resp = client.patch(
        f"/security/documents/{DOC_ID}", json={"security_level": 3},
        headers=_hdr("tok-emp"),
    )
    assert resp.status_code == 403, resp.text


def test_employee_cannot_escalate_object(client) -> None:
    resp = client.patch(
        f"/security/documents/{DOC_ID}/objects/{IMAGE_OBJECT_ID}",
        json={"security_level": 3}, headers=_hdr("tok-emp"),
    )
    assert resp.status_code == 403, resp.text


def test_settings_requires_platform_admin(client) -> None:
    assert client.get("/security/settings", headers=_hdr("tok-emp")).status_code == 403


def test_settings_reports_precondition(client, monkeypatch) -> None:
    async def _zero():
        return 0

    monkeypatch.setattr("app.api.security.counted_null_effective_levels", _zero)
    resp = client.get("/security/settings", headers=_hdr("tok-admin"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["null_effective_level_count"] == 0
    assert body["prefilter_strict_precondition_met"] is True
    assert "可安全开启" in body["note"]
