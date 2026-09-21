"""
T4 / 决策 16 —— 引用点击「再校验」的两个回源端点.

验收（设计 §16-3，逐条对应）：

    1. 图片单独提级后，低密级用户 ``GET /documents/{id}/images/{name}`` → 404，
       且文案与"文档不存在"**逐字一致**（存在性不泄露）。
    2. 图片被 ``excluded`` 后，其 OCR 派生的 table/code chunk 在
       ``GET /documents/{id}/chunks`` 的返回列表里**不存在**。
    3. 文档可见但某 chunk 被提级：返回列表**不含**该 chunk、**不含任何占位标记**。
    4. 三种剔除各产生一条 ``acl.drop.citation_open`` 审计。
    5. **阳性对照**：clearance=3 的同租户用户打开同一引用 → 200 且可见
       （防止"端点整个坏掉"导致的假通过）。
    6. ``?token=<jwt>`` 直载路径与 ``Authorization`` 头路径判定结果一致。
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

try:
    import pytest
    from fastapi import FastAPI, HTTPException, status
    from fastapi.testclient import TestClient

    from app.api import deps as api_deps
    from app.api.document_management import (
        _DOCUMENT_404_DETAIL,
        router as documents_router,
    )
    from app.db.security_models import make_object_id, object_type_from_content_type
    from app.services.security_policy import ScopePredicate
    from app.services.tenancy import DocumentScope
except ImportError as exc:      # 宿主机缺依赖 → 整份跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")

TENANT = "company_a"
DEPT = "d001"
DOC_ID = uuid.uuid4()
IMAGE_NAME = "page_1_image_1.png"


# ═══════════════════════════════════════════════════════════════════════════════
# 替身
# ═══════════════════════════════════════════════════════════════════════════════


class _StubUser:
    def __init__(self, username: str, *, clearance: int = 1, role: str = "user"):
        self.id = uuid.uuid4()
        self.username = username
        self.tenant_id = TENANT
        self.department_id = DEPT
        self.role = role
        self.clearance = clearance

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class _StubDoc:
    def __init__(self):
        self.id = DOC_ID
        self.filename = "受限文档.pdf"
        self.page_count = 1
        self.owner_id = uuid.uuid4()
        self.tenant_id = TENANT
        self.department_id = DEPT
        self.access_level = "tenant"
        self.security_level = 1
        self.visibility_mode = "tier"
        self.project_ids = []
        self.acl_allow = []
        self.acl_deny = []
        self.acl_expires_at = None


def _pred(clearance: int) -> ScopePredicate:
    return ScopePredicate(
        user_id=str(uuid.uuid4()),
        tenant_ids=frozenset({TENANT}),
        owns_tenant_ids=frozenset(),
        department_id=DEPT,
        tenant_wide=False,
        clearance=clearance,
        principals=frozenset(),
    )


class _FakeResult:
    def __init__(self, doc):
        self._doc = doc

    def scalar_one_or_none(self):
        return self._doc


class _FakeSession:
    def __init__(self, doc):
        self._doc = doc

    async def execute(self, *args, **kwargs):
        return _FakeResult(self._doc)


class _FakeQdrantClient:
    def __init__(self, points):
        self._points = points

    async def scroll(self, **kwargs):
        return list(self._points), None


class _Point:
    def __init__(self, pid, payload):
        self.id = pid
        self.payload = payload


# ═══════════════════════════════════════════════════════════════════════════════
# 端点级 fixtures（TestClient）
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def party():
    return SimpleNamespace(
        low=_StubUser("low", clearance=1),
        high=_StubUser("high", clearance=3),
        token_low="jwt-low",
        token_high="jwt-high",
    )


def _install_endpoint_patches(
    monkeypatch, party, *, doc, pred_by_token, image_info_factory, materialized=True,
    resolved_file=None,
):
    """把鉴权 / scope / DB / 图片反查 / 文件解析换成可控替身（路由与端点全是真实代码）."""

    async def _fake_resolve_user(token):
        mapping = {party.token_low: party.low, party.token_high: party.high}
        user = mapping.get(token or "")
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录")
        return user

    monkeypatch.setattr(api_deps, "_resolve_user", _fake_resolve_user)

    # 身份验证闸门（若开启）不得成为本测试的干扰项：直接视为已验证
    async def _fake_identity_verified(user, **kwargs):
        return True

    monkeypatch.setattr(
        "app.services.staff_service.is_identity_verified",
        _fake_identity_verified,
        raising=False,
    )

    @asynccontextmanager
    async def _fake_db():
        yield _FakeSession(doc)

    monkeypatch.setattr("app.db.postgres.get_db_session", _fake_db)

    async def _fake_request_scope(user):
        return DocumentScope(
            owner_id=user.id,
            tenant_ids=frozenset({TENANT}),
            owns_tenant_ids=frozenset(),
            department_id=DEPT,
            tenant_wide=False,
        )

    monkeypatch.setattr("app.api.document_management.request_scope", _fake_request_scope)
    monkeypatch.setattr("app.services.tenancy.request_scope", _fake_request_scope)

    async def _fake_security_scope(user, **kwargs):
        return SimpleNamespace(predicate=lambda: pred_by_token[user.username])

    monkeypatch.setattr(
        "app.api.document_management.request_security_scope", _fake_security_scope
    )
    monkeypatch.setattr(
        "app.services.security_scope.request_security_scope", _fake_security_scope
    )

    async def _fake_resolve_image_object_id(document_id, image_name, **kwargs):
        return image_info_factory(image_name)

    monkeypatch.setattr(
        "app.services.image_security.resolve_image_object_id", _fake_resolve_image_object_id
    )

    async def _fake_materialized(document_id, **kwargs):
        return materialized

    monkeypatch.setattr(
        "app.services.security_cascade.document_is_materialized", _fake_materialized
    )

    if resolved_file is not None:
        monkeypatch.setattr(
            "app.services.storage.resolve_image_path",
            lambda *a, **k: resolved_file,
        )

    drops: list[dict] = []

    async def _fake_record_acl_drop(stage, **kwargs):
        drops.append({"stage": stage, **kwargs})

    monkeypatch.setattr("app.services.audit_service.record_acl_drop", _fake_record_acl_drop)
    return drops


@pytest.fixture
def client(party, monkeypatch):
    app = FastAPI()
    app.include_router(documents_router)
    return TestClient(app)


def _get_image(client, token, *, bearer=True):
    if bearer:
        return client.get(
            f"/documents/{DOC_ID}/images/{IMAGE_NAME}",
            headers={"Authorization": f"Bearer {token}"},
        )
    return client.get(f"/documents/{DOC_ID}/images/{IMAGE_NAME}?token={token}")


def _image_info(*, effective_level: int, excluded: bool = False, security_level=None):
    return {
        "object_id": make_object_id(str(DOC_ID), "img_1", object_type="image"),
        "object_type": "image",
        "image_id": "img_1",
        "image_path": f"images/{IMAGE_NAME}",
        "tenant_id": TENANT,
        "owner_id": None,
        "department_id": DEPT,
        "access_level": "tenant",
        "visibility_mode": "tier",
        "project_ids": [],
        "security_level": security_level if security_level is not None else effective_level,
        "parent_security_level": 1,
        "effective_security_level": effective_level,
        "acl_allow": [],
        "acl_deny": [],
        "acl_expires_at": None,
        "excluded": excluded,
        "acl_sync_state": "synced",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ① 图片端点：提级 / 剔除 → 404，且文案与"文档不存在"逐字一致
# ═══════════════════════════════════════════════════════════════════════════════


def test_escalated_image_denied_and_indistinguishable_from_missing(
    client, party, monkeypatch, tmp_path
):
    img = tmp_path / IMAGE_NAME
    img.write_bytes(b"PNG")
    drops = _install_endpoint_patches(
        monkeypatch, party,
        doc=_StubDoc(),
        pred_by_token={"low": _pred(1), "high": _pred(3)},
        image_info_factory=lambda name: _image_info(effective_level=3),
        materialized=True,
        resolved_file=img,
    )

    denied = _get_image(client, party.token_low)
    assert denied.status_code == 404, denied.text
    assert denied.json() == {"detail": _DOCUMENT_404_DETAIL}

    # 与"文档不存在"逐字一致
    _install_endpoint_patches(
        monkeypatch, party,
        doc=None,
        pred_by_token={"low": _pred(1), "high": _pred(3)},
        image_info_factory=lambda name: None,
        materialized=False,
    )
    missing = _get_image(client, party.token_low)
    assert missing.status_code == 404
    assert denied.json() == missing.json(), "越权响应与'不存在'不一致 —— 泄露存在性"

    # 审计
    assert any(d["stage"] == "citation_open" for d in drops)


def test_excluded_image_denied(client, party, monkeypatch, tmp_path):
    img = tmp_path / IMAGE_NAME
    img.write_bytes(b"PNG")
    _install_endpoint_patches(
        monkeypatch, party,
        doc=_StubDoc(),
        pred_by_token={"low": _pred(3), "high": _pred(3)},
        image_info_factory=lambda name: _image_info(effective_level=1, excluded=True),
        materialized=True,
        resolved_file=img,
    )
    resp = _get_image(client, party.token_low)
    assert resp.status_code == 404, resp.text
    assert resp.json() == {"detail": _DOCUMENT_404_DETAIL}


def test_image_reverse_lookup_miss_fail_closed_when_materialized(
    client, party, monkeypatch, tmp_path
):
    img = tmp_path / IMAGE_NAME
    img.write_bytes(b"PNG")
    _install_endpoint_patches(
        monkeypatch, party,
        doc=_StubDoc(),
        pred_by_token={"low": _pred(3), "high": _pred(3)},
        image_info_factory=lambda name: None,      # 反查不到
        materialized=True,                          # 已物化 → fail-closed
        resolved_file=img,
    )
    resp = _get_image(client, party.token_low)
    assert resp.status_code == 404
    assert resp.json() == {"detail": _DOCUMENT_404_DETAIL}


def test_image_reverse_lookup_miss_falls_back_when_unmaterialized(
    client, party, monkeypatch, tmp_path
):
    """未物化（回填未覆盖）→ 回退文档级判定，不把存量图片集体判 404."""
    img = tmp_path / IMAGE_NAME
    img.write_bytes(b"PNG")
    _install_endpoint_patches(
        monkeypatch, party,
        doc=_StubDoc(),
        pred_by_token={"low": _pred(3), "high": _pred(3)},
        image_info_factory=lambda name: None,
        materialized=False,
        resolved_file=img,
    )
    resp = _get_image(client, party.token_low)
    assert resp.status_code == 200, resp.text


# ═══════════════════════════════════════════════════════════════════════════════
# ⑤ 阳性对照：高密级同租户用户必须能拿到原图
# ═══════════════════════════════════════════════════════════════════════════════


def test_positive_control_high_clearance_can_open_image(client, party, monkeypatch, tmp_path):
    img = tmp_path / IMAGE_NAME
    img.write_bytes(b"PNG-BYTES")
    _install_endpoint_patches(
        monkeypatch, party,
        doc=_StubDoc(),
        pred_by_token={"low": _pred(1), "high": _pred(3)},
        image_info_factory=lambda name: _image_info(effective_level=3),
        materialized=True,
        resolved_file=img,
    )
    ok = _get_image(client, party.token_high)
    assert ok.status_code == 200, ok.text
    assert ok.content == b"PNG-BYTES"


# ═══════════════════════════════════════════════════════════════════════════════
# ⑥ ?token= 与 Authorization 同判定
# ═══════════════════════════════════════════════════════════════════════════════


def test_query_token_channel_uses_same_decision(client, party, monkeypatch, tmp_path):
    img = tmp_path / IMAGE_NAME
    img.write_bytes(b"PNG")
    _install_endpoint_patches(
        monkeypatch, party,
        doc=_StubDoc(),
        pred_by_token={"low": _pred(1), "high": _pred(3)},
        image_info_factory=lambda name: _image_info(effective_level=3),
        materialized=True,
        resolved_file=img,
    )
    ok = _get_image(client, party.token_high, bearer=False)
    denied = _get_image(client, party.token_low, bearer=False)
    assert ok.status_code == 200, ok.text
    assert denied.status_code == 404, denied.text
    assert denied.json() == {"detail": _DOCUMENT_404_DETAIL}


# ═══════════════════════════════════════════════════════════════════════════════
# ②③④ 文本端点（service 层）：逐 chunk 过滤 / 无占位符 / 审计
# ═══════════════════════════════════════════════════════════════════════════════


def _points():
    return [
        _Point("p0", {"chunk_index": 0, "page_number": 1, "text": "正常正文",
                      "content_type": "text"}),
        _Point("p1", {"chunk_index": 1, "page_number": 1, "text": "机密表格",
                      "content_type": "table", "image_id": "img_1"}),
    ]


def _install_service_patches(monkeypatch, *, materialized=True, views=None, drops=None):
    @asynccontextmanager
    async def _fake_db():
        yield _FakeSession(_StubDoc())

    # 注意：document_query_service 是**模块级** `from app.db.postgres import
    # get_db_session`，因此必须打它自己的模块命名空间，不能只打 app.db.postgres。
    monkeypatch.setattr("app.services.document_query_service.get_db_session", _fake_db)
    monkeypatch.setattr("app.db.postgres.get_db_session", _fake_db)
    monkeypatch.setattr(
        "app.db.qdrant.get_qdrant_client", lambda: _FakeQdrantClient(_points())
    )

    async def _fake_materialized(document_id, **kwargs):
        return materialized

    async def _fake_load_object_views(object_ids, **kwargs):
        return views or {}

    monkeypatch.setattr(
        "app.services.security_cascade.document_is_materialized", _fake_materialized
    )
    monkeypatch.setattr(
        "app.services.security_cascade.load_object_views", _fake_load_object_views
    )

    recorded = drops if drops is not None else []

    async def _fake_record_acl_drop(stage, **kwargs):
        recorded.append({"stage": stage, **kwargs})

    monkeypatch.setattr("app.services.audit_service.record_acl_drop", _fake_record_acl_drop)
    return recorded


def _obj_id(pid: str, content_type: str) -> str:
    return make_object_id(str(DOC_ID), pid, object_type=object_type_from_content_type(content_type))


def _view_dict(object_id: str, *, excluded=False, effective_level=1):
    from app.services.security_policy import ObjectACLView

    return ObjectACLView(
        object_id=object_id,
        document_id=str(DOC_ID),
        tenant_id=TENANT,
        access_level="tenant",
        security_level=effective_level,
        effective_security_level=effective_level,
        excluded=excluded,
    )


def test_chunks_endpoint_service_drops_derived_and_no_placeholder(monkeypatch):
    from app.services.document_query_service import get_document_chunks

    views = {
        _obj_id("p0", "text"): _view_dict(_obj_id("p0", "text")),
        _obj_id("p1", "table"): _view_dict(_obj_id("p1", "table"), excluded=True),
    }
    drops = _install_service_patches(monkeypatch, materialized=True, views=views)

    result = asyncio.run(
        get_document_chunks(DOC_ID, owner_id=_StubDoc().owner_id,
                            tenant_ids=frozenset({TENANT}), pred=_pred(3))
    )
    texts = [c["text"] for c in result["chunks"]]
    assert "正常正文" in texts
    assert "机密表格" not in texts, "被剔除的派生块不得出现在返回列表"
    assert all("占位" not in t for t in texts)
    assert any(d["stage"] == "citation_open" for d in drops)


def test_chunks_endpoint_service_positive_control(monkeypatch):
    from app.services.document_query_service import get_document_chunks

    views = {
        _obj_id("p0", "text"): _view_dict(_obj_id("p0", "text")),
        _obj_id("p1", "table"): _view_dict(_obj_id("p1", "table")),
    }
    _install_service_patches(monkeypatch, materialized=True, views=views)
    result = asyncio.run(
        get_document_chunks(DOC_ID, owner_id=_StubDoc().owner_id,
                            tenant_ids=frozenset({TENANT}), pred=_pred(3))
    )
    assert result["total"] == 2
    assert {c["text"] for c in result["chunks"]} == {"正常正文", "机密表格"}


def test_chunks_endpoint_service_escalated_chunk_dropped(monkeypatch):
    from app.services.document_query_service import get_document_chunks

    views = {
        _obj_id("p0", "text"): _view_dict(_obj_id("p0", "text")),
        _obj_id("p1", "table"): _view_dict(_obj_id("p1", "table"), effective_level=3),
    }
    _install_service_patches(monkeypatch, materialized=True, views=views)
    result = asyncio.run(
        get_document_chunks(DOC_ID, owner_id=_StubDoc().owner_id,
                            tenant_ids=frozenset({TENANT}), pred=_pred(1))
    )
    assert result["total"] == 1
    assert result["chunks"][0]["text"] == "正常正文"


def test_chunks_endpoint_wiring_passes_pred(client, party, monkeypatch):
    """端点必须把重新签发的 ScopePredicate 传进 get_document_chunks（含 ?token 通道）."""
    captured = {}

    async def _fake_get_document_chunks(document_id, **kwargs):
        captured.update(kwargs)
        return {"document_id": str(document_id), "filename": "x", "page_count": 1,
                "total": 0, "chunks": []}

    monkeypatch.setattr(
        "app.api.document_management.get_document_chunks", _fake_get_document_chunks
    )
    _install_endpoint_patches(
        monkeypatch, party,
        doc=_StubDoc(),
        pred_by_token={"low": _pred(1), "high": _pred(3)},
        image_info_factory=lambda name: None,
        materialized=False,
    )
    resp = client.get(
        f"/documents/{DOC_ID}/chunks",
        headers={"Authorization": f"Bearer {party.token_low}"},
    )
    assert resp.status_code == 200, resp.text
    assert "pred" in captured and isinstance(captured["pred"], ScopePredicate)
