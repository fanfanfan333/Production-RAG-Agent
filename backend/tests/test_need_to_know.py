"""
T5 —— need-to-know 授予（P1-1）验收：带有效期例外生效 / 过期失效 / **禁止自我授予**
/ 无权限角色不能授予 / **阳性对照**（审批后确实能读到）.

设计依据：``docs/system_design_security_isolation.md`` 决策 13 / §4.5 / §15-13。

护栏（为什么阴性 + 阳性都要写）
──────────────────────────────
只写"无权者被 403""过期读不到"这类**阴性**用例是假测试：如果整条 need-to-know
链路坏掉（谁都授不上、或物化根本没写 acl_allow），阴性用例照样"通过"。
**阳性对照**负责证明合法路径仍然可用：批准后把主体写进 ``acl_allow``，
被授权人（clearance 不足）确实能读到。
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
    from app.db.security_models import GRANT_STATUS_APPROVED, GRANT_STATUS_PENDING
    from app.services.grant_service import (
        GrantError,
        compute_object_acl,
        is_self_grant,
        materialize_document_acl,
        parse_subject,
        request_grant,
        review_grant,
    )
    from app.services.security_policy import ObjectACLView, ScopePredicate, allows
except ImportError as exc:      # 宿主机缺依赖 → 整份跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")

FIXED_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
PAST = FIXED_NOW - timedelta(days=1)
FUTURE = FIXED_NOW + timedelta(days=1)

TENANT = "cleanA"
DOC_ID = str(uuid.uuid4())
GRANTEE = str(uuid.uuid4())
GRANTOR = str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════════════════════
# 替身
# ═══════════════════════════════════════════════════════════════════════════════


class _StubUser:
    def __init__(self, *, role: str = "kb_admin", is_admin: bool = False,
                 tenant_id: str = TENANT, uid: str | None = None):
        self.id = uuid.UUID(uid) if uid else uuid.uuid4()
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

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, *, by_entity=None, get_map=None):
        self.by_entity = by_entity or {}
        self.get_map = get_map or {}
        self.added = []

    async def get(self, model, ident):
        return self.get_map.get((model.__tablename__, _key(ident)))

    async def scalars(self, stmt):
        entity = stmt.column_descriptions[0].get("entity")
        return _FakeScalars(self.by_entity.get(entity, []))

    def add(self, obj):
        self.added.append(obj)

    async def flush(self) -> None:
        return None

    async def refresh(self, _obj) -> None:
        return None


@pytest.fixture(autouse=True)
def _silence_audit(monkeypatch):
    calls: list[tuple] = []

    async def _fake(action, **kwargs):
        calls.append((action, kwargs))

    monkeypatch.setattr("app.services.grant_service.record_audit", _fake)
    return calls


def _run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════════════
# ① 纯函数：主体解析 / 自我授予判定
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "subject,expected",
    [
        ("user:abc", ("user", "abc")),
        ("dept:d1", ("dept", "d1")),
        ("role:employee", ("role", "employee")),
        ("project:p_alpha", ("project", "p_alpha")),
        ("group:g1", ("group", "g1")),
    ],
)
def test_parse_subject_ok(subject, expected) -> None:
    assert parse_subject(subject) == expected


@pytest.mark.parametrize("bad", ["", "user", "user:", ":x", "bogus:1"])
def test_parse_subject_rejects(bad) -> None:
    with pytest.raises(GrantError):
        parse_subject(bad)


def test_is_self_grant() -> None:
    me = str(uuid.uuid4())
    assert is_self_grant(f"user:{me}", me) is True
    assert is_self_grant(f"user:{uuid.uuid4()}", me) is False
    assert is_self_grant("role:admin", me) is False
    assert is_self_grant(None, me) is False


# ═══════════════════════════════════════════════════════════════════════════════
# ② 物化计算（纯函数）：批准+未过期 → 进 acl_allow；pending/过期 → 不进；派生 → 空
# ═══════════════════════════════════════════════════════════════════════════════


def _grant(*, object_id, subject, status=GRANT_STATUS_APPROVED, expires=FUTURE,
           effect="allow"):
    return SimpleNamespace(
        object_id=object_id, subject=subject, effect=effect, status=status,
        expires_at=expires, document_id=DOC_ID,
    )


def _rows():
    return [
        SimpleNamespace(object_id=DOC_ID, object_type="doc", image_id=None),
        SimpleNamespace(object_id=f"{DOC_ID}::c0", object_type="text_chunk", image_id=None),
        SimpleNamespace(object_id=f"{DOC_ID}::img1", object_type="image", image_id="img1"),
        SimpleNamespace(object_id=f"{DOC_ID}::d0", object_type="text_chunk", image_id="img1"),
    ]


def test_compute_object_acl_approved_unexpired() -> None:
    grants = [_grant(object_id=DOC_ID, subject=f"user:{GRANTEE}")]
    mapping = compute_object_acl(_rows(), grants, doc_object_id=DOC_ID, now=FIXED_NOW)
    # 文档级授予 → doc 行 + 普通分块命中
    assert mapping[DOC_ID][0] == [f"user:{GRANTEE}"]
    assert mapping[f"{DOC_ID}::c0"][0] == [f"user:{GRANTEE}"]
    # 图片行继承文档级主体
    assert mapping[f"{DOC_ID}::img1"][0] == [f"user:{GRANTEE}"]
    # **OCR 派生块恒空集**（共享知识 9）
    assert mapping[f"{DOC_ID}::d0"][0] == []


def test_compute_object_acl_pending_does_not_grant() -> None:
    """pending 期间**不写** acl_allow ⇒ A7（中间态他人不可见）天然成立。"""
    grants = [_grant(object_id=DOC_ID, subject=f"user:{GRANTEE}", status=GRANT_STATUS_PENDING)]
    mapping = compute_object_acl(_rows(), grants, doc_object_id=DOC_ID, now=FIXED_NOW)
    assert mapping[DOC_ID][0] == []


def test_compute_object_acl_expired_does_not_grant() -> None:
    grants = [_grant(object_id=DOC_ID, subject=f"user:{GRANTEE}", expires=PAST)]
    mapping = compute_object_acl(_rows(), grants, doc_object_id=DOC_ID, now=FIXED_NOW)
    assert mapping[DOC_ID][0] == []


def test_compute_object_acl_takes_earliest_expiry() -> None:
    grants = [
        _grant(object_id=DOC_ID, subject="user:a", expires=FUTURE),
        _grant(object_id=DOC_ID, subject="user:b", expires=FUTURE + timedelta(days=5)),
    ]
    mapping = compute_object_acl(_rows(), grants, doc_object_id=DOC_ID, now=FIXED_NOW)
    assert mapping[DOC_ID][1] == FUTURE      # 取**最早**到期时间


# ═══════════════════════════════════════════════════════════════════════════════
# ③ **阳性对照**：批准 → 物化 → 被授权人（clearance 不足）确实能读到
# ═══════════════════════════════════════════════════════════════════════════════


def _secret_view(acl_allow, acl_expires_at):
    return ObjectACLView(
        object_id=DOC_ID, object_type="doc", document_id=DOC_ID,
        tenant_id=TENANT, owner_id=str(uuid.uuid4()),
        access_level="department", visibility_mode="tier",
        security_level=3, effective_security_level=3,
        acl_allow=frozenset(acl_allow), acl_expires_at=acl_expires_at,
    )


def _grantee_pred():
    return ScopePredicate(
        user_id=GRANTEE, tenant_ids=frozenset({TENANT}), department_id="d001",
        clearance=1, principals=frozenset({f"user:{GRANTEE}", "role:employee"}),
        now=FIXED_NOW,
    )


def test_positive_control_approved_grant_is_readable() -> None:
    """**阳性对照**：有审批权限的角色批准后，被授权人**确实能读到**（否则整体坏掉）。"""
    from app.db.security_models import AclGrant, DocumentObject

    grant = SimpleNamespace(
        object_id=DOC_ID, subject=f"user:{GRANTEE}", effect="allow",
        status=GRANT_STATUS_APPROVED, expires_at=FUTURE, document_id=DOC_ID,
    )
    doc_row = SimpleNamespace(
        id=uuid.UUID(DOC_ID), tenant_id=TENANT, acl_allow=[], acl_expires_at=None,
        acl_sync_state="synced",
    )
    object_rows = _rows()
    for r in object_rows:
        r.acl_allow = []
        r.acl_expires_at = None
        r.acl_sync_state = "synced"
        r.document_id = uuid.UUID(DOC_ID)

    session = _FakeSession(
        by_entity={DocumentObject: object_rows, AclGrant: [grant]},
        get_map={("documents", DOC_ID): doc_row},
    )

    result = _run(materialize_document_acl(DOC_ID, session=session, push=False))

    # 物化副本确实写进了 acl_allow
    assert doc_row.acl_allow == [f"user:{GRANTEE}"]
    assert result["doc_allow"] == [f"user:{GRANTEE}"]

    # 被授权人 clearance=1 < 文档密级 3 → 通常读不到；命中未过期例外 ⇒ **能读到**
    view = _secret_view(doc_row.acl_allow, doc_row.acl_expires_at)
    assert allows(_grantee_pred(), view).allowed is True


def test_expired_grant_is_not_readable() -> None:
    """过期即失效：同样的密级下，过期的例外**读不到**（阴性对照）。"""
    view = _secret_view(acl_allow=[f"user:{GRANTEE}"], acl_expires_at=PAST)
    assert allows(_grantee_pred(), view).allowed is False


# ═══════════════════════════════════════════════════════════════════════════════
# ④ 服务层：禁止自我授予 / 必带有效期 / 审批约束
# ═══════════════════════════════════════════════════════════════════════════════


def test_request_grant_self_is_forbidden() -> None:
    """**禁止自我授予**（服务层第一道）→ 403。"""
    actor = _StubUser(uid=GRANTOR)
    with pytest.raises(GrantError) as exc:
        _run(request_grant(
            actor, DOC_ID, f"user:{GRANTOR}", expires_at=FUTURE,
            session=_FakeSession(),
        ))
    assert exc.value.status_code == 403


def test_request_grant_requires_expiry() -> None:
    actor = _StubUser()
    with pytest.raises(GrantError) as exc:
        _run(request_grant(actor, DOC_ID, f"user:{GRANTEE}", expires_at=None,
                           session=_FakeSession()))
    assert exc.value.status_code == 400


def test_request_grant_rejects_past_expiry() -> None:
    actor = _StubUser()
    with pytest.raises(GrantError) as exc:
        _run(request_grant(actor, DOC_ID, f"user:{GRANTEE}", expires_at=PAST,
                           session=_FakeSession()))
    assert exc.value.status_code == 400


def test_review_rejects_self_approval() -> None:
    """审批人不得审批"授予给自己"的申请 → 403。"""
    grant = SimpleNamespace(
        id=uuid.uuid4(), object_id=DOC_ID, document_id=uuid.UUID(DOC_ID),
        subject=f"user:{GRANTOR}", effect="allow", status=GRANT_STATUS_PENDING,
        expires_at=FUTURE, reason=None, reviewer_id=None, reviewed_at=None,
    )
    reviewer = _StubUser(uid=GRANTOR)
    from app.db.security_models import AclGrant

    session = _FakeSession(get_map={("acl_grants", str(grant.id)): grant})
    with pytest.raises(GrantError) as exc:
        _run(review_grant(reviewer, grant.id, approve=True, session=session))
    assert exc.value.status_code == 403
    _ = AclGrant


# ═══════════════════════════════════════════════════════════════════════════════
# ⑤ API 权限门禁：无权限角色不能授予（阴性）+ 自我授予 403（API 层）
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(get_settings(), "REQUIRE_IDENTITY_VERIFICATION", False, raising=False)

    employee = _StubUser(role="employee")
    admin = _StubUser(role="kb_admin")
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
    tc._party = SimpleNamespace(employee=employee, admin=admin)
    return tc


def _hdr(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _grant_body(subject: str) -> dict:
    return {
        "document_id": DOC_ID,
        "subject": subject,
        "effect": "allow",
        "expires_at": FUTURE.isoformat(),
    }


def test_no_permission_role_cannot_request_grant(client) -> None:
    """**无权限角色不能授予**（阴性）：employee 没有 security.grant → 403。"""
    resp = client.post(
        "/security/grants", json=_grant_body(f"user:{GRANTEE}"), headers=_hdr("tok-emp")
    )
    assert resp.status_code == 403, resp.text


def test_self_grant_blocked_at_api_layer(client) -> None:
    """**禁止自我授予**（API 层拦截）：主体 = 自己 → 403。"""
    me = str(client._party.admin.id)
    resp = client.post(
        "/security/grants", json=_grant_body(f"user:{me}"), headers=_hdr("tok-admin")
    )
    assert resp.status_code == 403, resp.text
    assert "自我授予" in resp.text


def test_positive_control_admin_can_request_and_review(client, monkeypatch) -> None:
    """**阳性对照**：有权限的角色授给别人 → 201；审批通过 → 200，被授权人可读。"""
    created = SimpleNamespace(
        id=uuid.uuid4(), document_id=DOC_ID, object_id=DOC_ID,
        subject=f"user:{GRANTEE}", effect="allow", status=GRANT_STATUS_PENDING,
        granted_by=client._party.admin.id, reviewer_id=None, reason=None,
        expires_at=FUTURE, created_at=FIXED_NOW, reviewed_at=None,
    )
    approved = SimpleNamespace(**{**created.__dict__, "status": GRANT_STATUS_APPROVED,
                                  "reviewer_id": client._party.admin.id,
                                  "reviewed_at": FIXED_NOW})

    async def _fake_request(*_a, **_k):
        return created

    async def _fake_review(*_a, **_k):
        return approved

    monkeypatch.setattr("app.api.security.request_grant", _fake_request)
    monkeypatch.setattr("app.api.security.review_grant", _fake_review)

    r1 = client.post(
        "/security/grants", json=_grant_body(f"user:{GRANTEE}"), headers=_hdr("tok-admin")
    )
    assert r1.status_code == 201, r1.text
    grant_id = r1.json()["grant_id"]

    r2 = client.post(
        f"/security/grants/{grant_id}/review",
        json={"approve": True, "comment": "ok"},
        headers=_hdr("tok-admin"),
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["status"] == GRANT_STATUS_APPROVED

    # 物化副本（由服务层 compute_object_acl 决定）→ 被授权人可读（见 ③ 的阳性对照）
    mapping = compute_object_acl(
        _rows(), [_grant(object_id=DOC_ID, subject=f"user:{GRANTEE}")],
        doc_object_id=DOC_ID, now=FIXED_NOW,
    )
    view = _secret_view(mapping[DOC_ID][0], mapping[DOC_ID][1])
    assert allows(_grantee_pred(), view).allowed is True
