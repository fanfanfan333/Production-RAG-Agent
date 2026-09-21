"""
T5 —— 项目维度（P0）验收：跨部门成员可见 / 同部门非成员不可见 / tier 不受影响 /
跨租户项目不存在 / 临时成员到期失效.

设计依据：``docs/system_design_security_isolation.md`` 决策 4（项目作为
``_source_gate`` 的第四个 OR 分支）+ §4.4 + A6。

本文件不连数据库：DB 访问路径用**极小替身会话**，判定路径用**纯函数**
（``allows()``）。这样"项目只增加可见性、不改变三层知识库"这条红线有可执行、
可回归的断言。
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import pytest

    from app.db.security_models import Project, ProjectMember
    from app.services.project_service import (
        ProjectError,
        add_member,
        create_project,
        get_project,
        member_is_active,
        normalize_name,
        project_ids_of,
        remove_member,
        validate_project_id,
    )
    from app.services.security_policy import ObjectACLView, ScopePredicate, allows
except ImportError as exc:      # 宿主机缺依赖 → 整份跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")

FIXED_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
PAST = FIXED_NOW - timedelta(days=1)
FUTURE = FIXED_NOW + timedelta(days=1)

TENANT_A = "cleanA"
TENANT_B = "cleanB"
DEPT_D1 = "d001"
DEPT_D2 = "d002"
PROJECT_ALPHA = "p_alpha"

U1 = str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════════════════════
# 替身：用户 / 极小 AsyncSession
# ═══════════════════════════════════════════════════════════════════════════════


class _StubUser:
    def __init__(
        self,
        *,
        tenant_id: str = TENANT_A,
        department_id: str | None = DEPT_D1,
        role: str = "kb_admin",
        is_admin: bool = False,
    ):
        self.id = uuid.uuid4()
        self.username = f"u_{uuid.uuid4().hex[:6]}"
        self.tenant_id = tenant_id
        self.department_id = department_id
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
    """只实现被测路径用到的 get / scalars / add / delete / flush。"""

    def __init__(self, *, by_entity=None, get_map=None):
        self.by_entity = by_entity or {}
        self.get_map = get_map or {}
        self.added = []
        self.deleted = []

    async def get(self, model, ident):
        return self.get_map.get((model.__tablename__, _key(ident)))

    async def scalars(self, stmt):
        entity = stmt.column_descriptions[0].get("entity")
        return _FakeScalars(self.by_entity.get(entity, []))

    def add(self, obj):
        self.added.append(obj)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def flush(self) -> None:
        return None

    async def refresh(self, _obj) -> None:
        return None


@pytest.fixture(autouse=True)
def _silence_audit(monkeypatch):
    """审计写 DB —— 测试里换成内存记录（不碰真实库）。"""
    calls: list[tuple] = []

    async def _fake(action, **kwargs):
        calls.append((action, kwargs))

    monkeypatch.setattr("app.services.project_service.record_audit", _fake)
    return calls


# ═══════════════════════════════════════════════════════════════════════════════
# ① 纯函数：临时成员到期失效（P1-2）
# ═══════════════════════════════════════════════════════════════════════════════


def test_member_is_active_semantics() -> None:
    class _M:
        def __init__(self, exp):
            self.expires_at = exp

    assert member_is_active(_M(None), FIXED_NOW) is True          # 长期成员
    assert member_is_active(_M(FUTURE), FIXED_NOW) is True        # 未到期
    assert member_is_active(_M(PAST), FIXED_NOW) is False         # 已到期
    # naive 时间按 UTC 处理（前端可能回传无时区 ISO）
    assert member_is_active(_M(PAST.replace(tzinfo=None)), FIXED_NOW) is False


def test_project_ids_of_filters_expired() -> None:
    class _M:
        def __init__(self, pid, exp):
            self.project_id = pid
            self.expires_at = exp

    members = [
        _M("p_alpha", FUTURE),
        _M("p_beta", None),
        _M("p_gamma", PAST),          # 过期 → 不进入
    ]
    assert project_ids_of(members, FIXED_NOW) == frozenset({"p_alpha", "p_beta"})


def test_validate_project_id_rejects_unsafe() -> None:
    assert validate_project_id("p_alpha") == "p_alpha"
    for bad in ("", "  ", "p alpha", "a/b", "x" * 65, "p;drop"):
        with pytest.raises(ProjectError):
            validate_project_id(bad)


def test_normalize_name_rejects_empty() -> None:
    assert normalize_name("  项目甲 ") == "项目甲"
    with pytest.raises(ProjectError):
        normalize_name("   ")
    with pytest.raises(ProjectError):
        normalize_name("x" * 129)


# ═══════════════════════════════════════════════════════════════════════════════
# ② 判定矩阵（A6）：项目只增加可见性，tier 完全不受影响
# ═══════════════════════════════════════════════════════════════════════════════


def _pred(*, user_id=U1, dept=DEPT_D1, project_ids=frozenset({PROJECT_ALPHA})):
    return ScopePredicate(
        user_id=user_id,
        tenant_ids=frozenset({TENANT_A}),
        department_id=dept,
        clearance=1,
        project_ids=project_ids,
        principals=frozenset({f"user:{user_id}", f"dept:{dept}", "role:employee"}),
        now=FIXED_NOW,
    )


def _project_doc(*, dept, project_ids, visibility="project", level=1):
    return ObjectACLView(
        object_id=DOC_ID,
        object_type="doc",
        document_id=DOC_ID,
        tenant_id=TENANT_A,
        owner_id=str(uuid.uuid4()),
        department_id=dept,
        access_level="department",
        visibility_mode=visibility,
        project_ids=frozenset(project_ids),
        security_level=level,
        effective_security_level=level,
    )


DOC_ID = str(uuid.uuid4())


def test_cross_department_project_member_can_read() -> None:
    """跨部门项目成员**可见**：项目是横向维度，部门不参与判定（决策 4）。"""
    doc = _project_doc(dept=DEPT_D2, project_ids={PROJECT_ALPHA})   # 别的部门
    decision = allows(_pred(dept=DEPT_D1), doc)                     # 我属 D1，但在 p_alpha
    assert decision.allowed is True, decision


def test_same_department_non_member_cannot_read() -> None:
    """**同部门但非成员不可见**：project 模式闸门在部门闸门之前（判定顺序不可调）。"""
    doc = _project_doc(dept=DEPT_D1, project_ids={"p_someone_else"})
    decision = allows(_pred(dept=DEPT_D1, project_ids=frozenset({PROJECT_ALPHA})), doc)
    assert decision.allowed is False
    assert decision.gate == "project"


def test_tier_document_unaffected_by_project_dimension() -> None:
    """存量 ``visibility_mode=tier`` 文档**不受**项目维度影响（共享知识 5，零变化）。"""
    # 用户不属于任何项目，仍能看同部门库文档（tier 模式）
    doc = _project_doc(dept=DEPT_D1, project_ids=set(), visibility="tier")
    decision = allows(_pred(project_ids=frozenset()), doc)
    assert decision.allowed is True, decision

    # 甚至文档带了 project_ids，只要 visibility_mode=tier，就不参与判定
    doc2 = _project_doc(dept=DEPT_D1, project_ids={"p_unrelated"}, visibility="tier")
    assert allows(_pred(project_ids=frozenset()), doc2).allowed is True


def test_project_does_not_open_private_documents() -> None:
    """项目是 OR 分支，但**不能**撬开别人的个人库（产品红线 + B2/B6）。"""
    doc = ObjectACLView(
        object_id=DOC_ID, object_type="doc", document_id=DOC_ID,
        tenant_id=TENANT_A, owner_id=str(uuid.uuid4()), department_id=DEPT_D2,
        access_level="private", visibility_mode="project",
        project_ids=frozenset({PROJECT_ALPHA}), security_level=1,
        effective_security_level=1,
    )
    assert allows(_pred(), doc).allowed is False


# ═══════════════════════════════════════════════════════════════════════════════
# ③ 服务层：租户边界 / 跨部门 / 到期（替身会话）
# ═══════════════════════════════════════════════════════════════════════════════


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_create_project_non_admin_forced_to_own_tenant() -> None:
    actor = _StubUser(tenant_id=TENANT_A)
    project = _run(create_project(
        actor, "Alpha", project_id="p_alpha_x", session=_FakeSession()
    ))
    assert project.tenant_id == TENANT_A


def test_create_project_admin_can_target_tenant() -> None:
    admin = _StubUser(tenant_id="default", is_admin=True, role="admin")
    project = _run(create_project(
        admin, "Beta", project_id="p_beta_x", tenant_id=TENANT_B,
        session=_FakeSession(),
    ))
    assert project.tenant_id == TENANT_B


def test_project_in_other_tenant_is_invisible() -> None:
    """跨租户项目**不存在**（404，不泄露存在性）。"""
    actor = _StubUser(tenant_id=TENANT_B)
    foreign = Project(id="p_foreign", tenant_id=TENANT_A, name="别家公司项目")
    session = _FakeSession(get_map={("projects", "p_foreign"): foreign})
    with pytest.raises(ProjectError) as exc:
        _run(get_project(actor, "p_foreign", session=session))
    assert exc.value.status_code == 404


def test_add_member_cross_department_allowed() -> None:
    """成员**可跨部门**（跨部门项目组只看成员身份）。"""
    actor = _StubUser(tenant_id=TENANT_A, department_id=DEPT_D1, is_admin=True, role="admin")
    project = Project(id="p_multi", tenant_id=TENANT_A, name="跨部门项目")
    member_user = _StubUser(tenant_id=TENANT_A, department_id=DEPT_D2)
    session = _FakeSession(get_map={
        ("projects", "p_multi"): project,
        ("users", str(member_user.id)): member_user,
        ("project_members", ("p_multi", str(member_user.id))): None,
    })
    member, created = _run(add_member(
        actor, "p_multi", member_user.id, expires_at=FUTURE, session=session
    ))
    assert created is True
    assert str(member.user_id) == str(member_user.id)


def test_add_member_cross_tenant_rejected() -> None:
    """成员必须与项目**同租户**（项目不跨租户）→ 400。"""
    actor = _StubUser(tenant_id=TENANT_A, is_admin=True, role="admin")
    project = Project(id="p_solo", tenant_id=TENANT_A, name="本公司项目")
    outsider = _StubUser(tenant_id=TENANT_B)
    session = _FakeSession(get_map={
        ("projects", "p_solo"): project,
        ("users", str(outsider.id)): outsider,
    })
    with pytest.raises(ProjectError) as exc:
        _run(add_member(actor, "p_solo", outsider.id, session=session))
    assert exc.value.status_code == 400


def test_remove_missing_member_404() -> None:
    actor = _StubUser(tenant_id=TENANT_A, is_admin=True, role="admin")
    project = Project(id="p_solo2", tenant_id=TENANT_A, name="项目")
    session = _FakeSession(get_map={
        ("projects", "p_solo2"): project,
        ("project_members", ("p_solo2", U1)): None,
    })
    with pytest.raises(ProjectError) as exc:
        _run(remove_member(actor, "p_solo2", U1, session=session))
    assert exc.value.status_code == 404
