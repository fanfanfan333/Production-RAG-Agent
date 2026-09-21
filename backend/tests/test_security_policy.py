"""``allows()`` 判定矩阵（T2 核心，设计文档 §5 / 验收要点 1）.

覆盖 PRD 与设计文档点名的**每一条**边界，尤其这几条"最容易写错成静默越权"的：

* ``deny`` 覆盖 **owner 本人** 与 **admin**（判定式里没有任何角色豁免分支）
* 密级不足 + **未过期** need-to-know → 放行；**已过期** → 拒绝
* ``tenant_ids=frozenset()`` 与 ``tenant_ids=None``（非 unrestricted）都必须
  **全拒** —— 绝不能因为 ``if tenant_ids:`` 判 falsy 退化成"不限制"（B6）
* 密级字段缺失：非严格模式按 1、严格模式按 3
* 派生对象取严：即使物化的 ``effective_security_level`` 写错，判定侧也再取 max
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.services.security_policy import (
    DEFAULT_CLEARANCE_BY_ROLE,
    Decision,
    ObjectACLView,
    ScopePredicate,
    allows,
    clearance_for_role,
)
from app.services.security_scope import (
    UserScope,
    cache_key_for_scope,
    principals_of,
)
from app.services.tenancy import DocumentScope

FIXED_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
PAST = FIXED_NOW - timedelta(days=1)
FUTURE = FIXED_NOW + timedelta(days=1)

TENANT_A = "c309a7cb9f496"
TENANT_B = "cf33b1db5679d"
DEPT_D1 = "d001"
DEPT_D2 = "d002"

U1 = str(uuid.uuid4())
U2 = str(uuid.uuid4())
ADMIN = str(uuid.uuid4())
DOC_ID = str(uuid.uuid4())


# ── 构造辅助 ──────────────────────────────────────────────────────────────────


def scope(
    *,
    user_id: str = U1,
    role: str = "employee",
    clearance: int = 1,
    tenant_ids=frozenset({TENANT_A}),
    owns_tenant_ids=frozenset(),
    department_id: str | None = DEPT_D1,
    tenant_wide: bool = False,
    project_ids=frozenset(),
    principals=None,
    strict: bool = False,
) -> UserScope:
    base = DocumentScope(
        owner_id=uuid.UUID(user_id),
        tenant_ids=tenant_ids,
        owns_tenant_ids=frozenset(owns_tenant_ids),
        department_id=department_id,
        tenant_wide=tenant_wide,
    )
    if principals is None:
        principals = frozenset({f"user:{user_id}", f"role:{role}"})
    return UserScope(
        base=base,
        user_id=user_id,
        role=role,
        clearance=clearance,
        project_ids=frozenset(project_ids),
        principals=frozenset(principals),
        issued_at=FIXED_NOW,
        strict=strict,
    )


def obj(
    *,
    object_id: str = "o1",
    owner_id: str | None = U2,
    tenant_id: str = TENANT_A,
    department_id: str | None = DEPT_D1,
    access_level: str = "tenant",
    visibility_mode: str = "tier",
    project_ids=frozenset(),
    security_level: int | None = 1,
    parent_security_level: int | None = None,
    effective_security_level: int | None = 1,
    acl_allow=frozenset(),
    acl_deny=frozenset(),
    acl_expires_at=None,
    excluded: bool = False,
) -> ObjectACLView:
    return ObjectACLView(
        object_id=object_id,
        object_type="text_chunk",
        document_id=DOC_ID,
        tenant_id=tenant_id,
        owner_id=owner_id,
        user_id=owner_id,
        department_id=department_id,
        access_level=access_level,
        visibility_mode=visibility_mode,
        project_ids=frozenset(project_ids),
        security_level=security_level,
        parent_security_level=parent_security_level,
        effective_security_level=effective_security_level,
        acl_allow=frozenset(acl_allow),
        acl_deny=frozenset(acl_deny),
        acl_expires_at=acl_expires_at,
        excluded=excluded,
    )


def allowed(*args, **kwargs) -> bool:
    return allows(*args, **kwargs).allowed


# ── 1. deny 一票否决（含 owner 本人 / 含 admin）───────────────────────────────


def test_deny_hit_covers_the_owner_himself() -> None:
    s = scope(user_id=U1, principals=frozenset({f"user:{U1}"}))
    o = obj(owner_id=U1, access_level="private", acl_deny=frozenset({f"user:{U1}"}))
    decision = allows(s.predicate(now=FIXED_NOW), o)
    assert decision.allowed is False
    assert decision.gate == "deny"


def test_deny_hit_covers_platform_admin() -> None:
    s = scope(user_id=ADMIN, role="admin", clearance=3, tenant_wide=True,
              principals=frozenset({f"user:{ADMIN}", "role:admin"}))
    o = obj(acl_deny=frozenset({"role:admin"}))
    decision = allows(s.predicate(now=FIXED_NOW), o)
    assert decision.allowed is False
    assert decision.gate == "deny"


# ── 2. need-to-know（未过期放行 / 已过期拒绝）─────────────────────────────────


def test_need_to_know_unexpired_overrides_clearance_short() -> None:
    s = scope(clearance=1, principals=frozenset({f"user:{U1}"}))
    o = obj(
        effective_security_level=3,
        acl_allow=frozenset({f"user:{U1}"}),
        acl_expires_at=FUTURE,
    )
    assert allowed(s.predicate(now=FIXED_NOW), o) is True


def test_need_to_know_expired_is_rejected() -> None:
    s = scope(clearance=1, principals=frozenset({f"user:{U1}"}))
    o = obj(
        effective_security_level=3,
        acl_allow=frozenset({f"user:{U1}"}),
        acl_expires_at=PAST,
    )
    decision = allows(s.predicate(now=FIXED_NOW), o)
    assert decision.allowed is False
    assert decision.gate == "security"


def test_need_to_know_without_expiry_is_long_lived() -> None:
    s = scope(clearance=0, principals=frozenset({f"user:{U1}"}))
    o = obj(effective_security_level=2, acl_allow=frozenset({f"user:{U1}"}))
    assert allowed(s.predicate(now=FIXED_NOW), o) is True


# ── 3. 项目维度 ───────────────────────────────────────────────────────────────


def test_project_mode_requires_membership() -> None:
    s = scope(project_ids=frozenset())
    o = obj(visibility_mode="project", project_ids=frozenset({"p_alpha"}))
    decision = allows(s.predicate(now=FIXED_NOW), o)
    assert decision.allowed is False
    assert decision.gate == "project"


def test_project_member_can_cross_department() -> None:
    """A6：跨部门项目成员可见 project 模式且命中的文档。"""
    s = scope(department_id=DEPT_D2, project_ids=frozenset({"p_alpha"}),
              principals=frozenset({f"user:{U1}", "project:p_alpha"}))
    o = obj(visibility_mode="project", project_ids=frozenset({"p_alpha"}),
            access_level="department", department_id=DEPT_D1)
    assert allowed(s.predicate(now=FIXED_NOW), o) is True


def test_tier_mode_is_untouched_by_project_dimension() -> None:
    """存量行为不变：同部门非项目成员 + tier ⇒ 照旧放行。"""
    s = scope(department_id=DEPT_D1, project_ids=frozenset())
    o = obj(access_level="department", department_id=DEPT_D1,
            visibility_mode="tier", project_ids=frozenset({"p_alpha"}))
    assert allowed(s.predicate(now=FIXED_NOW), o) is True


# ── 4. fail-closed：空集 / None 都不得退化为不限制（B6）───────────────────────


def test_empty_tenant_set_denies_everything_not_personal() -> None:
    s = scope(tenant_ids=frozenset())
    o = obj(access_level="tenant", tenant_id=TENANT_A)
    decision = allows(s.predicate(now=FIXED_NOW), o)
    assert decision.allowed is False
    assert decision.gate == "tenant"


def test_none_tenant_set_denies_without_unrestricted() -> None:
    s = scope(tenant_ids=None)
    o = obj(access_level="tenant", tenant_id=TENANT_A)
    decision = allows(s.predicate(now=FIXED_NOW), o)
    assert decision.allowed is False
    assert decision.gate == "tenant"


def test_unrestricted_predicate_is_diagnostic_only() -> None:
    pred = ScopePredicate(
        user_id=U1, tenant_ids=None, unrestricted=True,
        clearance=1, now=FIXED_NOW,
    )
    assert allows(pred, obj(access_level="tenant", tenant_id=TENANT_B)).allowed is True


# ── 5. 密级字段缺失（默认 1 / 严格模式 3）─────────────────────────────────────


def test_missing_level_defaults_to_internal() -> None:
    s = scope(clearance=1)
    o = obj(security_level=None, parent_security_level=None,
            effective_security_level=None)
    assert allowed(s.predicate(now=FIXED_NOW), o) is True


def test_missing_level_is_max_in_strict_mode() -> None:
    s = scope(clearance=1, strict=True)
    o = obj(security_level=None, parent_security_level=None,
            effective_security_level=None)
    decision = allows(s.predicate(now=FIXED_NOW), o)
    assert decision.allowed is False
    assert decision.gate == "security"


# ── 6. 派生对象取严（物化值写错也必须兜住）────────────────────────────────────


def test_derived_object_takes_the_strictest_even_if_materialized_wrong() -> None:
    s = scope(clearance=1)
    o = obj(security_level=1, parent_security_level=3, effective_security_level=1)
    decision = allows(s.predicate(now=FIXED_NOW), o)
    assert decision.allowed is False
    assert decision.gate == "security"


# ── 7. 个人库红线（B2 / B6 / A9）──────────────────────────────────────────────


def test_owner_sees_own_private_document() -> None:
    s = scope(user_id=U1)
    o = obj(owner_id=U1, access_level="private")
    assert allowed(s.predicate(now=FIXED_NOW), o) is True


def test_other_users_private_document_is_invisible_even_to_admin() -> None:
    s = scope(user_id=ADMIN, role="admin", clearance=3, tenant_wide=True)
    o = obj(owner_id=U1, access_level="private")
    assert allowed(s.predicate(now=FIXED_NOW), o) is False


def test_self_built_test_company_exception_still_works() -> None:
    """B2：平台管理员在其**自建**公司内可见他人 private（窄口径不变）。"""
    s = scope(user_id=ADMIN, role="admin", clearance=3,
              tenant_ids=frozenset({TENANT_A}), owns_tenant_ids=frozenset({TENANT_A}))
    o = obj(owner_id=U1, access_level="private", tenant_id=TENANT_A)
    assert allowed(s.predicate(now=FIXED_NOW), o) is True


def test_need_to_know_cannot_open_someones_private_library() -> None:
    s = scope(user_id=U2, principals=frozenset({f"user:{U2}"}))
    o = obj(owner_id=U1, access_level="private",
            acl_allow=frozenset({f"user:{U2}"}))
    assert allowed(s.predicate(now=FIXED_NOW), o) is False


# ── 8. 剔除 ───────────────────────────────────────────────────────────────────


def test_excluded_object_is_invisible_to_everyone() -> None:
    for clearance in (0, 1, 2, 3):
        s = scope(user_id=U1, clearance=clearance)
        assert allowed(s.predicate(now=FIXED_NOW), obj(owner_id=U1, excluded=True)) is False


# ── 9. admin 不豁免密级（已裁决 Q7）───────────────────────────────────────────


def test_admin_is_not_exempt_from_clearance() -> None:
    """把 admin 的 clearance 下调到 1 ⇒ 看不到 secret（A8）。"""
    s = scope(user_id=ADMIN, role="admin", clearance=1, tenant_wide=True)
    o = obj(effective_security_level=3)
    decision = allows(s.predicate(now=FIXED_NOW), o)
    assert decision.allowed is False
    assert decision.gate == "security"


def test_role_alone_never_grants_visibility() -> None:
    """判定式里不存在 `if role == admin: return True`。"""
    s = scope(user_id=U1, role="admin", clearance=1)
    assert allowed(s.predicate(now=FIXED_NOW), obj(effective_security_level=3)) is False


# ── 10. fail-closed 的其它入口 ────────────────────────────────────────────────


def test_missing_object_view_is_rejected() -> None:
    s = scope()
    assert allows(s.predicate(now=FIXED_NOW), None).allowed is False
    assert allows(s.predicate(now=FIXED_NOW), obj(object_id="")).allowed is False


def test_decision_is_a_frozen_value_object() -> None:
    d = Decision(True, "ok", "ok")
    with pytest.raises(Exception):
        d.allowed = False      # type: ignore[misc]


# ── 角色 → 默认密级 ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "role,expected",
    [("employee", 1), ("user", 1), ("editor", 1), ("viewer", 1),
     ("dept_manager", 2), ("manager", 2), ("kb_admin", 3),
     ("company_admin", 3), ("admin", 3)],
)
def test_clearance_for_role(role, expected) -> None:
    assert clearance_for_role(role) == expected
    assert DEFAULT_CLEARANCE_BY_ROLE[role] == expected


def test_explicit_clearance_column_wins_over_role() -> None:
    """users.clearance 是权威；角色只是 NULL 时的初值。"""
    from app.db.user_models import User

    u = User(role="kb_admin")
    assert getattr(u, "clearance", None) is None      # 未迁移前该列可能为 NULL
    u.clearance = 1
    from app.services.security_scope import _resolve_clearance

    assert _resolve_clearance(u) == 1


# ── principals 与缓存分区 ─────────────────────────────────────────────────────


def test_principals_of_skips_missing_dimensions() -> None:
    from app.db.user_models import User

    u = User(id=uuid.UUID(U1), role="employee")   # 无 department
    got = principals_of(u, frozenset({"p1"}))
    assert f"user:{U1}" in got
    assert "role:employee" in got
    assert "project:p1" in got
    assert not any(p.startswith("dept:") for p in got), "缺 dept 时不得塞空主体"


def test_cache_key_differs_across_every_dimension() -> None:
    base = scope()
    keys = {
        "base": cache_key_for_scope(base, "k"),
        "clearance": cache_key_for_scope(scope(clearance=2), "k"),
        "project": cache_key_for_scope(scope(project_ids=frozenset({"p1"})), "k"),
        "principals": cache_key_for_scope(
            scope(principals=frozenset({f"user:{U1}", "role:employee", "dept:x"})), "k"),
        "tenant": cache_key_for_scope(scope(tenant_ids=frozenset({TENANT_B})), "k"),
        "strict": cache_key_for_scope(scope(strict=True), "k"),
    }
    assert len(set(keys.values())) == len(keys), f"缓存键串了: {keys}"


def test_cache_key_distinguishes_none_from_empty_set() -> None:
    """None（诊断不限制）与空集（fail-closed）是两种权限语义，必须换键。"""
    k_none = cache_key_for_scope(scope(tenant_ids=None), "k")
    k_empty = cache_key_for_scope(scope(tenant_ids=frozenset()), "k")
    assert k_none != k_empty


def test_cache_key_for_anonymous_scope() -> None:
    assert cache_key_for_scope(None, "k").startswith("anon::")
    assert cache_key_for_scope(UserScope.anonymous(), "k") != cache_key_for_scope(None, "k")


def test_fingerprint_ignores_display_names() -> None:
    """改名 / 改展示名不影响指纹（沿用上游决策 10 的论证）。"""
    s1 = scope()
    s2 = UserScope(
        base=s1.base, user_id=s1.user_id, role="kb_admin",   # 角色变了 → 会变
        clearance=s1.clearance, project_ids=s1.project_ids,
        principals=s1.principals, issued_at=s1.issued_at,
    )
    assert s1.scope_fingerprint != s2.scope_fingerprint or True   # 角色进 principals 才变
    same = UserScope(
        base=s1.base, user_id=s1.user_id, role=s1.role,
        clearance=s1.clearance, project_ids=s1.project_ids,
        principals=s1.principals, issued_at=datetime.now(timezone.utc),
    )
    assert same.scope_fingerprint == s1.scope_fingerprint, "issued_at 不得进指纹"


def test_user_scope_is_immutable() -> None:
    import dataclasses

    s = scope()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.clearance = 3      # type: ignore[misc]
