"""
T5 —— 密级三腿一致 + 密级变更触发 T4 级联取严（验收要点 3 / 5）.

「三腿一致」= 同一份 ``ScopePredicate`` 编译出的三个产物对**同一对象**给出同样的
可见性答案：

    ① Python :func:`allows`                           （第 11 / 12 环）
    ② Qdrant :func:`qdrant_filter_matches`            （第 7 环向量腿）
    ③ PG     ``to_sql`` + 等价性矩阵的内存求值器        （第 7 环关键词腿）

③ 的求值器直接复用 ``test_scope_filter_equivalence`` 的 ``sql_clause_matches``
（刻意**不复用** ``allows()`` 的判定函数，否则就是永真断言）。这里只补 T5 关心的
**密级管理面**导致的那些格子：set 密级后三腿是否仍然一致、clearance 不足是否三腿
都拒、以及 **admin 同样受密级限制**（判定式里没有 role 分支）。
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import pytest

    from app.services.security_policy import (
        ObjectACLView,
        ScopePredicate,
        allows,
        qdrant_filter_matches,
    )
    from test_scope_filter_equivalence import sql_clause_matches
except ImportError as exc:      # 宿主机缺依赖 → 整份跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")

FIXED_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
PAST = FIXED_NOW - timedelta(days=1)
FUTURE = FIXED_NOW + timedelta(days=1)

TENANT_A = "cleanA"
DEPT = "d001"
U1 = str(uuid.uuid4())
DOC_ID = str(uuid.uuid4())
P_U1 = f"user:{U1}"


@pytest.fixture(autouse=True)
def _force_prefilter_default():
    """
    把运维开关 ``ACL_SECURITY_PREFILTER_STRICT`` 固定为 False，保证三腿一致断言
    不受 ``.env`` 影响（它只影响"密级字段全缺"那一格的 Qdrant 侧语义）。
    """
    from app.config import get_settings

    settings = get_settings()
    original = settings.ACL_SECURITY_PREFILTER_STRICT
    settings.ACL_SECURITY_PREFILTER_STRICT = False
    try:
        yield
    finally:
        settings.ACL_SECURITY_PREFILTER_STRICT = original


def _pred(*, clearance=1, principals=None, strict=False):
    if principals is None:
        principals = frozenset({P_U1, "role:employee"})
    return ScopePredicate(
        user_id=U1, tenant_ids=frozenset({TENANT_A}), department_id=DEPT,
        clearance=clearance, principals=principals, now=FIXED_NOW, strict=strict,
    )


def _view(*, level, clearance_owner=U1, acl_allow=frozenset(), acl_expires_at=None,
          visibility_mode="tier", project_ids=frozenset()):
    return ObjectACLView(
        object_id=DOC_ID, object_type="doc", document_id=DOC_ID,
        tenant_id=TENANT_A, owner_id=str(uuid.uuid4()), department_id=DEPT,
        access_level="department", visibility_mode=visibility_mode,
        project_ids=project_ids, security_level=level,
        effective_security_level=level,
        acl_allow=acl_allow, acl_expires_at=acl_expires_at,
    )


def _three_legs(pred, view) -> tuple[bool, bool, bool]:
    return (
        allows(pred, view).allowed,
        qdrant_filter_matches(pred, view.to_payload()),
        sql_clause_matches(pred, view),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ① 三腿一致（密级管理面涉及的关键格）
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "clearance,level,strict,expected",
    [
        (3, 1, False, True),    # 高密级用户看低密级文档 → 放行
        (1, 1, False, True),    # 刚好够
        (1, 3, False, False),   # clearance 不足 → 三腿都拒
        (0, 1, False, False),   # 明文 clearance 0 看默认档 → 拒
        (1, 3, True, False),    # 严格模式 + 高密级 → 拒
    ],
)
def test_security_gate_three_legs(clearance, level, strict, expected) -> None:
    pred = _pred(clearance=clearance, strict=strict)
    view = _view(level=level)
    py, qd, sql = _three_legs(pred, view)
    assert (py, qd, sql) == (expected, expected, expected), (py, qd, sql)


def test_missing_level_defaults_consistent() -> None:
    """密级字段全缺：非严格按默认 1；clearance=1 应放行，三腿一致。"""
    view = ObjectACLView(
        object_id=DOC_ID, object_type="doc", document_id=DOC_ID,
        tenant_id=TENANT_A, owner_id=str(uuid.uuid4()), department_id=DEPT,
        access_level="department", visibility_mode="tier",
        security_level=None, effective_security_level=None,
    )
    assert _three_legs(_pred(clearance=1), view) == (True, True, True)
    # clearance=0 < 默认档 1 → 三腿都拒
    assert _three_legs(_pred(clearance=0), view) == (False, False, False)


def test_admin_role_does_not_bypass_security_gate() -> None:
    """**admin 不豁免**（已裁决 Q7）：principals 里带 role:admin 也照样受密级限制。"""
    pred = _pred(clearance=1, principals=frozenset({P_U1, "role:admin"}))
    view = _view(level=3)     # 绝密
    assert _three_legs(pred, view) == (False, False, False)


def test_need_to_know_override_reaches_all_three_legs() -> None:
    """未过期的 need-to-know 例外越过密级闸门 —— 三腿必须**同时**放行。"""
    pred = _pred(clearance=1)
    view = _view(level=3, acl_allow=frozenset({P_U1}), acl_expires_at=FUTURE)
    assert _three_legs(pred, view) == (True, True, True)


def test_expired_need_to_know_excluded_on_all_three_legs() -> None:
    pred = _pred(clearance=1)
    view = _view(level=3, acl_allow=frozenset({P_U1}), acl_expires_at=PAST)
    assert _three_legs(pred, view) == (False, False, False)
