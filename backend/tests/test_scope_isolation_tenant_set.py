"""
作用域集合隔离（T02）纯单元测试 —— 列表/检索**同一 SQL 入口** + admin 收敛 + 私库不动摇.

覆盖本轮三个 P0 断言（不依赖数据库连接）：

  * P0-6 admin 收敛：「无自建测试公司」→ ``tenant_ids`` 为空集 → fail-closed 返回空，
    **绝不**回退「全平台可见」。
  * P0-7 列表 == 检索：列表、检索、关键词腿全部经 ``document_scope_clause`` **唯一**
    入口组装可见条件；本测试断言该入口同时含「公司边界(tenant IN …)」与
    「个人库(private/NULL ∧ owner=我)」两条，二者是同一份谓词。
  * P0-8 私库不动摇：``owns_tenant_ids`` **只**放开 private 分支（admin 在自建公司内
    可见他人私库），不触碰部门/公司层的租户边界；且**不**进入删除判定
    （private 删除恒为「仅 owner_id == 我」，team-lead 裁决 10-A）。

运行方式（容器内）：
    docker exec -w /app rag_backend python -m pytest tests/test_scope_isolation_tenant_set.py -q
"""

from __future__ import annotations

import sys
import uuid

try:
    from sqlalchemy.dialects import postgresql

    from app.db.user_models import User
    from app.services.tenancy import (
        ACCESS_DEPARTMENT,
        ACCESS_PRIVATE,
        ACCESS_TENANT,
        can_access_document,
        delete_permission_for,
        document_scope_clause,
        scope_for,
        tenant_clause,
        tenant_scope_fingerprint,
    )

    _IMPORT_OK = True
except ImportError as exc:  # pragma: no cover — 宿主机缺依赖 → 跳过
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")
    _IMPORT_OK = False


def _sql(clause) -> str:
    """把 SQLAlchemy 条件编译成带字面量的 PostgreSQL 字符串，便于断言分支。"""
    return str(
        clause.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


class _Doc:
    """最小文档桩（只需 can_access_document / delete_permission_for 用到的字段）。"""

    def __init__(self, *, owner_id, tenant_id, access_level, department_id=None):
        self.owner_id = owner_id
        self.tenant_id = tenant_id
        self.access_level = access_level
        self.department_id = department_id


class _User:
    def __init__(self, *, role, tenant_id=None, department_id=None, is_admin=False):
        self.id = uuid.uuid4()
        self.role = role
        self.tenant_id = tenant_id
        self.department_id = department_id
        self.is_admin = is_admin


# ── 第一层：tenant_clause 四分支 ───────────────────────────────────────────────

def test_tenant_clause_branches():
    """None+unrestricted→true / None→false / 空集→false / 非空→IN (fail-closed)."""
    if not _IMPORT_OK:
        return
    assert _sql(tenant_clause(None, unrestricted=True)) == "true"
    assert _sql(tenant_clause(None)) == "false"
    assert _sql(tenant_clause(frozenset())) == "false", "空集必须 fail-closed，绝不等于不限制"
    in_sql = _sql(tenant_clause(frozenset({"c1", "c2"})))
    assert "IN (" in in_sql and "'c1'" in in_sql and "'c2'" in in_sql


def test_tenant_scope_fingerprint_three_states_distinct():
    if not _IMPORT_OK:
        return
    assert tenant_scope_fingerprint(None) == "all"
    assert tenant_scope_fingerprint(frozenset()) == "none"
    a = tenant_scope_fingerprint(frozenset({"c1"}))
    b = tenant_scope_fingerprint(frozenset({"c2"}))
    assert len({tenant_scope_fingerprint(None), tenant_scope_fingerprint(frozenset()), a, b}) == 4


# ── P0-7：列表/检索同一入口 ───────────────────────────────────────────────────

def test_scope_clause_is_single_entry_with_tenant_and_private():
    """可见条件 = 公司边界(tenant IN …) ∪ 自己的个人库(private/NULL ∧ owner=我)。"""
    if not _IMPORT_OK:
        return
    me = uuid.uuid4()
    sql = _sql(
        document_scope_clause(
            owner_id=me,
            department_id="tech",
            tenant_ids=frozenset({"company_a"}),
            owns_tenant_ids=frozenset(),
            tenant_wide=False,
        )
    )
    assert "IN ('company_a')" in sql, "公司边界必须落在 tenant IN 上"
    assert "private" in sql, "个人库分支必须存在"
    assert str(me) in sql, "个人库按 owner_id 归属"


def test_empty_tenant_set_still_keeps_own_private():
    """
    Rev2 正确性锚点：admin 无自建公司（空集）时，**自己的**个人库（含 default）
    仍必须出现在可见条件里 —— 个人库不参与公司过滤。
    """
    if not _IMPORT_OK:
        return
    me = uuid.uuid4()
    sql = _sql(
        document_scope_clause(
            owner_id=me,
            department_id=None,
            tenant_ids=frozenset(),          # fail-closed 的公司集合
            owns_tenant_ids=frozenset(),
            tenant_wide=True,
        )
    )
    assert "private" in sql, "空公司集合下 personal 分支必须保留（否则admin看不到自己的私库）"
    assert str(me) in sql


# ── P0-6：admin 收敛 + fail-closed ───────────────────────────────────────────

def test_admin_empty_tenant_set_is_fail_closed():
    if not _IMPORT_OK:
        return
    admin = _User(role=User.ROLE_ADMIN, tenant_id="default", is_admin=True)
    scope = scope_for(admin)  # 未传自建集合
    assert scope.tenant_ids == frozenset()
    assert scope.owns_tenant_ids == frozenset()
    assert scope.owner_id == admin.id, "admin 的个人库归属仍是自己"


def test_admin_owns_set_equals_tenant_set():
    if not _IMPORT_OK:
        return
    admin = _User(role=User.ROLE_ADMIN, tenant_id="default", is_admin=True)
    owned = frozenset({"c8111de986583", "cfb08c53677c4"})
    scope = scope_for(admin, owned_tenant_ids=owned)
    assert scope.tenant_ids == owned
    assert scope.owns_tenant_ids == owned
    assert scope.cross_tenant


# ── P0-8：私库只认归属人（owns 只在自建公司内放开读，不放开删）─────────────────

def test_private_doc_owner_only_and_owns_exception_is_read_only():
    if not _IMPORT_OK:
        return
    admin = _User(role=User.ROLE_ADMIN, tenant_id="default", is_admin=True)
    other = _User(role=User.ROLE_EMPLOYEE, tenant_id="company_a")

    # ① admin 自己的 default 私库：任何租户集合下都可见
    own_default = _Doc(owner_id=admin.id, tenant_id="default", access_level=ACCESS_PRIVATE)
    assert can_access_document(
        own_default, admin, tenant_ids=frozenset(), owns_tenant_ids=frozenset()
    )
    # ② 他人私库（不在自建集合内）→ 不可见
    others = _Doc(owner_id=other.id, tenant_id="company_a", access_level=ACCESS_PRIVATE)
    assert not can_access_document(
        others, admin, tenant_ids=frozenset({"company_b"}),
        owns_tenant_ids=frozenset({"company_b"}),
    )
    # ③ 他人私库落在自建集合内 → 可读（读放宽）
    inside = _Doc(owner_id=other.id, tenant_id="company_b", access_level=ACCESS_PRIVATE)
    assert can_access_document(
        inside, admin, tenant_ids=frozenset({"company_b"}),
        owns_tenant_ids=frozenset({"company_b"}),
    )
    # ④ 但**不可删**（10-A：读放宽、删不放宽；private 恒 owner-only）
    allowed, _reason = delete_permission_for(
        inside, admin, tenant_ids=frozenset({"company_b"}),
        owns_tenant_ids=frozenset({"company_b"}),
    )
    assert not allowed, "admin 对自建公司内他人私库可读不可删"


def test_admin_company_wide_read_within_owned_set():
    """admin 在自建集合内对公司库/部门库可见；集合外（非自建）不可见。"""
    if not _IMPORT_OK:
        return
    admin = _User(role=User.ROLE_ADMIN, tenant_id="default", is_admin=True)
    owned = frozenset({"company_b"})
    tenant_doc = _Doc(owner_id=uuid.uuid4(), tenant_id="company_b", access_level=ACCESS_TENANT)
    dept_doc = _Doc(
        owner_id=uuid.uuid4(), tenant_id="company_b",
        access_level=ACCESS_DEPARTMENT, department_id="other-dept",
    )
    assert can_access_document(tenant_doc, admin, tenant_ids=owned, owns_tenant_ids=owned)
    assert can_access_document(dept_doc, admin, tenant_ids=owned, owns_tenant_ids=owned)

    foreign = _Doc(owner_id=uuid.uuid4(), tenant_id="company_a", access_level=ACCESS_TENANT)
    assert not can_access_document(foreign, admin, tenant_ids=owned, owns_tenant_ids=owned)
