"""**两路同源等价性矩阵**（A2，设计文档 §6.2）—— T2 的验收要点 2.

同一份 :class:`ScopePredicate` 编译出的三个产物必须在**逐格**上给出同样的答案：

    ① Python  :func:`allows(pred, view)`                  （第 11 / 12 环口径）
    ② Qdrant  :func:`qdrant_filter_matches(pred, payload)`（第 7 环向量腿）
    ③ PG      ``to_sql(pred, STUB)`` + 内存求值器          （第 7 环关键词腿）

为什么必须有这个矩阵：「两路条件由两个不同函数各自拼装」（``_visibility_conditions``
vs ``document_scope_clause``）在本仓库里已经并存了三轮。测试只能覆盖被想到的
用例，而把两者做成**同一 IR 的两个编译器**之后，"分叉"唯一的可能就是某一格
不相等 —— 这正是本文件要抓的。

③ 的求值器刻意**不复用** ``allows()`` 的任何判定函数（否则就成了永真断言）：
它遍历 SQLAlchemy 表达式树，只认节点结构与运算符。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    SmallInteger,
    String,
    and_,
    false,
    not_,
    or_,
    true,
)
from sqlalchemy.sql import functions as sa_functions

from app.services.security_policy import (
    ObjectACLView,
    ScopePredicate,
    allows,
    qdrant_filter_matches,
    to_sql,
)
from app.services.security_scope import UserScope
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

PRINCIPAL_U1 = f"user:{U1}"
PRINCIPAL_U2 = f"user:{U2}"
PRINCIPAL_ROLE_EMPLOYEE = "role:employee"
PRINCIPAL_ROLE_ADMIN = "role:admin"


# ═══════════════════════════════════════════════════════════════════════════════
# ③ PG 侧的"内存替身模型" + 表达式树求值器
# ═══════════════════════════════════════════════════════════════════════════════

_STUB = sa.Table(
    "security_obj_stub",
    sa.MetaData(),
    sa.Column("tenant_id", String),
    sa.Column("owner_id", String),
    sa.Column("department_id", String),
    sa.Column("access_level", String),
    sa.Column("visibility_mode", String),
    sa.Column("project_ids", sa.JSON),
    sa.Column("security_level", SmallInteger),
    sa.Column("parent_security_level", SmallInteger),
    sa.Column("effective_security_level", SmallInteger),
    sa.Column("acl_allow", sa.JSON),
    sa.Column("acl_deny", sa.JSON),
    sa.Column("acl_expires_at", DateTime),
    sa.Column("excluded", Boolean),
)


def _row_of(obj: ObjectACLView) -> dict:
    return {
        "tenant_id": obj.tenant_id,
        "owner_id": obj.owner_id,
        "department_id": obj.department_id,
        "access_level": obj.access_level,
        "visibility_mode": obj.visibility_mode,
        "project_ids": sorted(obj.project_ids),
        "security_level": obj.security_level,
        "parent_security_level": obj.parent_security_level,
        "effective_security_level": obj.effective_security_level,
        "acl_allow": sorted(obj.acl_allow),
        "acl_deny": sorted(obj.acl_deny),
        "acl_expires_at": obj.acl_expires_at,
        "excluded": obj.excluded,
    }


def sql_clause_matches(pred: ScopePredicate, obj: ObjectACLView) -> bool:
    """把 ``to_sql`` 的产出套在替身模型上求值（PG 语义的 Python 镜像）。"""
    clause = to_sql(pred, _STUB.c)
    return bool(_eval(clause, _row_of(obj)))


def _eval(node, row: dict) -> bool:
    # SQLAlchemy 把需要括号的子表达式包成 Grouping：``to_sql`` 在 and_ 里嵌 or_
    # （或反之）时必然产生，语义与内层完全一致 —— 直接解开。
    # 漏掉这一支会让整张等价矩阵以 "未支持的表达式节点 Grouping" 集体误报，
    # 看起来像 578 条真实失败，其实是一条测试替身的缺口。
    if isinstance(node, sa.sql.elements.Grouping):
        return _eval(node.element, row)
    if isinstance(node, sa.sql.elements.True_):
        return True
    if isinstance(node, sa.sql.elements.False_):
        return False
    if isinstance(node, sa.sql.elements.BooleanClauseList):
        values = [_eval(c, row) for c in node.clauses]
        # 按**函数名**比较，不要用 ``is and_`` / ``is or_``：
        # SQLAlchemy 2.0 里 ``node.operator`` 是 ``sqlalchemy.sql.operators.or_``，
        # 与 ``from sqlalchemy import or_`` 拿到的对象**不是同一个**，同一性比较
        # 恒为假 —— 整张矩阵会以 "未支持的布尔组合 <built-in function or_>" 集体失败。
        # （下面的 UnaryExpression 分支本来就用 __name__ 比较，此处与之统一。）
        opname = getattr(node.operator, "__name__", "")
        if opname == "and_":
            return all(values)
        if opname == "or_":
            return any(values)
        raise AssertionError(f"未支持的布尔组合 {node.operator!r}")
    if isinstance(node, sa.sql.elements.AsBoolean):
        # ⚠️ ``AsBoolean`` 是 ``UnaryExpression`` 的**子类**，必须先于它判断。
        # SQLAlchemy 在把 ``false()`` / ``true()`` 之类的可空布尔字面量塞进
        # ``or_`` / ``and_`` 时，会包一层 ``AsBoolean``（运算符 ``is_true``），
        # 语义就是"求值后取真值"。它的 ``operator`` 不是 ``inv`` —— 漏掉这一支
        # 会让 ``_generic_scope_sql`` 里 ``or_(private, and_(not_(private), false()))``
        # 触发的每一格（如 ``empty_tenant_set`` 全列 44 格）以
        # "未支持的一元运算 is_true" 集体误报，看起来像 44 条生产缺陷，其实是
        # 测试替身的又一处缺口（与前面的 Grouping / BindParameter / 运算符同一性
        # 三个修复同类）。
        return bool(_eval(node.element, row))
    if isinstance(node, sa.sql.elements.UnaryExpression):
        # 只可能是 not_（SQLAlchemy 对可求反的二元运算会直接换成反运算符）
        assert node.operator.__name__ == "inv", f"未支持的一元运算 {node.operator!r}"
        return not _eval(node.element, row)
    if isinstance(node, sa.sql.elements.BinaryExpression):
        return _eval_binary(node, row)
    raise AssertionError(f"未支持的表达式节点 {type(node).__name__}")

def _eval_binary(node, row: dict) -> bool:
    left = _side(node.left, row)
    right = _side(node.right, row)
    op = node.operator

    opstring = getattr(op, "opstring", None)
    if opstring == "?|":      # JSONB 数组相交
        return bool(set(_strs(left)) & set(_strs(right)))

    name = getattr(op, "__name__", "")
    if name == "eq":
        return left == right
    if name == "ne":
        return left is not None and left != right      # SQL: NULL != x → NULL
    if name == "le":
        return left is not None and left <= right
    if name == "lt":
        return left is not None and left < right
    if name == "ge":
        return left is not None and left >= right
    if name == "gt":
        return left is not None and left > right
    if name == "is_":
        return left is None if right is None else left is right
    if name == "isnot":
        return left is not None if right is None else left is not right
    if name == "in_op":
        return left in (right or [])
    raise AssertionError(f"未支持的运算符 {op!r}（opstring={opstring!r}）")


def _side(node, row: dict):
    if isinstance(node, sa.sql.elements.Grouping):
        return _side(node.element, row)
    if isinstance(node, sa.sql.elements.Cast):
        return _side(node.clause, row)
    # 注意是 BindParameter，不是 BindParam —— ``sa.sql.elements`` 里没有
    # ``BindParam`` 这个名字，写成它会让**每一次** _side 调用都抛
    # AttributeError（本行位于无条件路径上），整张矩阵集体失败。
    if isinstance(node, sa.sql.elements.BindParameter):
        return node.value
    if isinstance(node, sa.sql.elements.Null):
        return None
    if isinstance(node, sa.sql.elements.True_):
        return True
    if isinstance(node, sa.sql.elements.False_):
        return False
    # ``greatest`` / ``coalesce`` 是 SQL 的 ``Function`` 节点（生产代码用它表达
    # "有效密级 = max(物化, 自身, 父级)"）。``Function`` 不在 elements 命名空间
    # 里（在 ``sqlalchemy.sql.functions``），且**不是** Column/Unary 等任何下面
    # 已有的分支 —— 不显式处理会掉到末尾的 ``getattr(node, "value", None)`` 返回
    # None，把密级表达式整体误算成 NULL（矩阵会在 legacy_* 上集体误报）。
    if isinstance(node, sa_functions.Function):
        return _eval_function(node, row)
    if isinstance(node, sa.sql.schema.Column):
        return row.get(node.name)
    if isinstance(node, (sa.sql.elements.BooleanClauseList,
                         sa.sql.elements.BinaryExpression,
                         sa.sql.elements.UnaryExpression)):
        return _eval(node, row)
    return getattr(node, "value", None)


def _eval_function(node, row):
    """SQL ``Function`` 节点求值（只支持生产代码用到的 ``greatest`` / ``coalesce``）。

    ``greatest`` 忽略 NULL（全为 NULL 才返回 NULL）—— 这正是 PG 的语义，也是
    ``_effective_level`` 与 ``_effective_level_sql`` 同构的关键。
    ``coalesce`` 返回第一个非 NULL；全 NULL 返回 NULL。
    """
    name = str(getattr(node, "name", "")).lower()
    args = [_side(c, row) for c in node.clauses]
    if name == "greatest":
        present = [a for a in args if a is not None]
        return max(present) if present else None
    if name == "coalesce":
        for a in args:
            if a is not None:
                return a
        return None
    raise AssertionError(f"未支持的函数 {name!r}")


def _strs(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v) for v in value]
    return [str(value)]


# ═══════════════════════════════════════════════════════════════════════════════
# 矩阵：13 个 Scope ×（存量 44 + 密级三键「部分缺失」中间态）个对象
# ═══════════════════════════════════════════════════════════════════════════════


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
    name: str,
    owner_id: str | None = U2,
    tenant_id: str = TENANT_A,
    department_id: str | None = DEPT_D1,
    access_level: str = "tenant",
    visibility_mode: str = "tier",
    project_ids=frozenset(),
    security_level: int | None = 1,
    parent_security_level: int | None = None,
    effective_security_level: int | None = None,
    acl_allow=frozenset(),
    acl_deny=frozenset(),
    acl_expires_at=None,
    excluded: bool = False,
) -> ObjectACLView:
    eff = effective_security_level
    if eff is None:
        vals = [v for v in (security_level, parent_security_level) if v is not None]
        eff = max(vals) if vals else 1
    return ObjectACLView(
        object_id=name,
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
        effective_security_level=eff,
        acl_allow=frozenset(acl_allow),
        acl_deny=frozenset(acl_deny),
        acl_expires_at=acl_expires_at,
        excluded=excluded,
    )


def level_obj(
    *,
    name: str,
    owner_id: str | None = U2,
    tenant_id: str = TENANT_A,
    department_id: str | None = DEPT_D1,
    access_level: str = "tenant",
    visibility_mode: str = "tier",
    project_ids=frozenset(),
    security_level: int | None = None,
    parent_security_level: int | None = None,
    effective_security_level: int | None = None,
    acl_allow=frozenset(),
    acl_deny=frozenset(),
    acl_expires_at=None,
    excluded: bool = False,
) -> ObjectACLView:
    """构造"密级三键**按需**取值"的对象 —— **不回填** ``effective_security_level``.

    这是比 :func:`obj` 更底层的入口。二者唯一区别：``obj()`` 总会把 effective
    回填成 ``max(security_level, parent_security_level)``（两者皆缺时回填 1），
    因此**表达不出**"三个密级键里仅缺其一/其二"的中间态；而 ``level_obj`` 原样
    保留 ``None``（``_payload_of`` 会把 None 丢弃 ⇒ payload 里就是"字段缺失"）。

    为什么必须有它：``_effective_level`` 的语义是
    ``max(三个键，忽略 None)``，**只有三者全 None** 才落回默认档（strict→3 否则→1）。
    这个语义最容易在"部分缺失"处被写错 —— 要么把"某键缺失"误当成"整行落默认档"，
    要么前置过滤只判 ``effective_security_level`` 一个键。矩阵里原先只有
    "三键齐全"与"三键全缺"两个极端，抓不到这两类错误。
    """
    return ObjectACLView(
        object_id=name,
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


def legacy(
    *,
    name: str,
    owner_id: str | None = U2,
    tenant_id: str = TENANT_A,
    department_id: str | None = DEPT_D1,
    access_level: str = "tenant",
) -> ObjectACLView:
    """迁移前写入的老向量：payload 里**没有**任何新增字段。"""
    return ObjectACLView(
        object_id=name,
        object_type="text_chunk",
        document_id=DOC_ID,
        tenant_id=tenant_id,
        owner_id=owner_id,
        user_id=owner_id,
        department_id=department_id,
        access_level=access_level,
        visibility_mode="tier",
        project_ids=frozenset(),
        security_level=None,
        parent_security_level=None,
        effective_security_level=None,
        acl_allow=frozenset(),
        acl_deny=frozenset(),
        acl_expires_at=None,
        excluded=False,
    )


SCOPE_MATRIX: dict[str, UserScope] = {
    "employee_c1": scope(),
    "employee_c3": scope(clearance=3),
    "employee_c0": scope(clearance=0),
    "employee_other_dept": scope(department_id=DEPT_D2),
    "employee_other_tenant": scope(tenant_ids=frozenset({TENANT_B}),
                                   department_id=None),
    "dept_manager_c2": scope(role="dept_manager", clearance=2),
    "kb_admin_wide": scope(role="kb_admin", clearance=3, tenant_wide=True),
    "admin_cross": scope(
        user_id=ADMIN, role="admin", clearance=3, tenant_wide=True,
        tenant_ids=frozenset({TENANT_A, TENANT_B}),
        owns_tenant_ids=frozenset({TENANT_A}),
        principals=frozenset({f"user:{ADMIN}", PRINCIPAL_ROLE_ADMIN}),
    ),
    "project_member_alpha": scope(project_ids=frozenset({"p_alpha"}),
                                  principals=frozenset({PRINCIPAL_U1,
                                                        PRINCIPAL_ROLE_EMPLOYEE,
                                                        "project:p_alpha"})),
    "project_member_beta": scope(project_ids=frozenset({"p_beta"}),
                                 principals=frozenset({PRINCIPAL_U1,
                                                       PRINCIPAL_ROLE_EMPLOYEE,
                                                       "project:p_beta"})),
    "empty_tenant_set": scope(tenant_ids=frozenset()),
    "strict_mode": scope(strict=True),
    "u2_employee": scope(user_id=U2, principals=frozenset({PRINCIPAL_U2,
                                                           PRINCIPAL_ROLE_EMPLOYEE})),
}

OBJECT_MATRIX: dict[str, ObjectACLView] = {
    # ── 存量形态（tier + 密级 1）：三个层级 × 本人 / 他人 ──────────────────────
    "own_private": obj(name="own_private", owner_id=U1, access_level="private"),
    "other_private": obj(name="other_private", owner_id=U2, access_level="private"),
    "dept_same": obj(name="dept_same", access_level="department", department_id=DEPT_D1),
    "dept_other": obj(name="dept_other", access_level="department", department_id=DEPT_D2),
    "dept_none": obj(name="dept_none", access_level="department", department_id=None),
    "tenant_level": obj(name="tenant_level", access_level="tenant"),
    "tenant_level_other_tenant": obj(name="tenant_level_b", access_level="tenant",
                                     tenant_id=TENANT_B),
    # ── 密级 ──────────────────────────────────────────────────────────────────
    "level_2_own": obj(name="level_2_own", owner_id=U1, security_level=2),
    "level_3_own": obj(name="level_3_own", owner_id=U1, security_level=3),
    "level_3_tenant": obj(name="level_3_tenant", security_level=3),
    "derived_takes_max": obj(name="derived_takes_max", security_level=1,
                             parent_security_level=3),
    "derived_materialized_wrong": obj(name="derived_wrong", security_level=1,
                                      parent_security_level=3,
                                      effective_security_level=1),
    "missing_level": obj(name="missing_level", security_level=None,
                         parent_security_level=None, effective_security_level=None),
    # ── need-to-know ──────────────────────────────────────────────────────────
    "ntu_fresh": obj(name="ntu_fresh", security_level=3,
                     acl_allow=frozenset({PRINCIPAL_U1}), acl_expires_at=FUTURE),
    "ntu_expired": obj(name="ntu_expired", security_level=3,
                       acl_allow=frozenset({PRINCIPAL_U1}), acl_expires_at=PAST),
    "ntu_no_expiry": obj(name="ntu_no_expiry", security_level=3,
                         acl_allow=frozenset({PRINCIPAL_U1})),
    "ntu_other_user": obj(name="ntu_other_user", security_level=3,
                          acl_allow=frozenset({PRINCIPAL_U2})),
    "ntu_on_private": obj(name="ntu_on_private", owner_id=U2, access_level="private",
                          acl_allow=frozenset({PRINCIPAL_U1})),
    # ── deny / 剔除 ───────────────────────────────────────────────────────────
    "deny_owner": obj(name="deny_owner", owner_id=U1, access_level="private",
                      acl_deny=frozenset({PRINCIPAL_U1})),
    "deny_role": obj(name="deny_role", acl_deny=frozenset({PRINCIPAL_ROLE_EMPLOYEE})),
    "deny_admin_role": obj(name="deny_admin_role",
                           acl_deny=frozenset({PRINCIPAL_ROLE_ADMIN})),
    "excluded_obj": obj(name="excluded_obj", owner_id=U1, excluded=True),
    # ── 项目维度 ──────────────────────────────────────────────────────────────
    "project_hit": obj(name="project_hit", visibility_mode="project",
                       project_ids=frozenset({"p_alpha"}), access_level="department",
                       department_id=DEPT_D2),
    "project_miss": obj(name="project_miss", visibility_mode="project",
                        project_ids=frozenset({"p_beta"}), access_level="tenant"),
    "project_no_ids": obj(name="project_no_ids", visibility_mode="project",
                          project_ids=frozenset()),
    "project_tier_fallback": obj(name="project_tier_fallback", visibility_mode="tier",
                                 project_ids=frozenset({"p_alpha"})),
    "project_private": obj(name="project_private", owner_id=U2,
                           access_level="private", visibility_mode="project",
                           project_ids=frozenset({"p_alpha"})),
    "project_dept_same": obj(name="project_dept_same", visibility_mode="project",
                             project_ids=frozenset({"p_alpha"}),
                             access_level="department", department_id=DEPT_D1),
    # ── 租户边界 ──────────────────────────────────────────────────────────────
    "foreign_tenant": obj(name="foreign_tenant", tenant_id=TENANT_B,
                          access_level="tenant", owner_id=U1),
    "foreign_tenant_private": obj(name="foreign_tenant_private", tenant_id=TENANT_B,
                                  access_level="private", owner_id=U1),
    "test_company_private": obj(name="test_company_private", tenant_id=TENANT_A,
                                access_level="private", owner_id=U2),
    # ── 层级 / 部门组合 ───────────────────────────────────────────────────────
    "dept_same_high": obj(name="dept_same_high", access_level="department",
                          department_id=DEPT_D1, security_level=3),
    "dept_other_high": obj(name="dept_other_high", access_level="department",
                           department_id=DEPT_D2, security_level=3),
    "tenant_high_ntu": obj(name="tenant_high_ntu", access_level="tenant",
                           security_level=3, acl_allow=frozenset({PRINCIPAL_U1}),
                           acl_expires_at=FUTURE),
    "tenant_high_ntu_expired": obj(name="tenant_high_ntu_expired",
                                   access_level="tenant", security_level=3,
                                   acl_allow=frozenset({PRINCIPAL_U1}),
                                   acl_expires_at=PAST),
    "owner_high_level": obj(name="owner_high_level", owner_id=U1, security_level=3,
                            access_level="tenant"),
    "owner_high_excluded": obj(name="owner_high_excluded", owner_id=U1,
                               security_level=3, excluded=True),
    "no_owner_tenant": obj(name="no_owner_tenant", owner_id=None,
                           access_level="tenant"),
    "no_owner_private": obj(name="no_owner_private", owner_id=None,
                            access_level="private"),
    # ── 老向量（缺全部新字段）──────────────────────────────────────────────────
    "legacy_own_private": legacy(name="legacy_own_private", owner_id=U1,
                                 access_level="private"),
    "legacy_other_private": legacy(name="legacy_other_private", owner_id=U2,
                                   access_level="private"),
    "legacy_tenant": legacy(name="legacy_tenant", access_level="tenant"),
    "legacy_dept": legacy(name="legacy_dept", access_level="department"),
    "legacy_foreign": legacy(name="legacy_foreign", tenant_id=TENANT_B,
                             access_level="tenant"),
}


# ── 密级三键「按需给值」的中间态（`_effective_level` 的 max 语义最易错处）─────────
# 每个对象只给**部分**密级键，其余为 None（payload 里即"字段缺失"）。它们既进入
# `test_three_legs_agree` 的 13-Scope × 全对象矩阵，又被
# `test_partial_level_keys_three_legs_agree` 按 clearance∈{0,1,2,3} × strict∈{off,on}
# 逐格全扫 —— 覆盖 1) 只给 security_level 2) 只给 parent_security_level
# 3) 只给 effective_security_level 4) security_level+parent 5) 三者全缺 五种形态。
PARTIAL_LEVEL_OBJECTS: dict[str, ObjectACLView] = {
    # 1) 只给 security_level（parent / effective 均缺）
    "partial_only_own_0": level_obj(name="partial_only_own_0", security_level=0),
    "partial_only_own_1": level_obj(name="partial_only_own_1", security_level=1),
    "partial_only_own_3": level_obj(name="partial_only_own_3", security_level=3),
    # 2) 只给 parent_security_level（security / effective 均缺）
    "partial_only_parent_2": level_obj(name="partial_only_parent_2",
                                       parent_security_level=2),
    "partial_only_parent_3": level_obj(name="partial_only_parent_3",
                                       parent_security_level=3),
    # 3) 只给 effective_security_level（`obj()` 的默认物化形态）
    "partial_only_eff_1": level_obj(name="partial_only_eff_1",
                                    effective_security_level=1),
    "partial_only_eff_3": level_obj(name="partial_only_eff_3",
                                    effective_security_level=3),
    # 4) security_level + parent_security_level（缺 effective）
    "partial_own_parent_0_0": level_obj(name="partial_own_parent_0_0",
                                        security_level=0, parent_security_level=0),
    "partial_own_parent_1_3": level_obj(name="partial_own_parent_1_3",
                                        security_level=1, parent_security_level=3),
    # 5) 三者全缺（与 legacy_* 同形，这里再显式留一条便于扫描）
    "partial_all_missing": level_obj(name="partial_all_missing"),
}

# 让上述"中间态"一并进入 test_three_legs_agree 的 13-Scope 矩阵
OBJECT_MATRIX.update(PARTIAL_LEVEL_OBJECTS)


def _payload_of(obj: ObjectACLView) -> dict:
    """视图 → payload（**丢弃 None**，等价于"字段缺失"的老向量形态）。"""
    return {k: v for k, v in obj.to_payload().items() if v is not None}


@pytest.mark.parametrize("obj_name", sorted(OBJECT_MATRIX))
@pytest.mark.parametrize("scope_name", sorted(SCOPE_MATRIX))
def test_three_legs_agree(scope_name: str, obj_name: str) -> None:
    sc = SCOPE_MATRIX[scope_name]
    view = OBJECT_MATRIX[obj_name]
    pred = sc.predicate(now=FIXED_NOW)

    expect = allows(pred, view).allowed
    got_qdrant = qdrant_filter_matches(pred, _payload_of(view))
    got_sql = sql_clause_matches(pred, view)

    detail = (
        f"scope={scope_name} obj={obj_name} "
        f"(level={view.access_level} vis={view.visibility_mode} "
        f"eff={view.effective_security_level} clr={pred.clearance})"
    )
    assert got_qdrant == expect, f"Qdrant 腿与判定内核不一致 — {detail}"
    assert got_sql == expect, f"SQL 腿与判定内核不一致 — {detail}"


# ═══════════════════════════════════════════════════════════════════════════════
# 密级三键「部分缺失」的 clearance × strict 全扫（QA 指出的覆盖缺口）
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("strict", [False, True], ids=["lenient", "strict"])
@pytest.mark.parametrize("clearance", [0, 1, 2, 3])
@pytest.mark.parametrize("obj_name", sorted(PARTIAL_LEVEL_OBJECTS))
def test_partial_level_keys_three_legs_agree(
    obj_name: str, clearance: int, strict: bool
) -> None:
    """密级三键"仅缺其一/其二"时，三腿必须逐格一致.

    这正是 ``_effective_level = max(三键，忽略 None)`` 的三条镜像最容易分叉之处：

      * 内核 ``allows()``：``max`` 后再对"三键全缺"兜默认档；
      * SQL ``to_sql()``：``greatest``(忽略 NULL) + ``coalesce(..., 默认)``；
      * Qdrant ``to_qdrant()``：Deny-4 判"任一键 > clearance" + "三键**全缺**且
        默认档 > clearance 才排除"。

    任一键缺失若被错误地当成"整行落默认档"，或 Deny-4 只判 ``effective_security_level``
    一个键，这些格子就会红。扫描面：5 类形态 × 4 档 clearance × 2 档 strict。
    """
    sc = scope(clearance=clearance, strict=strict)
    pred = sc.predicate(now=FIXED_NOW)
    view = PARTIAL_LEVEL_OBJECTS[obj_name]

    expect = allows(pred, view).allowed
    got_qdrant = qdrant_filter_matches(pred, _payload_of(view))
    got_sql = sql_clause_matches(pred, view)

    detail = (
        f"obj={obj_name} clr={clearance} strict={strict} "
        f"(own={view.security_level} par={view.parent_security_level} "
        f"eff={view.effective_security_level})"
    )
    assert got_qdrant == expect, f"Qdrant 腿与判定内核不一致 — {detail}"
    assert got_sql == expect, f"SQL 腿与判定内核不一致 — {detail}"


# ── 阳性对照：证明上述扫描"真的能抓到回归"，不是恒真断言 ──────────────────────────
# 每条都显式算出"退化语义"的结果，并断言它与正确语义**相反** —— 这样一旦实现
# 退化成那个 bug，对应格子必然翻红。


def test_positive_control_zero_own_with_missing_others_is_allowed() -> None:
    """对照 A（抓"把某键缺失当作整行默认档"）.

    ``security_level=0``、``parent``/``effective`` 均缺、``clearance=0``、非严格：
    内核取 ``max(0)=0``，``0 <= 0`` **放行**。

    退化语义（"存在缺失键 ⇒ 整行按默认档 1"）会算成 ``1 > 0`` 而**拒绝** ——
    与正确语义相反，故此格能区分二者。
    """
    pred = scope(clearance=0).predicate(now=FIXED_NOW)
    view = level_obj(name="pc_zero_own", security_level=0)

    # 反证：退化语义的结果必须与正确语义相反，否则本对照无牙齿。
    any_key_missing = any(
        v is None for v in (view.security_level, view.parent_security_level,
                            view.effective_security_level)
    )
    degenerate_level = 1 if any_key_missing else max(
        v for v in (view.security_level, view.parent_security_level,
                    view.effective_security_level) if v is not None
    )
    assert (degenerate_level <= pred.clearance) is False, "对照失效：退化语义未拒绝"

    assert allows(pred, view).allowed is True
    assert qdrant_filter_matches(pred, _payload_of(view)) is True
    assert sql_clause_matches(pred, view) is True


def test_positive_control_high_parent_with_missing_effective_is_denied() -> None:
    """对照 B（抓"Deny-4 只判 effective_security_level 一个键"）.

    只给 ``parent_security_level=3``（``security``/``effective`` 均缺）、
    ``clearance=1``：内核取 ``max(3)=3``，``3 > 1`` **拒绝**。

    退化语义（Deny-4 只 Range 在 ``effective_security_level`` 上，该字段缺失 ⇒
    Range 不匹配 ⇒ 不排除 = fail-open）会**放行** —— 与正确语义相反，可区分。
    """
    pred = scope(clearance=1).predicate(now=FIXED_NOW)
    view = level_obj(name="pc_parent_only", parent_security_level=3)

    # 反证：只判 effective 的 fail-open 形态必然放行该对象（effective 缺）。
    degenerate_fail_open = (
        view.effective_security_level is None
        or view.effective_security_level <= pred.clearance
    )
    assert degenerate_fail_open is True, "对照失效：退化语义未放行"

    assert allows(pred, view).allowed is False
    assert qdrant_filter_matches(pred, _payload_of(view)) is False
    assert sql_clause_matches(pred, view) is False


def test_positive_control_strict_all_missing_defaults_to_max() -> None:
    """对照 C（抓"严格模式下三键全缺未按默认档 3 排除"）.

    三键全缺、``clearance=1``、严格：内核落默认档 ``3``，``3 > 1`` **拒绝**。

    退化语义（缺失即 fail-open、不排除）会**放行** —— 与正确语义相反，可区分。
    """
    pred = scope(clearance=1, strict=True).predicate(now=FIXED_NOW)
    view = level_obj(name="pc_all_missing_strict")

    # 反证：仅靠 Deny-4（判"任一键 > clearance"）不会排除该对象（三键全缺）。
    deny4_would_match = any(
        v is not None and v > pred.clearance
        for v in (view.security_level, view.parent_security_level,
                  view.effective_security_level)
    )
    assert deny4_would_match is False, "对照失效：Deny-4 本可排除"

    assert allows(pred, view).allowed is False
    assert qdrant_filter_matches(pred, _payload_of(view)) is False
    assert sql_clause_matches(pred, view) is False


# ═══════════════════════════════════════════════════════════════════════════════
# 单列的定向断言（矩阵跑绿了也要有"人能读懂"的验收点）
# ═══════════════════════════════════════════════════════════════════════════════


def test_old_vectors_without_new_fields_are_not_excluded() -> None:
    """B4：老向量缺 ``effective_security_level`` ⇒ fail-open 放行，交 PG 终判。"""
    pred = scope(clearance=1).predicate(now=FIXED_NOW)
    payload = {"document_id": DOC_ID, "tenant_id": TENANT_A,
               "access_level": "tenant", "user_id": U2,
               "department_id": DEPT_D1}
    assert qdrant_filter_matches(pred, payload) is True


def test_over_clearance_is_excluded_on_both_legs() -> None:
    pred = scope(clearance=1).predicate(now=FIXED_NOW)
    view = obj(name="x", security_level=3, access_level="tenant")
    assert allows(pred, view).allowed is False
    assert qdrant_filter_matches(pred, _payload_of(view)) is False
    assert sql_clause_matches(pred, view) is False


def test_need_to_know_survives_the_prefilter() -> None:
    """密级不足 + 未过期授权：前置过滤**不得**先把候选剪掉（否则第 11 环看不到）。"""
    pred = scope(clearance=1, principals=frozenset({PRINCIPAL_U1,
                                                    PRINCIPAL_ROLE_EMPLOYEE})
                 ).predicate(now=FIXED_NOW)
    view = obj(name="ntu", security_level=3, access_level="tenant",
               acl_allow=frozenset({PRINCIPAL_U1}), acl_expires_at=FUTURE)
    payload = _payload_of(view)
    payload["acl_expires_at_ts"] = FUTURE.timestamp()
    assert allows(pred, view).allowed is True
    assert qdrant_filter_matches(pred, payload) is True


def test_expired_need_to_know_is_excluded_on_both_legs() -> None:
    pred = scope(clearance=1, principals=frozenset({PRINCIPAL_U1,
                                                    PRINCIPAL_ROLE_EMPLOYEE})
                 ).predicate(now=FIXED_NOW)
    view = obj(name="ntu_exp", security_level=3, access_level="tenant",
               acl_allow=frozenset({PRINCIPAL_U1}), acl_expires_at=PAST)
    payload = _payload_of(view)
    payload["acl_expires_at_ts"] = PAST.timestamp()
    assert allows(pred, view).allowed is False
    assert qdrant_filter_matches(pred, payload) is False
    assert sql_clause_matches(pred, view) is False


def test_project_miss_is_excluded_on_both_legs() -> None:
    pred = scope().predicate(now=FIXED_NOW)
    view = obj(name="pm", visibility_mode="project",
               project_ids=frozenset({"p_beta"}), access_level="tenant")
    assert allows(pred, view).allowed is False
    assert qdrant_filter_matches(pred, _payload_of(view)) is False
    assert sql_clause_matches(pred, view) is False


def test_deny_and_excluded_are_excluded_on_both_legs() -> None:
    pred = scope(clearance=3).predicate(now=FIXED_NOW)
    for view in (
        obj(name="d1", owner_id=U1, access_level="private",
            acl_deny=frozenset({PRINCIPAL_U1})),
        obj(name="d2", owner_id=U1, excluded=True),
    ):
        assert allows(pred, view).allowed is False
        assert qdrant_filter_matches(pred, _payload_of(view)) is False
        assert sql_clause_matches(pred, view) is False


def test_unrestricted_predicate_excludes_nothing() -> None:
    pred = ScopePredicate(user_id=U1, tenant_ids=None, unrestricted=True,
                          clearance=0, now=FIXED_NOW)
    view = obj(name="secret", security_level=3, access_level="tenant",
               tenant_id=TENANT_B)
    assert qdrant_filter_matches(pred, _payload_of(view)) is True


def test_fail_closed_tenant_sets_exclude_everything_non_personal() -> None:
    for tenant_ids in (frozenset(), None):
        pred = scope(tenant_ids=tenant_ids).predicate(now=FIXED_NOW)
        view = obj(name="t", access_level="tenant")
        assert allows(pred, view).allowed is False
        assert qdrant_filter_matches(pred, _payload_of(view)) is False
        assert sql_clause_matches(pred, view) is False


def test_to_sql_on_document_model_compiles_and_is_a_strict_subset() -> None:
    """
    ``to_sql(pred, Document)`` 必须在既有 ``document_scope_clause`` 之上**追加**，
    而不是替换 —— 否则列表 / 检索 / 关键词五条路径的语义会一起变。
    """
    from sqlalchemy import and_

    from app.db.models import Document
    from app.services.tenancy import document_scope_clause

    pred = scope().predicate(now=FIXED_NOW)
    clause = to_sql(pred, Document)

    assert isinstance(clause, sa.sql.elements.BooleanClauseList)
    # 按**函数名**比较，不要用 ``is and_``：SQLAlchemy 2.0 里 ``clause.operator``
    # 是 ``sqlalchemy.sql.operators.and_``，与 ``from sqlalchemy import and_``
    # 拿到的对象不是同一个（与上面 ``_eval`` 里 or_ 的坑同源）。
    assert getattr(clause.operator, "__name__", "") == "and_"
    base = document_scope_clause(
        owner_id=uuid.UUID(pred.user_id),
        department_id=pred.department_id,
        tenant_ids=pred.tenant_ids,
        owns_tenant_ids=pred.owns_tenant_ids,
        tenant_wide=pred.tenant_wide,
        unrestricted=pred.unrestricted,
    )
    # 第一个子条件就是既有三维。``and_(base, _security_sql(...))`` 会把 ``base``
    # 包成一个 ``Grouping``（渲染时多一对括号），故先解开再比较字符串。
    first = clause.clauses[0]
    if isinstance(first, sa.sql.elements.Grouping):
        first = first.element
    assert str(first) == str(base)
    import sqlalchemy.dialects.postgresql as _pg
    compiled = str(clause.compile(dialect=_pg.dialect()))
    assert "security_level" in compiled or "effective_security_level" in compiled


def test_to_sql_raises_rather_than_degrading_to_no_filter() -> None:
    """共享知识 7：编译失败必须抛 ``ScopeCompileError``，不得静默放行。"""
    from app.services.security_policy import ScopeCompileError

    class _Broken:
        __tablename__ = "broken_model"

    with pytest.raises(ScopeCompileError):
        to_sql(scope().predicate(now=FIXED_NOW), _Broken)


def test_to_sql_rejects_none_predicate() -> None:
    from app.services.security_policy import ScopeCompileError

    with pytest.raises(ScopeCompileError):
        to_sql(None, _STUB.c)      # type: ignore[arg-type]
