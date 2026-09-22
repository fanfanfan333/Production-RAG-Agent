"""
``scripts/run_eval_baseline.py`` 回归门禁脚本的**门禁测试**.

被锁的东西（本轮修复的两处真实缺陷 + 一处返回契约）
──────────────────────────────────────────────────────
该脚本是「检索质量回归门禁」，但它**不在** ``app/**`` 下，所以不受
``tests/test_no_unscoped_retrieval.py`` 的 AST 扫描覆盖。它历史上：

1. ``check_negatives`` / ``measure_band_headroom`` 用旧的**三维**
   ``tenancy.request_scope`` 展开 scope，而 HTTP 端点 ``POST /eval/run`` 用
   **五维** ``security_scope.request_security_scope``（经 ``content_scope`` →
   ``exclude_test_tenants`` 剔除测试公司）。于是两个辅助函数在**比端点更宽**的
   语料上测量，脚本里"完全同源"的注释是假的。现改为 ``main()`` 一次签发
   ``UserScope`` 并透传三处，检索统一走 ``retrieve_chunks_scoped(scope=...)``。
2. ``resolve_labels`` 过去按 filename 命中就无条件把 ``document_id::chunk_index``
   塞进 ``relevant``，从不检查调用者能否看见该文档 —— 权限范围外的标注被算成
   "检索漏了"，与真正的漏召回在输出上无法区分。现在它用**与检索链路同一个判定
   内核**（``security_policy.allows`` + ``build_predicate``）复核可达性，不可达的
   标注进 ``unattainable``（``{query, filename, gate, reason}``），可达标注全空的
   整条用例从评测集剔除，返回三元组 ``(cases, problems, unattainable)``。

测试分组
────────
A. 可达性过滤（真判定内核 + 假 DB 会话，行为断言）
B. 部分不可达 / 全部不可达 / 阴性对照
C. 源码接线断言（锁"忘了改回去 / 改回来"）
D. ``judge()`` 必须把 ``unattainable`` 交出来（返回契约）

为什么 C 组必须做源码断言（沿用本仓库 ``test_access_level_object_sync.py`` 的先例）
──────────────────────────────────────────────────────────────────────────
这三处接线断了，在 DB 层表现为"结果看起来还挺好"：三维 scope 比五维宽 →
测量口径悄悄变宽但数字仍然漂亮；检索函数漏传 ``scope=`` → 要么报错要么退化成
更宽口径。**只跑行为的测试根本察觉不到**，因为失效方向是"变宽松"而非"报错"。
所以这里用 ``inspect.signature`` / ``ast`` 直接对源码结构断言。
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import importlib.util
import uuid
from pathlib import Path

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# 加载被测脚本（沿用仓库既有约定：``test_access_level_object_sync.py`` 的
# ``_load_ops_script()``）—— 该脚本有 ``if __name__ == "__main__":`` 守卫，
# import 时**不会**执行 main，因此安全。
# ═══════════════════════════════════════════════════════════════════════════════

_HARNESS_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_eval_baseline.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("_eval_harness", _HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)      # 只定义函数；main 有 __main__ 守卫
    return module


harness = _load_harness()
SRC = _HARNESS_PATH.read_text(encoding="utf-8")

# 固定文档 id（字符串形态，模拟 ``id::text`` 出来的坐标）
A_ID = "aaaaaaaa-0000-0000-0000-000000000001"
B_ID = "bbbbbbbb-0000-0000-0000-000000000002"
C_ID = "cccccccc-0000-0000-0000-000000000003"


# ═══════════════════════════════════════════════════════════════════════════════
# 测试夹具：真 UserScope + 假 DB 会话
# ═══════════════════════════════════════════════════════════════════════════════


def _scope():
    """
    调用者本人的**真** :class:`UserScope`（五维）.

    ⚠️ ``clearance`` 必须显式给 3（admin 档），**不能用默认值 0**：
    对象行密级缺省会被判定为 ``DEFAULT_SECURITY_LEVEL = 1``，clearance=0 的
    predicate 会**先被密级闸门拒掉**（gate=``security``），根本走不到我们要测的
    ``tenant`` 闸门 —— 那样 A/B 组就变成"测试自己造的错误前提"。
    """
    from app.services.security_scope import UserScope
    from app.services.tenancy import DocumentScope

    return UserScope(
        base=DocumentScope(
            owner_id=uuid.UUID("00000000-0000-0000-0000-0000000000aa"),
            tenant_ids=frozenset({"default"}),
            owns_tenant_ids=frozenset(),
            department_id=None,
            tenant_wide=True,
        ),
        user_id="u-self",
        role="admin",
        clearance=3,
    )


def _doc_obj(doc_id: str, *, tenant_id: str, access_level: str, owner_id: str,
             department_id: str | None = None) -> dict:
    """一条 ``document_objects`` 里 ``object_type='doc'`` 的行（mapping 形态）。"""
    return {
        "object_id": doc_id,
        "object_type": "doc",
        "document_id": doc_id,
        "tenant_id": tenant_id,
        "owner_id": owner_id,
        "department_id": department_id,
        "access_level": access_level,
        "security_level": None,
        "parent_security_level": None,
        "effective_security_level": None,
        "visibility_mode": "tier",
        "project_ids": [],
        "acl_allow": [],
        "acl_deny": [],
        "acl_expires_at": None,
        "excluded": False,
        "acl_sync_state": "synced",
    }


# 固定语料：A 可达（default/private/本人）；B、C 不可达（t-test，不在 scope tenant_ids）
_DOCUMENTS = [("a.txt", A_ID), ("b.txt", B_ID), ("c.txt", C_ID)]
_OBJECTS = [
    _doc_obj(A_ID, tenant_id="default", access_level="private", owner_id="u-self"),
    _doc_obj(B_ID, tenant_id="t-test", access_level="department", owner_id="u-other"),
    _doc_obj(C_ID, tenant_id="t-test", access_level="department", owner_id="u-other"),
]


class _Rows:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list:
        return list(self._rows)


class _Result:
    """极简 execute 结果：``.all()`` 给 documents 行，``.mappings().all()`` 给对象行。"""

    def __init__(self, rows: list, mapping_rows: list) -> None:
        self._rows = rows
        self._mapping_rows = mapping_rows

    def all(self) -> list:
        return list(self._rows)

    def mappings(self) -> _Rows:
        return _Rows(self._mapping_rows)


class _FakeSession:
    def __init__(self, documents: list, objects: list) -> None:
        self._documents = documents
        self._objects = objects

    async def execute(self, stmt, params=None):      # noqa: ANN001
        sql = getattr(stmt, "text", "") or str(stmt)
        params = params or {}
        if "document_objects" in sql:                # 第二条查询
            wanted = set(params.get("ids") or [])
            return _Result([], [r for r in self._objects if r["document_id"] in wanted])
        wanted = set(params.get("names") or [])      # 第一条查询（from documents）
        return _Result([r for r in self._documents if r[0] in wanted], [])


class _FakeSessionCM:
    """异步上下文管理器，``yield`` 一个假 session（对齐 ``get_db_session`` 的用法）。"""

    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *exc) -> bool:
        return False


def _install_fake_db(monkeypatch) -> None:
    """替换 ``app.db.postgres.get_db_session``（函数内延迟导入，故 patch 模块属性生效）。"""

    def _factory() -> _FakeSessionCM:
        return _FakeSessionCM(_FakeSession(_DOCUMENTS, _OBJECTS))

    monkeypatch.setattr("app.db.postgres.get_db_session", _factory)


def _case(query: str, *expects: tuple[str, int]) -> dict:
    return {
        "query": query,
        "expect": [{"filename": f, "chunk_index": i} for f, i in expects],
    }


def _resolve(monkeypatch, cases: list[dict]):
    _install_fake_db(monkeypatch)
    golden = {"name": "t", "cases": cases, "negative_cases": []}
    return asyncio.run(harness.resolve_labels(golden, _scope()))


# ═══════════════════════════════════════════════════════════════════════════════
# A. 可达性过滤：范围外标注必须进 unattainable（而不是混进 relevant）
# ═══════════════════════════════════════════════════════════════════════════════


def test_unreachable标注被剔除且gate是tenant(monkeypatch) -> None:
    """case1 标 A（可达）、case2 标 B（tenant 范围外）→ 只留 case1。"""
    cases, problems, unattainable = _resolve(monkeypatch, [
        _case("q1", ("a.txt", 0)),
        _case("q2", ("b.txt", 0)),
    ])

    assert [c["query"] for c in cases] == ["q1"]
    assert cases[0]["relevant"] == [f"{A_ID}::0"]
    assert problems == []

    assert len(unattainable) == 1, unattainable
    record = unattainable[0]
    # 断言 gate 而不是只断言条数 —— 否则测不出"是哪一个闸门在拒"。
    # B 的 access_level=department 且 tenant='t-test' 不在 scope.tenant_ids 内，
    # 因此必须由 **tenant** 闸门拒掉。
    assert record["gate"] == "tenant", record
    assert record["filename"] == "b.txt", record
    assert record["query"] == "q2", record
    assert {"query", "filename", "gate", "reason"} == set(record), record


def test_reachable_document_is_not_flagged(monkeypatch) -> None:
    """阴性对照：本人 private 文档可达 → 不进 unattainable（证明过滤不是"一律剔几条"）。"""
    cases, _, unattainable = _resolve(monkeypatch, [
        _case("q1", ("a.txt", 0), ("a.txt", 1)),
    ])

    assert unattainable == []
    assert len(cases) == 1
    assert cases[0]["relevant"] == [f"{A_ID}::0", f"{A_ID}::1"], "可达标注数必须原样保留"


# ═══════════════════════════════════════════════════════════════════════════════
# B. 部分不可达 / 全部不可达
# ═══════════════════════════════════════════════════════════════════════════════


def test_partial_unreachable_case_is_kept(monkeypatch) -> None:
    """一条用例一个可达 + 一个不可达 → 用例**保留**（relevant 只剩可达那条）。"""
    cases, _, unattainable = _resolve(monkeypatch, [
        _case("q1", ("a.txt", 0), ("b.txt", 0)),
    ])

    assert len(cases) == 1, cases
    assert cases[0]["relevant"] == [f"{A_ID}::0"], cases
    assert len(unattainable) == 1
    assert unattainable[0]["gate"] == "tenant"


def test_all_unreachable_case_is_dropped(monkeypatch) -> None:
    """一条用例两个标注都不可达 → 整条用例**被剔除**（不留必然 0 召回的用例）。"""
    cases, _, unattainable = _resolve(monkeypatch, [
        _case("q-drop", ("b.txt", 0), ("c.txt", 0)),
    ])

    assert all(c["query"] != "q-drop" for c in cases), "全不可达的用例必须被剔除"
    assert cases == []
    # 剔除整条不等于吞掉留痕：两条都应在 unattainable 里
    assert len(unattainable) == 2
    assert {u["gate"] for u in unattainable} == {"tenant"}


# ═══════════════════════════════════════════════════════════════════════════════
# C. 源码接线断言（锁"忘了改回去 / 改回来"）
# ═══════════════════════════════════════════════════════════════════════════════


def _func_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _module_tree() -> ast.Module:
    return ast.parse(SRC)


def test_old_three_dim_signing_is_gone() -> None:
    """旧的三维签发（``tenancy.request_scope``）不得再被调用。"""
    # 注意：源码注释里**故意**留了 ``tenancy.request_scope`` 作为反例说明（无括号），
    # 所以这里断言的是**调用形态** ``request_scope(``，而不是字符串 ``request_scope``。
    assert "request_scope(" not in SRC, "旧三维 tenancy.request_scope( 仍被调用"
    assert "from app.services.tenancy import request_scope" not in SRC


def test_all_scoped_retrieval_calls_pass_scope() -> None:
    """每一处 ``retrieve_chunks_scoped(...)`` 调用都必须带 ``scope=``。"""
    assert "retrieve_chunks_scoped(" in SRC

    calls = [
        n for n in ast.walk(_module_tree())
        if isinstance(n, ast.Call) and _func_name(n.func) == "retrieve_chunks_scoped"
    ]
    assert calls, "源码里找不到 retrieve_chunks_scoped 调用（接线可能被回退）"
    for call in calls:
        kwarg_names = {kw.arg for kw in call.keywords}
        assert "scope" in kwarg_names, (
            f"retrieve_chunks_scoped 调用漏传 scope=（行 {call.lineno}）"
        )


@pytest.mark.parametrize("func_name", ["check_negatives", "measure_band_headroom"])
def test_helper_signatures_expose_scope(func_name: str) -> None:
    """两个辅助函数的签名里必须有 ``scope`` 形参（用 inspect，比 grep 更硬）。"""
    func = getattr(harness, func_name)
    params = inspect.signature(func).parameters
    assert "scope" in params, f"{func_name} 签名缺少 scope 形参：{list(params)}"


def test_request_security_scope_is_called_inside_main() -> None:
    """``request_security_scope(`` 必须在 ``main()`` 函数体内被调用（唯一签发点）。"""
    tree = _module_tree()
    main_def = next(
        (n for n in tree.body
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main"),
        None,
    )
    assert main_def is not None, "找不到 main()"

    calls = [
        n for n in ast.walk(main_def)
        if isinstance(n, ast.Call) and _func_name(n.func) == "request_security_scope"
    ]
    assert calls, "main() 里没有调用 request_security_scope（五维 scope 未签发）"


# ═══════════════════════════════════════════════════════════════════════════════
# D. judge() 必须把 unattainable 交出来（返回契约）
# ═══════════════════════════════════════════════════════════════════════════════


def _count_unattainable(value) -> int:      # noqa: ANN001
    """
    读出 ``unattainable`` 的条数.

    接受两种 shape：``{"count": N, "items": [...]}``（当前实现）/ ``{"total": N}``，
    以及裸 list/tuple。断言的是"**能读出条数**"，而不是某一具体字段名 —— 契约的
    要点是"该字段存在且可消费"，字段内部结构留实现自由。
    """
    if isinstance(value, dict):
        if "count" in value:
            return int(value["count"])
        if "total" in value:
            return int(value["total"])
        if "items" in value:
            return len(value["items"] or [])
    if isinstance(value, (list, tuple)):
        return len(value)
    raise AssertionError(f"unattainable 形状不可识别：{type(value)!r} -> {value!r}")


def _passing_report() -> dict:
    return {
        "recall": {"10": 1.0},
        "precision": {"3": 0.8},
        "mrr": 1.0,
        "evidence_slices": {"multi_evidence": {"all_gold_found_rate": 1.0}},
    }


_THRESHOLDS = {
    "min_recall_at_10": 0.8,
    "min_mrr": 0.8,
    "min_precision_at_3": 0.5,
    "min_multi_evidence_all_found_rate": 0.8,
}


def test_judge_surfaces_unattainable() -> None:
    """docstring 承诺返回里有 ``unattainable`` 字段 —— 必须真的存在且可读条数。"""
    one = [{"query": "q", "filename": "b.txt", "gate": "tenant", "reason": "tenant_mismatch"}]
    verdict = harness.judge(
        report=_passing_report(),
        negatives=[{"refused": True, "query": "x"}],
        thresholds=_THRESHOLDS,
        band=None,
        unattainable=one,
    )

    assert "unattainable" in verdict, (
        "judge() 返回里缺 unattainable 字段 —— docstring 已声明'见返回里的 "
        "unattainable 字段'，缺失会让消费方分不清『标注在权限范围外』与『没实现』"
    )
    assert _count_unattainable(verdict["unattainable"]) == 1

    # 权限事实不是检索质量回归 → 不得因 unattainable 非空而判失败
    assert verdict["passed"] is True, verdict.get("failures")


def test_judge_surfaces_empty_unattainable() -> None:
    """阴性对照：``unattainable=[]`` 时该键仍存在、计数为 0（不能"没有就省略"）。"""
    verdict = harness.judge(
        report=_passing_report(),
        negatives=[{"refused": True, "query": "x"}],
        thresholds=_THRESHOLDS,
        band=None,
        unattainable=[],
    )

    assert "unattainable" in verdict, "空集也必须交出来，否则消费方分不清'没有'与'没实现'"
    assert _count_unattainable(verdict["unattainable"]) == 0
    assert verdict["passed"] is True
