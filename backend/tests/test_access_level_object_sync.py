"""
三层知识库「层级变更 → 对象级权限行」同步的回归门禁.

背景（这正是被修复的真实缺陷）
──────────────────────────────
三层知识库的可见性在**两处**记录，而"发布 / 收回 / 转为部门文档"历史上只改了
其中一处：

    documents          access_level / department_id   ← 列表、详情、文档级判定
    document_objects   access_level / department_id   ← 第 11/12 环逐块 allows()、
                                                         GET /{id}/chunks 逐块复核

对象行是**入库那一刻的快照**（文档入库时必然是 private）。于是「发布到公司库 /
转为部门文档」之后：

    * 列表看得见、点得开、文档级判定全对；
    * "原文预览"对**除 owner 外**的所有人返回 0 条分块
      （界面显示成"该文档暂无索引内容（可能仍在处理中）"）；
    * 检索第 11 / 12 环把该文档的全部 chunk 丢弃 → 表现为拒答。

owner 走 ``_source_gate`` 的 owner 分支恒放行，所以**文档作者自测永远正常**，
必须换账号才复现 —— 这是它长期没被发现的原因。

本文件锁三件事：
    A. 语义：陈旧的 ``access_level='private'`` 对象行确实会挡住部门同事（问题成立）
    B. 修复：``security_cascade.sync_access_level`` 写入正确的两列（含归一化）
    C. 接线：``knowledge_tier_service.set_document_access_level`` 必须调用它
       （源码断言 —— 少一次调用就静默复发，而 DB 层测试抓不到"忘了调"）
"""

from __future__ import annotations

import asyncio
import inspect
import uuid

import pytest

DOC_ID = uuid.UUID("11111111-2222-3333-4444-555555555555")


# ═══════════════════════════════════════════════════════════════════════════════
# A. 语义：陈旧对象行 = 部门同事看不见（问题成立性）
# ═══════════════════════════════════════════════════════════════════════════════


def _dept_colleague_pred(*, clearance: int = 1):
    """
    部门同事的 Scope（真实形态：employee 默认密级 1）.

    ⚠️ ``clearance`` 必须显式给 1：``ScopePredicate.clearance`` 的默认值是
    ``SECURITY_LEVEL_MIN = 0``（"密级解析失败"的兜底），而对象密级是
    ``DEFAULT_SECURITY_LEVEL = 1``。用默认值构造的 predicate 会**先被密级闸门
    拒掉**（``clearance_short``，gate=``security``），根本走不到我们要测的
    ``source`` 闸门 —— 那样 A 组就变成"测试自己造的错误前提"，而不是在验证
    对象行陈旧的后果。真实用户走 ``build_predicate`` 时密级来自
    ``DEFAULT_CLEARANCE_BY_ROLE``（employee/editor/user 均为 1）。
    """
    from app.services.security_policy import ScopePredicate

    return ScopePredicate(
        user_id="22222222-2222-2222-2222-222222222222",
        tenant_ids=frozenset({"c_a"}),
        department_id="d_tech",
        clearance=clearance,
    )


def _doc_view(access_level: str, department_id: str | None):
    from app.services.security_policy import ObjectACLView

    return ObjectACLView(
        object_id=str(DOC_ID),
        object_type="text_chunk",
        document_id=str(DOC_ID),
        tenant_id="c_a",
        owner_id="99999999-9999-9999-9999-999999999999",   # 不是这位同事
        access_level=access_level,
        department_id=department_id,
        security_level=1,
        effective_security_level=1,
    )


def test_stale_private_object_row_blocks_department_colleague() -> None:
    """问题成立：文档已发布到部门库，但对象行还写着 private → 同事被拒。"""
    from app.services.security_policy import allows

    pred = _dept_colleague_pred()
    stale = allows(pred, _doc_view("private", None))
    assert stale.allowed is False, "陈旧对象行必须复现「部门同事看不到分块」"
    # 必须是 **source** 闸门拒的：这才是"对象行 access_level 陈旧"的直接后果。
    # 若这里变成 security/tenant，说明测试前提（密级 / 公司）造错了，
    # 缺陷本身并没有被 L 到 —— 断言闸门名是这条测试的防自欺装置。
    assert stale.gate == "source", stale


def test_gate_isolation_clearance_zero_is_blocked_by_security_gate() -> None:
    """
    对照：密级 0 的用户被 **security** 闸门挡住，与对象行同步无关.

    存在的意义是防"测试自己造错误前提"：本文件 A 组要证明的是
    「access_level 陈旧 ⇒ source 闸门拒」，因此 predicate 的密级必须与对象同档
    （1），否则失败原因会是 ``clearance_short`` 而让人误以为修复没生效。
    这条测试把两种闸门的边界钉住。
    """
    from app.services.security_policy import allows

    zero = _dept_colleague_pred(clearance=0)
    # 即便对象行**已经是** department + 本部门（最理想状态），密级不足仍然拒
    denied = allows(zero, _doc_view("department", "d_tech"))
    assert denied.allowed is False
    assert denied.gate == "security", denied
    assert "clearance_short" in denied.reason, denied


def test_synced_object_row_lets_department_colleague_in() -> None:
    """修复后：同一对象行写成 department + 本部门 → 同事可见（阳性对照）。"""
    from app.services.security_policy import allows

    pred = _dept_colleague_pred()
    fixed = allows(pred, _doc_view("department", "d_tech"))
    assert fixed.allowed is True, f"同步后的对象行应放行，实际 {fixed}"

    # 阴性对照：别的部门仍不可见（修复不能变成"人人可见"）
    other = _doc_view("department", "d_finance")
    assert allows(pred, other).allowed is False


def test_synced_tenant_object_row_is_company_wide() -> None:
    """公司库：同步后本公司任意同事可见，且不再带旧部门归属。"""
    from app.services.security_policy import allows

    pred = _dept_colleague_pred()
    assert allows(pred, _doc_view("tenant", None)).allowed is True
    assert allows(pred, _doc_view("tenant", "d_tech")).allowed is True


# ═══════════════════════════════════════════════════════════════════════════════
# B. 修复：sync_access_level 的写入内容与归一化
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _CapturingSession:
    """只记录被执行的语句（async）；不连库、不开事务。"""

    def __init__(self, rowcount: int = 3) -> None:
        self.statements: list = []
        self._rowcount = rowcount

    async def execute(self, stmt):      # noqa: ANN001
        self.statements.append(stmt)
        return _FakeResult(self._rowcount)


def _compiled_params(stmt) -> dict:      # noqa: ANN001
    from sqlalchemy.dialects import postgresql

    return dict(stmt.compile(dialect=postgresql.dialect()).params)


@pytest.mark.parametrize(
    ("raw_level", "raw_dept", "want_level", "want_dept"),
    [
        ("department", "d_tech", "department", "d_tech"),
        # 大小写 / 空格归一化
        ("  DEPARTMENT  ", " d_tech ", "department", "d_tech"),
        # 公司库（tenant）**必须清空**部门归属，避免旧值残留日后误判
        ("tenant", "d_tech", "tenant", None),
        # 个人库同样清空
        ("private", "d_tech", "private", None),
        # 未识别的取值回落 private（绝不把"没认出来"实现成"更宽松"）
        ("supertier", "d_tech", "private", None),
        ("", "d_tech", "private", None),
        # 部门库但没给部门 → 仍记 department，部门为空（调用方负责拒绝这种输入）
        ("department", "", "department", None),
    ],
)
def test_sync_access_level_values(raw_level, raw_dept, want_level, want_dept) -> None:
    from app.services.security_cascade import sync_access_level

    sess = _CapturingSession(rowcount=7)
    written = asyncio.run(
        sync_access_level(
            DOC_ID, access_level=raw_level, department_id=raw_dept, session=sess
        )
    )

    assert written == 7, "应把 rowcount 透传给调用方（0 是'一行都没改'的信号）"
    assert len(sess.statements) == 1, "必须是**一条** UPDATE（不是逐行写）"
    params = _compiled_params(sess.statements[0])
    assert params.get("access_level") == want_level, params
    assert params.get("department_id") == want_dept, params


def test_sync_access_level_failure_is_not_fatal() -> None:
    """写对象行失败不得把调用方（层级变更）拖崩 —— 返回 0 并记日志。"""

    class _Boom:
        async def execute(self, stmt):      # noqa: ANN001, ARG002
            raise RuntimeError("db down")

    from app.services.security_cascade import sync_access_level

    assert asyncio.run(
        sync_access_level(
            DOC_ID, access_level="tenant", department_id=None, session=_Boom()
        )
    ) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# C. 接线：层级变更必须真的调用同步（源码门禁）
# ═══════════════════════════════════════════════════════════════════════════════


def test_set_document_access_level_calls_sync_access_level() -> None:
    """
    ``set_document_access_level`` 是层级变更的**唯一实现点**，必须同时更新
    ``documents`` 与 ``document_objects``。

    刻意用源码断言而不是 DB 行为断言：「忘了调用」在 DB 层表现为"对象行没变"，
    而这恰好与"文档本来就没发布过"无法区分；源码断言能精确抓到漏接线。
    """
    from app.services import knowledge_tier_service as kts

    src = inspect.getsource(kts.set_document_access_level)
    assert "sync_access_level" in src, "层级变更必须同步 document_objects 的 access_level"
    # 阳性对照：拼接词不代表调用（防止变量名/注释里出现就算过）
    assert "await sync_access_level(" in src, src


def test_transfer_document_to_department_goes_through_set_document_access_level() -> None:
    """「转为部门文档」（用户报的场景）必须复用同一个实现点，不得另写一条路径。"""
    from app.services import knowledge_tier_service as kts

    src = inspect.getsource(kts.transfer_document_to_department)
    assert "set_document_access_level(" in src, src


def test_both_view_loaders_share_key_rules() -> None:
    """
    单文档版与批量版的**键规则必须一致**（设计文档标注 #11 的分叉点）。

    分叉后果：父块行只在一版里查得到，走另一版的调用点会把它当"缺行"按
    fail-closed 丢掉 —— 同一份数据在两条入口上给出相反结论。
    """
    from app.services import security_cascade as sc

    single = inspect.getsource(sc.load_document_view_index)
    batch = inspect.getsource(sc.load_view_indexes)
    for token in ("img:", "pc:", "ci:"):
        assert token in single and token in batch, (
            f"键规则分叉：{token!r} 只出现在 "
            f"{'单文档版' if token in single else '批量版'}里"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# D. 存量回填脚本：报告条件必须与修复条件同构
# ═══════════════════════════════════════════════════════════════════════════════


def _load_ops_script():
    """加载 ``scripts/repair_object_access_level.py``（不执行 main）。"""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "repair_object_access_level.py"
    )
    spec = importlib.util.spec_from_file_location("_ops_objfix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ops_script_report_matches_fix_predicate() -> None:
    """
    回填脚本的**报告条件**与 ``sync_all_access_levels`` 的 **UPDATE 条件**必须同构.

    这是运维脚本最容易失去信任的地方：报告里列出 3 条、``--apply`` 却改 0 行
    （或反过来）。两边各写一份条件就必然在某次改动后分叉，所以用断言把关键的
    谓词片段钉在一起 —— 改了一边没改另一边，这条测试会红。
    """
    from app.services import security_cascade as sc

    ops = _load_ops_script()
    report_pred = ops._STALE_PREDICATE
    fix_src = inspect.getsource(sc.sync_all_access_levels)

    for token in (
        "IS DISTINCT FROM",
        "LOWER(COALESCE(d.access_level, 'private'))",
        "'department'",
        "d.department_id",
    ):
        assert token in report_pred, f"报告条件缺 {token!r}：{report_pred}"
        assert token in fix_src, f"修复条件缺 {token!r}（两侧已分叉）"

    # 报告也只读：SELECT ... FROM document_objects JOIN documents
    assert "FROM document_objects" in ops._REPORT_SELECT
    assert "JOIN documents" in ops._REPORT_SELECT
    for write_verb in ("UPDATE ", "DELETE ", "INSERT "):
        assert write_verb not in ops._REPORT_SELECT, (
            f"报告语句里出现 {write_verb!r} —— dry-run 必须只读"
        )


def test_ops_script_is_dry_run_by_default() -> None:
    """默认不写：``--apply`` 是唯一的写入开关（与 repair_acl_payload.py 同约定）。"""
    ops = _load_ops_script()

    src = inspect.getsource(ops.main)
    assert "--apply" in src, src
    # 未给出 --apply 时必须 return，且 return 出现在调用写入函数之前
    guard = src.index("if not args.apply:")
    tail = src[guard:]
    assert "return" in tail, "缺少 dry-run 早退"
    assert tail.index("return") < tail.index("sync_all_access_levels"), (
        "dry-run 分支必须在写入之前返回"
    )

