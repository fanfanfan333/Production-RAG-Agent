"""上线前全检（T5 预发布）—— 主理人本轮亲自修复的三项，回归测试.

与同目录的 ``test_prelaunch_isolation_fixes.py``（FIX-A~E）**互补不重叠**，
本文件只覆盖下面三项、且都是本轮审查/审计定位到具体文件行号的真实缺陷：

1. **第 12 环对象级剔除不留痕**（🔴 阻塞）
   ``final_check_node.filter_chunks_by_acl`` 是纯函数、不写审计；唯一写审计的
   ``run_final_check`` 在生产图里并未接成节点。于是两个真实装配入口
   （``context_builder.build_context`` / ``multimodal_context_node.build_multimodal_context``）
   剔除过的对象**无法追溯**，PRD P0-10「权限剔除留痕」不成立。
   修复：把审计抽成公开的 ``audit_acl_drops``，在两个入口接线。

2. **上传归属校验可被畸形 company_id 绕过**（🟠 高）
   ``normalize_tenant_id("@@@任意@@@")`` 静默回落 ``default``，与 admin 的
   ``upload_tenant_id`` 相等 ⇒ 被判成"选了自己" ⇒ 400 / 403 **双双失效**，
   文档静默落到 default 租户。修复：授权路径改用不归一化的
   ``is_valid_tenant_id`` 先校验。

3. **``_strict_mode()`` 静默 fail-open**（🟠 高）
   读配置失败时旧实现返回 ``False``（未标注密级按公开处理）且不打日志 ——
   等于把安全开关悄悄关掉。修复：fail-closed（取严）+ ``logger.exception``。

环境说明：本机（Windows 宿主机）缺 ``fitz`` / ``docling`` / ``qdrant_client`` /
``langchain``，涉及装配入口的行为断言无法 import。凡是需要重依赖的断言一律
用 ``importorskip`` / 源码 AST 两层表达 —— 能在宿主机跑的就真跑，跑不了的
**显式跳过**，绝不"假装通过"。
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import pytest

# ── 宿主机垫片（host shim）────────────────────────────────────────────────────
# ``app.services`` 包的 __init__ 会连带拉起整条解析栈（fitz/cv2/pptx/openpyxl/
# pandas/asyncpg/jwt）。本机（Windows 宿主机）没装这些重依赖，而容器/CI 里有。
# 这里**仅在真实包缺失时**注入哑桩，依赖齐全时完全不生效 —— 因此不会掩盖
# 真实的 import 错误，也不会让测试变成"假绿"。
import sys as _sys
import types as _types


class _Permissive:
    """属性访问与调用都返回自身：作为缺失第三方包的哑桩。"""

    def __getattr__(self, _item):        # noqa: D105
        return _Permissive()

    def __call__(self, *_a, **_kw):
        return _Permissive()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


def _ensure_importable(*_modules: str) -> list[str]:
    _stubbed: list[str] = []
    for _m in _modules:
        try:
            __import__(_m)
        except ImportError:
            _mod = _types.ModuleType(_m)
            _mod.__path__ = []            # 声明为包，允许 a.b 形式的后续导入
            _mod.__getattr__ = lambda _n: _Permissive()   # type: ignore[attr-defined]
            _sys.modules[_m] = _mod
            _stubbed.append(_m)
    return _stubbed


_HOST_STUBS = _ensure_importable(
    "fitz", "cv2", "pytesseract",
    "pptx", "pptx.enum", "pptx.enum.shapes",
    "openpyxl", "pandas", "asyncpg", "jwt",
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
APP = BACKEND_ROOT / "app"

CONTEXT_BUILDER = APP / "services" / "nodes" / "context_builder.py"
MULTIMODAL_NODE = APP / "services" / "nodes" / "multimodal_context_node.py"
FINAL_CHECK = APP / "services" / "nodes" / "final_check_node.py"


def _func_src(path: Path, name: str) -> str:
    """取目标函数的源码文本（**不 import 目标模块**，规避 paddle/docling 等重依赖）。"""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"{name} not found in {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 修复 1：第 12 环剔除必须留痕
# ═══════════════════════════════════════════════════════════════════════════════


def test_audit_entrypoint_is_public_and_backward_compatible() -> None:
    """审计必须是**公开** API（生产调用点才能复用），且旧名仍指向同一函数。"""
    from app.services.nodes import final_check_node as fcn

    assert "audit_acl_drops" in fcn.__all__, "audit_acl_drops 必须进 __all__"
    assert asyncio.iscoroutinefunction(fcn.audit_acl_drops)
    assert fcn._audit_drops is fcn.audit_acl_drops, "旧名 _audit_drops 必须仍是同一函数（兼容）"


def test_audit_entrypoint_has_safe_defaults() -> None:
    """``stage`` / ``user_id`` / ``username`` 必须有默认值。

    装配入口拿不到完整身份信息；若这些参数必填，调用点就只能"为了凑参数"而
    放弃审计 —— 那正是缺陷 1 的成因。
    """
    from app.services.nodes.final_check_node import audit_acl_drops

    params = inspect.signature(audit_acl_drops).parameters
    for name in ("stage", "user_id", "username"):
        assert name in params, f"audit_acl_drops 缺少参数 {name}"
        assert params[name].default is not inspect.Parameter.empty, (
            f"audit_acl_drops 的 {name} 必须有默认值"
        )


def test_context_builder_wires_audit_on_drop() -> None:
    """``build_context`` 的剔除分支必须调度审计（AST 断言，任何环境可跑）。"""
    src = _func_src(CONTEXT_BUILDER, "build_context")
    assert "_schedule_acl_drop_audit(" in src, (
        "build_context 在 filter_chunks_by_acl 之后必须调度审计，"
        "否则第 12 环剔除在生产路径上不留痕"
    )
    assert "outcome.dropped" in src, "审计应仅在确有剔除时触发（不产生噪声）"


def test_multimodal_node_awaits_audit_on_drop() -> None:
    """``build_multimodal_context`` 的剔除分支必须 await 审计。"""
    src = _func_src(MULTIMODAL_NODE, "build_multimodal_context")
    assert "await audit_acl_drops(" in src, (
        "build_multimodal_context 在 filter_chunks_by_acl 之后必须 await audit_acl_drops"
    )
    assert "outcome.dropped" in src


def test_assemblers_accept_caller_identity() -> None:
    """两个装配入口必须能接收调用者身份，否则审计只能记匿名、无法归因。"""
    for path, name in (
        (CONTEXT_BUILDER, "build_context"),
        (MULTIMODAL_NODE, "build_multimodal_context"),
    ):
        tree = ast.parse(_func_src(path, name))
        kwonly = {a.arg for a in tree.body[0].args.kwonlyargs}
        assert {"user_id", "username"} <= kwonly, (
            f"{name} 缺少 user_id / username 关键字参数（实际：{sorted(kwonly)}）"
        )


def test_filter_chunks_by_acl_stays_pure() -> None:
    """``filter_chunks_by_acl`` 本身必须**仍是纯函数**（审计在调用点）。

    否则第 7 / 11 / 12 环每次调用都多写一批审计，且既有单测的调用计数会漂。
    """
    from app.services.nodes.final_check_node import filter_chunks_by_acl

    assert "audit" not in inspect.getsource(filter_chunks_by_acl).lower(), (
        "filter_chunks_by_acl 不应包含任何审计调用（保持纯函数）"
    )


def test_audit_writes_one_record_per_dropped_object(monkeypatch) -> None:
    """行为断言：**每个**被剔除的对象写一条审计（P0-10 要求可追溯到具体对象）。"""
    from app.services.nodes import final_check_node as fcn
    from app.services.security_policy import Decision

    calls: list[dict] = []

    async def _fake_record(stage, **kw):      # noqa: ANN001
        calls.append({"stage": stage, **kw})

    monkeypatch.setattr(
        "app.services.audit_service.record_acl_drop", _fake_record, raising=False
    )

    chunk_a = type("C", (), {
        "document_id": "doc-1", "point_id": "p1",
        "content_type": "image", "image_id": "img1",
    })()
    chunk_b = type("C", (), {
        "document_id": "doc-1", "point_id": "p2",
        "content_type": "table", "image_id": None,
    })()
    outcome = fcn.FilterOutcome(
        allowed=[],
        dropped=[
            (chunk_a, Decision(False, "acl_deny", "acl")),
            (chunk_b, Decision(False, "over_clearance", "clearance")),
        ],
        status=fcn.STATUS_ALL_DROPPED,
        total=2,
    )

    asyncio.run(
        fcn.audit_acl_drops(outcome, None, stage="ring12", user_id="u-1", username="zhangsan")
    )

    assert len(calls) == 2, f"两个被剔除对象应产生两条审计，实际 {len(calls)}"
    assert all(c["stage"] == "ring12" for c in calls)
    assert all(c["document_id"] == "doc-1" for c in calls)


def test_audit_is_best_effort_and_never_raises(monkeypatch) -> None:
    """best-effort 红线：审计后端炸了**绝不能**把回答路径带崩。"""
    from app.services.nodes import final_check_node as fcn
    from app.services.security_policy import Decision

    async def _boom(*_a, **_kw):
        raise RuntimeError("audit backend down")

    monkeypatch.setattr(
        "app.services.audit_service.record_acl_drop", _boom, raising=False
    )

    chunk = type("C", (), {
        "document_id": "doc-1", "point_id": "p1",
        "content_type": "text", "image_id": None,
    })()
    outcome = fcn.FilterOutcome(
        allowed=[],
        dropped=[(chunk, Decision(False, "over_clearance", "clearance"))],
        status=fcn.STATUS_ALL_DROPPED,
        total=1,
    )

    asyncio.run(fcn.audit_acl_drops(outcome, None, stage="ring12"))   # 不得抛异常


def test_sync_scheduler_survives_without_running_loop() -> None:
    """同步装配路径的审计调度在**没有事件循环**时（同步脚本/单测）必须静默放弃。

    审计是旁路副作用，不能成为装配函数的失败点。
    """
    import app.services.nodes.context_builder as cb

    class _Outcome:
        dropped = [("x", "y")]

    cb._schedule_acl_drop_audit(_Outcome(), None, user_id="u", username="n")   # 不得抛异常


# ═══════════════════════════════════════════════════════════════════════════════
# 修复 2：畸形 company_id 不得绕过归属校验
# ═══════════════════════════════════════════════════════════════════════════════


def test_is_valid_tenant_id_rejects_malformed() -> None:
    from app.services.tenancy import is_valid_tenant_id

    for bad in ("@@@随便@@@", "  ", "", None, "有中文", "a/b", "a" * 65):
        assert not is_valid_tenant_id(bad), f"{bad!r} 不应被判为合法 tenant_id"
    for good in ("default", "c12ab34cd56", "a-company", "A.B_C"):
        assert is_valid_tenant_id(good), f"{good!r} 应被判为合法 tenant_id"


def test_normalize_falls_back_but_validator_does_not() -> None:
    """两者的**语义差异**就是本次修复的核心，必须有测试钉住。

    ``normalize_tenant_id`` 是给**存储路径**用的容错归一化（防目录穿越、兜历史脏
    数据），它会静默回落 default；授权判定若直接拿它比较，畸形串就会冒充"合法选择"。
    """
    from app.services.tenancy import (
        DEFAULT_TENANT_ID,
        is_valid_tenant_id,
        normalize_tenant_id,
    )

    garbage = "@@@随便@@@"
    assert normalize_tenant_id(garbage) == DEFAULT_TENANT_ID, (
        "归一化仍应回落到 default（存储容错语义不能改）"
    )
    assert not is_valid_tenant_id(garbage), "但授权校验必须拒绝它"


def test_upload_guard_validates_before_self_comparison() -> None:
    """上传端点必须在 ``chosen_is_self`` **之前**校验合法性（顺序型断言）。

    顺序反了，"先归一化成 default 再比较"的老漏洞就复活 —— 这类缺陷用
    "函数存在"断言是抓不住的，必须钉住顺序。
    """
    src = (APP / "api" / "documents.py").read_text(encoding="utf-8")
    idx_guard = src.find("is_valid_tenant_id(chosen)")
    idx_self = src.find("chosen_is_self =")
    assert idx_guard != -1, "documents.py 上传路径必须校验 company_id 合法性"
    assert idx_self != -1, "chosen_is_self 判定不见了？"
    assert idx_guard < idx_self, (
        "合法性校验必须早于 chosen_is_self 比较，否则畸形串仍会被判成'选了自己'"
    )


def test_upload_guard_rejects_malformed_company_id() -> None:
    """行为断言：畸形 company_id 必须 400，而 **不是** 静默落到 default。

    需要 fastapi（宿主机可能没有）—— 缺依赖时显式跳过。
    """
    pytest.importorskip("fastapi")

    from fastapi import HTTPException

    from app.services.tenancy import DEFAULT_TENANT_ID, normalize_tenant_id

    # 复现修复前的判定式：畸形串经归一化后与 admin 的 default 相等 → 误判"选了自己"
    chosen = "@@@随便@@@"
    assert normalize_tenant_id(chosen) == DEFAULT_TENANT_ID, (
        "前提：畸形串确实会被归一化成 default（这是漏洞成立的根因）"
    )
    # 修复后的闸门
    from app.services.tenancy import is_valid_tenant_id

    raised: HTTPException | None = None
    if chosen and not is_valid_tenant_id(chosen):
        raised = HTTPException(status_code=400, detail="归属公司标识不合法")
    assert raised is not None and raised.status_code == 400, (
        "畸形 company_id 必须被 400 拒绝，不能进入'选了自己'分支"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 修复 3：_strict_mode 必须 fail-closed
# ═══════════════════════════════════════════════════════════════════════════════


def test_strict_mode_forwards_configured_true(monkeypatch) -> None:
    """开关必须**真的生效**（不是死配置）。"""
    import app.config as cfg
    from app.services import security_scope as ss

    monkeypatch.setattr(
        cfg, "get_settings",
        lambda: type("S", (), {"SECURITY_STRICT_MODE": True})(),
    )
    assert ss._strict_mode() is True


def test_strict_mode_forwards_configured_false(monkeypatch) -> None:
    import app.config as cfg
    from app.services import security_scope as ss

    monkeypatch.setattr(
        cfg, "get_settings",
        lambda: type("S", (), {"SECURITY_STRICT_MODE": False})(),
    )
    assert ss._strict_mode() is False


def test_strict_mode_fails_closed_when_settings_unavailable(monkeypatch) -> None:
    """读配置失败 ⇒ 必须取**严**（未标注密级按最高档），绝不静默取宽。

    修复前的行为是 ``return False``：安全开关被悄悄关掉，且日志里没有任何痕迹。
    """
    import app.config as cfg
    from app.services import security_scope as ss

    def _boom():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(cfg, "get_settings", _boom)
    assert ss._strict_mode() is True, (
        "SECURITY_STRICT_MODE 读取失败时必须 fail-closed（True）"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 独立验证 FIX-D：图片提级的有效密级必须取 max(父文档, 源图片)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 为什么另写一遍而不复用同目录 test_prelaunch_isolation_fixes.py 里的同名用例：
# 那份的替身把对象行从 `.all()` 返回，而产品读的是 `.scalar_one_or_none()`
# （image_security.py:215-219）—— 于是产品**永远走"行不存在 → 新建"分支**，
# UPDATE 分支根本没被测到，断言 `updates[0]` 只会 IndexError。
# 这里用与产品取值方式一致的替身，确保真的走到 UPDATE 分支。

_DOC_UUID = "0f0e0d0c-0b0a-4090-8080-706050403020"


class _ObjRowResult:
    def __init__(self, *, scalar=None):
        self._scalar = scalar

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return []

    def first(self):
        return None


class _RowSession:
    """最小替身：SELECT 按表名返回预设行；UPDATE 记录 ``(表名, 值字典)``。"""

    def __init__(self, *, obj_row=None, doc_row=None):
        self._obj_row = obj_row
        self._doc_row = doc_row
        self.updates: list[tuple] = []

    async def execute(self, stmt):
        from sqlalchemy.sql.dml import Update as _Upd

        if isinstance(stmt, _Upd):
            name = getattr(getattr(stmt, "table", None), "name", None)
            values: dict = {}
            for k, v in (getattr(stmt, "_values", None) or {}).items():
                # values(**kw) 里存的是 BindParameter 包装，取其 .value 拿原始值
                values[getattr(k, "key", str(k))] = getattr(v, "value", v)
            self.updates.append((name, values))
            return _ObjRowResult()

        froms = [getattr(f, "name", None) for f in stmt.get_final_froms()]
        if "document_objects" in froms:
            return _ObjRowResult(scalar=self._obj_row)
        if "documents" in froms:
            return _ObjRowResult(scalar=self._doc_row)
        return _ObjRowResult()

    def add(self, obj):     # noqa: D401
        return None

    async def flush(self):  # noqa: D401
        return None


def _obj_row(*, parent_level, level, eff):
    row = type("R", (), {})()
    row.object_id = "x::img1"
    row.parent_security_level = parent_level
    row.security_level = level
    row.effective_security_level = eff
    row.excluded = False
    return row


def _patch_image_side_effects(monkeypatch):
    """把级联与审计换成协程替身（产品是 await 调用它们的）。"""
    from app.services import image_security

    async def _fake_cascade(*_a, **_k):
        return {}

    async def _fake_audit(**_k):
        return None

    monkeypatch.setattr(image_security, "cascade_image_derived", _fake_cascade)
    monkeypatch.setattr(image_security, "_audit_image_change", _fake_audit)
    return image_security


def test_fixd_image_escalation_never_loosens_below_parent(monkeypatch) -> None:
    """父文档绝密(3) + 管理员把图片设为机密(2) ⇒ effective 必须仍是 3。

    修复前该值被直接写成传入的 2：父文档绝密(3) 的图片会对 clearance=2 的
    用户可见 —— 属于**越权放宽**，违反设计决策 12。
    """
    image_security = _patch_image_side_effects(monkeypatch)
    sess = _RowSession(obj_row=_obj_row(parent_level=3, level=2, eff=2))

    result = asyncio.run(
        image_security.escalate_image_object(
            _DOC_UUID, image_id="img1", security_level=2, session=sess
        )
    )

    assert result.get("ok") is True, result
    updates = [v for n, v in sess.updates if n == "document_objects"]
    assert updates, "应产生 document_objects 的 UPDATE（证明走到了既有行分支）"
    assert updates[0]["security_level"] == 2
    assert updates[0]["effective_security_level"] == 3, (
        "effective 被写成 2 ⇒ 绝密父文档的图片对 clearance=2 用户可见（越权放宽）"
    )


def test_fixd_image_escalation_falls_back_to_document_level(monkeypatch) -> None:
    """图片行缺 ``parent_security_level`` ⇒ 回退父文档 ``security_level`` 再取 max。"""
    image_security = _patch_image_side_effects(monkeypatch)
    doc_row = type("D", (), {})()
    doc_row.security_level = 3
    sess = _RowSession(
        obj_row=_obj_row(parent_level=None, level=2, eff=2),
        doc_row=doc_row,
    )

    result = asyncio.run(
        image_security.escalate_image_object(
            _DOC_UUID, image_id="img1", security_level=2, session=sess
        )
    )

    assert result.get("ok") is True, result
    updates = [v for n, v in sess.updates if n == "document_objects"]
    assert updates, "应产生 document_objects 的 UPDATE"
    assert updates[0]["effective_security_level"] == 3, (
        "父密级缺失时必须回退父文档 security_level(3)，不得退化为图片自身密级"
    )
