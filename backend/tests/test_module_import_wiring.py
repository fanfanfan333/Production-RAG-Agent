"""跨模块导入接线门禁（纯 AST 静态解析，不真的 import）。

背景
────
2026-09-21 重烤镜像后实测发现：``app/main.py`` 的 ``lifespan`` 里写着

    from app.config import (
        ALLOW_INSECURE_JWT,          # ← config.py 里没有这个模块级名字
        is_jwt_secret_weak,
        jwt_secret_must_fail_startup,
    )

而 ``ALLOW_INSECURE_JWT`` 在 ``config.py`` 里只是 ``Settings`` 的**类字段**
（第 100 行，缩进在类体内），并不是模块级名字。于是：

    ImportError: cannot import name 'ALLOW_INSECURE_JWT' from 'app.config'
    ERROR:    Application startup failed. Exiting.

容器因此反复重启（RestartCount=3）；而**宿主机全量单元测试 1807 条全绿** ——
因为 ``lifespan`` 只在 FastAPI 真正启动时执行，测试从不启动 app。
这就是本项目最怕的那类失效：**测试全绿、服务起不来**。

为什么用 AST 而不是 ``import``
──────────────────────────────
宿主机缺 fitz/cv2/docling 等依赖（见 ``conftest.py`` 的 host-shim），
真的 import ``app.main`` 会拉进 FastAPI + 全部路由 + 全部服务模块 → 既慢又可能
因缺依赖而失败，把"接线错误"和"环境缺依赖"混在一起。AST 解析零依赖、零副作用，
且能**一次性扫全仓**，把整类"从一个模块 import 一个它不存在的顶层名字"钉死在 CI 里。

判定规则（一条 import 视为有效，当且仅当满足其一）
────────────────────────────────────────────────
1. 名字是该模块的顶层定义（def / class / 赋值 / 嵌套导入 / ``__all__`` 声明）；
2. 名字是目标包下的**子模块**（``from app.db import models`` → ``app/db/models.py``）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_SCAN_DIRS = (_BACKEND / "app", _BACKEND / "scripts")


# ── AST 工具 ─────────────────────────────────────────────────────────────────
def _names_in_target(target: ast.expr) -> set[str]:
    """解构赋值/注解赋值左侧的绑定名（支持元组解包）。"""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for elt in target.elts:
            out |= _names_in_target(elt)
        return out
    return set()


def _collect_from_body(body: list[ast.stmt]) -> set[str]:
    """收集一段语句体里会绑定到**当前模块命名空间**的名字。

    递归进入 ``if`` / ``try`` / ``with`` —— 条件定义（``if TYPE_CHECKING:``、
    ``try: import x except ImportError:`` 的兜底）同样是模块级名字。
    """
    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                names |= _names_in_target(tgt)
        elif isinstance(node, ast.AnnAssign):
            names |= _names_in_target(node.target)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.If):
            names |= _collect_from_body(node.body) | _collect_from_body(node.orelse)
        elif isinstance(node, ast.Try):
            names |= (
                _collect_from_body(node.body)
                | _collect_from_body(node.orelse)
                | _collect_from_body(node.finalbody)
            )
            for handler in node.handlers:
                names |= _collect_from_body(handler.body)
        elif isinstance(node, ast.With):
            names |= _collect_from_body(node.body)
    return names


def _module_file(dotted: str) -> Path | None:
    """``app.services.retrieval_service`` → ``backend/app/services/retrieval_service.py``。"""
    if not dotted.startswith("app"):
        return None
    base = _BACKEND.joinpath(*dotted.split("."))
    if base.is_dir():
        init = base / "__init__.py"
        return init if init.is_file() else None
    return base.with_suffix(".py") if base.with_suffix(".py").is_file() else None


def _top_level_names(path: Path) -> set[str]:
    """模块的顶层可导出名（含 ``__all__`` 里声明的字符串）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = _collect_from_body(tree.body)
    for node in tree.body:  # __all__ 显式声明的名字也算（可能是再导出）
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        names.add(elt.value)
    return names


def _iter_internal_imports():
    """产出 ``(源文件, 行号, 目标模块, 被导入名)``，仅限 ``from app.x import y`` 形式。"""
    for base in _SCAN_DIRS:
        if not base.is_dir():
            continue
        for py in sorted(base.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            except SyntaxError as exc:  # 语法错误单独报，别伪装成接线错误
                pytest.fail(f"语法错误：{py} → {exc}")
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level != 0 or not node.module or not node.module.startswith("app"):
                    continue  # 只查绝对导入的 app.* 内部引用
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    yield py, node.lineno, node.module, alias.name


# ── 门禁 ─────────────────────────────────────────────────────────────────────
def test_all_app_internal_imports_resolve() -> None:
    """全仓 ``from app.* import <name>`` 必须真的解析得到。

    这条门禁的引入契机就是 ``main.py`` 那行 ``ALLOW_INSECURE_JWT``：
    单元测试全绿、容器启动即崩。任何"接线断裂"都应在这里变红。
    """
    broken: list[str] = []
    checked = 0
    for src, lineno, module, name in _iter_internal_imports():
        target = _module_file(module)
        rel = src.relative_to(_BACKEND)
        if target is None:
            broken.append(f"{rel}:{lineno}  目标模块不存在：{module}")
            continue
        checked += 1
        if name in _top_level_names(target):
            continue
        # `from app.db import models` —— 名字是包下的子模块，同样有效
        sibling = target.parent
        if (sibling / f"{name}.py").is_file() or (sibling / name / "__init__.py").is_file():
            continue
        broken.append(f"{rel}:{lineno}  {module} 无顶层名 {name!r}")

    assert checked > 0, "没有扫到任何 app.* 内部导入 —— 扫描逻辑本身失效了（fail-closed）"
    assert not broken, "跨模块导入接线断裂：\n  " + "\n  ".join(broken)


def test_guard_scanner_can_detect_a_missing_name() -> None:
    """阳性对照：证明门禁**抓得住**这类缺陷，而不是"因为没扫到所以绿"。"""
    names = _top_level_names(_BACKEND / "app" / "config.py")
    # 模块级真实存在的函数
    assert "jwt_secret_must_fail_startup" in names
    # Settings 的类字段**不是**模块级名字 —— 这正是当初那次 ImportError 的形状
    assert "ALLOW_INSECURE_JWT" not in names
