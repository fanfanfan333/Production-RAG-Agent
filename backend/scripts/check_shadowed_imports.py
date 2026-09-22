"""
扫全仓：找出"函数内 import 遮蔽了模块级同名导入"的隐患.

为什么值得单独写一个检查器
──────────────────────────
Python 的作用域规则是"**整个函数**内只要有一次赋值/import，该名字就是局部
变量"。于是

    from app.services.tenancy import normalize_tenant_id     # 模块级

    def f(...):
        if cond:
            from app.services.tenancy import normalize_tenant_id   # 局部导入
            ...
        return normalize_tenant_id(x)      # ← cond 为假时 UnboundLocalError

这段代码在 cond 为真时完全正常，只在**特定权限组合 / 分支**下崩溃，而报错点
与根因隔了几百行。pyflakes 也抓不到（它只看"导入是否被使用"）。

本检查器用 AST 精确报出这类遮蔽，附带行号。

── 与 check_shadowing.py 的分工 ─────────────────────────────────────────────
``check_shadowing.py`` 是**超集**（额外扫"复合块内参数/常量再绑定"），候选多、
需人工逐条三查，适合当**排查工具**。本文件只扫"条件分支内 import 遮蔽模块级
导入"这一类，判据窄、误报少，因此适合当 **CI 门禁**（退出码 0/1 直接可用）。

── 真缺陷 vs 良性误报（2026-09 收紧）──────────────────────────────────────
原实现把"条件分支内的 import 遮蔽了模块级同名导入"一律算失败。但同一文件末尾
的判据早就写明：这是缺陷的**必要**条件、不是充分条件 —— 如果该名字在函数里的
**全部使用点**都在**同一个块内**，行为完全正确：

    def stream_master(...):
        ...
        from app.services.nodes.evidence_gate import is_refusal   # 块内导入
        refused = is_refusal(final_answer)                        # 紧挨着的唯一使用点
        yield {...}

``is_refusal`` 在 ``stream_master`` 里只有这一个使用点、且与导入同处一个块，
未走到该块的路径根本不会读它 —— 不会 UnboundLocalError。旧实现把它报成候选，
于是把它当门禁会让 CI **从第一天起就常红**（门禁一旦长期红就等于没有门禁）。

现在按上面那条判据自动分流：

  * **失败项（退出码 1）**—— 该名字在函数里**存在**位于"导入所在块之外"的
    使用点。这才是真正的 UnboundLocalError 风险。
  * **良性项（不影响退出码，仍打印）**—— 全部使用点都被导入所在的块"包住"。
    打印出来是为了**可审计**：万一将来判断口径要复核，名单还在。

残留局限（与 ``check_shadowing.py`` 同）：同一块内"先用后导入"（使用点在
import 之前）不会被本口径拦住 —— 它同样是 UnboundLocalError，但极罕见，且当前
代码库为 0 例。需要覆盖时应扩展为"使用点行号 > 导入行号"的检查。

用法：
    python check_shadowed_imports.py <项目根>
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

#: 会引入新作用域层级的复合语句
_CONDITIONAL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With,
                ast.AsyncWith, ast.Match)
#: 有独立作用域的嵌套函数（收集使用点时要跳过）
_NESTED_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


def module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def walk_functions(node: ast.AST):
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child


def conditional_imports_with_scope(func: ast.AST) -> list[tuple[str, int, ast.AST]]:
    """
    该函数作用域内**位于条件/循环/异常块里**的 import，附其"外层条件块"节点.

    为什么只看这些：import 写在函数体第一层（无条件）时，函数里任何一行用到它
    的位置都在它之后，行为与模块级导入完全一致 —— 只是风格问题，不是缺陷。
    真正会炸的是"**某个分支里**才导入、而**分支之外**还要用"。

    返回 ``(名字, 行号, 外层条件块节点)``。第三个值用于后续判定"使用点是否都在
    这个块内"（是这个块或其后代节点即算在内）。
    """

    out: list[tuple[str, int, ast.AST]] = []

    def visit(stmt: ast.AST, enclosing: ast.AST | None) -> None:
        if isinstance(stmt, _NESTED_FUNC):
            return                       # 嵌套函数有独立作用域，各自单独处理
        if isinstance(stmt, ast.Import):
            if enclosing is not None:
                for alias in stmt.names:
                    out.append((alias.asname or alias.name.split(".")[0],
                                stmt.lineno, enclosing))
            return
        if isinstance(stmt, ast.ImportFrom):
            if enclosing is not None:
                for alias in stmt.names:
                    out.append((alias.asname or alias.name, stmt.lineno, enclosing))
            return
        # enclosing 只记**最外层**的那个条件块（沿缩进向下传递、不再被覆盖）
        deeper = enclosing if enclosing is not None else (
            stmt if isinstance(stmt, _CONDITIONAL) else None
        )
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                visit(child, deeper)

    for stmt in func.body:
        visit(stmt, None)
    return out


def conditional_local_imports(func: ast.AST) -> list[tuple[str, int]]:
    """兼容旧签名的薄封装：只返回 ``(名字, 行号)``."""
    return [(name, lineno) for name, lineno, _ in conditional_imports_with_scope(func)]


def uses_in_own_scope(func: ast.AST, name: str) -> list[ast.Name]:
    """
    收集函数**自身作用域**内对 *name* 的读取点（``ast.Name`` 且 ctx=Load）.

    跳过嵌套函数的函数体 —— 它们有独立作用域，那里的同名名字与本次遮蔽无关。
    """
    uses: list[ast.Name] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _NESTED_FUNC):
                continue
            if (isinstance(child, ast.Name) and child.id == name
                    and isinstance(child.ctx, ast.Load)):
                uses.append(child)
            visit(child)

    for stmt in func.body:
        if isinstance(stmt, _NESTED_FUNC):
            continue
        visit(stmt)
    return uses


def check_file(path: Path) -> tuple[list[str], list[str]]:
    """
    返回 ``(失败项, 良性项)``.

    失败项 = 真风险（有使用点落在导入所在块之外）；
    良性项 = 全部使用点都被导入所在块包住（行为正确，仅留痕）。
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"{path}: 无法解析（{type(exc).__name__}: {exc}）"], []

    top = module_level_names(tree)
    if not top:
        return [], []

    findings: list[str] = []
    benign: list[str] = []
    for func in walk_functions(tree):
        for name, lineno, enclosing in conditional_imports_with_scope(func):
            if name not in top:
                continue
            uses = uses_in_own_scope(func, name)
            subtree = {id(node) for node in ast.walk(enclosing)}
            outside = [u for u in uses if id(u) not in subtree]
            if outside:
                lines = ", ".join(str(u.lineno) for u in outside[:5])
                findings.append(
                    f"{path}:{lineno}: 条件分支内的导入 '{name}' 遮蔽了模块级导入 "
                    f"（函数 {func.name} 中未走到该分支的路径会在使用点抛 "
                    f"UnboundLocalError；分支外的使用点行号 {lines}）"
                )
            else:
                benign.append(
                    f"{path}:{lineno}: '{name}'（函数 {func.name}）的导入与其全部"
                    f"{len(uses)} 个使用点同处一个块内 —— 行为正确，仅留痕"
                )
    return findings, benign


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    findings: list[str] = []
    benign: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in {".venv", "venv", "node_modules", "__pycache__", "legacy"}
               for part in path.parts):
            continue
        f, b = check_file(path)
        findings.extend(f)
        benign.extend(b)

    if benign:
        print(f"良性留痕 {len(benign)} 处（使用点全在该块内，不影响退出码）：")
        for line in benign:
            print("  " + line)
        print()

    if findings:
        print(f"发现 {len(findings)} 处**风险项**（退出码 1）：")
        for line in findings:
            print("  " + line)
        print(
            "\n判据：『条件分支内的 import 遮蔽模块级同名导入』且该名字**存在**位于"
            "该分支之外的使用点 —— 此时未走到分支的路径会在使用点抛 "
            "UnboundLocalError。全部使用点都在该块内的情形已归入『良性留痕』。"
        )
    else:
        print("未发现条件分支内 import 遮蔽模块级导入的风险项。")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
