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

用法：
    python check_shadowed_imports.py <项目根>
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


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


def conditional_local_imports(func: ast.AST) -> list[tuple[str, int]]:
    """
    该函数作用域内**位于条件/循环/异常块里**的 import.

    为什么只看这些：import 写在函数体第一层（无条件）时，函数里任何一行用到它
    的位置都在它之后，行为与模块级导入完全一致 —— 只是风格问题，不是缺陷。
    真正会炸的是"**某个分支里**才导入、而**分支之外**还要用"：Python 把该名字
    判定为整个函数的局部变量，分支没走到时使用点就是 UnboundLocalError。

    因此：递归遍历函数的直接语句，一旦发现"某个 import 的祖先里有 if/for/
    while/try/with"，才认为它可疑。
    """

    out: list[tuple[str, int]] = []
    CONDITIONAL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With,
                   ast.AsyncWith, ast.Match)

    def visit(stmt: ast.AST, nested: bool) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return                       # 嵌套函数有独立作用域，各自单独处理
        if isinstance(stmt, ast.Import):
            if nested:
                for alias in stmt.names:
                    out.append((alias.asname or alias.name.split(".")[0],
                                stmt.lineno))
            return
        if isinstance(stmt, ast.ImportFrom):
            if nested:
                for alias in stmt.names:
                    out.append((alias.asname or alias.name, stmt.lineno))
            return
        deeper = nested or isinstance(stmt, CONDITIONAL)
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                visit(child, deeper)

    for stmt in func.body:
        visit(stmt, False)
    return out


def check_file(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"{path}: 无法解析（{type(exc).__name__}: {exc}）"]

    top = module_level_names(tree)
    if not top:
        return []

    findings: list[str] = []
    for func in walk_functions(tree):
        for name, lineno in conditional_local_imports(func):
            if name in top:
                findings.append(
                    f"{path}:{lineno}: 条件分支内的导入 '{name}' 遮蔽了模块级导入 "
                    f"（函数 {func.name} 中未走到该分支的路径会在使用点抛 "
                    f"UnboundLocalError）"
                )
    return findings


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    findings: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in {".venv", "venv", "node_modules", "__pycache__", "legacy"}
               for part in path.parts):
            continue
        findings.extend(check_file(path))

    if findings:
        print(f"发现 {len(findings)} 处**候选**（需人工确认最后一步）：")
        for line in findings:
            print("  " + line)
        print(
            "\n判据与局限：本检查器用 AST 找出『条件分支内的 import 遮蔽了模块级同名\n"
            "导入』。这是 UnboundLocalError 的必要条件，但不是充分条件 —— 如果该\n"
            "名字在函数里的**全部使用点**都位于同一个分支内（import 紧挨着它的\n"
            "唯一使用处），行为仍然正确。确认方式：看函数内是否存在『同一名字、\n"
            "在更外层缩进处被使用』。"
        )
    else:
        print("未发现条件分支内 import 遮蔽模块级导入的情形。")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
