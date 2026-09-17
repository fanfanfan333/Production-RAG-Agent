"""
扫全仓：找出"作用域遮蔽"型隐患（import 遮蔽 + 参数/常量被再绑定）.

为什么值得单独写一个检查器
──────────────────────────
Python 的作用域规则是"**整个函数**内只要有一次赋值/import，该名字就是局部
变量"。于是有两类隐蔽缺陷，都在**特定分支**下才炸，报错点与根因隔几百行：

**类型 A — 条件分支内的 import 遮蔽模块级导入**

    from app.services.tenancy import normalize_tenant_id     # 模块级

    def f(...):
        if cond:
            from app.services.tenancy import normalize_tenant_id   # 局部导入
            ...
        return normalize_tenant_id(x)      # ← cond 为假时 UnboundLocalError

**类型 B — 复合语句块内的赋值遮蔽函数参数（或模块级常量）**

    def pg_part(rows, batch):                 # rows 是"行数"
        ...
        async with get_db_session() as s:
            rows = (await s.execute(...)).all()   # ← 同名再绑定，rows 变成行列表
        ...
        rule(f"插入 {rows:,} 行")              # ← TypeError: list.__format__

类型 B 尤其阴——同样的名字在同一函数里表示两种东西，读代码时"看起来没错"。
本文件里 `rows` 是行数、`user_rows` 才该是查询结果；把参数换个名字就没事了，
但没人会一眼看出来。

为什么 pyflakes 抓不到：它只看"导入是否被使用"，不做作用域污染分析。

用法：
    python check_shadowing.py <项目根>

退出码：0 = 干净，1 = 有候选（**候选**，需人工确认最后一步，见文件末尾判据）。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

# 会引入新作用域层级的复合语句：名字在块内被绑定后，块外仍能看到新值
CONDITIONAL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With,
               ast.AsyncWith, ast.Match)


def module_level_names(tree: ast.Module) -> set[str]:
    """模块级导入 + 模块级赋值目标的名字."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for tgt in _target_names(node):
                names.add(tgt)
    return names


def _target_names(node: ast.AST) -> list[str]:
    """取出赋值语句里所有被绑定的裸名字（跳过属性/下标目标）."""
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, (ast.AugAssign, ast.NamedExpr)):
        targets = [node.target]

    out: list[str] = []
    stack = list(targets)
    while stack:
        t = stack.pop()
        if isinstance(t, ast.Name):
            out.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            stack.extend(t.elts)
    return out


def walk_functions(node: ast.AST):
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child


def _param_names(func: ast.AST) -> set[str]:
    args = func.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    return names


def _declared_globals(func: ast.AST) -> set[str]:
    """函数内 ``global`` / ``nonlocal`` 声明的名字（有意改写外层，不算遮蔽）."""
    out: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            out.update(node.names)
    return out


def _expr_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _is_none_guard(test: ast.AST | None, name: str) -> bool:
    """
    判断条件是否是针对该名字的"空值守卫"（默认值模式）.

        if settings is None:
            settings = get_settings()      # ← 有意替换，不是遮蔽缺陷

    这是项目里最普遍的写法，必须排除，否则候选列表全是噪音。
    """
    if test is None:
        return False
    names = _expr_names(test)
    if name not in names:
        return False
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return True
    if isinstance(test, ast.Compare) and all(
        isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)) for op in test.ops
    ):
        return True
    return False


def scan_function(func: ast.AST) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """
    返回 (条件块内 import, 复合块内再绑定名).

    为什么只看"块内"的绑定：
      · import 写在函数体第一层（无条件）时，函数里任何一行用到它的位置都在
        它之后，行为与模块级导入完全一致 —— 只是风格问题，不是缺陷。
      · 赋值写在函数体第一层时，它本来就是普通局部变量，遮蔽参数属于有意为之
        （虽然不推荐），且不依赖分支走向。

    真正会炸的是"**某个块内**才绑定、而**块外**还要读"：绑定是否发生取决于
    分支是否走到，读到的值因此不确定。

    B 类（再绑定）判据收紧到"**把外层同名值整个丢弃后重建**"——
      · ``stmt = stmt.where(...)``          右值引用了自己 → 链式收窄，放行
      · ``x += 1``                          增量赋值 → 放行
      · ``global _engine`` / ``nonlocal x``  已声明 → 有意改写外层，放行
      · ``if s is None: s = get_settings()`` 空值守卫 → 默认值模式，放行
      · ``if c: c = c or d``                右值引用了自己 → 收窄，放行
    剩下才报：右值与旧值无关，等于"这个块把参数换成了另一个东西"，
    正是 ``rows = (await ...).all()`` 那类——块之后读到的语义已经变了。
    """

    imports: list[tuple[str, int]] = []
    rebinds: list[tuple[str, int]] = []
    declared = _declared_globals(func)

    def visit(stmt: ast.AST, nested: bool, guards: tuple[ast.AST, ...]) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return                       # 嵌套函数有独立作用域，各自单独处理
        if isinstance(stmt, ast.Import):
            if nested:
                for alias in stmt.names:
                    imports.append((alias.asname or alias.name.split(".")[0],
                                    stmt.lineno))
            return
        if isinstance(stmt, ast.ImportFrom):
            if nested:
                for alias in stmt.names:
                    imports.append((alias.asname or alias.name, stmt.lineno))
            return

        if nested and isinstance(stmt, (ast.Assign, ast.AnnAssign,
                                        ast.AugAssign, ast.NamedExpr)):
            if isinstance(stmt, ast.AugAssign):
                value = None             # x += 1 天然引用旧值，不报
            elif isinstance(stmt, ast.Assign):
                value = stmt.value
            else:
                value = getattr(stmt, "value", None)
            for name in _target_names(stmt):
                if name in declared:
                    continue
                if value is not None and name in _expr_names(value):
                    continue             # 右值引用旧值 → 收窄，非丢弃
                if any(_is_none_guard(g, name) for g in guards):
                    continue             # 空值守卫 → 默认值模式
                rebinds.append((name, stmt.lineno))

        # 条件块的"守卫条件"沿缩进向下传递，供内层赋值判断
        new_guards = guards
        if isinstance(stmt, ast.If):
            new_guards = (*guards, stmt.test)
        deeper = nested or isinstance(stmt, CONDITIONAL)
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                visit(child, deeper, new_guards)

    for stmt in func.body:
        visit(stmt, False, ())
    return imports, rebinds


def check_file(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [f"{path}: 无法解析（{type(exc).__name__}: {exc}）"]

    top = module_level_names(tree)
    findings: list[str] = []

    for func in walk_functions(tree):
        imports, rebinds = scan_function(func)
        params = _param_names(func)

        for name, lineno in imports:
            if name in top:
                findings.append(
                    f"{path}:{lineno}: [A] 条件分支内的导入 '{name}' 遮蔽了模块级"
                    f"导入（函数 {func.name} 中未走到该分支的路径会在使用点抛 "
                    f"UnboundLocalError）"
                )
        seen: set[tuple[str, int]] = set()
        for name, lineno in rebinds:
            if name in params:
                kind = "函数参数"
            elif name in top:
                kind = "模块级名"
            else:
                continue
            key = (name, lineno)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                f"{path}:{lineno}: [B] 复合语句块内把 {kind} '{name}' 就地再绑定"
                f"（函数 {func.name} 中该块之后读到的值取决于块是否执行）"
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
            "\n判据与局限：本检查器用 AST 找出『块内绑定（import / 赋值）遮蔽了外层\n"
            "同名名（模块级导入、模块级常量、函数参数）』。这是缺陷的**必要**条件，\n"
            "不是充分条件 —— 如果该名字在函数里的**全部使用点**都在同一个块内，\n"
            "行为仍然正确。确认方式：看是否存在『同一名字、在更外层缩进处被使用』。"
        )
    else:
        print("未发现作用域遮蔽情形（块内 import / 参数再绑定）。")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
