"""pytest 收集期的「整份文件跳过」助手.

这些单测都在模块顶层做 `try: <导入重依赖> except ImportError: 跳过` —— 宿主机
没有 torch / docling / paddleocr 时应当优雅跳过，而不是报错。

**为什么不能再用 `print(...); sys.exit(0)`**：pytest 是在**收集阶段**导入测试
模块的。此时抛出 SystemExit 不会被当成"跳过"，而是让整个 pytest 会话
INTERNALERROR 崩掉 —— 同目录其它用例一条都跑不了。一个"本机缺依赖"的问题被
放大成"整套测试不可用"，这是测试套件最不该有的性质（tests/test_upload.py 的
文档里已经踩过同一个坑：早先它在模块顶层 open() 夹具，夹具不在场就在收集时炸）。

`pytest.skip(..., allow_module_level=True)` 抛的是 `Skipped`，收集器能正确
识别为"跳过这一个文件"。本助手把"有 pytest 就 skip、没有 pytest 就直接退出"
的差异收在一处，避免每个文件各抄一遍 try/except。
"""
from __future__ import annotations

import sys


def skip_module(reason: str) -> None:
    """整份测试文件跳过：pytest 下抛 Skipped，直接运行时安静退出。"""
    try:
        import pytest
    except ImportError:  # 直接 `python tests/xxx.py` 且环境里没有 pytest
        print(f"SKIP: {reason}")
        raise SystemExit(0) from None
    pytest.skip(reason, allow_module_level=True)


def host_shimmed(*names: str) -> bool:
    """给定的第三方包是否**被宿主机哑桩顶替**（即本机真的没装这个包）。

    与 ``skip_module`` 的分工：
    * ``skip_module`` —— **收集期**整份文件跳过（模块顶层 import 就失败时用）；
    * ``host_shimmed`` —— **运行期**判断，给"只在某个用例里才用到重依赖"的场景用。

    判据是 ``tests/conftest.py`` 的垫片在哑桩模块上打的 ``__host_shim__`` 标记。
    依赖齐全时垫片完全不生效、标记不存在 → 恒返回 False，
    因此容器 / CI 里这些用例会**照常真跑**，条件跳过不会变成"永久豁免"。
    """
    import importlib

    for name in names:
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = importlib.import_module(name)
            except Exception:      # noqa: BLE001 —— 导不进来就等价于"缺"
                return True
        if getattr(mod, "__host_shim__", False):
            return True
    return False


def skip_if_host_shimmed(*names: str) -> None:
    """某个重依赖在本机被顶替时，跳过**当前用例**（不是整份文件）。

    为什么单独做这个助手：一个文件里往往只有 1–2 个用例真的需要 cv2 / python-pptx，
    其余用例是纯逻辑。用 ``pytestmark`` 整文件跳过会连带把能跑的用例一起关掉，
    白白丢掉覆盖率。放在用例体内则只跳该跳的。
    """
    missing = [n for n in names if host_shimmed(n)]
    if not missing:
        return
    import pytest

    pytest.skip(
        f"宿主机未安装 {', '.join(missing)}（被 conftest 哑桩顶替）"
        f" → 该用例需在 backend 容器内跑（容器里依赖齐全，会自动真跑）"
    )
