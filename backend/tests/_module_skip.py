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
