"""守卫自身的守卫：宿主机垫片 + 运行期条件跳过，必须"该跳才跳".

为什么专门测这个
----------------
本轮把 5 个「宿主机缺重依赖」的失败用例（cv2 ×2 / python-pptx ×2 / asyncpg ×1）
从**假红**改成**运行期条件跳过**（``_module_skip.skip_if_host_shimmed``）。

条件跳过有个天生的风险：**守卫一旦恒真，这 5 个用例就被永久豁免** —— 那就从
"假红"变成了"假绿"，比原来更糟（只是换了个姿势掩盖同一件事）。所以把两条前提
钉死在测试里：

1. 守卫对**真实安装**的包必须返回 False —— 否则容器里也不会跑；
2. 哑桩标记 ``__host_shim__`` 只能打在**垫片造的假模块**上：
   打在真包上 → 无故跳过（假绿）；忘了打 → 条件跳过失效（假红照旧）。

容器内（依赖齐全）本文件应**全通过**：此时 pad 集合为空，第 ③ 条退化为空真，
而第 ①② 条恰恰是容器里最需要守住的。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 与同目录其它用例一致：保证 ``_module_skip`` 可导入（本文件不依赖 pytest 的补路径）
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _module_skip import host_shimmed, skip_if_host_shimmed  # noqa: E402

#: 测试环境里必定真实安装的包（requirements.txt 直接依赖）
_REAL_PACKAGES = ("PIL", "sqlalchemy", "pytest", "numpy")


def _carries_shim_marker(obj: object) -> bool:
    """对象是否**真的**被垫片打了标记 —— 只读它自身的 ``__dict__``。

    为什么不能用 ``getattr(obj, "__host_shim__", False)``
    ────────────────────────────────────────────────────
    ``torch.ops`` 是 ``torch._ops._Ops`` 的**实例**（不是模块），它实现了自定义
    ``__getattr__``：访问任意未定义的名字都会**动态构造**并返回一个 operator
    namespace 模块。于是

        getattr(torch.ops, "__host_shim__", False)
        → <module 'torch.ops.__host_shim__' from 'torch.ops'>   （truthy！）

    守卫据此把它判成"带哑桩标记"，再去要 ``__spec__`` 就必然是 ``None`` → 用例
    在**容器内**变红（宿主机 torch 被垫片顶替、根本没有 ``torch.ops``，所以一直
    没暴露）。实测：``vars(torch.ops)`` 里**没有**这个键，即垫片从未标记过它。

    垫片（``conftest._pad``）是**显式写入模块 ``__dict__``** 的
    （``mod.__host_shim__ = True``），所以按 ``__dict__`` 判断才是准确判据 ——
    而且比 ``getattr`` **更严格**（不再被自定义属性协议骗过）。
    """
    d = getattr(obj, "__dict__", None)
    return isinstance(d, dict) and bool(d.get("__host_shim__", False))


def test_guard_is_not_vacuous() -> None:
    """① 真装了的包不能被判成"缺失" —— 否则容器里也会被无辜跳过。"""
    for name in _REAL_PACKAGES:
        assert host_shimmed(name) is False, f"{name} 是真实安装的包，不该被判为缺失"


def test_skip_helper_returns_for_real_packages() -> None:
    """② 对真包调用跳过助手必须直接返回（不抛异常、不误跳）。"""
    skip_if_host_shimmed("PIL", "sqlalchemy")


def test_real_modules_never_carry_host_shim_marker() -> None:
    """③ 真包不得带哑桩标记（带了 → 无故跳过 → 假绿）。"""
    for name in _REAL_PACKAGES:
        mod = sys.modules.get(name)
        assert not _carries_shim_marker(mod), f"{name} 是真包却带哑桩标记"


def test_marked_modules_are_actually_fakes() -> None:
    """④ 凡带标记的模块，必须是"没有真加载器"的假模块。

    宿主机上应存在若干带标记的包（fitz / cv2 / torch …）；容器内一个都没有 ——
    两种都合法，所以这里**不**断言"必须非空"（那会让本文件在容器里变红），
    只保证"有标记的确实是假的"。
    """
    marked = [
        n for n, m in list(sys.modules.items())
        if _carries_shim_marker(m)
    ]
    for name in marked:
        spec = getattr(sys.modules[name], "__spec__", None)
        assert spec is not None, f"{name} 带哑桩标记却没有 __spec__"
        assert spec.loader is None, f"{name} 带哑桩标记，但它有真加载器（是真包）"


def test_unknown_package_is_reported_as_missing() -> None:
    """⑤ 根本不存在的包名必须判为"缺失"（守卫的主路径确实在工作）。"""
    assert host_shimmed("definitely_not_a_real_package_zzz") is True
