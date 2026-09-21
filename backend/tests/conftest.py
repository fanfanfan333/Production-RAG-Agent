"""pytest 全局夹具 —— 兼「宿主机轻依赖垫片」。

背景
----
本项目 ``app/services/__init__.py`` 会连带拉起整条解析栈（fitz / cv2 / pandas /
python-pptx / openpyxl / asyncpg / PyJWT …）。在**开发机（Windows 宿主机）**上
没装 torch / paddle / docling 这些重依赖时，任何 ``import app.services.*`` 的单测
都会直接 collection error —— 于是本地"跑不了测试"，只能去容器里跑。

这层垫片的作用：**仅在真实第三方包 import 失败时**，往 ``sys.modules`` 注入一个
哑桩模块，让"只依赖纯逻辑（ACL / 租户 / 策略 / 节点装配）"的单测能在宿主机真跑。

安全性
------
* **依赖齐全时完全不生效**（容器 / CI 里 ``__import__`` 成功 → 不注入任何桩），
  因此不会掩盖真实的 import 错误，也不会造成"假绿"。
* 每次注入都会打一条醒目的 WARNING，列出被顶替的包名 —— 看得到、查得清。
* 只顶替**列在 _HOST_PAD_NAMES 里**的包；其它缺失照常报错。
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
import warnings

#: 允许被哑桩顶替的第三方顶层包。仅限「本模块链路上用不到其行为」的重依赖。
_HOST_PAD_NAMES: tuple[str, ...] = (
    "fitz",            # PyMuPDF
    "cv2",             # opencv-python-headless
    "pytesseract",
    "pptx",
    "pptx.enum",
    "pptx.enum.shapes",
    "openpyxl",
    "pandas",
    "asyncpg",
    "jwt",             # PyJWT
    "docling",
    "docling_core",
    "paddle",
    "paddleocr",
    "FlagEmbedding",
    "torch",
    "torchvision",
    "transformers",
    "timm",
    "albumentations",
    "sentence_transformers",
)


class _Permissive:
    """属性访问与调用都返回自身：作为缺失第三方包的哑桩。"""

    def __getattr__(self, _item):          # noqa: D105
        return _Permissive()

    def __call__(self, *_a, **_kw):
        return _Permissive()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


def _pad(name: str) -> None:
    mod = types.ModuleType(name)
    mod.__path__ = []                       # 声明为包，允许 a.b 形式的后续导入
    # 补一个合法 spec：否则 `docling.__spec__ is None` 之类的断言会直接炸
    # （已被 test_docling_rich_cell 触发过）。
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
    mod.__getattr__ = lambda _n: _Permissive()     # type: ignore[attr-defined]
    # ★ 打标记：这是"宿主机哑桩"而不是真包。用例可以据此做**条件跳过**
    #   （tests/_module_skip.host_shimmed）——只在依赖真缺失时跳过，
    #   容器/CI 里依赖齐全 → 没有该标记 → 用例照常**真跑**，不会掩盖缺陷。
    mod.__host_shim__ = True                       # type: ignore[attr-defined]
    sys.modules[name] = mod


_padded: list[str] = []

for _name in _HOST_PAD_NAMES:
    if _name in sys.modules:
        continue
    try:
        __import__(_name)
    except ImportError:
        _pad(_name)
        _padded.append(_name)

if _padded:
    warnings.warn(
        "host-shim: 宿主机缺少以下第三方包，已注入哑桩（容器/CI 中不会发生）："
        + ", ".join(_padded),
        stacklevel=1,
    )
    print(
        "[host-shim] 已顶替 %d 个缺失的重依赖：%s" % (len(_padded), ", ".join(_padded)),
        file=sys.stderr,
    )
