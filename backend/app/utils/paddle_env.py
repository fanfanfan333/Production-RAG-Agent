"""
Paddle 系引擎的公共前置：导入顺序守卫 + 模型缓存目录.

本文件里的两块"创可贴"都是**真实踩过的坑**，删掉就会退回一个个难以定位的
崩溃。改动前请先读完注释。

────────────────────────────────────────────────────────────────────────────
坑 1：导入顺序 → SIGSEGV（zlib 符号冲突）
────────────────────────────────────────────────────────────────────────────
在只 ``import paddleocr`` 的裸进程里，PaddleOCR 会在导入期加载 OpenCV/zlib，
与系统 zlib 撞符号，崩在 ``inflateReset2``：

    FatalError: `Segmentation fault` ... 0  inflateReset2

只要**先** ``import numpy`` / ``import cv2``，让 OpenCV 先把 zlib 符号固定下来，
后续 paddleocr 导入就完全正常。这也解释了为什么"在完整应用里能跑、单跑脚本就崩"
—— 应用里其它模块早就把 cv2 导进来了。

因此本模块（app/utils/paddle_env.py）顶层强制 ``import numpy, cv2``，并把它作为所有 Paddle 引擎的**唯一**
入口：任何要 import paddleocr 的地方都必须先 import 本模块。

────────────────────────────────────────────────────────────────────────────
坑 2：HOME 不可写 → 模型下载失败
────────────────────────────────────────────────────────────────────────────
PaddleOCR 把模型下到 ``~/.paddleocr``。容器里 ``appuser`` 是 system 用户，
``HOME=/nonexistent``，于是：

    PermissionError: [Errno 13] Permission denied: '/nonexistent'

:func:`ensure_model_home` 会在 HOME 不可写时把 ``HOME`` 指向一个可写的缓存目录
（默认 ``/app/.cache``，可用 ``PADDLE_CACHE_DIR`` 覆盖）。

────────────────────────────────────────────────────────────────────────────
坑 3：PP-OCRv4 中文模型 → SIGILL（非法指令）
────────────────────────────────────────────────────────────────────────────
默认 ``ocr_version='PP-OCRv4'`` 的中文识别模型会在 IR 优化阶段触发
``SelfAttentionFusePass``，在**不支持对应指令集**的 CPU 上直接

    FatalError: `Illegal instruction` ... SelfAttentionFusePass::ApplyImpl

改用 ``ocr_version='PP-OCRv3'`` 即可绕开（本环境已验证 v3 中英文均正常）。
默认值由 ``settings.PADDLE_OCR_VERSION`` 控制，默认就是 ``PP-OCRv3``。

────────────────────────────────────────────────────────────────────────────
坑 4：并发推理 → SIGSEGV（**整个进程消失**，try/except 抓不住）
────────────────────────────────────────────────────────────────────────────
Paddle 的 predictor 不是线程安全的：内部共享 workspace 与 oneDNN primitive
cache，两个线程同时推理会写坏原生内存，先报

    could not create a primitive descriptor for a reorder primitive
    InvalidArgumentError: Broadcast dimension mismatch ... db_fpn.py:249

再升级成 ``FatalError: Segmentation fault`` —— 进程直接死。

这坑是"入库改异步"之后才浮现的：多份文档的 OCR 会经 ``asyncio.to_thread``
**真的并行**跑在不同线程里，且共用同一个 predictor 实例。单文档串行跑永远
测不出来。因此本模块导出 :func:`predict_lock`：**任何**调用
``PaddleOCR.ocr()`` / ``PPStructure()`` 的地方都必须持有它。

复现/验证：``backend/scripts/repro_paddle_threads.py``
（4 线程共享实例 ⇒ 秒崩、退出码 139；加锁 ⇒ 288 次调用零崩溃）。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

# ── 坑 1：必须先于 paddleocr 导入（顺序不可调整）─────────────────────────────
import numpy  # noqa: F401  (import order matters — see module docstring)
import cv2  # noqa: F401

from app.utils.logging import get_logger

logger = get_logger(__name__)

_home_fixed = False

# ── 坑 4：所有 Paddle 原生推理的全局串行锁（见模块 docstring）──────────────
# 一把锁覆盖全部引擎（通用 OCR / 版面 / 公式）：它们跑在同一个 Paddle
# runtime 上、共享底层内存池，分别上锁等于没上。
_PREDICT_LOCK = threading.Lock()


def predict_lock() -> threading.Lock:
    """返回推理用的全局锁。调用点：``with predict_lock(): predictor(...)``"""
    return _PREDICT_LOCK


def ensure_model_home() -> str:
    """
    保证 ``~/.paddleocr`` 所在目录可写；不可写则把 HOME 指向缓存目录.

    返回最终生效的 HOME。幂等，可重复调用。
    """
    global _home_fixed

    target = os.environ.get("PADDLE_CACHE_DIR") or "/app/.cache"
    current = os.environ.get("HOME", "")

    if _home_fixed:
        return current or target

    def _writable(path: str) -> bool:
        try:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".paddle_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return True
        except Exception:      # noqa: BLE001
            return False

    if current and _writable(current):
        _home_fixed = True
        return current

    try:
        Path(target).mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = target
        logger.info("PaddleOCR model cache: HOME redirected to %s (was %r)", target, current)
        _home_fixed = True
        return target
    except Exception as exc:      # noqa: BLE001
        logger.warning("Cannot prepare a writable PaddleOCR cache dir (%s); using %r", exc, current)
        return current


def paddle_ocr_version() -> str:
    """PP-OCR 模型版本。默认 PP-OCRv3 —— 见坑 3。"""
    try:
        from app.config import get_settings

        return getattr(get_settings(), "PADDLE_OCR_VERSION", "PP-OCRv3") or "PP-OCRv3"
    except Exception:      # noqa: BLE001
        return "PP-OCRv3"


def paddle_lang() -> str:
    """识别语言。``ch`` 模型中英混排都能认，是中文知识库的正确默认。"""
    try:
        from app.config import get_settings

        return getattr(get_settings(), "PADDLE_OCR_LANG", "ch") or "ch"
    except Exception:      # noqa: BLE001
        return "ch"


def import_paddleocr():
    """
    安全导入 paddleocr（先落实 HOME 与导入顺序守卫）.

    返回模块对象；不可用时返回 None。**不要**在别处直接 import paddleocr。
    """
    ensure_model_home()
    try:
        import paddleocr  # noqa: PLC0415

        return paddleocr
    except Exception as exc:      # noqa: BLE001
        logger.warning("PaddleOCR import failed: %s", exc)
        return None


def build_paddle_ocr():
    """
    构造一个 ``PaddleOCR`` 实例（语言/版本按配置，绕开 v4 中文 SIGILL）.

    失败返回 None。构造是**重**操作（要加载检测/识别/方向三套模型），
    因此调用方必须自己做单例缓存。
    """
    module = import_paddleocr()
    if module is None:
        return None
    try:
        return module.PaddleOCR(
            use_angle_cls=False,
            lang=paddle_lang(),
            ocr_version=paddle_ocr_version(),
            show_log=False,
        )
    except Exception as exc:      # noqa: BLE001
        logger.warning("PaddleOCR init failed (lang=%s): %s", paddle_lang(), exc)
        return None


def paddle_available() -> bool:
    """轻量可用性探测：只测 import，不构造实例、不下载模型。"""
    return import_paddleocr() is not None


def pp_structure_v3_class():
    """
    取 PP-StructureV3 的类；拿不到就返回 None.

    PP-StructureV3（版式 + 表格 + 公式 + 阅读顺序的完整链路）只在
    paddleocr ≥ 3.0 暴露。2.9.x 只有 ``PPStructure``（v2），因此这里做
    **能力探测**而不是硬编码 —— 升到 3.x 后本函数自动返回 V3。
    """
    module = import_paddleocr()
    if module is None:
        return None
    for attr in ("PPStructureV3", "PPStructureV3Pipeline"):
        cls = getattr(module, attr, None)
        if cls is not None:
            return cls
    return None


def pp_structure_v2_class():
    """取 ``PPStructure``（v2）的类；拿不到返回 None."""
    module = import_paddleocr()
    if module is None:
        return None
    return getattr(module, "PPStructure", None)


def formula_class():
    """
    取公式识别类；拿不到返回 None.

    公式推理 API（PP-FormulaNet / LaTeX-OCR）同样只在 paddleocr ≥ 3.0 暴露
    （``FormulaRecognition`` 等）。2.9.x 只带训练/数据工具，没有推理入口 ——
    此时 Formula 引擎会如实报告"不可用"并让管线走 Vision 兜底，而不是假装成功。
    """
    module = import_paddleocr()
    if module is None:
        return None
    for attr in ("FormulaRecognition", "FormulaRecognitionPipeline", "TextRecognition"):
        cls = getattr(module, attr, None)
        if cls is not None and attr.startswith("Formula"):
            return cls
    return None


__all__ = [
    "ensure_model_home",
    "paddle_ocr_version",
    "paddle_lang",
    "predict_lock",
    "import_paddleocr",
    "build_paddle_ocr",
    "paddle_available",
    "pp_structure_v3_class",
    "pp_structure_v2_class",
    "formula_class",
]
