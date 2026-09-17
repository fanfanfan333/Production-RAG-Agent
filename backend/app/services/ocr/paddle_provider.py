"""
PaddleOCR provider（通用 OCR 的主力引擎）.

⚠️ 所有 PaddleOCR 相关的坑（导入顺序 / HOME 可写 / PP-OCR 版本）都集中在
``app/utils/paddle_env.py``。本模块**只**通过它来构造实例，不要在这里直接
``import paddleocr`` 或自己调 ``PaddleOCR(...)``。
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from app.services.ocr.base import OCRLine, OCRProvider
from app.utils.paddle_env import build_paddle_ocr, predict_lock

# 构造失败后的冷却时间（秒）。见 PaddleOCRProvider._ensure 的说明。
_FAILURE_RETRY_COOLDOWN_S = 60.0


def _bbox(coords) -> tuple[float, float, float, float] | None:
    """把 PaddleOCR 的 4 点多边形压成外接矩形 (x0, y0, x1, y1)."""
    try:
        xs = [float(p[0]) for p in coords]
        ys = [float(p[1]) for p in coords]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


class PaddleOCRProvider(OCRProvider):
    """
    PaddleOCR 包装.

    **惰性构造**：``__init__`` 只记标志，第一次 ``analyze`` 才真正加载模型。
    这样"import 一个 provider"不再等于"付出几秒 + 几百 MB"。
    """

    def __init__(self):
        self._ocr = None
        self._init_attempted = False
        self._failed_at: float | None = None

    def _ensure(self):
        """
        惰性构造 + **失败带冷却的重试**.

        不要把"构造失败"永久缓存：历史上容器缺 `albumentations` 时
        `import paddleocr` 会抛 ModuleNotFoundError，一次失败后 `_ocr` 被钉死为
        None，即使事后补装了依赖、进程也没重启，PaddleOCR 也永远不会再被尝试，
        中文文档的图片 OCR 全程静默退化成 Tesseract(eng)。

        但也不能每次调用都重试 —— 依赖真缺失时，一份几百张图的文档会触发
        几百次注定失败的 import，把日志刷成"PaddleOCR import failed"瀑布。
        因此失败后进入 60s 冷却：装完依赖最迟一分钟自动恢复，期间不重复试。
        """
        if self._ocr is not None:
            return self._ocr
        if self._init_attempted:
            return None          # 同一次冷却窗口内不再重试
        if self._failed_at is not None:
            if time.monotonic() - self._failed_at < _FAILURE_RETRY_COOLDOWN_S:
                return None
            # 冷却结束，重新尝试一次
        self._ocr = build_paddle_ocr()
        self._init_attempted = self._ocr is not None
        self._failed_at = None if self._ocr is not None else time.monotonic()
        return self._ocr

    def name(self) -> str:
        return "PaddleOCR"

    def analyze(self, image: Image.Image) -> tuple[str, list[OCRLine]]:
        """
        一次推理产出文本 + 带坐标的行.

        表格结构识别需要"哪一行文字落在哪个 x 区间"，因此这里必须保留
        PaddleOCR 返回的四边形坐标，而不是像旧实现那样直接丢掉。
        """
        ocr = self._ensure()
        if ocr is None:
            raise RuntimeError("PaddleOCR is not installed or failed to initialize.")

        # Convert PIL Image to cv2 format (numpy array), RGB → BGR
        img_arr = np.array(image.convert("RGB"))
        img_arr = img_arr[:, :, ::-1].copy()

        # 串行化：Paddle 的 predictor 不是线程安全的，并发推理会 SIGSEGV
        # （整个进程消失，try/except 抓不住）。详见 app/utils/paddle_env.py 坑 4。
        with predict_lock():
            result = ocr.ocr(img_arr, cls=False)
        if not result or not result[0]:
            return "", []

        # result is a list of lines, each line is [coords, (text, confidence)]
        lines: list[OCRLine] = []
        for line in result[0]:
            if not line or len(line) < 2:
                continue
            coords, payload = line[0], line[1]
            text = str(payload[0]) if payload else ""
            if not text.strip():
                continue
            try:
                confidence = float(payload[1]) if len(payload) > 1 else 1.0
            except (TypeError, ValueError):
                confidence = 1.0
            lines.append(
                OCRLine(text=text, box=_bbox(coords), confidence=confidence)
            )
        return "\n".join(item.text for item in lines), lines

    def extract_text(self, image: Image.Image) -> str:
        return self.analyze(image)[0]

    def extract_lines(self, image: Image.Image) -> list[OCRLine]:
        return self.analyze(image)[1]
