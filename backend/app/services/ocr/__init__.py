"""
OCR 服务工厂.

**惰性初始化**是刻意的：PaddleOCR 构造要加载检测/识别/方向三套模型（数秒 + 数百
MB 内存），而且在本环境里还踩过导入顺序导致 segfault 的坑。历史上这里是模块级
``PaddleOCRProvider()`` —— 意味着"只要 import 这个包"就付出全部代价，任何
import 顺序问题也会在启动期直接带走进程。

现在改成：import 本模块零成本，第一次 ``get_ocr_service()`` 才构造。
"""

from __future__ import annotations

import logging
import threading

from app.services.ocr.base import OCRService, OCRProvider
from app.services.ocr.paddle_provider import PaddleOCRProvider
from app.services.ocr.tesseract_provider import TesseractProvider

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_ocr_service: OCRService | None = None


def get_ocr_service() -> OCRService:
    """返回进程级单例 OCR 服务（首次调用才构造 providers）."""
    global _ocr_service
    if _ocr_service is None:
        with _lock:
            if _ocr_service is None:
                primary = PaddleOCRProvider()
                fallback = TesseractProvider()
                _ocr_service = OCRService(primary_provider=primary, fallback_provider=fallback)
                logger.info(
                    "OCR service ready: primary=%s fallback=%s",
                    primary.name(), fallback.name(),
                )
    return _ocr_service


__all__ = ["OCRService", "OCRProvider", "get_ocr_service"]
