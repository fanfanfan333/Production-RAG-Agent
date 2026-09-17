"""
Second OCR 引擎（Tesseract）.

对应流程图 "Fallback → Second OCR"：当专用引擎置信度低、而 Vision 又不可用
（没配多模态模型）时，用**另一个 OCR 引擎**再读一遍。换引擎而不是原样重试，
是因为不同 OCR 的失败模式不一样（PaddleOCR 擅长中文/弯折文本，Tesseract 对
规整英文/数字表格有时更稳），重试同一个引擎收益极低。
"""

from __future__ import annotations

from PIL import Image

from app.services.image_understanding.engines.base import (
    EngineKind,
    EngineOutput,
    ImageEngine,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class TesseractEngine(ImageEngine):
    """Tesseract 通用 OCR（作为第二 OCR 引擎）."""

    name = "tesseract"
    kind = EngineKind.OCR

    def is_available(self) -> bool:
        try:
            import pytesseract  # noqa: F401  # noqa: PLC0415

            return True
        except Exception:      # noqa: BLE001
            return False

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "pytesseract / tesseract 二进制未安装"

    def process(self, image: Image.Image, **kwargs) -> EngineOutput:
        try:
            from app.services.ocr.tesseract_provider import TesseractProvider

            text, lines = TesseractProvider().analyze(image)
        except Exception as exc:      # noqa: BLE001
            logger.warning("Tesseract inference failed: %s", exc)
            return EngineOutput(engine=self.name, ok=False, error=str(exc))

        text = (text or "").strip()
        if not text:
            return EngineOutput(engine=self.name, ok=False, error="无文本", lines=list(lines or []))

        confs = [float(getattr(l, "confidence", 0.0) or 0.0) for l in (lines or [])]
        confidence = round(sum(confs) / len(confs), 4) if confs else 0.5

        return EngineOutput(
            text=text,
            confidence=confidence,
            engine=self.name,
            ok=True,
            lines=list(lines or []),
        )


__all__ = ["TesseractEngine"]
