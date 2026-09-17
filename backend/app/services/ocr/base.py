import abc
from dataclasses import dataclass

from PIL import Image


@dataclass
class OCRLine:
    """
    一行 OCR 结果（带版面坐标）.

    坐标是**原图像素**下的 (x0, y0, x1, y1) 外接矩形。表格结构识别依赖它把
    文字行还原成行列网格：没有坐标就只能拿到一串按行拼接的文本，"哪一格写了
    什么"就丢了。

    未提供坐标的引擎会把 box 置为 None —— 此时上层会退化为纯文本行处理。
    """

    text: str
    box: tuple[float, float, float, float] | None = None
    confidence: float = 1.0

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "box": list(self.box) if self.box else None,
            "confidence": round(float(self.confidence), 4),
        }


class OCRProvider(abc.ABC):
    """
    Abstract base class for OCR Engines.
    """
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name of the OCR engine."""
        pass

    @abc.abstractmethod
    def extract_text(self, image: Image.Image) -> str:
        """
        Extract text from a PIL Image.
        Returns the extracted text as a string.
        """
        pass

    def extract_lines(self, image: Image.Image) -> list[OCRLine]:
        """
        返回带坐标的文字行（默认实现：退化为无坐标的纯文本行）.

        有版面能力的引擎（PaddleOCR / Tesseract）应覆写本方法；未覆写时
        上层仍能工作，只是表格结构识别拿不到对齐信息。
        """
        text = self.extract_text(image)
        return [
            OCRLine(text=line.strip())
            for line in (text or "").splitlines()
            if line.strip()
        ]

    def analyze(self, image: Image.Image) -> tuple[str, list[OCRLine]]:
        """
        一次调用同时产出文本与带坐标的行.

        默认实现 = extract_text + extract_lines（可能跑两遍 OCR）。有版面
        能力的引擎应覆写本方法，让"分类 + 表格还原 + 文本"共享同一次推理。
        """
        return self.extract_text(image), self.extract_lines(image)


class OCRService:
    """
    Service facade for OCR providers.
    Uses PaddleOCR by default, falls back to Tesseract.
    """
    def __init__(self, primary_provider: OCRProvider, fallback_provider: OCRProvider | None = None):
        self.primary = primary_provider
        self.fallback = fallback_provider

    def extract_text(self, image: Image.Image) -> tuple[str, str]:
        """
        Extract text and return a tuple: (extracted_text, engine_used).
        """
        text, engine, _ = self.extract_text_with_lines(image)
        return text, engine

    def extract_text_with_lines(
        self, image: Image.Image
    ) -> tuple[str, str, list[OCRLine]]:
        """
        一次 OCR 同时拿到：纯文本、引擎名、带坐标的行.

        表格结构识别与图片分类都需要行坐标，而 OCR 是整条链路里最贵的一步 ——
        因此统一在这里做一次，避免为"分类"和"表格还原"各跑一遍 OCR。
        """
        try:
            text, lines = self.primary.analyze(image)
            return text, self.primary.name(), list(lines or [])
        except Exception:      # noqa: BLE001
            if not self.fallback:
                raise
            text, lines = self.fallback.analyze(image)
            return text, self.fallback.name(), list(lines or [])
