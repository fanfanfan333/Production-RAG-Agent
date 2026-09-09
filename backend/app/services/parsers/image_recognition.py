"""
Embedded-image recognition helpers (问题3).

Used by the document parsers to recognise the images that live INSIDE a
document (PDF pages, DOCX packages, PPTX slides):

    1. OCR  — always on (ENABLE_IMAGE_OCR), via the existing OCRService
              (PaddleOCR with Tesseract fallback).
    2. Vision caption — optional, only when OLLAMA_VISION_MODEL is configured;
              asks a multimodal Ollama model to describe the image, so charts
              and photos without text still become searchable content.

Every produced text block is merged into the document's extracted text with a
``[图片N内容]`` marker so it flows through chunking/embedding like any other
content, with the OCR engine recorded in the document metadata.
"""

import base64
import hashlib
import io

import httpx
from PIL import Image

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Content-marker prefix used in the merged document text.
IMAGE_MARKER_PREFIX = "图片"

# Prompt for the optional Ollama vision captioning.
_VISION_CAPTION_PROMPT = (
    "请用中文简要描述这张图片的内容（如果是图表，请说明图表类型和主要数据；"
    "如果是照片，请描述其中的对象和场景）。只输出描述本身，不要任何前缀。"
)


def _is_usable_image(img: Image.Image) -> bool:
    """Filter out icons / separators / 1px decorations."""
    width, height = img.size
    min_dim = get_settings().MIN_IMAGE_DIMENSION
    if width < min_dim or height < min_dim:
        return False
    # Extreme aspect ratios are usually borders or rules, not content.
    if width / max(height, 1) > 25 or height / max(width, 1) > 25:
        return False
    return True


def _to_rgb(img: Image.Image) -> Image.Image:
    """Normalise colour modes — OCR and vision models want RGB (or L)."""
    if img.mode in ("RGB", "L"):
        return img
    if img.mode == "RGBA" or img.mode == "P" or img.mode == "LA":
        return img.convert("RGB")
    if img.mode == "CMYK":
        return img.convert("RGB")
    return img.convert("RGB")


def caption_with_vision(img: Image.Image, filename: str = "") -> str | None:
    """
    Describe an image with the configured Ollama vision model.

    Returns None when vision captioning is disabled, not configured, or the
    model call fails — OCR-only recognition then remains as the fallback.
    """
    settings = get_settings()
    if not settings.OLLAMA_VISION_MODEL:
        return None

    try:
        buffer = io.BytesIO()
        _to_rgb(img).save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        response = httpx.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={
                "model": settings.OLLAMA_VISION_MODEL,
                "prompt": _VISION_CAPTION_PROMPT,
                "images": [b64],
                "stream": False,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        caption = str(response.json().get("response", "")).strip()
        return caption or None
    except Exception as exc:
        logger.warning(
            "Vision caption failed for image in '%s' (model=%s): %s",
            filename or "?",
            settings.OLLAMA_VISION_MODEL,
            exc,
        )
        return None


class EmbeddedImageRecognizer:
    """
    Recognises a stream of embedded images for one document.

    Usage:
        recognizer = EmbeddedImageRecognizer(filename)
        for pil_image in images:
            recognizer.recognize(pil_image)
        result_text = recognizer.merged_text()   # "[图片1内容] ..." block
    """

    def __init__(self, filename: str, max_images: int | None = None):
        self.filename = filename
        settings = get_settings()
        self.enabled = settings.ENABLE_IMAGE_OCR
        self.max_images = max_images or settings.MAX_IMAGES_PER_DOCUMENT
        self.texts: list[str] = []
        self.engines: set[str] = set()
        self._seen_hashes: set[str] = set()
        self._skipped = 0

        if self.enabled:
            # Imported lazily so modules using only text parsers never pay for it
            from app.services.ocr import get_ocr_service
            self._ocr = get_ocr_service()
        else:
            self._ocr = None

    @property
    def has_content(self) -> bool:
        return bool(self.texts)

    def recognize(self, img: Image.Image) -> bool:
        """
        OCR (and optionally caption) one embedded image.

        Returns True when recognisable text was produced for it.
        """
        if not self.enabled or len(self.texts) >= self.max_images:
            if self.enabled:
                self._skipped += 1
            return False

        try:
            img = _to_rgb(img)
            if not _is_usable_image(img):
                return False

            # Skip duplicated images (logos repeated on every page/slide).
            digest = hashlib.sha1(img.tobytes()).hexdigest()
            if digest in self._seen_hashes:
                return False
            self._seen_hashes.add(digest)

            blocks: list[str] = []

            # 1. OCR — text inside the image
            try:
                ocr_text, engine = self._ocr.extract_text(img)
                ocr_text = (ocr_text or "").strip()
                if ocr_text:
                    settings = get_settings()
                    blocks.append(ocr_text[: settings.MAX_IMAGE_OCR_CHARS])
                    self.engines.add(engine)
            except Exception as exc:
                logger.warning(
                    "Embedded-image OCR failed in '%s': %s", self.filename, exc
                )

            # 2. Optional vision caption — meaning when the image has no text
            caption = caption_with_vision(img, self.filename)
            if caption:
                blocks.append(f"图片描述: {caption}")

            if not blocks:
                return False

            self.texts.append("\n".join(blocks))
            return True
        except Exception as exc:
            logger.warning(
                "Embedded-image recognition failed in '%s': %s", self.filename, exc
            )
            return False

    def merged_text(self, *, section_header: str = "图片识别内容") -> str:
        """
        Render all recognised image texts as one block that parsers append
        to the document text. Empty string when nothing was recognised.
        """
        if not self.texts:
            return ""
        lines = [
            f"[{IMAGE_MARKER_PREFIX}{i}内容] {text}"
            for i, text in enumerate(self.texts, start=1)
        ]
        header = f"[{section_header}]" if section_header else ""
        body = "\n".join(lines)
        return f"{header}\n{body}" if header else body

    @property
    def skipped_count(self) -> int:
        return self._skipped
