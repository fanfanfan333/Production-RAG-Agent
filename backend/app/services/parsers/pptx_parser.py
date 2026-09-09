import io
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from PIL import Image
from app.services.parsers.base import DocumentParser, ExtractionResult, ExtractedPage
from app.services.parsers.image_recognition import EmbeddedImageRecognizer
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _iter_picture_blobs(shape):
    """Yield image blobs from a shape, recursing into grouped shapes."""
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        try:
            yield shape.image.blob
        except Exception:
            return
    elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for sub in getattr(shape, "shapes", []):
            yield from _iter_picture_blobs(sub)


class PptxParser(DocumentParser):
    """
    Parses PPTX files.

    Embedded-image recognition (问题3): pictures on each slide are OCR'd
    (optionally captioned by an Ollama vision model) and merged into that
    slide's text, so recognition keeps the correct slide attribution.
    """

    def parse(self, content: bytes, filename: str) -> ExtractionResult:
        try:
            prs = Presentation(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Cannot open PPTX '{filename}': {exc}") from exc

        pages: list[ExtractedPage] = []
        cursor = 0
        total_pages = len(prs.slides)

        recognizer = EmbeddedImageRecognizer(filename)
        all_image_texts: list[str] = []

        for i, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    slide_text.append(shape.text.strip())

            # ── Embedded pictures on this slide (问题3) ──────────────────────
            slide_image_texts_before = len(all_image_texts)
            for blob in _iter_picture_blobs_for_slide(slide):
                try:
                    img = Image.open(io.BytesIO(blob))
                except Exception:
                    continue
                if recognizer.recognize(img):
                    all_image_texts.append(recognizer.texts[-1])

            page_text = "\n".join(t for t in slide_text if t).strip()
            slide_images = all_image_texts[slide_image_texts_before:]
            if slide_images:
                image_block = "\n".join(
                    f"[图片{i}内容] {t}"
                    for i, t in enumerate(slide_images, start=slide_image_texts_before + 1)
                )
                page_text = f"{page_text}\n\n[图片识别内容]\n{image_block}".strip()

            if not page_text:
                continue

            start = cursor
            end = start + len(page_text)

            pages.append(ExtractedPage(
                page_number=i + 1,
                text=page_text,
                char_start=start,
                char_end=end
            ))
            cursor = end + 2  # account for \n\n

        if not pages:
            raise ValueError(f"PPTX '{filename}' contains no extractable text.")

        full_text = "\n\n".join(p.text for p in pages)

        image_ocr_used = recognizer.has_content
        engines = ",".join(sorted(recognizer.engines)) if recognizer.engines else None
        extraction_method = "native"
        if image_ocr_used:
            extraction_method = "native+image_ocr"

        if image_ocr_used:
            logger.info(
                "PPTX '%s': recognised %d embedded image(s)",
                filename,
                len(recognizer.texts),
            )

        return ExtractionResult(
            pages=pages,
            full_text=full_text,
            page_count=total_pages,
            char_count=len(full_text),
            file_type="pptx",
            parser_used="python-pptx",
            ocr_used=image_ocr_used,
            ocr_engine=engines,
            extraction_method=extraction_method,
            image_count=len(all_image_texts),
            image_texts=all_image_texts,
        )


def _iter_picture_blobs_for_slide(slide):
    """Walk every shape on a slide (including groups) and yield picture blobs."""
    for shape in slide.shapes:
        yield from _iter_picture_blobs(shape)
