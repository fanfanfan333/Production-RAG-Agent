import io
from docx import Document as DocxDocument
from PIL import Image
from app.services.parsers.base import DocumentParser, ExtractionResult, ExtractedPage
from app.services.parsers.image_recognition import EmbeddedImageRecognizer
from app.utils.logging import get_logger

logger = get_logger(__name__)

class DocxParser(DocumentParser):
    """
    Parses DOCX files.

    Embedded-image recognition (问题3): every image stored in the DOCX
    package is OCR'd (optionally captioned by an Ollama vision model) and
    appended to the extracted text so its content is searchable.
    """

    def parse(self, content: bytes, filename: str) -> ExtractionResult:
        try:
            doc = DocxDocument(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Cannot open DOCX '{filename}': {exc}") from exc

        full_text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())

        # ── Embedded images (问题3) ────────────────────────────────────────────
        recognizer = EmbeddedImageRecognizer(filename)
        try:
            for part in doc.part.package.iter_parts():
                if not part.content_type.startswith("image/"):
                    continue
                try:
                    img = Image.open(io.BytesIO(part.blob))
                except Exception:
                    continue  # not a raster image PIL can decode
                recognizer.recognize(img)
        except Exception as exc:
            logger.warning("DOCX image extraction failed for '%s': %s", filename, exc)

        image_block = recognizer.merged_text()
        if image_block:
            full_text = f"{full_text}\n\n{image_block}".strip()

        if not full_text:
            raise ValueError(f"DOCX '{filename}' contains no extractable text.")

        # Treat whole doc as a single page for chunking purposes
        page = ExtractedPage(
            page_number=1,
            text=full_text,
            char_start=0,
            char_end=len(full_text)
        )

        image_ocr_used = recognizer.has_content
        engines = ",".join(sorted(recognizer.engines)) if recognizer.engines else None
        extraction_method = "native"
        if image_ocr_used:
            extraction_method = "native+image_ocr"

        if image_ocr_used:
            logger.info(
                "DOCX '%s': recognised %d embedded image(s)",
                filename,
                len(recognizer.texts),
            )

        return ExtractionResult(
            pages=[page],
            full_text=full_text,
            page_count=1,
            char_count=len(full_text),
            file_type="docx",
            parser_used="python-docx",
            ocr_used=image_ocr_used,
            ocr_engine=engines,
            extraction_method=extraction_method,
            image_count=len(recognizer.texts),
            image_texts=list(recognizer.texts),
        )
