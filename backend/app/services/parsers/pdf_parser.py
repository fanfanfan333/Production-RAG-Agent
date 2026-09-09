import fitz  # PyMuPDF
from PIL import Image
import io
from app.services.parsers.base import DocumentParser, ExtractionResult, ExtractedPage
from app.services.ocr import get_ocr_service
from app.services.parsers.image_recognition import EmbeddedImageRecognizer
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

class PDFParser(DocumentParser):
    """
    Parses PDFs. Automatically falls back to OCR if the native text extraction
    yields very little or no content.

    Embedded-image recognition (问题3): images inside text-rich pages are
    extracted and OCR'd (optionally captioned by an Ollama vision model) so
    their content becomes searchable alongside the page text.
    """

    def __init__(self, min_chars_per_page_for_ocr: int = 20):
        self.min_chars = min_chars_per_page_for_ocr
        self.ocr_service = get_ocr_service()

    def parse(self, content: bytes, filename: str) -> ExtractionResult:
        try:
            doc: fitz.Document = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Cannot open PDF '{filename}': {exc}") from exc

        pages: list[ExtractedPage] = []
        parts: list[str] = []
        cursor = 0
        total_pages: int = len(doc)

        ocr_was_used = False
        engines_used = set()

        recognizer = EmbeddedImageRecognizer(filename)
        seen_xrefs: set[int] = set()   # same image reused across pages → once
        max_images_per_page = get_settings().MAX_IMAGES_PER_PAGE

        for page_index in range(total_pages):
            page: fitz.Page = doc[page_index]
            page_text: str = page.get_text("markdown").strip()
            page_needed_page_ocr = False

            # If native text is too little, we attempt OCR
            if len(page_text) < self.min_chars:
                logger.debug("Page %d of '%s' has little/no text, attempting OCR", page_index + 1, filename)
                try:
                    # Render page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution for better OCR
                    img = Image.open(io.BytesIO(pix.tobytes("png")))

                    ocr_text, engine_name = self.ocr_service.extract_text(img)

                    if ocr_text.strip():
                        page_text = ocr_text.strip()
                        ocr_was_used = True
                        page_needed_page_ocr = True
                        engines_used.add(engine_name)
                except Exception as e:
                    logger.warning("OCR failed on page %d of '%s': %s", page_index + 1, filename, e)

            # ── Embedded images on this page (问题3) ─────────────────────────
            # Pages already OCR'd as a whole render were OCR'd WITH their
            # images, so skip per-image OCR there to avoid duplicates.
            image_texts_before = len(recognizer.texts)
            if recognizer.enabled and not page_needed_page_ocr:
                per_page_used = 0
                try:
                    for img_info in page.get_images(full=True):
                        if per_page_used >= max_images_per_page:
                            break
                        xref = img_info[0]
                        if xref in seen_xrefs:
                            continue
                        seen_xrefs.add(xref)
                        try:
                            extracted = doc.extract_image(xref)
                            img = Image.open(io.BytesIO(extracted["image"]))
                            if recognizer.recognize(img):
                                per_page_used += 1
                        except Exception as exc:
                            logger.debug(
                                "Skipping unextractable image xref=%s on page %d of '%s': %s",
                                xref, page_index + 1, filename, exc,
                            )
                except Exception as exc:
                    logger.warning(
                        "Image extraction failed on page %d of '%s': %s",
                        page_index + 1, filename, exc,
                    )
                if len(recognizer.texts) > image_texts_before:
                    engines_used.update(recognizer.engines)

            # Merge recognised image content into the page text so it is
            # chunked/embedded with the correct page attribution.
            page_new = recognizer.texts[image_texts_before:]
            if page_new:
                page_image_block = "\n".join(
                    f"[图片{i}内容] {t}"
                    for i, t in enumerate(page_new, start=image_texts_before + 1)
                )
                page_text = f"{page_text}\n\n[图片识别内容]\n{page_image_block}".strip()

            if not page_text:
                continue

            start = cursor
            end = start + len(page_text)

            pages.append(ExtractedPage(
                page_number=page_index + 1,
                text=page_text,
                char_start=start,
                char_end=end,
            ))
            parts.append(page_text)
            cursor = end + 2

        doc.close()

        if not pages:
            raise ValueError(f"PDF '{filename}' contains no extractable text even after OCR.")

        full_text = "\n\n".join(p.text for p in pages)

        image_ocr_used = recognizer.has_content
        engine_str = ",".join(engines_used) if engines_used else None
        extraction_method = "ocr" if ocr_was_used else "native"
        if image_ocr_used:
            extraction_method += "+image_ocr"

        if image_ocr_used:
            logger.info(
                "PDF '%s': recognised %d embedded image(s)%s",
                filename,
                len(recognizer.texts),
                f", skipped {recognizer.skipped_count}" if recognizer.skipped_count else "",
            )

        return ExtractionResult(
            pages=pages,
            full_text=full_text,
            page_count=total_pages,
            char_count=len(full_text),
            file_type="pdf",
            parser_used="PyMuPDF",
            ocr_used=ocr_was_used or image_ocr_used,
            ocr_engine=engine_str,
            extraction_method=extraction_method,
            image_count=len(recognizer.texts),
            image_texts=list(recognizer.texts),
        )
