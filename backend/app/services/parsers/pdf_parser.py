import fitz  # PyMuPDF
from PIL import Image
import io
from app.services.parsers.base import DocumentParser, ExtractionResult, ExtractedPage
from app.services.ocr import get_ocr_service
from app.services.parsers.image_recognition import EmbeddedImageRecognizer
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _image_bbox_on_page(page, xref: int) -> tuple[float, float, float, float] | None:
    """
    图片在页面上的边界框（点坐标，原点左上、y 向下）.

    ``page.get_images()`` 只给 xref（图片对象），不给它画在页面的哪里；
    ``get_image_rects()`` 才回答"这张图出现在页面的哪个矩形里"。同一张图
    在一页里可能出现多次（水印平铺），取第一个矩形即可 —— 引用定位只需要
    "大概在哪"，不需要枚举全部实例。

    拿不到矩形（矢量图 / 已被旋转裁剪 / PyMuPDF 版本差异）时返回 None，
    由上层降级为只显示"第几页 + 第几张图"。
    """
    try:
        rects = page.get_image_rects(xref)
    except Exception:      # noqa: BLE001 — 坐标是"锦上添花"，失败不影响主流程
        return None
    if not rects:
        return None
    rect = rects[0]
    return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))


class PDFParser(DocumentParser):
    """
    Parses PDFs. Automatically falls back to OCR if the native text extraction
    yields very little or no content.

    Embedded images (部分1 + 部分2): every picture inside the PDF is extracted,
    OCR'd, optionally captioned by the vision model, and **persisted to disk**
    as a structured ExtractedImage. When IMAGE_AS_INDEPENDENT_OBJECT is on
    (default) their text is no longer merged into the page text — they are
    indexed as independent retrieval objects instead.

    Scanned pages (no native text) are themselves images: the rendered page is
    persisted too, so the original scan can be returned as the source picture.
    """

    def __init__(self, min_chars_per_page_for_ocr: int = 20):
        self.min_chars = min_chars_per_page_for_ocr
        self.ocr_service = get_ocr_service()

    def parse(
        self,
        content: bytes,
        filename: str,
        *,
        document_id: str | None = None,
        tenant_id: str | None = None,
    ) -> ExtractionResult:
        try:
            doc: fitz.Document = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Cannot open PDF '{filename}': {exc}") from exc

        settings = get_settings()
        independent = settings.IMAGE_AS_INDEPENDENT_OBJECT

        pages: list[ExtractedPage] = []
        parts: list[str] = []
        cursor = 0
        total_pages: int = len(doc)

        ocr_was_used = False
        engines_used = set()

        recognizer = EmbeddedImageRecognizer(filename, document_id=document_id, tenant_id=tenant_id)
        seen_xrefs: set[int] = set()   # same image reused across pages → once
        max_images_per_page = settings.MAX_IMAGES_PER_PAGE
        image_only_lost = 0   # 纯图片页（无文字且无可用图片文本）被跳过的计数

        for page_index in range(total_pages):
            page: fitz.Page = doc[page_index]
            page_number = page_index + 1
            page_text: str = page.get_text("markdown").strip()
            page_needed_page_ocr = False
            page_render: Image.Image | None = None

            # If native text is too little, we attempt OCR
            if len(page_text) < self.min_chars:
                logger.debug("Page %d of '%s' has little/no text, attempting OCR", page_number, filename)
                try:
                    # Render page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution for better OCR
                    page_render = Image.open(io.BytesIO(pix.tobytes("png")))

                    ocr_text, engine_name = self.ocr_service.extract_text(page_render)

                    if ocr_text.strip():
                        page_text = ocr_text.strip()
                        ocr_was_used = True
                        page_needed_page_ocr = True
                        engines_used.add(engine_name)
                except Exception as e:
                    logger.warning("OCR failed on page %d of '%s': %s", page_number, filename, e)

            # ── Scanned page → persisted image object（部分2）─────────────────
            # A page that needed whole-page OCR is essentially a scan: keep the
            # rendered image so the original page can be returned as a picture.
            if page_needed_page_ocr and page_render is not None:
                recognizer.register_page_scan(
                    page_render, page_number, page_text,
                    # 整页扫描的"位置"就是整页矩形
                    bbox=(0.0, 0.0, float(page.rect.width), float(page.rect.height)),
                )

            # ── Embedded images on this page（部分1+2）────────────────────────
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
                            # 保留图片在页面上的位置（细粒度引用：第几页的第几处）
                            bbox = _image_bbox_on_page(page, xref)
                            if recognizer.recognize(
                                img, page_number=page_number, bbox=bbox
                            ):
                                per_page_used += 1
                        except Exception as exc:
                            logger.debug(
                                "Skipping unextractable image xref=%s on page %d of '%s': %s",
                                xref, page_number, filename, exc,
                            )
                except Exception as exc:
                    logger.warning(
                        "Image extraction failed on page %d of '%s': %s",
                        page_number, filename, exc,
                    )
                if len(recognizer.texts) > image_texts_before:
                    engines_used.update(recognizer.engines)

            # Legacy path: merge recognised image content into the page text so
            # it is chunked/embedded with the correct page attribution. Skipped
            # when images are indexed as independent objects (部分1).
            if not independent:
                page_new = recognizer.texts[image_texts_before:]
                if page_new:
                    page_image_block = "\n".join(
                        f"[图片{i}内容] {t}"
                        for i, t in enumerate(page_new, start=image_texts_before + 1)
                    )
                    page_text = f"{page_text}\n\n[图片识别内容]\n{page_image_block}".strip()

            if not page_text:
                # 纯图片页：有图但没抽出任何文字 → 整页内容静默消失（已知缺陷 B11）。
                # 至少计数告警，便于排查"页数 vs 实际内容"不一致。
                had_images = (len(recognizer.texts) > image_texts_before) or page_needed_page_ocr
                if had_images:
                    image_only_lost += 1
                continue

            start = cursor
            end = start + len(page_text)

            pages.append(ExtractedPage(
                page_number=page_number,
                text=page_text,
                char_start=start,
                char_end=end,
            ))
            parts.append(page_text)
            cursor = end + 2

        doc.close()

        if image_only_lost:
            # 纯图片页有图无文、整页被跳过：内容未入库但 page_count 仍按总页数报，
            # 页数 vs 实际内容会不一致。告警让该现象可见（B11）。
            logger.warning(
                "PDF '%s': %d 个纯图片页无可用文字被跳过（仅保留独立图片对象，"
                "正文未入库）；page_count 仍为总页数 %d 而非实际页数 %d",
                filename, image_only_lost, total_pages, len(pages),
            )

        if not pages:
            raise ValueError(f"PDF '{filename}' contains no extractable text even after OCR.")

        full_text = "\n\n".join(p.text for p in pages)

        image_ocr_used = recognizer.has_content or bool(recognizer.images)
        engine_str = ",".join(engines_used) if engines_used else None
        extraction_method = "ocr" if ocr_was_used else "native"
        if recognizer.images:
            extraction_method += "+image"

        if recognizer.images:
            logger.info(
                "PDF '%s': extracted %d image object(s) (%d persisted)%s",
                filename,
                len(recognizer.images),
                recognizer.saved_count,
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
            images=list(recognizer.images),
            image_count=len(recognizer.images),
            image_texts=list(recognizer.texts),
        )
