import io
import hashlib
from PIL import Image
from app.services.parsers.base import (
    DocumentParser,
    ExtractionResult,
    ExtractedPage,
    ExtractedImage,
)
from app.services.image_understanding import (
    fallback_understanding,
    understand_image,
)
from app.services.parsers.image_recognition import (
    _is_usable_image,
    _to_rgb,
    encode_png,
)
from app.services.storage import save_image
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 处理路径 → extraction_method（也是 DB 里可见的"这张图是怎么被读懂的"）
_METHOD_BY_ROUTE = {
    "table_parser": "table",
    "formula_parser": "formula",
    "code_parser": "code",
    "vision": "vision",
    "ocr": "ocr",
}


class ImageParser(DocumentParser):
    """
    Parses Image files.

    独立上传的图片与文档内嵌图片走**同一套三层处理**（部分1 + 部分2）：

        Image → Picture Classification ─┬─ Table   → Table Parser（Markdown 表格）
                                        ├─ Chart   → Vision（数据点/坐标轴）
                                        ├─ Diagram → Vision（节点/连线）
                                        └─ 其他     → OCR

    产出物落盘为图片对象，检索命中后可回显原始图片；若它是表格图片，则
    入库为 content_type="table" 的真表格，"表格检索"因此对图片同样生效。
    """

    def parse(
        self,
        content: bytes,
        filename: str,
        *,
        document_id: str | None = None,
        tenant_id: str | None = None,
    ) -> ExtractionResult:
        try:
            img = Image.open(io.BytesIO(content))
            # verify image is valid
            img.verify()

            # reopening because verify() seeks to end
            img = Image.open(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Cannot open Image '{filename}': {exc}") from exc

        settings = get_settings()
        img = _to_rgb(img)

        # 小图标 / 分隔线不值得进知识库
        if not _is_usable_image(img):
            raise ValueError(
                f"Image '{filename}' is too small or too thin to be meaningful "
                f"content ({img.size[0]}x{img.size[1]})."
            )

        # ── 落盘（部分2）——先落盘，PNG 字节同时喂给 Vision，只编码一次 ──────
        image_path: str | None = None
        png_bytes: bytes | None = None
        if settings.ENABLE_IMAGE_SAVE and document_id:
            png_bytes = encode_png(img)
            image_path = save_image(str(document_id), 1, 1, png_bytes, "png", tenant_id=tenant_id)

        # ── 三层图片处理 ─────────────────────────────────────────────────────
        try:
            understanding = understand_image(
                img, page_number=1, filename=filename, png_bytes=png_bytes,
            )
        except Exception as exc:      # noqa: BLE001
            logger.warning("Image understanding failed for '%s': %s", filename, exc)
            understanding = fallback_understanding(img, reason=str(exc))

        base = str(document_id) if document_id else hashlib.sha1(
            filename.encode("utf-8", "ignore")
        ).hexdigest()[:12]
        extracted = ExtractedImage(
            image_id=f"{base}-p1-i1",
            page_number=1,
            ocr_text=understanding.ocr_text,
            vision_caption=understanding.vision_caption,
            image_path=image_path,
            width=img.size[0],
            height=img.size[1],
            ocr_engine=understanding.ocr_engine,
            image_type=understanding.image_type,
            structured_content=understanding.structured_content,
            analyze_engine=understanding.analyze_engine,
            classify_engine=understanding.classification.engine,
            classification_signals=dict(understanding.classification.signals),
            analyze_confidence=understanding.confidence,
            analyze_decision=understanding.decision,
            manual_review=understanding.manual_review,
        )

        searchable = extracted.searchable_text
        if not searchable and not image_path:
            raise ValueError(
                f"Image '{filename}' contains no extractable text even after "
                f"classification + OCR + vision."
            )

        # 部分1：图片不再当成纯文本 —— 开启独立对象模式时，正文留空，
        # 由 document_service 只生成 image chunk（避免同一内容重复入库）。
        full_text = "" if settings.IMAGE_AS_INDEPENDENT_OBJECT else searchable

        # Determine format extension
        fmt = (img.format or "image").lower()

        return ExtractionResult(
            pages=[ExtractedPage(
                page_number=1,
                text=searchable,
                char_start=0,
                char_end=len(searchable),
            )],
            full_text=full_text,
            page_count=1,
            char_count=len(searchable),
            file_type=fmt,
            parser_used="ImageParser",
            ocr_used=bool(understanding.ocr_engine),
            ocr_engine=understanding.ocr_engine,
            extraction_method=_METHOD_BY_ROUTE.get(
                understanding.route, "image"
            ),
            # 表格图片产出的是真表格（content_type="table"）
            is_tabular=extracted.content_type == "table",
            images=[extracted],
            image_count=1,
            image_texts=[searchable] if searchable else [],
        )
