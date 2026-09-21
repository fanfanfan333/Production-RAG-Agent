import io
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from PIL import Image
from app.services.parsers.base import DocumentParser, ExtractionResult, ExtractedPage
from app.services.parsers.image_recognition import EmbeddedImageRecognizer
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


# 1 点 = 12700 EMU（English Metric Unit）。python-pptx 的 left/top/width/height
# 都是 EMU 整数，换算成"点"才能与 PDF 的 bbox 用同一套坐标系（1/72 英寸）。
_EMU_PER_POINT = 12700.0


def _shape_bbox(shape) -> tuple[float, float, float, float] | None:
    """
    形状在幻灯片上的边界框（点坐标，原点左上、y 向下）.

    幻灯片尺寸固定（16:9 或 4:3），所以形状坐标本身就是稳定的页面坐标 ——
    这正是 PPTX 能给出 bbox 而 DOCX 不能的原因：PPTX 是绝对定位，DOCX 是
    流式排版，段落位置要等渲染引擎排版后才存在。
    """
    try:
        left = int(shape.left)
        top = int(shape.top)
        width = int(shape.width)
        height = int(shape.height)
    except (TypeError, ValueError, AttributeError):
        return None
    return (
        left / _EMU_PER_POINT,
        top / _EMU_PER_POINT,
        (left + width) / _EMU_PER_POINT,
        (top + height) / _EMU_PER_POINT,
    )


def _shape_offset(shape, parent: tuple[float, float]) -> tuple[float, float]:
    """形状左上角在页面坐标系中的位置（叠加父级组的偏移）."""
    try:
        left = int(shape.left) / _EMU_PER_POINT
        top = int(shape.top) / _EMU_PER_POINT
    except (TypeError, ValueError, AttributeError):
        return parent
    return (parent[0] + left, parent[1] + top)


def _iter_picture_blobs(shape, offset: tuple[float, float] = (0.0, 0.0)):
    """
    Yield ``(blob, bbox)`` from a shape, recursing into grouped shapes.

    组合形状（group）里的图片，其 left/top 是**相对组内**的偏移，直接当页面
    坐标用会偏。递归时把组自身的位置作为 offset 传下去，子图片的矩形才能
    平移到真实的页面坐标。
    """
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        try:
            blob = shape.image.blob
        except Exception:
            return
        bbox = _shape_bbox(shape)
        if bbox is not None and offset != (0.0, 0.0):
            dx, dy = offset
            bbox = (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)
        yield blob, bbox
    elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        group_offset = _shape_offset(shape, offset)
        for sub in getattr(shape, "shapes", []):
            yield from _iter_picture_blobs(sub, group_offset)


class PptxParser(DocumentParser):
    """
    Parses PPTX files.

    Embedded images (部分1 + 部分2): pictures on each slide are OCR'd,
    optionally captioned by the vision model, and **persisted** as structured
    ExtractedImage objects, keeping their correct slide attribution.
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
            prs = Presentation(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Cannot open PPTX '{filename}': {exc}") from exc

        settings = get_settings()
        independent = settings.IMAGE_AS_INDEPENDENT_OBJECT

        pages: list[ExtractedPage] = []
        cursor = 0
        total_pages = len(prs.slides)
        has_tables = False
        has_charts = False

        recognizer = EmbeddedImageRecognizer(filename, document_id=document_id, tenant_id=tenant_id)
        all_image_texts: list[str] = []

        for i, slide in enumerate(prs.slides):
            slide_number = i + 1
            slide_text = []
            for shape in slide.shapes:
                # 表格（GraphicFrame）没有 .text 属性，旧实现里整张表被静默跳过；
                # 图表（内嵌 workbook 数据）同理。这里显式识别并转成可检索文本。
                if shape.has_table:
                    md = _pptx_table_to_markdown(shape.table)
                    if md:
                        slide_text.append(md)
                        has_tables = True
                    continue
                if shape.has_chart:
                    chart_text = _pptx_chart_to_text(shape.chart)
                    if chart_text:
                        slide_text.append(chart_text)
                        has_charts = True
                    continue
                if shape.has_text_frame and shape.text:
                    slide_text.append(shape.text.strip())

            # ── Embedded pictures on this slide (部分1+2) ─────────────────────
            slide_image_texts_before = len(all_image_texts)
            for blob, bbox in _iter_picture_blobs_for_slide(slide):
                try:
                    img = Image.open(io.BytesIO(blob))
                except Exception:
                    continue
                if recognizer.recognize(img, page_number=slide_number, bbox=bbox):
                    all_image_texts.append(recognizer.texts[-1])

            page_text = "\n".join(t for t in slide_text if t).strip()

            # Legacy merge path only; independent mode keeps images separate.
            if not independent:
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
                page_number=slide_number,
                text=page_text,
                char_start=start,
                char_end=end
            ))
            cursor = end + 2  # account for \n\n

        if not pages:
            raise ValueError(f"PPTX '{filename}' contains no extractable text.")

        full_text = "\n\n".join(p.text for p in pages)

        image_ocr_used = recognizer.has_content or bool(recognizer.images)
        engines = ",".join(sorted(recognizer.engines)) if recognizer.engines else None
        extraction_method = "native"
        if recognizer.images:
            extraction_method = "native+image"
        # 表格 / 图表内容已入库，在 extraction_method 上体现（旧实现整块丢失却仍报 native）。
        if has_tables:
            extraction_method += "+table"
        if has_charts:
            extraction_method += "+chart"

        if recognizer.images:
            logger.info(
                "PPTX '%s': extracted %d image object(s) (%d persisted)",
                filename,
                len(recognizer.images),
                recognizer.saved_count,
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
            images=list(recognizer.images),
            image_count=len(recognizer.images),
            image_texts=list(all_image_texts),
        )


def _iter_picture_blobs_for_slide(slide):
    """Walk every shape on a slide (including groups) and yield picture blobs."""
    for shape in slide.shapes:
        yield from _iter_picture_blobs(shape)


def _pptx_table_to_markdown(table) -> str:
    """
    把 PPTX 里的表格（GraphicFrame）转成 Markdown 表格入库.

    与 DOCX 同一思路：转 Markdown 才能让下游 chunker 识别为 content_type="table"，
    支持表格检索，且 Document Agent 能把它还原成真正表格。旧实现只取
    ``shape.text``，而 GraphicFrame 没有该属性 → 整张表被静默丢弃。
    """
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [
            cell.text.strip().replace("\n", " ").replace("|", "\\|")
            for cell in row.cells
        ]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""

    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    lines = [
        "| " + " | ".join(padded[0]) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for r in padded[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _pptx_chart_to_text(chart) -> str:
    """
    把 PPTX 里内嵌的图表（chart）导出为"系列名: 数值序列"文本.

    图表的数据存在内嵌 workbook 里，旧实现取不到（同 GraphicFrame 问题）。
    导出系列名 + 各点数值，至少让图里的数字能被检索到。
    """
    parts: list[str] = []
    try:
        for plot in chart.plots:
            for series in plot.series:
                name = (series.name or "").strip()
                values = series.values
                if not values:
                    continue
                vals = ", ".join(str(v) for v in values)
                parts.append(f"{name}: {vals}" if name else vals)
    except Exception as exc:      # noqa: BLE001
        logger.debug("PPTX chart export skipped for '%s': %s", exc, exc)
        return ""
    return "\n".join(parts)
