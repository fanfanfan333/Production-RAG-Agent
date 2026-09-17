import io
import re
from collections.abc import Sequence

from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from PIL import Image
from app.services.parsers.base import (
    DocumentParser,
    ExtractedImage,
    ExtractionResult,
    ExtractedPage,
)
from app.services.parsers.docling_support import docling_text
from app.services.parsers.image_recognition import EmbeddedImageRecognizer
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: Docling 的 Markdown 导出遇到图片时只写一个 HTML 注释占位，不带任何图片本体
#: 或说明。原样入库的后果：用户打开"文档内容"看到的是一串 `<!-- image -->`，
#: 图片等于**没有被标注**（既看不出是第几张，也不知道图里是什么）。
_IMAGE_PLACEHOLDER_RE = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)

#: image_type → 中文标签，与前端引用卡片的徽标说法保持一致。
_IMAGE_TYPE_LABEL = {
    "table": "表格",
    "chart": "图表",
    "diagram": "流程图",
    "screenshot": "截图",
    "photo": "图片",
    "formula": "公式",
    "code": "代码",
}

#: 单张图标注里说明文字的上限（防止把整篇 Markdown 表格塞进正文）。
_ANNOTATION_MAX_CHARS = 220


def _image_description(image: ExtractedImage) -> str:
    """
    给一张图挑一句最能说明"这是什么"的文字.

    优先级与 ``ExtractedImage.searchable_text`` 相反：检索要"结构化优先"
    （表格用单元格网格召回更准），而**给人读的标注**要"语义优先" ——
    Vision 的图义描述（"风扇系统，input→Fan→output"）比一堆单元格更能
    让人一眼看懂这张图在讲什么。
    """
    for candidate in (
        image.vision_caption,
        image.ocr_text,
        image.structured_content,
    ):
        text = " ".join((candidate or "").split())
        if text:
            if len(text) > _ANNOTATION_MAX_CHARS:
                text = text[:_ANNOTATION_MAX_CHARS] + "…"
            return text
    return ""


def annotate_image_placeholders(
    text: str,
    images: Sequence[ExtractedImage],
    source_ordinals: Sequence[int | None] | None = None,
) -> str:
    """
    把 Docling 正文里的 `<!-- image -->` 占位符换成**带编号与说明的图片标注**.

    正文里的占位符按阅读顺序编号，而 ``recognizer.images`` 是"**通过了过滤**的
    图片"列表 —— 尺寸过小（< MIN_IMAGE_DIMENSION 的图标/分隔条）、重复、超过
    数量上限的图都会被丢掉，**不再占号**。因此不能简单地按位置一一对应：
    真实案例里一份文档有 3 个占位符，第 2 张图是 262×33 的窄条被过滤掉，
    直接按顺序对齐会把"第 3 张图（数据流图）"的说明错标到第 2 个占位符上。

    所以对齐依据是 ``source_ordinals``（每张收下的图在来源枚举里排第几）。
    该信息缺失时退化为按位置对齐（旧调用方 / 无过滤场景仍然正确）。

    占位符没有对应图片对象时给出 ``［图 N］``（至少标出"这里原本有一张图"）；
    一张图都没有时退化为 ``［图片］``。

    Args:
        text:   Docling 导出的 Markdown 正文。
        images: ``EmbeddedImageRecognizer.images``。
        source_ordinals: 与 *images* 等长的来源序号列表。

    Returns:
        标注替换后的正文；没有占位符时原样返回。
    """
    if not text or not _IMAGE_PLACEHOLDER_RE.search(text):
        return text

    # 占位符序号 → 图片对象
    by_ordinal: dict[int, ExtractedImage] = {}
    if source_ordinals is not None and len(source_ordinals) == len(images):
        for ordinal, image in zip(source_ordinals, images):
            if ordinal:
                by_ordinal[int(ordinal)] = image
    else:
        by_ordinal = {i + 1: image for i, image in enumerate(images)}

    counter = 0

    def _replace(_match: re.Match) -> str:
        nonlocal counter
        counter += 1
        if not images:
            return "［图片］"
        image = by_ordinal.get(counter)
        if image is None:
            # 这个位置的图被过滤掉了（过小 / 重复 / 超上限）—— 标号仍在，
            # 但不编造说明
            return f"［图 {counter}］"
        label = _IMAGE_TYPE_LABEL.get(image.image_type or "", "图片")
        description = _image_description(image)
        head = f"［图 {counter} · {label}］"
        return f"{head}{description}" if description else head

    annotated = _IMAGE_PLACEHOLDER_RE.sub(_replace, text)
    logger.info(
        "DOCX image placeholders annotated: %d placeholder(s), %d image object(s), "
        "%d matched by source ordinal",
        counter,
        len(images),
        len(by_ordinal),
    )
    return annotated

# python-docx 默认模板自带一张 docProps/thumbnail.jpeg（文档缩略图）。
# 它不是文档内容，但也是一张 image/* part —— 若按"包内所有图片"遍历，
# 每份 DOCX 都会多出一张假图片（既落盘占空间，又可能被当成检索对象）。
_DECORATIVE_PART_PREFIXES = ("/docprops/",)


def _iter_body_image_parts(doc):
    """
    按正文出现顺序产出**文档真正引用**的图片 part.

    只遍历包内所有 image/* part 会把模板缩略图也算进来；这里改为解析
    ``<a:blip r:embed="rIdN">``（正文里的每张内嵌图都会有一个），再通过关系
    表拿到对应的 part —— 这才是"文档里的图片"的准确定义。

    若解析不到任何 blip（老版本文件 / 异常文档），退化为包内扫描并跳过
    docProps 之类的装饰性 part。
    """
    seen: set[str] = set()
    found = False
    try:
        for blip in doc.element.body.iter(qn("a:blip")):
            rid = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
            if not rid:
                continue
            rel = doc.part.rels.get(rid)
            part = getattr(rel, "target_part", None)
            if part is None:
                continue
            if not str(part.content_type).startswith("image/"):
                continue
            key = str(part.partname)
            if key in seen:
                continue          # 同一张图在正文里复用多次只处理一次
            seen.add(key)
            found = True
            yield part
    except Exception as exc:      # noqa: BLE001
        logger.warning("Cannot enumerate DOCX body images (%s) — falling back", exc)
        found = False

    if found:
        return

    for part in doc.part.package.iter_parts():
        if not str(part.content_type).startswith("image/"):
            continue
        partname = str(part.partname).lower()
        if any(partname.startswith(prefix) for prefix in _DECORATIVE_PART_PREFIXES):
            continue
        yield part


def _iter_block_items(doc):
    """
    按文档顺序产出 Paragraph / Table.

    python-docx 的 ``doc.paragraphs`` **不包含表格内的段落**，只看它会把 Word
    表格内容整段丢掉（表格检索失效）。这里遍历 body 的 XML 子元素，才能拿到
    真实阅读顺序，表格也就不会被遗漏。
    """
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield DocxParagraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield DocxTable(child, doc)


def _table_to_markdown(table: DocxTable) -> str:
    """
    把 Word 表格转成 Markdown 表格.

    转 Markdown 而不是纯文本，是为了让下游 chunker 的
    ``detect_content_type`` 识别为 content_type="table"，从而支持表格检索，
    并让 Document Agent 能把它还原成真正的 Word 表格。
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

    def _pad(r: list[str]) -> list[str]:
        return r + [""] * (width - len(r))

    lines = [
        "| " + " | ".join(_pad(rows[0])) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for r in rows[1:]:
        lines.append("| " + " | ".join(_pad(r)) + " |")
    return "\n".join(lines)


class DocxParser(DocumentParser):
    """
    Parses DOCX files.

    Embedded images (部分1 + 部分2): every image stored in the DOCX package is
    OCR'd, optionally captioned by the vision model, and **persisted** as a
    structured ExtractedImage. DOCX has no fixed pagination, so images are
    attributed to page 1 (the same convention used for the text).
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
            doc = DocxDocument(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Cannot open DOCX '{filename}': {exc}") from exc

        # ── 正文：优先 Docling，失败回退 python-docx ──────────────────────────
        # Docling 的优势在阅读顺序、标题层级与原生表格的结构化导出；DOCX 正文
        # 本就没有分页概念（这里统一归到第 1 页），因此切换过去**没有页码回归**。
        docling_text_value = docling_text(content, filename)
        parser_used = "python-docx"

        if docling_text_value:
            full_text = docling_text_value
            parser_used = "docling"
        else:
            # 按阅读顺序抽取正文：段落 + 表格（表格转 Markdown）
            blocks: list[str] = []
            for block in _iter_block_items(doc):
                if isinstance(block, DocxParagraph):
                    text = block.text.strip()
                    if text:
                        blocks.append(text)
                else:  # DocxTable
                    markdown = _table_to_markdown(block)
                    if markdown:
                        blocks.append(markdown)
            full_text = "\n\n".join(blocks)

        # ── Embedded images (部分1+2) ─────────────────────────────────────────
        settings = get_settings()
        independent = settings.IMAGE_AS_INDEPENDENT_OBJECT
        recognizer = EmbeddedImageRecognizer(filename, document_id=document_id, tenant_id=tenant_id)
        try:
            # 注意：这里**不传 bbox**。DOCX 是流式排版 —— 图片的页面坐标由
            # Word 的排版引擎在渲染时才决定，OOXML 里并没有"绝对位置"。
            # 硬凑一个假坐标会让引用卡片画出错误的框，不如老实降级：
            # position（文档内第几张图）仍然准确，前端据此定位。
            for ordinal, part in enumerate(_iter_body_image_parts(doc), start=1):
                try:
                    img = Image.open(io.BytesIO(part.blob))
                except Exception:
                    continue  # not a raster image PIL can decode
                # source_ordinal 必须传：被过滤掉的图不占 self.images 的名额，
                # 正文标注要靠它才能对回"第几个占位符"。
                recognizer.recognize(img, page_number=1, source_ordinal=ordinal)
        except Exception as exc:
            logger.warning("DOCX image extraction failed for '%s': %s", filename, exc)

        # ── 图片标注：Docling 正文里的 <!-- image --> 换成可读的图片说明 ────────
        # 必须放在图片抽取**之后** —— 标注要用到每张图的类型、图注，以及
        # "这张图对应正文里的第几个占位符"（source_ordinals）。
        if parser_used == "docling":
            full_text = annotate_image_placeholders(
                full_text, recognizer.images, recognizer.source_ordinals
            )

        if not independent:
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

        image_ocr_used = recognizer.has_content or bool(recognizer.images)
        engines = ",".join(sorted(recognizer.engines)) if recognizer.engines else None
        # 正文解析引擎（docling / python-docx）+ 是否有图片对象
        extraction_method = "docling" if parser_used == "docling" else "native"
        if recognizer.images:
            extraction_method += "+image"

        if recognizer.images:
            logger.info(
                "DOCX '%s': extracted %d image object(s) (%d persisted)",
                filename,
                len(recognizer.images),
                recognizer.saved_count,
            )

        return ExtractionResult(
            pages=[page],
            full_text=full_text,
            page_count=1,
            char_count=len(full_text),
            file_type="docx",
            parser_used=parser_used,
            ocr_used=image_ocr_used,
            ocr_engine=engines,
            extraction_method=extraction_method,
            images=list(recognizer.images),
            image_count=len(recognizer.images),
            image_texts=list(recognizer.texts),
        )
