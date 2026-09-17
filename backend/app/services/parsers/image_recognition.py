"""
Embedded-image recognition（三层图片处理：分类 → 分流 → 结构化）.

Used by the document parsers to turn the images that live INSIDE a document
(PDF pages, DOCX packages, PPTX slides) into **structured, persisted images**
instead of losing them into the text stream.

处理链路（对应设计稿）::

    Document
      ├─ Native Text
      └─ Embedded Image
            ├─ filter    丢掉图标 / 分隔线 / 1px 装饰
            ├─ de-dup    同一张 logo 在每页重复出现只处理一次
            ├─ OCR       一次推理拿到文本 + 行坐标（分类与表格还原共用）
            ├─ classify  Picture Classification（table/chart/diagram/screenshot/photo）
            ├─ route     ┌ Table  → Table Parser（框线/对齐 → Markdown 表格）
            │            ├ Chart  → Vision（数据点 / 坐标轴 / 图例）
            │            ├ Diagram→ Vision（节点 / 连线 / 方向）
            │            └ 其他    → OCR
            └─ persist   图片本体落盘 uploads/{document_id}/images/page_3_image_1.png

产出 ``list[ExtractedImage]``：每张图片带 image_id / image_path / page_number /
image_type / ocr_text / vision_caption / structured_content，由 document_service
转成独立 chunk 入库 —— 表格图片产出的是 content_type="table" 的真表格。

与旧版的关键区别：旧版把 OCR 文本拼进正文后即丢弃图片本体，且对所有图片
一律只做 OCR；现在图片先判类型再分流，"图里的表格"能变成可检索的表格。
"""

from __future__ import annotations

import hashlib
import io

from PIL import Image

from app.config import get_settings
from app.services.image_understanding import (
    IMAGE_TYPE_PHOTO,
    ImageUnderstanding,
    fallback_understanding,
    understand_image,
)
from app.services.parsers.base import ExtractedImage
from app.services.storage import save_image
from app.services.vision import get_vision_service
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Content-marker prefix used by the legacy merged-text path.
IMAGE_MARKER_PREFIX = "图片"


def _normalize_bbox(
    bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """
    规整边界框：丢弃非法值、保证 (x1 ≤ x2, y1 ≤ y2).

    解析器给的坐标来自第三方库（PyMuPDF / python-pptx），偶尔会出现旋转页
    导致的逆序或 None 分量。这里统一收敛，避免把脏坐标写进 payload 后
    前端画出一个负宽高的框。
    """
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if any(v != v for v in (x1, y1, x2, y2)):     # NaN
        return None
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    if x2 <= x1 or y2 <= y1:
        return None
    return (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))


def _is_usable_image(img: Image.Image) -> bool:
    """
    Filter out icons / separators / 1px decorations.

    判据分三层，缺一不可（曾经只看"任一边 < MIN_IMAGE_DIMENSION"，
    把**真实内容**也误杀了 —— 例如一行终端输出截图 ``torch.Size([64, 10])``
    只有 262×33，被当成图标丢掉，于是文档里那张图既没被分析、也没被标注）：

        1. 任一边小到退化（< MIN_IMAGE_THIN_SIDE）→ 边框 / 分隔线
        2. **两边都小**（< MIN_IMAGE_DIMENSION）→ 图标 / 装饰
           （只小一边不算：宽而矮的窄条往往是代码/输出截图，是内容）
        3. 极端长宽比（> 25）→ 横线 / 竖线
    """
    width, height = img.size
    settings = get_settings()
    min_dim = settings.MIN_IMAGE_DIMENSION
    thin_side = settings.MIN_IMAGE_THIN_SIDE
    if min(width, height) < thin_side:
        return False
    if width < min_dim and height < min_dim:
        return False
    if width / max(height, 1) > 25 or height / max(width, 1) > 25:
        return False
    return True


def _to_rgb(img: Image.Image) -> Image.Image:
    """Normalise colour modes — OCR and vision models want RGB (or L)."""
    if img.mode in ("RGB", "L"):
        return img
    return img.convert("RGB")


def encode_png(img: Image.Image) -> bytes:
    """Serialise a PIL image to PNG bytes (used for persistence + vision)."""
    buffer = io.BytesIO()
    _to_rgb(img).save(buffer, format="PNG")
    return buffer.getvalue()


def caption_with_vision(img: Image.Image, filename: str = "") -> str | None:
    """
    用通用提示词让多模态模型描述一张图片（保留的历史入口）.

    三层路由之后，入库期的图片理解统一走 ``image_understanding``：图片先分类，
    再按类型选提示词（图表要数据点、流程图要节点连线）。本函数保留给
    "没有分类上下文、只想拿一句描述"的调用方（如独立图片上传的兜底）。

    返回 None 表示 vision 未开启 / 模型未拉取 / 调用失败 —— 此时 OCR 文本
    仍是可用的检索信号。
    """
    settings = get_settings()
    if not settings.VISION_ENABLED or not settings.OLLAMA_VISION_MODEL:
        return None
    try:
        caption = get_vision_service().describe_image_sync(
            encode_png(img), timeout=settings.VISION_TIMEOUT_SECONDS
        )
        if caption:
            return caption[: settings.VISION_CAPTION_MAX_CHARS]
        return None
    except Exception as exc:      # noqa: BLE001
        logger.warning(
            "Vision caption failed for image in '%s' (model=%s): %s",
            filename or "?",
            settings.OLLAMA_VISION_MODEL,
            exc,
        )
        return None


def _safe_understand(
    img: Image.Image,
    *,
    page_number: int,
    filename: str,
    png_bytes: bytes | None,
) -> ImageUnderstanding:
    """三层路由的不可抛入口：任何异常都退化为"普通图片 + OCR"."""
    try:
        return understand_image(
            img,
            page_number=page_number,
            filename=filename,
            png_bytes=png_bytes,
        )
    except Exception as exc:      # noqa: BLE001
        logger.warning("Image understanding failed for '%s': %s", filename, exc)
        return fallback_understanding(img, reason=str(exc))


class EmbeddedImageRecognizer:
    """
    Recognises + persists a stream of embedded images for one document.

    Usage:
        recognizer = EmbeddedImageRecognizer(filename, document_id=doc_id)
        for pil_image, page_number in images:
            recognizer.recognize(pil_image, page_number=page_number)
        result.images = recognizer.images          # 结构化图片（部分1）
        block = recognizer.merged_text()           # 兼容旧的合并文本路径
    """

    def __init__(
        self,
        filename: str,
        max_images: int | None = None,
        *,
        document_id: str | None = None,
        tenant_id: str | None = None,
    ):
        self.filename = filename
        self.document_id = document_id
        # 第三层隔离：图片落盘到 uploads/{tenant_id}/{document_id}/images/
        self.tenant_id = tenant_id
        settings = get_settings()
        self.enabled = settings.ENABLE_IMAGE_OCR or settings.IMAGE_CLASSIFICATION_ENABLED
        self.max_images = max_images or settings.MAX_IMAGES_PER_DOCUMENT
        self.save_enabled = bool(settings.ENABLE_IMAGE_SAVE and document_id)

        self.texts: list[str] = []                 # 兼容字段：识别出的文本块
        self.images: list[ExtractedImage] = []     # 结构化图片（部分1）
        # 与 self.images **逐项对齐**的来源序号：该图片在被枚举时排第几
        # （1-based）。有些来源图会被过滤掉（尺寸过小、重复、超上限），
        # 此时"第 n 张被收下的图" ≠ "原文里的第 n 张图" —— 想在正文里按
        # 位置标注图片（DOCX 的 <!-- image --> 占位符）必须靠这个序号对齐。
        self.source_ordinals: list[int | None] = []
        self.engines: set[str] = set()
        # 三层路由的运行统计（日志 / 排查用）
        self.type_counts: dict[str, int] = {}
        self.route_counts: dict[str, int] = {}
        self._seen_hashes: set[str] = set()
        self._page_counters: dict[int, int] = {}
        # 文档级图片序号（跨页累计）—— 与 _page_counters 的"页内序号"不同，
        # 它回答"这是整份文档里的第几张图"，写入 chunk 的 position 字段。
        self._position = 0
        self._skipped = 0

    # ── 查询 ──────────────────────────────────────────────────────────────

    @property
    def has_content(self) -> bool:
        return bool(self.texts)

    @property
    def saved_count(self) -> int:
        return sum(1 for img in self.images if img.image_path)

    @property
    def skipped_count(self) -> int:
        return self._skipped

    def summary(self) -> dict:
        """入库统计（写进日志，便于确认"分流真的发生了"）."""
        return {
            "images": len(self.images),
            "saved": self.saved_count,
            "skipped": self._skipped,
            "types": dict(sorted(self.type_counts.items())),
            "routes": dict(sorted(self.route_counts.items())),
            "ocr_engines": sorted(self.engines),
        }

    # ── 主流程 ────────────────────────────────────────────────────────────

    def recognize(
        self,
        img: Image.Image,
        page_number: int = 1,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        source_ordinal: int | None = None,
    ) -> bool:
        """
        对一张内嵌图片执行：过滤 → 去重 → OCR → 分类 → 分流 → 落盘.

        :param bbox: 图片在所在页面上的边界框 (x1, y1, x2, y2)（点坐标，原点
            左上、y 向下）。PDF / PPTX 能拿到，DOCX 拿不到（流式布局无页面
            几何）→ 传 None，前端降级为只显示页码 + 序号。
        :param source_ordinal: 这张图在**来源枚举顺序**里的位置（1-based）。
            被过滤掉的图不占 self.images 的名额，调用方若要按原文位置回填
            图片标注，就必须显式传这个序号（见 ``self.source_ordinals``）。

        Returns True when searchable text was produced for it. The image is
        appended to ``self.images`` (and written to disk) whenever it passes
        the usability + dedup filters, regardless of the return value.
        """
        if not self.enabled or len(self.images) >= self.max_images:
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

            width, height = img.size
            page_number = int(page_number or 1)
            self._page_counters[page_number] = self._page_counters.get(page_number, 0) + 1
            index = self._page_counters[page_number]
            image_id = self._make_image_id(page_number, index)
            # 文档级序号：只对"真的被收下"的图片递增，被过滤/去重的不占号，
            # 保证 position 连续且可复现。
            self._position += 1
            position = self._position

            # ── 1. 落盘原图（部分2）────────────────────────────────────────
            # PNG 只编码一次：落盘与 Vision 复用同一份字节。
            png_bytes: bytes | None = None
            image_path: str | None = None
            if self.save_enabled:
                png_bytes = encode_png(img)
                image_path = save_image(
                    str(self.document_id), page_number, index, png_bytes, "png",
                    tenant_id=self.tenant_id,
                )

            # ── 2. 三层图片处理：OCR → 分类 → 分流 → 结构化 ─────────────────
            understanding = _safe_understand(
                img,
                page_number=page_number,
                filename=self.filename,
                png_bytes=png_bytes,
            )

            extracted = ExtractedImage(
                image_id=image_id,
                page_number=page_number,
                ocr_text=understanding.ocr_text,
                vision_caption=understanding.vision_caption,
                image_path=image_path,
                width=width,
                height=height,
                ocr_engine=understanding.ocr_engine,
                # ── 位置信息（细粒度引用）────────────────────────────────────
                position=position,
                bbox=_normalize_bbox(bbox),
                # ── 分类与结构化 ────────────────────────────────────────────
                image_type=understanding.image_type,
                structured_content=understanding.structured_content,
                analyze_engine=understanding.analyze_engine,
                classify_engine=understanding.classification.engine,
                classification_signals={
                    k: v for k, v in understanding.classification.signals.items()
                },
                # ── 置信度门控结果 ──────────────────────────────────────────
                analyze_confidence=understanding.confidence,
                analyze_decision=understanding.decision,
                manual_review=understanding.manual_review,
                # ── 产出质检 + 双通道融合（可验证的事实，前端据此打警示标）──
                analyze_quality=dict(understanding.quality),
                analyze_fusion=dict(understanding.fusion),
            )
            self.images.append(extracted)
            # 与 self.images 同步增长（两者必须等长，索引才一一对应）
            self.source_ordinals.append(source_ordinal)

            self.type_counts[understanding.image_type] = (
                self.type_counts.get(understanding.image_type, 0) + 1
            )
            self.route_counts[understanding.route] = (
                self.route_counts.get(understanding.route, 0) + 1
            )
            if understanding.ocr_engine:
                self.engines.add(understanding.ocr_engine)

            # ── 3. 兼容旧路径的合并文本 ─────────────────────────────────────
            text = extracted.searchable_text.strip()
            if not text:
                return False
            self.texts.append(text)
            return True
        except Exception as exc:      # noqa: BLE001
            logger.warning(
                "Embedded-image recognition failed in '%s': %s", self.filename, exc
            )
            return False

    def _make_image_id(self, page_number: int, index: int) -> str:
        """Deterministic, human-readable image id."""
        if self.document_id:
            base = str(self.document_id)
        else:
            base = hashlib.sha1(self.filename.encode("utf-8", "ignore")).hexdigest()[:12]
        return f"{base}-p{int(page_number)}-i{int(index)}"

    def register_page_scan(
        self,
        img: Image.Image,
        page_number: int,
        ocr_text: str,
        *,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> str | None:
        """
        Persist a rendered page (scanned page / image-only page) as an image object.

        A scanned page IS an image, so it gets the same treatment as an embedded
        picture: the original render is saved to disk and can be returned as the
        source picture. Its OCR text becomes the image's searchable text.

        注意：整页扫描**不走**类型分流 —— 页面级 OCR 文本已由 PDF 解析器给出，
        再对整页跑一遍"表格结构识别"既慢又容易把整页误判成一张大表。
        因此这里固定按"普通图片 + OCR"入库，保留原图回显能力。

        :param bbox: 整页扫描的边界框 = 整页矩形（由调用方按页面尺寸给出）。

        Returns the stored relative path (or None).
        """
        if not self.enabled or len(self.images) >= self.max_images:
            return None
        try:
            img = _to_rgb(img)
            settings = get_settings()
            page_number = int(page_number or 1)
            self._page_counters[page_number] = self._page_counters.get(page_number, 0) + 1
            index = self._page_counters[page_number]
            self._position += 1

            image_path: str | None = None
            if self.save_enabled:
                image_path = save_image(
                    str(self.document_id), page_number, index, encode_png(img), "png",
                    tenant_id=self.tenant_id,
                )

            text = (ocr_text or "").strip()
            text = text[: settings.MAX_IMAGE_OCR_CHARS] if text else ""

            self.images.append(
                ExtractedImage(
                    image_id=self._make_image_id(page_number, index),
                    page_number=page_number,
                    ocr_text=text,
                    vision_caption=None,
                    image_path=image_path,
                    width=img.size[0],
                    height=img.size[1],
                    # 位置信息：整页扫描同样占据一个文档级序号
                    position=self._position,
                    bbox=_normalize_bbox(bbox),
                    image_type=IMAGE_TYPE_PHOTO,
                    analyze_engine="ocr",
                    classify_engine="page-scan",
                )
            )
            # 整页扫描不对应正文里的任何"第 N 张内嵌图" → 来源序号为空。
            # 必须与 self.images 同步追加，否则两个列表会错位。
            self.source_ordinals.append(None)
            self.type_counts[IMAGE_TYPE_PHOTO] = self.type_counts.get(IMAGE_TYPE_PHOTO, 0) + 1
            return image_path
        except Exception as exc:      # noqa: BLE001
            logger.warning(
                "Failed to persist page scan for '%s' page %d: %s",
                self.filename, page_number, exc,
            )
            return None

    def merged_text(self, *, section_header: str = "图片识别内容") -> str:
        """
        Render all recognised image texts as one block (legacy merge path).

        Empty string when nothing was recognised.
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


__all__ = [
    "EmbeddedImageRecognizer",
    "encode_png",
    "caption_with_vision",
    "_is_usable_image",
    "_to_rgb",
    "IMAGE_MARKER_PREFIX",
]
