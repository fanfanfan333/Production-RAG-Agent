from dataclasses import dataclass, field
import abc

from app.utils.logging import get_logger

logger = get_logger(__name__)


# 解码失败字符（U+FFFD）占比超过该阈值即认为"解码彻底失败"，直接拒收——
# 与其把一坨乱码写进向量库，不如报错让用户换编码 / 转 .docx。
# 与 doc_parser._MAX_REPLACEMENT_RATIO 同一口径（0.05）。
_MAX_REPLACEMENT_RATIO = 0.05

# 编码回退链：先试带 BOM 的 UTF-8（Windows 记事本默认），再试中文最常遇到的
# GBK/GB18030，再试繁体 Big5，最后才用 UTF-8 + replace（replace 会把解码不了的
# 字节变成 U+FFFD，全篇中文会变成一堆不可见方块，比直接报错更糟）。
_TEXT_FALLBACK_ENCODINGS = ("utf-8-sig", "gb18030", "big5")


def decode_text_with_fallback(content: bytes, *, filename: str = "") -> str:
    """
    把文件字节解码成文本，按回退链尝试常见中文编码.

    返回解码后的文本。命中非 UTF-8 编码时打 warning（可见的降级），若最终
    U+FFFD 占比超过阈值则抛 ``ValueError`` 拒收，避免静默乱码入库。

    旧实现直接 ``content.decode("utf-8", errors="replace")``：GBK 中文会被整篇
    替换成 U+FFFD 且**不报错不告警**，是"看着成功了其实坏了"的典型。
    """
    # 先试 utf-8（是否带 BOM 都覆盖），用 replace 以便统计失败字符占比
    text, _ = _try_decode(content, "utf-8", errors="replace")
    if "\ufffd" not in text:
        return text

    # UTF-8 含替换字符 → 依次试带 BOM / GBK / Big5（strict：失败即换下一个）
    for enc in _TEXT_FALLBACK_ENCODINGS:
        candidate, _ = _try_decode(content, enc, errors="strict")
        if candidate is not None:
            logger.warning(
                "文本 '%s' 非 UTF-8，已按 %s 解码（UTF-8 解码含替换字符）",
                filename, enc,
            )
            return candidate

    # 回退链都失败：用 utf-8 + replace 兜底，但检查乱码占比，过高直接拒收
    replacement_ratio = text.count("\ufffd") / max(1, len(text))
    if replacement_ratio > _MAX_REPLACEMENT_RATIO:
        raise ValueError(
            f"文件「{filename}」解码失败（U+FFFD 占比 {replacement_ratio:.0%} 超过 "
            f"{_MAX_REPLACEMENT_RATIO:.0%}）：可能是非 UTF-8 编码或非文本文件，"
            f"请转成 UTF-8 后重新上传。"
        )
    logger.warning(
        "文本 '%s' 按 UTF-8(replace) 解码，含 %.0f%% 替换字符，可能部分乱码",
        filename, replacement_ratio * 100,
    )
    return text


def _try_decode(content: bytes, encoding: str, *, errors: str) -> tuple[str | None, str | None]:
    """
    尝试用 *encoding* 解码；errors="strict" 失败时返回 (None, None)，
    errors="replace" 总是成功并顺带返回是否含替换字符。
    """
    try:
        return content.decode(encoding, errors=errors), encoding
    except (UnicodeDecodeError, LookupError):
        return None, encoding


@dataclass
class ExtractedPage:
    """Text content and position metadata for a single page or section."""
    page_number: int        # 1-indexed
    text: str
    char_start: int         # offset of this page's text in the full document string
    char_end: int           # exclusive end offset

@dataclass
class ExtractedImage:
    """
    一张从文档中抽出的内嵌图片（部分1：图片不再是"纯文本"）.

    与历史实现的关键区别：历史上内嵌图片的 OCR 文本被直接拼进文档正文，
    图片本体被丢弃 —— 图片因此退化成一个普通文本块，既无法回显原始图片，
    也无法作为独立检索对象。现在每张图片都被抽成一个结构化对象：

        image_id       稳定唯一 ID（document_id + 页码 + 序号）
        image_path     落盘相对路径 uploads/{document_id}/images/page_3_image_1.png
        page_number    所在页码（1-indexed）
        ocr_text       图片内文字的 OCR 结果（可为空串）
        vision_caption 多模态模型对图片的语义描述（未开启 vision 时为 None）

    ── 三层图片处理（Image → Classification → 分流）──────────────────────────
    图片进入文档后先判类型再分流，因此还带上：

        image_type          table | formula | code | chart | diagram | screenshot | photo
        structured_content  结构化还原结果：表格图片 → Markdown 表格，
                            公式 → LaTeX，代码截图 → 围栏代码块
        analyze_engine      真正产出内容的引擎：table-transformer | table_parser |
                            ocr+code-parser | paddleocr-formula | vision | paddleocr |
                            tesseract | ocr-degraded

    该对象会被 document_service 转成独立的 image chunk 写入 Qdrant，
    并在检索命中后回显原图（部分2 / 部分3）。
    """

    image_id: str
    page_number: int
    ocr_text: str = ""
    vision_caption: str | None = None
    image_path: str | None = None       # None = 未落盘（保存关闭或写盘失败）
    width: int | None = None
    height: int | None = None
    ocr_engine: str | None = None

    # ── 位置信息（保留图片位置 → 细粒度引用）──────────────────────────────────
    # position：该图片在**整篇文档**中的序号（1-based，跨页累计）。
    #   与 image_id 里的页内序号是两回事 —— 后者是 "第 3 页的第 1 张"，
    #   前者回答"这是整份文档里的第几张图"，引用时更贴近人的说法。
    # bbox：图片在所在页面上的边界框 (x1, y1, x2, y2)，单位为点（1/72 英寸），
    #   原点在页面左上角，y 轴向下。PDF 由 ``page.get_image_rects()`` 得到，
    #   PPTX 由形状的 left/top/width/height 换算得到。
    #   DOCX 是**流式布局、没有页面几何**，因此恒为 None —— 前端据此降级为
    #   只显示"第几页 + 第几张图"，不假装有坐标。
    position: int | None = None
    bbox: tuple[float, float, float, float] | None = None

    # ── 分类与结构化（三层图片处理 + 置信度门控）──────────────────────────────
    image_type: str = "photo"
    structured_content: str | None = None
    analyze_engine: str | None = None
    classify_engine: str | None = None
    classification_signals: dict = field(default_factory=dict)
    # 门控结果：最终置信度 / 走过哪条路（accept|fallback|pass|manual_review）
    analyze_confidence: float = 0.0
    analyze_decision: str | None = None
    # True = 兜底后仍未通过校验，需要人工复核（前端会打标）
    manual_review: bool = False
    # ── 产出质量校验与双通道融合（新增）────────────────────────────────────
    # analyze_quality：QualityReport.to_dict() —— 代码语法是否通过、OCR 行置信度
    #   评估、VLM 幻觉检查（提示词泄漏/复读/数字锚点）、结构校验，以及总评分。
    #   前端据此显示"代码语法未通过""OCR 置信度偏低"这类**可验证**的警示，
    #   而不是只给一个说不清来源的置信度数字。
    # analyze_fusion：FusionResult.to_dict() —— 双通道策略（互补/视觉优先/
    #   OCR 优先）、最终选了哪个通道、两个通道各自的产出与质检、融合说明。
    analyze_quality: dict = field(default_factory=dict)
    analyze_fusion: dict = field(default_factory=dict)

    @property
    def content_type(self) -> str:
        """
        该图片入库时的 content_type.

        表格图片产出的是**真表格**（Markdown 网格），因此 content_type 取
        "table" —— "表格检索"对图片表格与正文表格一视同仁。其余类型是 "image"，
        检索期由 multimodal 节点按"原图 + Vision"处理。
        """
        if self.image_type == "table" and (self.structured_content or "").strip():
            return "table"
        return "image"

    def _ocr_is_trustworthy(self) -> bool:
        """
        OCR 文本是否值得进检索正文.

        质检（``analyze_quality``）已经算过"行置信度均值/低置信占比"并给出
        ``ocr.passed``。这里把它接上：只有当质检**明确判否**时才算不可信。
        没有质检信息（旧数据 / 未开启质检）时一律按可信处理，避免误伤。

        触发场景（实测）：中文文档里 Tesseract(eng) 认出的图形标题会变成
        ``H |=+2 x padding|0|—dilation...`` 这类噪声，置信度均值 0.44、
        67% 的行低于 0.60。这段噪声与高质量 Vision 描述一起被拼进正文后，
        既稀释了嵌入语义，又给 BM25 灌进一堆假词。
        """
        quality = self.analyze_quality or {}
        ocr_report = quality.get("ocr") if isinstance(quality, dict) else None
        if isinstance(ocr_report, dict) and ocr_report.get("passed") is False:
            return False
        return True

    @property
    def searchable_text(self) -> str:
        """
        该图片用于向量 / BM25 检索的文本表示.

        优先级体现"先结构化、后原文"：
            1. structured_content —— 表格图片的 Markdown 单元格网格
               （已含全部文字，不能再叠 OCR 文本，否则同一份内容被重复计权）
            2. ocr_text           —— 图内文字（普通图片的主信号）
            3. vision_caption     —— 图义描述（图表 / 流程图的"结构化描述"）

        低质 OCR 的处置：若质检判定 OCR 不合格**且**有 vision caption 兜底，
        则丢弃 OCR 文本只留描述 —— 用一段读得懂的语义描述，换掉一串认错的字符。
        没有 caption 时仍保留 OCR（聊胜于无，且删掉就等于这张图彻底检索不到）。
        """
        parts: list[str] = []
        structured = (self.structured_content or "").strip()
        if structured:
            parts.append(structured)
        if not structured and self.ocr_text:
            if self._ocr_is_trustworthy() or not self.vision_caption:
                parts.append(self.ocr_text)
        if self.vision_caption:
            parts.append(f"图片描述: {self.vision_caption}")
        return "\n".join(parts)


@dataclass
class ExtractionResult:
    """Result returned by extractors."""
    pages: list[ExtractedPage] = field(default_factory=list)
    full_text: str = ""
    page_count: int = 0
    char_count: int = 0

    # Metadata about how it was extracted
    file_type: str = "unknown"
    parser_used: str = "unknown"
    ocr_used: bool = False
    ocr_engine: str | None = None
    extraction_method: str = "native"

    # 表格类文档（csv/xlsx）标记为 True —— 其 chunk 会被标记
    # content_type="table"，检索结果可据此区分"表格检索"（表格检索能力）。
    is_tabular: bool = False

    # ── 结构化内嵌图片（部分1）─────────────────────────────────────────────
    # 每张可用的内嵌图片抽成一个 ExtractedImage。图片本体已落盘到
    # uploads/{document_id}/images/，由 document_service 转成独立 image chunk。
    # 注意：当 IMAGE_AS_INDEPENDENT_OBJECT 开启时，图片文本 **不再** 拼进
    # full_text —— 图片是独立检索对象，不是正文的一部分。
    images: list[ExtractedImage] = field(default_factory=list)

    # 兼容字段：仍保留供旧调用方 / 统计 / 日志使用
    image_count: int = 0                 # embedded images recognised
    image_texts: list[str] = field(default_factory=list)

    def page_for_offset(self, char_offset: int) -> int:
        """Return the 1-indexed page number that contains the given character offset."""
        for page in self.pages:
            if page.char_start <= char_offset < page.char_end:
                return page.page_number
        # Fallback: return last page
        return self.pages[-1].page_number if self.pages else 1


class DocumentParser(abc.ABC):
    """
    Abstract base class for all document parsers.
    """

    @abc.abstractmethod
    def parse(
        self,
        content: bytes,
        filename: str,
        *,
        document_id: str | None = None,
        tenant_id: str | None = None,
    ) -> ExtractionResult:
        """
        Parse raw bytes and return an ExtractionResult.

        *document_id*（可选）用于给内嵌图片落盘并生成稳定 image_id：
        ``uploads/{tenant_id}/{document_id}/images/page_3_image_1.png``
        （第三层：图片按租户隔离；*tenant_id* 缺省时走旧布局
        ``uploads/{document_id}/``）。为 None 时图片只被识别、不落盘
        （向后兼容旧调用方式）。
        """
        pass
