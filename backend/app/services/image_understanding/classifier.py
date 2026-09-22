"""
Picture Classification（图片类型判断）.

设计稿的第一层：

    Image → Picture Classification → Table → Table Structure Recognition → 结构化表格

判断本身**不依赖任何模型**，只用三条线上下文的确定性信号：

    1. 版面线条   —— 横/竖"ruling line"投影（表格最强信号）
    2. 色彩统计   —— 饱和度、大面积同色块（图表信号）、近白/近黑占比（线稿信号）
    3. OCR 版面   —— 文字行的行/列对齐度（无边框表格的兜底信号）

为什么坚持规则而不是让多模态模型来分类？
    · 分类发生在**入库期**，一张文档可能有几十张图，逐张调模型会让入库慢一个量级；
    · 设计稿明确要求"没有多模态模型就先放弃 Vision"，分类若依赖 Vision，
      那么"表格 → Table Parser"这条最有价值的支路在无模型时也会一起失效；
    · 规则信号可解释、可单测、可调参（signals 会一并返回，便于排查误判）。

需要更准时可把 ``IMAGE_CLASSIFIER_ENGINE`` 设为 ``rules+vision``，
在规则判不出来（photo/diagram 边界模糊）时用多模态模型复核。

判定顺序（与设计稿伪代码一致）：table → chart → diagram → 其他（screenshot/photo）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from app.config import get_settings
from app.services.image_understanding.imaging import (
    background_is_dark,
    bands,
    count_bands,
    ink_profiles,
    line_art_ratio,
    to_gray_pixels,
)
from app.services.image_understanding.structured_content import (
    ALL_IMAGE_TYPES,
    IMAGE_TYPE_CHART,
    IMAGE_TYPE_CODE,
    IMAGE_TYPE_DIAGRAM,
    IMAGE_TYPE_FORMULA,
    IMAGE_TYPE_PHOTO,
    IMAGE_TYPE_SCREENSHOT,
    IMAGE_TYPE_TABLE,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 分析用的缩略图长边上限。分类只需要版面/色彩统计，不需要原始分辨率；
# 但也不能压得太狠 —— 表格框线通常只有 2~3px 宽（见 make_doc_photo_fixture），
# 把 1000px 的图缩到 256（因子 ~3.9）会让细线在 BILINEAR 重采样时被相邻白底
# 平均掉、只剩零星几行还够"墨"，实测 t0 的 6 条横线只活下来 2 条 —— 一张
# 铺满整页的表格因此够不上 framed，被判成 diagram/screenshot（2026-09 实测）。
# 384 是这个权衡的落点：2px 线稳定存活、纯 Python 像素遍历仍在几十毫秒级。
_GRID_MAX = 384

# 一条线"墨"占**内容区**的比例超过该值才算表格线（相对于整幅图的旧口径已废弃）
_RULE_RATIO = 0.55
# 一条横线最多允许的间断（把粗细不均的线合并成一条）
_RULE_MAX_GAP = 2
# 一条"线"在分析分辨率（≤256）下的最大**厚度**（像素）。这是把"表格框线"与
# "实心色块/文字行"区分开的关键：框线细（1~3px），柱状图的柱子、密集文字行都
# 厚（≥5px）。只认细带，能从根上堵掉"柱状图被当成表格"（柱顶横边/柱身竖边
# 又长又厚，旧口径把它们当成了表格线）。
_RULE_MAX_THICK = 4
# "文字行"检测：一行里墨占比超过该值即认为存在一排文字
_TEXT_ROW_RATIO = 0.10
# 文字行的最大间断（字形间隙比线条大，放宽到 2）
_TEXT_ROW_MAX_GAP = 2

# ── 表格 vs 图表 / 界面截图的判别阈值（2026-09 重标定，见 _decide）────────────
# 表格是"无彩色"的：柱/饼/散点图必有明显彩色面积。彩色像素占比 ≥ 该值即视为
# 图表，不再考虑判成表格（实测柱状图 0.18/0.20，全部表格夹具一律 0.0）。
# 可用 settings.TABLE_MAX_COLORFUL_RATIO 覆盖。
_TABLE_MAX_COLORFUL_RATIO = 0.12
# 无框线表格的 OCR 兜底：列稳定性（每列被多少行支撑的均值）下限。真表格的列
# 被绝大多数行填充；UI 截图/图表的文字左边界零散、支撑率低。0.20 是真实管线
# （含预处理）两类的中间点：倾斜表格 t5=0.219，非表格侧 ≤0.18（ui_documents 等）。
# 可用 settings.TABLE_OCR_MIN_COL_STABILITY 覆盖。
_TABLE_OCR_MIN_COL_STABILITY = 0.20

# ── 彩色结构示意图（diagram）的兜底判据（2026-09，见 _decide 第 6.5 步）──────────
# 背景：一张**结构示意图**（三栏对比图、带彩色标题条 / 卡片面板的示意图）会同时
# 踩空既有的三条判据 ——
#   · 它的彩色面积够不上图表（图表要求 colorful_ratio ≥ 0.15），
#   · 它的 line_art_ratio 又够不上流程图 / 结构图的线稿判据（要求 ≥ 0.8），
# 于是掉进最后的 "default" → photo → 走 OCR、**永远**拿不到图意总结。实测
# 「企业知识库RAG系统建设实施手册.docx」的第 3 张图就是这样丢掉了 image_caption
# （Qdrant payload image_caption=None）。
# 这里补一条"结构图"兜底：**统一背景 + 可辨结构 + 彩度不足以判图表**即认作
# diagram（→ 走多模态，拿得到图意描述）。用已有信号即可判定，无需新模型。
#
# 关于"确有结构"的析取项：**只认几何结构**（粗带 / 彩色描边块 / 细直线），
# **不**把 ``ocr_rows >= 2`` 算进去。理由：一张"白底黑字、无线无色带"的**纯文字页**
# 翻拍件同样满足"统一背景 + 彩度低 + OCR 多行"，若把 OCR 行数当结构信号，就会把
# 纯文字页误升格成 diagram —— 而纯文字页的正解是 OCR（要文字），不是 vision
# （要图意）。这条边界由 test_image_structured_diagram.py 的红-绿用例守住。
#
# 阈值锚点（2026-09 真实管线实测，DOMINANT 高 / 彩度低 / 线稿占比中等是这类图的
# 共同特征；真实照片三条信号全部落在门外）：
#
#                    dominant  line_art  colorful  raw_bands  期望
#   image 3（真实）    0.4761    0.6519    0.0889    1          diagram
#   合成结构示意图      0.5823    0.7081    0.0971    8          diagram
#   _photo_image      0.1679    0.0000    1.0000    3          必须仍为 photo
#
# photo 的 dominant 极低（连续渐变里没有大面积同色块）、line_art≈0（连续色调中
# 几乎没有近白/近黑像素）、colorful≈1.0（高饱和）—— 所以"真实照片必须仍判 photo"
# 这条红线由信号本身保证，而不是靠把阈值调紧去碰运气。
_STRUCTURED_MIN_DOMINANT_RATIO = 0.45
_STRUCTURED_MIN_LINE_ART = 0.55
# 与图表判据的 colorful_ratio ≥ 0.15 互补：达到 0.15 就已是图表，这边只接"彩度
# 不足"的那一类。两个分支因此构成对彩色的干净二分，不会互相抢。
_STRUCTURED_MAX_COLORFUL_RATIO = 0.15


def _rule_count(
    profile: list[int], limit: int, ratio: float, max_gap: int, max_thick: int
) -> int:
    """
    数"细长直线"的条数（表格框线口径）.

    与 ``count_bands`` 的区别：**额外要求每条带足够薄**。文档表格的框线只有
    1~3px；柱状图的柱子（厚）、密集文字行（厚）都会被这条滤掉，从而不再
    被误当成表格线。
    """
    return sum(
        1
        for start, end in bands(profile, limit, ratio, max_gap)
        if (end - start + 1) <= max_thick
    )


@dataclass
class ImageClassification:
    """Picture Classification 的产出."""

    image_type: str
    confidence: float = 0.0
    signals: dict = field(default_factory=dict)
    engine: str = "rules"

    @property
    def is_table(self) -> bool:
        return self.image_type == IMAGE_TYPE_TABLE

    @property
    def is_chart(self) -> bool:
        return self.image_type == IMAGE_TYPE_CHART

    @property
    def is_diagram(self) -> bool:
        return self.image_type == IMAGE_TYPE_DIAGRAM

    @property
    def is_screenshot(self) -> bool:
        return self.image_type == IMAGE_TYPE_SCREENSHOT

    @property
    def is_photo(self) -> bool:
        return self.image_type == IMAGE_TYPE_PHOTO

    @property
    def is_code(self) -> bool:
        return self.image_type == IMAGE_TYPE_CODE

    @property
    def is_formula(self) -> bool:
        return self.image_type == IMAGE_TYPE_FORMULA

    def to_dict(self) -> dict:
        return {
            "image_type": self.image_type,
            "confidence": round(float(self.confidence), 3),
            "engine": self.engine,
            "signals": self.signals,
        }


def vision_analyze_types() -> set[str]:
    """需要走 Vision（看图理解）的图片类型集合."""
    raw = (get_settings().VISION_ANALYZE_TYPES or "").strip()
    if not raw:
        return set()
    wanted = {part.strip().lower() for part in raw.split(",") if part.strip()}
    return {t for t in wanted if t in ALL_IMAGE_TYPES}


# ─────────────────────────────────────────────────────────────────────────────
# 低层信号（像素投影 / 线条检测统一走 imaging，保证与表格识别口径一致）
# ─────────────────────────────────────────────────────────────────────────────


def _color_signals(img: Image.Image) -> dict:
    """
    色彩统计（用更小的网格，色彩只需要分布不需要细节）.

    返回：
        saturation        非白像素的平均饱和度
        flat_blocks       大面积同色块数量（≥3% 像素）
        chromatic_blocks  **有彩色**的大面积同色块数量 —— 柱/饼/折线的信号。
                          照片的量化噪声也会产生很多"同色块"，但它们零散且
                          背景不统一，因此还需要 dominant_ratio 一起判定。
        dominant_ratio    最大同色占比 —— 背景统一度（图表/截图/线稿都高）
        colorful_ratio    明显有彩色像素的占比
    """
    small = img.convert("RGB")
    width, height = small.size
    scale = 96.0 / float(max(width, height, 1))
    if scale < 1.0:
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
        small = small.resize((width, height), Image.BILINEAR)

    pixels = list(small.getdata())
    total = len(pixels) or 1

    sat_sum = 0.0
    colored = 0
    quantized: dict[tuple[int, int, int], int] = {}

    for r, g, b in pixels:
        mx, mn = max(r, g, b), min(r, g, b)
        sat = 0.0 if mx == 0 else (mx - mn) / float(mx)
        if mx > 40 and sat > 0.12:          # 忽略近白/近黑背景的噪声
            sat_sum += sat
            if sat > 0.25:
                colored += 1
        # 4 bit/通道量化，统计"同色块"
        key = (r >> 4, g >> 4, b >> 4)
        quantized[key] = quantized.get(key, 0) + 1

    flat = 0
    chromatic = 0
    for (r, g, b), count in quantized.items():
        if count / total < 0.03:
            continue
        flat += 1
        # 还原到 0-255 区间判断"这块色是不是彩色"（而不是白底/黑线）
        rr, gg, bb = (r << 4) + 8, (g << 4) + 8, (b << 4) + 8
        mx, mn = max(rr, gg, bb), min(rr, gg, bb)
        sat = 0.0 if mx == 0 else (mx - mn) / float(mx)
        if sat >= 0.2 and mx >= 60:
            chromatic += 1

    dominant = max(quantized.values()) / total if quantized else 0.0

    return {
        "saturation": round(sat_sum / total, 4),
        "flat_blocks": flat,
        "chromatic_blocks": chromatic,
        "dominant_ratio": round(dominant, 4),
        "colorful_ratio": round(colored / total, 4),
    }


def _ocr_layout(lines: list, width: int, height: int) -> dict:
    """
    由 OCR 行框推断版面：行数、列数、每行多单元格的比例.

    *lines* 是带 ``box=(x0,y0,x1,y1)`` 与 ``text`` 的对象列表（见
    ``app.services.ocr.base.OCRLine``）。没有坐标信息时退化为纯文本行统计。
    """
    boxes = []
    for line in lines or []:
        text = (getattr(line, "text", "") or "").strip()
        box = getattr(line, "box", None)
        if not text or not box:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append((x0, y0, x1, y1, text))

    if not boxes:
        return {
            "rows": 0, "cols": 0, "multi_cell_rows": 0,
            "alignment": 0.0, "col_stability": 0.0,
        }

    boxes.sort(key=lambda b: (b[1], b[0]))
    heights = sorted(b[3] - b[1] for b in boxes)
    median_h = heights[len(heights) // 2] or 1.0
    row_tol = median_h * 0.7

    # ── 行聚类（按 y 中心）────────────────────────────────────────────────
    rows: list[list[tuple[float, float, float, float, str]]] = []
    for item in boxes:
        cy = (item[1] + item[3]) / 2.0
        placed = False
        for row in rows:
            ref = sum((b[1] + b[3]) / 2.0 for b in row) / len(row)
            if abs(cy - ref) <= row_tol:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])
    for row in rows:
        row.sort(key=lambda b: b[0])

    # ── 列聚类（按左边界全局聚类）─────────────────────────────────────────
    lefts = sorted({round(b[0], 1) for b in boxes})
    col_tol = max(6.0, (width or 1) * 0.025)
    column_starts: list[float] = []
    for x in lefts:
        if not column_starts or x - column_starts[-1] > col_tol:
            column_starts.append(x)

    multi_cell = sum(1 for row in rows if len(row) >= 2)
    alignment = multi_cell / len(rows) if rows else 0.0

    # ── 列稳定性（区分"真表格"与"UI 截图/图表"的关键，2026-09 新增）──────────
    # alignment 只看"每行有几个单元格"，对界面截图同样很高 —— 聊天/文档列表
    # 页面上并排的文字一样会形成"多单元格行"。真表格的额外特征是"同一组列被
    # **每一行**反复使用"：每一列都能在多数行里找到落在它附近的单元格。UI 的
    # 文字左边界零散分布，列被支撑的比例自然低。实测：t0..t8 = 0.40~0.80，
    # 界面截图/图表 ≤0.29，阈值 0.35 两边都留足余量（见 TABLE_OCR_MIN_COL_STABILITY）。
    if rows:
        support = [
            sum(1 for row in rows if any(abs(b[0] - col) <= col_tol for b in row))
            for col in column_starts
        ]
        col_stability = round(sum(support) / len(support) / len(rows), 4)
    else:
        col_stability = 0.0

    return {
        "rows": len(rows),
        "cols": len(column_starts),
        "multi_cell_rows": multi_cell,
        "alignment": round(alignment, 4),
        "col_stability": col_stability,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────


def compute_signals(img: Image.Image, *, ocr_lines: list | None = None) -> dict:
    """算出全部判定信号（分离出来便于调试与单测）."""
    pixels, width, height = to_gray_pixels(img, _GRID_MAX)
    # 极性（哪一侧是纸）先定，投影才可能正确。**不要**改回"全图多数像素"：
    # 白底页面里嵌一块大面积深色图时那条规则会把白色页边距当成墨迹，投影里
    # 凭空多出贯穿全图的竖线，流程图随即被判成表格（见 imaging.background_is_dark）。
    inverted, polarity_source = background_is_dark(pixels, width, height)
    row_ink, col_ink, _ = ink_profiles(pixels, width, height, inverted=inverted)

    # ── 表格框线：**相对内容区**、且只认**细线**（2026-09 重标定）────────────
    # 旧口径把"墨占整幅图的比例 ≥0.55"当表格线，于是两类失效同时发生：
    #   · 有页边距/标题的表格，竖线只占画幅 40% → v_lines=0 → 表被判成 diagram
    #     /screenshot（连铺满整页的 t0 都因子表只到 2 条横线而不达标）；
    #   · 柱状图柱子又长又厚，却满足"整幅图比例" → 图表被当成表格。
    # 新口径改为：线的长度相对**内容外接框**（有墨的行/列范围），并只统计**细**
    # 带（≤_RULE_MAX_THICK）。表格框线细且贯穿内容区 → 命中；柱子/文字行粗 →
    # 被厚度滤掉；流程图小方框的边短 → 不达标。
    content_w = sum(1 for value in col_ink if value > 0) or width
    content_h = sum(1 for value in row_ink if value > 0) or height
    h_lines = _rule_count(row_ink, content_w, _RULE_RATIO, _RULE_MAX_GAP, _RULE_MAX_THICK)
    v_lines = _rule_count(col_ink, content_h, _RULE_RATIO, _RULE_MAX_GAP, _RULE_MAX_THICK)
    # "粗带"（不滤厚度，旧口径的等价物）—— 专供**流程图/结构图**判据使用：
    # 流程图的大色块/面板会形成又长又厚的带，线条判据刻意把它滤掉（否则图表
    # 会被当表格），但"图里有一整块结构"这件事本身仍是 diagram 的正信号。
    raw_h_bands = count_bands(row_ink, width, _RULE_RATIO, _RULE_MAX_GAP)
    raw_v_bands = count_bands(col_ink, height, _RULE_RATIO, _RULE_MAX_GAP)
    # 文字行密度（**不依赖 OCR**）：代码截图 / 文档扫描页会产生大量"有墨"的行。
    # 这是把"截图"和"线稿流程图"区分开的关键 —— 两者 line_art_ratio 都高，
    # 区别只在"到底有多少排文字"。
    text_bands = count_bands(row_ink, width, _TEXT_ROW_RATIO, _TEXT_ROW_MAX_GAP)
    color = _color_signals(img)
    layout = _ocr_layout(ocr_lines or [], width, height)
    text_stats = _text_signals(ocr_lines or [])

    return {
        "width": width,
        "height": height,
        "h_lines": h_lines,
        "v_lines": v_lines,
        # 内容外接框（有墨的行/列数）—— 表格框线判据的分母，排查"为何没判成表"
        # 时第一眼就该看它：内容框远小于画幅 = 图有页边距/标题。
        "content_w": content_w,
        "content_h": content_h,
        "raw_h_bands": raw_h_bands,
        "raw_v_bands": raw_v_bands,
        "inverted": inverted,
        # 极性是怎么判出来的（page-ring / global-majority）—— 排查"这张图
        # 为什么被判成表格"时，第一眼就该看这两个字段。
        "polarity_source": polarity_source,
        "text_bands": text_bands,
        "line_art_ratio": line_art_ratio(pixels),
        "saturation": color["saturation"],
        "flat_blocks": color["flat_blocks"],
        "chromatic_blocks": color["chromatic_blocks"],
        "dominant_ratio": color["dominant_ratio"],
        "colorful_ratio": color["colorful_ratio"],
        "ocr_rows": layout["rows"],
        "ocr_cols": layout["cols"],
        "ocr_multi_cell_rows": layout["multi_cell_rows"],
        "ocr_alignment": layout["alignment"],
        # 列稳定性：无框线表格的 OCR 兜底判据（区分真表格与界面截图），见 _ocr_layout
        "ocr_col_stability": layout["col_stability"],
        # ── 公式 / 代码（文本层信号，OCR 不可用时全部为 0）─────────────────
        "code_score": text_stats["code_score"],
        "code_language": text_stats["code_language"],
        "math_ratio": text_stats["math_ratio"],
        "avg_line_chars": text_stats["avg_line_chars"],
        # 中文（CJK）字符占比：公式判定的护栏。括号、百分号这类符号普通中文
        # 段落里也很常见，只有"几乎没有汉字"才敢当公式。
        "cjk_ratio": text_stats["cjk_ratio"],
    }


#: 数学符号（公式判定）。取自 Unicode 数学运算符/希腊字母的常见子集，外加
#: LaTeX 里最常出现的 ASCII 组合。PaddleOCR 的中文模型对希腊字母识别不错，
#: 因此这条规则在中文知识库里也能生效。
_MATH_CHARS = set(
    "=+−-×÷±∓⋅·∗∑∏∫∮√∝∞≈≠≤≥≪≫∈∉⊂⊃∪∩∀∃∂∇"
    "αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"
    "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉→←↔⇒⇔"
    # ASCII 运算符与括号。**这一行是补上的**：此前集合里只有上标/希腊字母一类
    # 少见符号，斜杠、星号、括号、百分号一律不计入，于是
    #
    #     y = (a+b)/(c-d) * 100% = 12.5
    #
    # 这种一眼就是公式的图，math_ratio 只算出 0.17（只有 2 个 = 加 1 个 + / -），
    # 卡在 0.18 阈值下方 → 被判成"普通图片"走整图 OCR。实测复现于
    # `_audit_916/audit_parse.py` 的公式 fixture。
    "()[]{}/\\*^_%"
)


def _text_signals(ocr_lines: list) -> dict:
    """
    从 OCR 文本行里提取"公式 / 代码"判据.

    公式：**符号占比高 + 行短 + 几乎没有汉字**。正文里的 "=" 通常夹杂大量汉字，
    符号占比很低；公式行反过来 —— 符号多、字少、无成句的中文。
    """
    texts = [(getattr(l, "text", "") or "").strip() for l in ocr_lines]
    texts = [t for t in texts if t]
    if not texts:
        return {
            "code_score": 0.0, "code_language": "text", "math_ratio": 0.0,
            "avg_line_chars": 0.0, "cjk_ratio": 0.0,
        }

    from app.services.image_understanding.engines.code_parser import (
        code_likeness,
        detect_language,
    )

    blob = "".join(texts)
    non_space = [c for c in blob if not c.isspace()]
    math_hits = sum(1 for c in non_space if c in _MATH_CHARS)
    math_ratio = round(math_hits / len(non_space), 4) if non_space else 0.0
    # 汉字占比：把 ASCII 括号/百分号也算进"数学符号"之后，光看符号密度会把
    # "（见附录 A）占比 30%" 这类中文句子也算成公式，必须再加一道"几乎无汉字"
    # 的护栏。公式截图里出现成句中文的概率极低，这道护栏代价小、收益明确。
    cjk_hits = sum(1 for c in non_space if "\u4e00" <= c <= "\u9fff")
    cjk_ratio = round(cjk_hits / len(non_space), 4) if non_space else 0.0

    language, _ = detect_language(texts)
    return {
        "code_score": code_likeness(texts),
        "code_language": language,
        "math_ratio": math_ratio,
        "avg_line_chars": round(len(blob) / len(texts), 2),
        "cjk_ratio": cjk_ratio,
    }


def classify_image(
    img: Image.Image,
    *,
    ocr_lines: list | None = None,
    filename: str = "",
) -> ImageClassification:
    """
    判断一张图片属于哪一类（table / chart / diagram / screenshot / photo）.

    *ocr_lines* 是可选的带坐标 OCR 行（``OCRLine``）。传进来能显著提高
    无边框表格的召回率；不传也能工作（只用版面 + 色彩信号）。
    """
    settings = get_settings()
    if not getattr(settings, "IMAGE_CLASSIFICATION_ENABLED", True):
        # 关闭分类 → 全部按普通图片走 OCR（旧行为）
        return ImageClassification(
            image_type=IMAGE_TYPE_PHOTO, confidence=0.0,
            signals={"classification": "disabled"}, engine="disabled",
        )

    signals = compute_signals(img, ocr_lines=ocr_lines)
    image_type, confidence, reason = _decide(signals, settings)
    signals["reason"] = reason

    logger.info(
        "Image classification%s: %s (conf=%.2f, reason=%s) lines=%d/%d align=%.2f sat=%.2f flat=%d",
        f" [{filename}]" if filename else "",
        image_type, confidence, reason,
        signals["h_lines"], signals["v_lines"], signals["ocr_alignment"],
        signals["saturation"], signals["flat_blocks"],
    )
    return ImageClassification(
        image_type=image_type, confidence=confidence, signals=signals, engine="rules",
    )


def _decide(signals: dict, settings) -> tuple[str, float, str]:
    """
    规则判定（顺序即设计稿伪代码的顺序：table → chart → diagram → 其他）.

    返回 (类型, 置信度, 判定依据)。

    调参经验（都踩过坑）：
      · 图表不能只看"有彩色块" —— 照片量化后同样会碎成很多色块。真正的判别
        信号是"**少数几块**大面积彩色 + **统一背景**"，因此要求
        chromatic_blocks 在 2–8 之间且 dominant_ratio 足够高。
      · 流程图的小方框（宽 100px / 图宽 520px）达不到"整行都是墨"的阈值，
        所以竖线往往检测不到。判定条件因此放宽到 (h_lines+v_lines) >= 2，
        但要配合 line_art_ratio 高，避免把照片卷进来。
    """
    h_lines = signals["h_lines"]
    v_lines = signals["v_lines"]
    min_h = getattr(settings, "TABLE_MIN_H_LINES", 3)
    min_v = getattr(settings, "TABLE_MIN_V_LINES", 2)
    # 粗带（含大色块/面板）：图表"有坐标轴/结构"与流程图"有一整块结构"的旁证。
    # 缺省回退到细线计数 —— 手工拼 signals 的用例（test_dark_node_diagram_not_chart）
    # 不带 raw_* 键，这样它们的行为与旧版一致。
    raw_bands = signals.get("raw_h_bands", h_lines) + signals.get("raw_v_bands", v_lines)
    # 表格是"无彩色"的（黑字灰线白底）。柱/饼/散点图必有明显彩色面积 ——
    # 用它把"彩色图表被判成表格"堵掉（实测柱状图 colorful_ratio 0.18/0.20，
    # 全部表格夹具一律 0.0）。缺省 0.0 保证手工拼 signals 的旧用例不受影响。
    colorful_ratio = signals.get("colorful_ratio", 0.0)
    max_colorful = getattr(settings, "TABLE_MAX_COLORFUL_RATIO", _TABLE_MAX_COLORFUL_RATIO)
    is_colorful = colorful_ratio >= max_colorful

    # ── 1. 表格：有真实框线，或 OCR 版面呈现稳定的多列对齐 ────────────────
    framed = h_lines >= min_h and v_lines >= min_v and not is_colorful
    grid_by_layout = (
        signals["ocr_rows"] >= 3
        and signals["ocr_cols"] >= 2
        and signals["ocr_alignment"] >= 0.6
        and signals["ocr_multi_cell_rows"] >= 2
        # 列稳定性：真表格的列被绝大多数行支撑；界面截图/图表的文字左边界零散，
        # 支撑率低（实测表格 ≥0.40，UI/图表 ≤0.29）。这是把"UI 截图被判成表格"
        # 堵在门外的那道判据（单看 alignment 分不开：t5=0.71 vs ui_documents=0.73）。
        and signals.get("ocr_col_stability", 0.0)
        >= getattr(settings, "TABLE_OCR_MIN_COL_STABILITY", _TABLE_OCR_MIN_COL_STABILITY)
        and not is_colorful
    )
    if framed:
        # 线越多越像表；横线权重更高（表格横线通常多于竖线）
        confidence = min(0.98, 0.6 + 0.06 * h_lines + 0.05 * v_lines)
        return IMAGE_TYPE_TABLE, confidence, f"ruling-lines(h={h_lines},v={v_lines})"
    if grid_by_layout:
        confidence = min(0.9, 0.4 + 0.2 * signals["ocr_alignment"]
                        + 0.2 * signals.get("ocr_col_stability", 0.0))
        return IMAGE_TYPE_TABLE, confidence, "ocr-grid-alignment"

    # ── 2. 公式：符号占比高 + 行很短 + 无线框 ──────────────────────────────
    # 公式和正文的区别是"符号密度"：正文里 "=" 夹在大量汉字中占比极低，
    # 公式反过来（符号多、字少、行短）。放这么靠前是因为公式图常被判成
    # "截图"或"图表"，而它其实有专门的识别引擎。
    math_ratio = signals.get("math_ratio", 0.0)
    formula_like = (
        math_ratio >= 0.18
        and signals["ocr_rows"] <= 6
        and signals.get("avg_line_chars", 999) <= 60
        and signals.get("code_score", 0.0) < 0.5
        and signals["ocr_alignment"] <= 0.5
        # 几乎不含汉字：ASCII 括号/百分号也算数学符号之后，单看密度会把
        # "（见附录）占比 30%" 这类中文句子拉进公式分支，这道护栏把它挡住。
        # 缺省取 1.0（= 直接不通过）—— 调用方没算这个信号时宁可判成非公式，
        # 也不要凭空把一张图丢给公式引擎。
        and signals.get("cjk_ratio", 1.0) <= 0.10
    )
    if formula_like:
        confidence = min(0.9, 0.5 + math_ratio * 1.2)
        return IMAGE_TYPE_FORMULA, confidence, f"math-symbols({math_ratio:.2f})"

    # ── 3. 代码截图：文本"像代码" + 有多行 ─────────────────────────────────
    # code_score 综合了符号密度/缩进比/行尾标点/语言关键字命中（见
    # engines.code_parser.code_likeness）。必须同时有多行文字，避免把
    # 一句带分号的话误判成代码。
    code_score = signals.get("code_score", 0.0)
    if code_score >= 0.5 and (signals["text_bands"] >= 4 or signals["ocr_rows"] >= 4):
        confidence = min(0.92, 0.45 + code_score * 0.5)
        return (
            IMAGE_TYPE_CODE,
            confidence,
            f"code-like({code_score:.2f},{signals.get('code_language')})",
        )

    # ── 4. 图表：少量大面积**彩色**块 + 统一背景 + 文字行不多 ──────────────
    # "有几个色块"远不足以判定图表。深色底的**节点/结构图**（TensorBoard 的
    # graph、Keras `plot_model` 的输出）也有 1~2 块低饱和的彩色**描边**，单看
    # chromatic_blocks 会把它误判成图表，随后套用"横轴/纵轴/数据点"模板，吐出
    # 一堆"无具体数值"的空话。真正的图表必然满足下面至少一条：
    #   · 数据色块占据可观面积（柱/饼/热力图）—— colorful_ratio 高。此判据与
    #     主题无关：深色主题的图表色块一样明亮，同样能命中；
    #   · 存在坐标轴线（散点/折线图）：线条细、彩色面积小，但一定有轴线。
    # 节点图的两条都不满足（彩色只是描边、图内无长直线），于是落到下面的 diagram。
    chromatic = signals["chromatic_blocks"]
    chart_evidence = (
        signals["colorful_ratio"] >= 0.15
        or raw_bands >= 2
    )
    chart_like = (
        2 <= chromatic <= 8
        and signals["dominant_ratio"] >= 0.3
        and signals["colorful_ratio"] >= 0.02
        # 代码截图也会有一两块彩色（高亮）但它有几十行文字
        and signals["ocr_rows"] <= 15
        and chart_evidence
    )
    # 大面积彩色 + 统一背景：柱/饼/分组柱这类"彩色块占可观面积"的图表，
    # 即使没被量化成 ≥3% 的同色块（细柱/渐变），彩色像素占比依然很高。
    # 单凭 chromatic_blocks 会把它们漏掉（实测柱状图 chromatic=0），于是落到
    # screenshot/photo —— 图意（趋势、数值）永远读不到。
    colorful_chart = (
        signals["colorful_ratio"] >= 0.15
        and signals["saturation"] >= 0.06
        and signals["dominant_ratio"] >= 0.3
        and signals["ocr_rows"] <= 15
    )
    if chart_like or colorful_chart:
        confidence = min(0.9, 0.5 + 0.1 * chromatic + signals["dominant_ratio"] * 0.2)
        return IMAGE_TYPE_CHART, confidence, f"color-blocks({chromatic})"

    # ── 5. 截图：文字行密集 + 背景统一 + 基本无彩色 ────────────────────────
    # text_bands 来自像素投影，ocr_rows 来自真实 OCR；两者取或，保证
    # OCR 不可用时（无模型/无引擎）依然能把截图认出来。
    text_dense = (
        (signals["text_bands"] >= 8 or signals["ocr_rows"] >= 8)
        and signals["ocr_alignment"] <= 0.45
    )
    uniform_bg = (
        signals["dominant_ratio"] >= 0.45
        and signals["saturation"] <= 0.12
        and signals["line_art_ratio"] >= 0.75
    )
    if text_dense and uniform_bg:
        confidence = min(0.88, 0.5 + 0.02 * max(signals["text_bands"], signals["ocr_rows"]))
        return IMAGE_TYPE_SCREENSHOT, confidence, "text-dense-uniform-bg"

    # ── 6. 流程图 / 结构图：线稿（近白近黑为主）+ 有直线段或文字行 ──────────
    # chromatic 上限放宽到 3：节点图常带 1~3 块低饱和的彩色描边（TensorBoard
    # 的橄榄色/蓝色圆角框），老阈值 `<=1` 会把它们挡在 diagram 之外、掉进
    # photo 走 OCR —— 而 OCR 对深底浅字只能读出一两个单词，等于没读。
    # 真正的图表上一步已被 chart_evidence 拦走，放宽这里不会把图表卷进来。
    line_art = signals["line_art_ratio"] >= 0.8
    structured = raw_bands >= 2 or signals["ocr_rows"] >= 2
    if line_art and structured and chromatic <= 3:
        confidence = min(
            0.85, 0.45 + signals["line_art_ratio"] * 0.3 + 0.03 * (h_lines + v_lines)
        )
        return IMAGE_TYPE_DIAGRAM, confidence, "line-art-structured"

    # ── 6.5 彩色结构示意图兜底：统一背景 + 可辨结构 + 彩度不足以判图表 ────────
    # 有些"结构示意图"（三栏对比图、带彩色标题条 / 卡片面板的示意图）彩色面积
    # 够不上图表（colorful_ratio < 0.15），line_art_ratio 又够不上上面 0.8 的
    # 流程图线稿判据，于是掉进 default → photo → 走 OCR、永远没有图意总结。
    # 这类图有**统一背景 + 可辨结构**（面板 / 色带 / 直线等几何结构），把它认成
    # diagram 让它走多模态，才是"图能被读懂"的正确归宿（见模块级常量注释里的
    # 实测锚点）。真实照片的三条信号都落在门外（dominant 低、line_art≈0、
    # colorful 极高），因此不会被误收。
    structured_color_panels = (
        signals["dominant_ratio"]
        >= getattr(settings, "STRUCTURED_MIN_DOMINANT_RATIO", _STRUCTURED_MIN_DOMINANT_RATIO)
        and signals["line_art_ratio"]
        >= getattr(settings, "STRUCTURED_MIN_LINE_ART", _STRUCTURED_MIN_LINE_ART)
        and signals["colorful_ratio"]
        < getattr(settings, "STRUCTURED_MAX_COLORFUL_RATIO", _STRUCTURED_MAX_COLORFUL_RATIO)
        # 必须确有**几何结构**（粗带 / 彩色描边块 / 细直线之一）；
        # 三者皆空的（纯空白页、纯色块、以及"有文字但没有任何线/带"的纯文字页）
        # 都不能凭"背景统一"就升格成 diagram —— 纯文字页的正解是 OCR（要文字），
        # 不是 vision（要图意）。⚠️ 特别注意：**不要**把 `ocr_rows >= 2` 当结构
        # 信号加回来 —— 一张"白底黑字、无线无色带"的打印页翻拍件恰好会满足
        # "统一背景 + 彩度低 + OCR 多行"，一旦把 OCR 行数算作结构，它就会被误
        # 升格成 diagram，把本该走 OCR 的文字页推去白烧一次多模态推理。
        and (
            raw_bands >= 1
            or chromatic >= 1
            or (h_lines + v_lines) >= 1
        )
    )
    if structured_color_panels:
        confidence = min(
            0.8,
            0.45 + signals["line_art_ratio"] * 0.25 + signals["dominant_ratio"] * 0.2,
        )
        return IMAGE_TYPE_DIAGRAM, confidence, "structured-panels"

    # ── 7. 其余：普通图片 → OCR ────────────────────────────────────────────
    return IMAGE_TYPE_PHOTO, 0.4, "default"


def classify_image_safe(
    img: Image.Image,
    *,
    ocr_lines: list | None = None,
    filename: str = "",
) -> ImageClassification:
    """永不抛出的分类入口：任何异常都退化为 photo（走 OCR 的保守路径）."""
    try:
        return classify_image(img, ocr_lines=ocr_lines, filename=filename)
    except Exception as exc:      # noqa: BLE001
        logger.warning("Image classification failed for '%s': %s", filename, exc)
        return ImageClassification(
            image_type=IMAGE_TYPE_PHOTO, confidence=0.0,
            signals={"error": str(exc)}, engine="fallback",
        )


__all__ = [
    "ImageClassification",
    "classify_image",
    "classify_image_safe",
    "compute_signals",
    "vision_analyze_types",
]
