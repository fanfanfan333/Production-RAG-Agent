"""
底层图像工具（分类器与表格识别共用）.

这里的函数只做"把 PIL 图片变成可判定的投影信号"这一件事，不含任何业务
逻辑。抽出来是因为 classifier 与 table_recognizer 都需要**完全一致**的
墨迹阈值与线条检测 —— 两处各写一遍必然会在调参时漂移。

一个容易踩的坑：**深色底截图**（代码截图、深色主题 UI）里"墨"是浅色的。
所有投影函数因此都先判断底色极性，再决定哪一侧算墨迹。

## 极性为什么不能看"全图多数像素"（2026-09 修的 BUG）

判极性的目的只有一个：**搞清哪一侧是纸、哪一侧是字**。历史上用的是
"全图深色像素 > 50% 就算深底浅字"，这条在**混合极性**的图上会翻车：

    白底页面 + 内嵌一块大面积深色图（深色主题流程图截图、终端窗口截图、
    暗色卡片），深色面积一旦超过一半，全图多数判定就会说"这是深底浅字"，
    于是**白色页边距被当成墨**，在投影里形成两条贯穿整幅图的"竖线"，
    分类器据此判出 ruling-lines → 这张流程图被当成表格送进 Table Parser。

判极性因此改为看**页面背景环**（图像最外圈几条像素）：一张文档图片的
纸面颜色就体现在页边距上，而"深色主题 UI 铺满整幅图"的情况它的最外圈
同样是深色 —— 两种情形都能判对。背景环本身不统一时（满幅照片、无边框
的复杂图）才退回全图多数法，并且把用的是哪条依据一并返回，便于排查。
"""

from __future__ import annotations

from PIL import Image

# 灰度阈值：低于此值视为"墨"（浅底深字）
INK_THRESHOLD = 165
# 深底浅字时，"墨"是亮像素；用 255 - INK_THRESHOLD 作为分界
INVERTED_THRESHOLD = 255 - INK_THRESHOLD
# 近白 / 近黑阈值（判断线稿图）
NEAR_WHITE = 235
NEAR_BLACK = 55

# ── 页面背景（纸面）估计 ─────────────────────────────────────────────────────
#: 背景环的采样宽度上限（像素）。取最外圈这么宽的边框像素来估"纸"的颜色。
BG_RING_MAX_PX = 6
#: 背景环宽度相对短边的比例（小图按比例取，避免整幅图都被当成边框）
BG_RING_RATIO = 0.02
#: 背景环亮度中位数低于该值 → 纸是深色的（深底浅字）
BG_DARK_LEVEL = 128
#: 背景环中"深/浅"两侧的纯度：一侧占比达到该值才认为背景是单一颜色。
#: 低于它说明最外圈本身就花（满幅照片 / 贴边内容），据此判定不可靠。
BG_PURITY = 0.75
#: 全图多数法（退路）的分界：深色占比超过该值算深底浅字
MAJORITY_DARK_RATIO = 0.5


def to_gray_pixels(img: Image.Image, max_side: int) -> tuple[list[int], int, int]:
    """转灰度并限制长边（返回 (像素列表, 宽, 高)）."""
    width, height = img.size
    scale = max_side / float(max(width, height, 1))
    if scale < 1.0:
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
        img = img.resize((width, height), Image.BILINEAR)
    gray = img.convert("L")
    return list(gray.getdata()), width, height


def background_stats(pixels: list[int], width: int, height: int) -> tuple[int, float]:
    """
    估计"纸面"（页面背景）的亮度：返回 ``(中位亮度, 纯度)``.

    *纯度* 是背景环里"深 / 浅"两侧中占优一侧的比例。纯度高说明最外圈是
    单一颜色（真正的页面背景）；纯度低说明最外圈本身就花（满幅照片、
    贴边内容），此时中位数没有代表性。
    """
    if not pixels or width <= 0 or height <= 0:
        return 255, 0.0

    band = max(1, min(BG_RING_MAX_PX, int(min(width, height) * BG_RING_RATIO) or 1))
    ring: list[int] = []
    for y in range(height):
        base = y * width
        near_y_edge = y < band or y >= height - band
        for x in range(width):
            if near_y_edge or x < band or x >= width - band:
                ring.append(pixels[base + x])
    if not ring:
        return 255, 0.0

    ring.sort()
    median = ring[len(ring) // 2]
    dark = 0
    for value in ring:
        if value < BG_DARK_LEVEL:
            dark += 1
    purity = max(dark, len(ring) - dark) / float(len(ring))
    return int(median), round(purity, 4)


def background_is_dark(
    pixels: list[int], width: int, height: int
) -> tuple[bool, str]:
    """
    页面底色（纸面）是否为深色 —— 即这张图是不是"深底浅字".

    返回 ``(inverted, 判定依据)``，依据字符串会写进分类信号便于排查：

        page-ring(...)        —— 背景环是单一颜色，直接采信（常规情形）
        global-majority(...)  —— 背景环本身很花，退回"全图深色占多数"

    注意这里**不能**无条件用"全图多数像素"。白底页面 + 内嵌大面积深色图
    （深色主题流程图 / 终端窗口截图）会让深色面积超过一半，全图多数法
    于是误判成深底浅字，白色页边距随即被当成墨迹、在投影里形成贯穿全图的
    假"表格框线"—— 这是 [2026-09] 修掉的那个误判的直接成因。
    """
    median, purity = background_stats(pixels, width, height)
    if purity >= BG_PURITY:
        return median < BG_DARK_LEVEL, f"page-ring(median={median},purity={purity})"

    total = len(pixels) or 1
    dark = 0
    for value in pixels:
        if value < INK_THRESHOLD:
            dark += 1
    ratio = dark / float(total)
    return ratio > MAJORITY_DARK_RATIO, f"global-majority(dark={round(ratio, 4)})"


def ink_profiles(
    pixels: list[int], width: int, height: int, *, inverted: bool | None = None
) -> tuple[list[int], list[int], bool]:
    """
    每行 / 每列的墨迹像素计数.

    返回 ``(row_ink, col_ink, inverted)``。``inverted=True`` 表示底是深色、
    墨是浅色（深色主题截图 / 代码截图），此时投影统计的是亮像素。

    *inverted* 传给调用方已经算好的极性，避免同一张图重复估计两次底色。
    """
    if inverted is None:
        inverted, _ = background_is_dark(pixels, width, height)

    row_ink = [0] * height
    col_ink = [0] * width
    if inverted:
        for y in range(height):
            base = y * width
            count = 0
            for x in range(width):
                if pixels[base + x] > INVERTED_THRESHOLD:
                    count += 1
                    col_ink[x] += 1
            row_ink[y] = count
    else:
        for y in range(height):
            base = y * width
            count = 0
            for x in range(width):
                if pixels[base + x] < INK_THRESHOLD:
                    count += 1
                    col_ink[x] += 1
            row_ink[y] = count
    return row_ink, col_ink, inverted


def bands(
    profile: list[int], limit: int, ratio: float, max_gap: int = 2
) -> list[tuple[int, int]]:
    """
    找出 profile 中"墨占比超过阈值"的连续区间（允许 max_gap 个像素的断裂）.

    一条粗细不均 / 抗锯齿的线会因此合并成**一个** band，而不是被数成多条。
    """
    threshold = limit * ratio
    result: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for index, value in enumerate(profile):
        if value >= threshold:
            if start is None:
                start = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                result.append((start, index - gap))
                start = None
                gap = 0
    if start is not None:
        result.append((start, len(profile) - 1 - gap))
    return result


def count_bands(
    profile: list[int], limit: int, ratio: float, max_gap: int = 2
) -> int:
    """bands() 的数量版（分类器只需要"有几条线"）."""
    return len(bands(profile, limit, ratio, max_gap))


def line_art_ratio(pixels: list[int]) -> float:
    """近白或近黑像素占比 —— 线稿 / 结构图 / 截图都很高，照片很低."""
    if not pixels:
        return 0.0
    hits = 0
    for value in pixels:
        if value >= NEAR_WHITE or value <= NEAR_BLACK:
            hits += 1
    return round(hits / len(pixels), 4)


def encode_png_safe(img: Image.Image) -> bytes:
    """
    PIL Image → PNG 字节（失败时返回空 bytes，绝不抛）.

    引擎层要往 Vision 传图，但编码失败不该让整条理解链路挂掉 —— 调用方拿到
    空 bytes 时会走"没有图"的分支，而不是崩。
    """
    import io as _io

    try:
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception:      # noqa: BLE001
        return b""


__all__ = [
    "INK_THRESHOLD",
    "INVERTED_THRESHOLD",
    "NEAR_WHITE",
    "NEAR_BLACK",
    "BG_RING_MAX_PX",
    "BG_DARK_LEVEL",
    "BG_PURITY",
    "to_gray_pixels",
    "background_stats",
    "background_is_dark",
    "ink_profiles",
    "bands",
    "count_bands",
    "line_art_ratio",
    "encode_png_safe",
]
