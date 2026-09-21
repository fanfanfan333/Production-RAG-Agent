"""
图片极性（哪一侧是"纸"）单元测试 —— 混合极性误判的回归用例.

背景（2026-09 修的真实 BUG）：

    一张 **白底页面上嵌着深色流程图面板** 的截图（深色主题流程图粘进白底
    文档，非常常见）走完管线后：

        classify → table (conf=0.94) → Table Parser → 兜底全失败
                 → manual_review / analyze_engine="ocr-degraded"

    根因不在 OCR（OCR 读出了图里的字），而在**极性判错**。历史上判极性用的是
    "全图深色像素 > 50% 就是深底浅字"。深色面板面积一旦过半，这条规则就把整张
    图当成深底浅字，于是**白色页边距被算成墨迹**，投影里凭空出现贯穿全幅的
    竖线（实测 v_lines=2），加上最宽那个节点框的上下两条横边（h_lines=4），
    凑够 `framed = h_lines>=3 and v_lines>=2` —— 一张流程图被判成"有框线的表格"。

    判极性因此改成看**页面背景环**（最外圈像素）：页面的"纸"颜色体现在页边距
    上。深色主题铺满整幅图时最外圈同样是深色，两种情形都能判对。

本文件锁死以下不变量，防止有人"优化"回全图多数法：

    1. 白底 + 大面积深色面板 → 判定为**浅底**（纸是白的），且依据是 page-ring；
    2. 深色主题铺满整幅图       → 判定为**深底浅字**；
    3. 页边距不得被算成表格框线（h/v_lines 不因页边距而虚高）；
    4. 混合极性流程图必须落在 diagram（→ Vision），**不能**落到 table_parser；
    5. 深底图的二值化必须先归正极性，否则内容被抹平；
    6. 混合极性的图跳过二值化（按哪一侧归正都会毁掉另一侧）；
    7. 兜底 OCR 对非浅底图要补一次"反色重读"。

宿主机缺依赖时优雅跳过（与既有单测同一约定）；推荐在 backend 容器内运行：
    docker exec rag_backend python tests/test_image_polarity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    import importlib

    from PIL import Image, ImageDraw

    from app.services.image_understanding import (
        IMAGE_TYPE_DIAGRAM,
        IMAGE_TYPE_TABLE,
        classify_image_safe,
        compute_signals,
        resolve_route,
    )
    from app.services.image_understanding.engines.base import EngineOutput
    from app.services.image_understanding.imaging import (
        background_is_dark,
        background_stats,
        count_bands,
        ink_profiles,
        to_gray_pixels,
    )

    # 注意：包的 __init__ 里 `preprocess` 这个名字被**函数**占用（同名遮蔽了
    # 子模块），`pipeline` 同理可能撞名，所以这里一律按模块路径显式取。
    pp = importlib.import_module("app.services.image_understanding.preprocess")
    image_pipeline = importlib.import_module("app.services.image_understanding.pipeline")
except ImportError as exc:  # 宿主机缺依赖 → 跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    # 不能用 sys.exit()：pytest 在收集阶段导入本模块，抛 SystemExit 会让整个
    # 会话 INTERNALERROR，同目录其它用例全部跑不了。
    skip_module(f"missing dependency ({exc}) — run inside the backend container")


# ── 夹具 ─────────────────────────────────────────────────────────────────────
# 全部**程序化生成**，不依赖任何二进制样本文件：

def _mixed_polarity_flowchart() -> Image.Image:
    """
    复刻用户那张截图：白底页面 + 页眉注释 + 一整块深色流程图面板（混合极性）.

    深色面板刻意占到 ~55% 面积 —— 正是让"全图多数法"翻车的比例。
    """
    img = Image.new("RGB", (289, 313), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(11):                                    # 页眉注释（黑字白底）
        draw.line([(14 + i * 13, 6), (14 + i * 13, 20)], fill=(40, 40, 40), width=2)
    draw.rectangle([20, 34, 200, 285], fill=(10, 10, 10))  # 深色面板
    for x0, y0, x1, y1, colour in (                        # 三个节点框
        (40, 62, 170, 102, (150, 150, 120)),
        (32, 150, 190, 195, (140, 140, 140)),
        (40, 248, 170, 288, (150, 150, 120)),
    ):
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=3)
    draw.line([(111, 103), (111, 149)], fill=(150, 150, 150), width=2)   # 连线
    draw.line([(111, 196), (111, 247)], fill=(150, 150, 150), width=2)
    return img


def _text_page(background: tuple[int, int, int], ink: tuple[int, int, int]) -> Image.Image:
    """造一页"规则文字"（用竖条模拟字形，不需要真实字体）."""
    img = Image.new("RGB", (320, 220), background)
    draw = ImageDraw.Draw(img)
    for row in range(8):
        for i in range(12):
            draw.line(
                [(16 + i * 22, 20 + row * 24), (16 + i * 22, 36 + row * 24)],
                fill=ink, width=2,
            )
    return img


def _dark_theme_page() -> Image.Image:
    """深色主题铺满整幅图（真·深底浅字）."""
    return _text_page((18, 18, 20), (200, 200, 205))


def _light_page() -> Image.Image:
    """白底黑字文档页."""
    return _text_page((255, 255, 255), (30, 30, 30))


def _dark_theme_text_page() -> Image.Image:
    """深底浅字 + 较粗笔画（用于验证二值化不再把内容抹平）."""
    img = Image.new("RGB", (400, 180), (12, 12, 14))
    draw = ImageDraw.Draw(img)
    for row in range(5):
        for i in range(10):
            draw.line(
                [(20 + i * 36, 20 + row * 30), (20 + i * 36, 40 + row * 30)],
                fill=(230, 230, 235), width=4,
            )
    return img


def _gradient_photo() -> Image.Image:
    """渐变照片：最外圈本身就花，背景环不纯度 → 该走全图多数法."""
    img = Image.new("RGB", (320, 220))
    pixels = img.load()
    for y in range(220):
        for x in range(320):
            base = int(90 + 70 * (x / 320.0) + 40 * (y / 220.0))
            pixels[x, y] = (base, int(base * 0.72), int(base * 0.5))
    return img


# ── 1. 极性判定 ──────────────────────────────────────────────────────────────

def test_mixed_polarity_image_is_light_paper() -> None:
    """
    **核心回归**：白底 + 大面积深色面板 → 纸是白的（不能按深底浅字处理）.

    这条断言就是那个 BUG 的靶心：旧规则 `全图深色 > 50%` 在这里返回 True，
    于是白色页边距被当成墨迹。
    """
    img = _mixed_polarity_flowchart()
    pixels, w, h = to_gray_pixels(img, 256)

    # 先确认这张夹具确实"深色过半" —— 否则测不到旧规则的坑
    dark_ratio = sum(1 for v in pixels if v < 165) / len(pixels)
    assert dark_ratio > 0.5, f"夹具应深色过半（实际 {dark_ratio:.3f}），否则测不出旧 BUG"

    inverted, source = background_is_dark(pixels, w, h)
    assert inverted is False, f"白底页面的纸是白的，不该判成深底浅字（依据 {source}）"
    assert source.startswith("page-ring"), f"应走背景环判据，实际 {source}"

    median, purity = background_stats(pixels, w, h)
    assert median >= 200, f"页边距（背景环）应是白的，实际中位亮度 {median}"
    assert purity >= 0.9, f"白底页边距应是单一颜色，实际纯度 {purity}"
    print("  ok test_mixed_polarity_image_is_light_paper")


def test_full_bleed_dark_theme_is_inverted() -> None:
    """深色主题铺满整幅图 → 仍是深底浅字（背景环本身就是深色）."""
    for img in (_dark_theme_page(), _dark_theme_text_page()):
        pixels, w, h = to_gray_pixels(img, 256)
        inverted, source = background_is_dark(pixels, w, h)
        assert inverted is True, f"深色主题应判为深底浅字（依据 {source}）"
        assert source.startswith("page-ring"), source
    print("  ok test_full_bleed_dark_theme_is_inverted")


def test_light_document_is_not_inverted() -> None:
    """白底黑字文档页 → 浅底深字（最常见的情形，不能被改动搞坏）."""
    pixels, w, h = to_gray_pixels(_light_page(), 256)
    inverted, _ = background_is_dark(pixels, w, h)
    assert inverted is False
    print("  ok test_light_document_is_not_inverted")


def test_busy_border_falls_back_to_global_majority() -> None:
    """背景环本身很花（渐变照片）→ 退回全图多数法，且把依据讲清楚."""
    pixels, w, h = to_gray_pixels(_gradient_photo(), 256)
    _, purity = background_stats(pixels, w, h)
    assert purity < 0.75, f"渐变照片的边框不该被当成统一背景，纯度 {purity}"
    _, source = background_is_dark(pixels, w, h)
    assert source.startswith("global-majority"), source
    assert "dark=" in source
    print("  ok test_busy_border_falls_back_to_global_majority")


def test_ink_profiles_accepts_polarity_override() -> None:
    """`ink_profiles` 支持传入已算好的极性（避免同图重复估计底色）."""
    pixels, w, h = to_gray_pixels(_mixed_polarity_flowchart(), 256)
    row_a, col_a, inv_a = ink_profiles(pixels, w, h, inverted=False)
    row_b, col_b, inv_b = ink_profiles(pixels, w, h, inverted=True)
    assert inv_a is False and inv_b is True
    assert sum(row_a) != sum(row_b), "两种极性下墨迹分布必须不同"
    assert len(col_a) == w and len(row_b) == h
    print("  ok test_ink_profiles_accepts_polarity_override")


# ── 2. 页边距不得被算成表格框线 ──────────────────────────────────────────────

def test_page_margins_are_not_ruling_lines() -> None:
    """
    白底页边距不能变成"贯穿全幅的竖线".

    这是 BUG 的传导路径：页边距被当成墨 → col_ink 在页面左右两条白边上
    满格 → 数出 2 条"竖框线"。修好极性后它们不再是墨，自然消失。
    """
    signals = compute_signals(_mixed_polarity_flowchart())
    assert signals["inverted"] is False, signals["polarity_source"]
    assert signals["polarity_source"].startswith("page-ring")

    pixels, w, h = to_gray_pixels(_mixed_polarity_flowchart(), 256)
    row_ink, col_ink, _ = ink_profiles(pixels, w, h, inverted=False)
    v_lines = count_bands(col_ink, h, 0.55, 2)
    # 旧行为是 2（左右页边距各一条）。深色面板本身是一整块，最多贡献 1 条。
    assert v_lines <= 1, f"页边距不该形成竖框线（v_lines={v_lines}）"
    assert signals["v_lines"] <= 1, signals
    assert signals["h_lines"] <= 2, f"h_lines={signals['h_lines']}（深色面板至多 1 条）"
    print("  ok test_page_margins_are_not_ruling_lines")


def test_mixed_polarity_flowchart_is_diagram_not_table() -> None:
    """
    **端到端回归**：混合极性流程图必须判成 diagram → 路由到 Vision.

    这条把"判错类型 → 送进 Table Parser → 兜底全失败 → 人工复核"整条错误
    链路钉死：判成 table 就再也不会走 Vision，图里的节点/连线永远拿不到。
    """
    result = classify_image_safe(_mixed_polarity_flowchart(), filename="flowchart")
    assert result.image_type != IMAGE_TYPE_TABLE, (
        f"流程图被判成了表格（signals={result.signals}）"
    )
    assert result.image_type == IMAGE_TYPE_DIAGRAM, result.signals
    assert resolve_route(result.image_type) == "vision"
    assert result.signals["reason"] == "line-art-structured"
    assert 0.0 < result.confidence <= 1.0
    print("  ok test_mixed_polarity_flowchart_is_diagram_not_table")


# ── 3. 二值化必须归正极性 ────────────────────────────────────────────────────

def test_binarize_normalises_dark_theme_polarity() -> None:
    """
    深底浅字的图二值化后必须是"浅底 + 深字"（OCR 模型只认这一侧）.

    旧实现直接把深底图丢给 adaptiveThreshold：背景整块比邻域均值暗 →
    被判成前景，字形与背景糊在一起，OCR 通道等于废掉。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_if_host_shimmed

    # 同 test_image_noise_quality：宿主机无 cv2 时 _binarize 静默跳过，
    # applied 里不会有 'binarize' —— 环境缺库，不是极性归正逻辑坏了。
    skip_if_host_shimmed("cv2")
    img = _dark_theme_text_page()
    assert pp.page_polarity(img) == "dark"

    result = pp.preprocess(img, mode="text")
    assert "binarize" in result.applied, result.applied
    assert result.ocr_image is not None, "text 档应产出二值化的 OCR 输入"

    levels = list(result.ocr_image.convert("L").getdata())
    mean = sum(levels) / len(levels)
    foreground = sum(1 for v in levels if v < 128) / len(levels)

    assert mean > 200, f"OCR 通道的纸应是浅色（均值 {mean:.1f}）"
    assert 0.005 < foreground < 0.20, (
        f"字形应作为少数派前景保留下来（前景占比 {foreground*100:.2f}%）"
    )
    # 原图通道不能被二值化：明暗层次必须保住
    image_levels = len(set(result.image.convert("L").getdata()))
    assert image_levels > 2, "image 通道不能被二值化"
    print("  ok test_binarize_normalises_dark_theme_polarity")


def test_binarize_skips_mixed_polarity() -> None:
    """
    混合极性的图跳过二值化 —— 按哪一侧归正都会毁掉另一侧.

    白底页面 + 深色面板：按白底归正 → 深色面板整块变白（内容全没）；
    按深底归正 → 白色页边距整块变黑。与其毁掉一半，不如不做。
    """
    img = _mixed_polarity_flowchart()
    assert pp.page_polarity(img) == "mixed"

    for mode in ("text", "scan"):
        result = pp.preprocess(img, mode=mode)
        assert "binarize" not in result.applied, (
            f"{mode} 档对混合极性图不该做二值化，实际 {result.applied}"
        )
        assert result.ocr_image is None
        assert result.ocr_input is result.image
    print("  ok test_binarize_skips_mixed_polarity")


def test_invert_for_ocr_flips_light_background() -> None:
    """反色副本的"纸"必须翻到浅色侧（校验兜底重读用的那张图）."""
    img = _dark_theme_text_page()
    flipped = pp.invert_for_ocr(img)
    assert pp.page_polarity(flipped) == "light"
    assert flipped.size == img.size
    print("  ok test_invert_for_ocr_flips_light_background")


# ── 4. 兜底 OCR 的反色重读 ───────────────────────────────────────────────────

class _FakeEngine:
    """只认"浅底深字"的假引擎（模拟 Tesseract：深底图直接返回空串）."""

    def __init__(self, name: str, text: str):
        self.name = name
        self._text = text
        self.seen: list[Image.Image] = []

    def is_available(self) -> bool:
        return True

    def process(self, image: Image.Image) -> EngineOutput:
        self.seen.append(image)
        levels = list(image.convert("L").getdata())
        readable = sum(levels) / len(levels) > 200      # 纸是浅色的才读得出来
        return EngineOutput(
            engine=self.name,
            text=self._text if readable else "",
            confidence=0.9,
            ok=readable,
        )


class _FakeRegistry:
    def __init__(self, engines: dict):
        self._engines = engines

    def get(self, name: str):
        return self._engines.get(name)


def test_second_ocr_retries_with_inverted_copy() -> None:
    """
    深底图兜底时，第二个引擎要拿到"反色副本"再读一遍.

    否则深色主题截图在兜底路径上永远是空 —— Tesseract 在这类图上直接返回
    空串（实测），"换引擎重读"也就失去了意义。
    """
    fake = _FakeEngine("tesseract", "recovered text from inverted copy")
    origin = image_pipeline.get_registry
    image_pipeline.get_registry = lambda: _FakeRegistry({  # type: ignore[assignment]
        "tesseract": fake,
        "paddleocr": None,
    })
    try:
        # 主 OCR 已经用过 paddleocr（假引擎里为 None，只需它别被再调用）
        primary = EngineOutput(engine="paddleocr", text="", confidence=0.0,
                               ok=False, error="empty")
        out = image_pipeline._second_ocr(_dark_theme_text_page(), primary)
    finally:
        image_pipeline.get_registry = origin              # type: ignore[assignment]

    assert out is not None, "反色副本应能读出文字，不该返回 None"
    assert out.text == "recovered text from inverted copy"
    assert out.engine == "tesseract+inverted", f"来源应标注出来，实际 {out.engine}"
    assert len(fake.seen) >= 2, "原始图与反色副本各应被读过一次"
    print("  ok test_second_ocr_retries_with_inverted_copy")


def test_second_ocr_keeps_original_when_longer() -> None:
    """反色重读只取"字数更多"的结果 —— 不能把本来读得好的图弄坏."""
    class _BilingualEngine:
        def __init__(self):
            self.name = "tesseract"
            self.seen = []

        def is_available(self) -> bool:
            return True

        def process(self, image: Image.Image) -> EngineOutput:
            self.seen.append(image)
            levels = list(image.convert("L").getdata())
            light_paper = sum(levels) / len(levels) > 200
            text = "short" if light_paper else ("original result is clearly longer " * 3)
            return EngineOutput(engine="tesseract", text=text, confidence=0.8, ok=True)

    fake = _BilingualEngine()
    origin = image_pipeline.get_registry
    image_pipeline.get_registry = lambda: _FakeRegistry({  # type: ignore[assignment]
        "tesseract": fake,
        "paddleocr": None,
    })
    try:
        primary = EngineOutput(engine="paddleocr", text="", confidence=0.0, ok=False)
        out = image_pipeline._second_ocr(_dark_theme_text_page(), primary)
    finally:
        image_pipeline.get_registry = origin              # type: ignore[assignment]

    assert out is not None
    assert out.text.startswith("original result is clearly longer")
    assert out.engine == "tesseract", "没反色时应保持原引擎名"
    print("  ok test_second_ocr_keeps_original_when_longer")


def test_light_image_does_not_trigger_inverted_retry() -> None:
    """浅底图不该多跑一次反色重读（省一次推理，也避免引入噪声结果）."""
    fake = _FakeEngine("tesseract", "light background text")
    origin = image_pipeline.get_registry
    image_pipeline.get_registry = lambda: _FakeRegistry({  # type: ignore[assignment]
        "tesseract": fake,
        "paddleocr": None,
    })
    try:
        primary = EngineOutput(engine="paddleocr", text="", confidence=0.0, ok=False)
        out = image_pipeline._second_ocr(_light_page(), primary)
    finally:
        image_pipeline.get_registry = origin              # type: ignore[assignment]

    assert out is not None and out.text == "light background text"
    assert len(fake.seen) == 1, f"浅底图只该读一次，实际 {len(fake.seen)} 次"
    print("  ok test_light_image_does_not_trigger_inverted_retry")


# ── 执行 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("图片极性（混合极性误判）回归测试")

    print("\n[一] 极性判定")
    test_mixed_polarity_image_is_light_paper()
    test_full_bleed_dark_theme_is_inverted()
    test_light_document_is_not_inverted()
    test_busy_border_falls_back_to_global_majority()
    test_ink_profiles_accepts_polarity_override()

    print("\n[二] 页边距不得变成表格框线")
    test_page_margins_are_not_ruling_lines()
    test_mixed_polarity_flowchart_is_diagram_not_table()

    print("\n[三] 二值化极性归正")
    test_binarize_normalises_dark_theme_polarity()
    test_binarize_skips_mixed_polarity()
    test_invert_for_ocr_flips_light_background()

    print("\n[四] 兜底 OCR 的反色重读")
    test_second_ocr_retries_with_inverted_copy()
    test_second_ocr_keeps_original_when_longer()
    test_light_image_does_not_trigger_inverted_retry()

    print("\nAll image polarity regression tests passed.")


if __name__ == "__main__":
    main()
