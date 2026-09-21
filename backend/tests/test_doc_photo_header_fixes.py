"""
「文档照片里的表格表头」修复回归测试（任务 A/B/C/D）.

覆盖四类**行为不变量**，防止后续改动把这几处修复碰回去：

  任务 A（P0 · 题注行顶掉表头）
    · 题注识别（单块行 / 「表 N …」被拆成两块的都算题注）；
    · 题注**不占表头位**：剥离后首行才是真表头；
    · 题注**不丢**：作为 caption 放在 Markdown 表格之前、并写进 meta。

  任务 C（P1 · 退化表必须显式失败）
    · ``TableStructure.ok`` 必须否掉 **1 列**（透视/阴影把框线切崩的塌列）；
    · 退化成单列的网格不得产出可用表（ok=False）—— 不许输出 1 列垃圾表。

  任务 D（P1 · vision 不可用要"可见"）
    · 非 Vision 引擎的产出**绝不**写进 ``vision_caption``（不拿 OCR 冒充图意）；
    · 表格类在人工复核时清空 ``structured_content``（避免伪表格入库）；
    · vision 不可用时，chart 的 route 从 vision **降级为 ocr** 且 meta 如实标记。

  任务 B（P1 · 分类器误判）
    · 框线表格 → table；
    · 彩色柱图/分组柱 → **不判 table**（判 chart）；
    · UI 截图（高 OCR alignment 但列稳定性低）→ **不判 table**。

宿主机缺依赖时优雅跳过（与既有单测同一约定）；推荐在 backend 容器内运行：
    docker exec -w /app -e PYTHONPATH=/app rag_backend \
        python -m pytest tests/test_doc_photo_header_fixes.py -q
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    from PIL import Image

    from app.config import get_settings
    from app.services.image_understanding import (
        IMAGE_TYPE_CHART,
        IMAGE_TYPE_SCREENSHOT,
        IMAGE_TYPE_TABLE,
    )
    from app.services.image_understanding.classifier import (
        ImageClassification,
        _decide,
    )
    from app.services.image_understanding.engines.base import EngineOutput
    from app.services.image_understanding.pipeline import (
        ImageUnderstanding,
        _apply_output,
        _keep_ocr_text,
    )
    from app.services.image_understanding.table_recognizer import (
        TableStructure,
        _from_alignment,
        _is_caption_texts,
        _Line,
        _strip_caption_grid,
        to_markdown,
    )

    pl = importlib.import_module("app.services.image_understanding.pipeline")
    ve = importlib.import_module("app.services.image_understanding.engines.vision_engine")
except ImportError as exc:  # 宿主机缺依赖 → 跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")


# ─────────────────────────────────────────────────────────────────────────────
# 任务 A · 题注行不得占据表头位
# ─────────────────────────────────────────────────────────────────────────────


def test_caption_line_is_detected() -> None:
    """题注判据：单块行 → 题注；「表 N …」拆成两块 → 题注；真表头 → 不是."""
    assert _is_caption_texts(["表 32024 年度核心经营指标"]) is True
    assert _is_caption_texts(["表 3", "2024 年度核心经营指标"]) is True
    assert _is_caption_texts(["Table 3", "2024 core metrics"]) is True
    assert _is_caption_texts(["图 1", "系统架构"]) is True
    # 真表头（≥3 个非空单元、首块不是「表 N」）绝不能被误判成题注
    assert _is_caption_texts(["指标", "2023年", "2024年", "同比"]) is False
    assert _is_caption_texts(["Metric", "2023", "2024", "YoY"]) is False
    print("  ok test_caption_line_is_detected")


def test_caption_grid_not_header_and_not_lost() -> None:
    """题注行被剥离后：表头是真表头；题注作为 caption 保留在表格之前."""
    grid = [
        ["表 3 2024 年度核心经营指标", "", "", ""],
        ["指标", "2023年", "2024年", "同比"],
        ["营业收入", "30240", "33500", "+10.8%"],
    ]
    captions, body = _strip_caption_grid(grid, min_cols=2)
    assert captions == ["表 3 2024 年度核心经营指标"], captions
    assert body[0] == ["指标", "2023年", "2024年", "同比"], body[0]

    md = to_markdown(body, caption=" ".join(captions))
    lines = md.splitlines()
    assert lines[0] == "表 3 2024 年度核心经营指标", "题注必须保留在表格之前"
    header_line = next(line for line in lines if line.startswith("|"))
    for token in ("指标", "2023年", "2024年", "同比"):
        assert token in header_line, f"表头行缺 {token}: {header_line}"
    print("  ok test_caption_grid_not_header_and_not_lost")


def _mk(text: str, x0: float, y0: float, w: float = 60.0, h: float = 20.0) -> "_Line":
    return _Line(text=text, x0=x0, y0=y0, x1=x0 + w, y1=y0 + h)


def test_alignment_keeps_real_header_after_caption_strip() -> None:
    """无框线对齐法：题注行在最上方时，还原出的表头仍是真表头（t1 的症状）."""
    lines = [_mk("表 3 2024 年度核心经营指标", 200, 40, 260, 24)]
    cols_x = [80.0, 220.0, 360.0, 500.0]
    for cells, y in (
        (["指标", "2023年", "2024年", "同比"], 110.0),
        (["营业收入", "30240", "33500", "+10.8%"], 140.0),
        (["净利润", "2180", "2460", "+12.8%"], 170.0),
        (["毛利率", "24.1%", "25.0%", "+0.9pp"], 200.0),
    ):
        for cell, x in zip(cells, cols_x):
            lines.append(_mk(cell, x, y))

    structure = _from_alignment(
        lines, min_rows=2, min_cols=2, max_rows=200, max_cols=30, page_number=1,
    )
    assert structure.ok, structure.meta
    assert structure.header[0] == "指标", structure.header
    assert "2023年" in structure.header and "同比" in structure.header
    assert "表 3" in structure.meta.get("caption", ""), structure.meta
    print("  ok test_alignment_keeps_real_header_after_caption_strip")


# ─────────────────────────────────────────────────────────────────────────────
# 任务 C · 退化表必须显式失败（不许输出 1 列垃圾表）
# ─────────────────────────────────────────────────────────────────────────────


def test_ok_predicate_rejects_single_column() -> None:
    """``TableStructure.ok`` 必须否掉 1 列（塌列），2 列才通过."""
    one_col = TableStructure(
        markdown="| 只有一列 |\n|---|\n| x |", rows=2, cols=1,
    )
    assert one_col.ok is False, "1 列表必须判为不可入库（ok=False）"

    two_col = TableStructure(
        markdown="| a | b |\n|---|---|\n| 1 | 2 |", rows=2, cols=2,
    )
    assert two_col.ok is True
    print("  ok test_ok_predicate_rejects_single_column")


def test_collapsed_alignment_table_fails() -> None:
    """退化成单列的网格 → 对齐法必须显式失败（ok=False），而不是吐 1 列表."""
    lines = [_mk(f"第 {i} 行内容", 100, 40 + i * 30, 120, 20) for i in range(5)]
    structure = _from_alignment(
        lines, min_rows=2, min_cols=2, max_rows=200, max_cols=30, page_number=1,
    )
    assert structure.ok is False, structure.meta
    assert structure.cols < 2
    print("  ok test_collapsed_alignment_table_fails")


# ─────────────────────────────────────────────────────────────────────────────
# 任务 D · vision 不可用要"可见"（不得把 OCR 当 vision_caption）
# ─────────────────────────────────────────────────────────────────────────────


def _understanding(image_type: str) -> "ImageUnderstanding":
    return ImageUnderstanding(
        image_type=image_type,
        route="vision",
        classification=ImageClassification(image_type=image_type, confidence=0.5, signals={}),
    )


def test_ocr_output_never_written_as_vision_caption() -> None:
    """非 Vision 引擎（tesseract/paddleocr）的文本只能进 ocr_text，绝不冒充 vision_caption."""
    for image_type in (IMAGE_TYPE_CHART, IMAGE_TYPE_SCREENSHOT, "diagram", "photo"):
        result = _understanding(image_type)
        _apply_output(result, EngineOutput(text="图里的文字", engine="tesseract", ok=True))
        assert result.vision_caption is None, f"{image_type}: OCR 被写成了 vision_caption"
        assert result.ocr_text == "图里的文字"
    print("  ok test_ocr_output_never_written_as_vision_caption")


def test_vision_engine_output_written_as_caption() -> None:
    """只有真正的 Vision 产出才写 ``vision_caption``."""
    result = _understanding(IMAGE_TYPE_CHART)
    _apply_output(result, EngineOutput(text="模型对图意的解读", engine="vision", ok=True))
    assert result.vision_caption == "模型对图意的解读"
    print("  ok test_vision_engine_output_written_as_caption")


def test_keep_ocr_text_never_caption_and_clears_table() -> None:
    """人工复核兜底：OCR 文本只留 ocr_text；表格类清空 structured_content."""
    shot = _understanding(IMAGE_TYPE_SCREENSHOT)
    _keep_ocr_text(shot, EngineOutput(text="界面文字", engine="paddleocr", ok=True))
    assert shot.vision_caption is None
    assert shot.ocr_text == "界面文字"

    table = _understanding(IMAGE_TYPE_TABLE)
    table.structured_content = "| a | b |\n|---|---|\n| 1 | 2 |"
    _keep_ocr_text(table, EngineOutput(text="残留文字", engine="ocr", ok=True))
    assert table.structured_content is None, "表格复核时不得保留伪表格"
    print("  ok test_keep_ocr_text_never_caption_and_clears_table")


def test_vision_unavailable_downgrades_route_and_does_not_fake_caption() -> None:
    """
    多模态不可用时：chart 的 route 从 vision 降级为 ocr，meta 如实标记，
    且``vision_caption`` 为空 —— 结果里不能出现"route=vision 却是 OCR 产出"。
    """
    from PIL import ImageDraw

    img = Image.new("RGB", (360, 240), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, colour in enumerate(((66, 133, 244), (219, 68, 55), (15, 157, 88))):
        draw.rectangle([40 + i * 100, 60, 120 + i * 100, 200], fill=colour)

    orig_cls = pl.classify_image_safe
    orig_avail = ve.VisionEngine.is_available
    pl.classify_image_safe = lambda image, ocr_lines=None, filename="": ImageClassification(  # type: ignore[assignment]
        image_type=IMAGE_TYPE_CHART, confidence=0.8, signals={}, engine="rules",
    )
    ve.VisionEngine.is_available = lambda self: False  # type: ignore[assignment]
    try:
        result = pl.understand_image(img, filename="chart-vision-off")
    finally:
        pl.classify_image_safe = orig_cls        # type: ignore[assignment]
        ve.VisionEngine.is_available = orig_avail  # type: ignore[assignment]

    assert result.image_type == IMAGE_TYPE_CHART
    assert result.meta.get("vision_unavailable") is True, result.meta
    assert result.route == "ocr", f"vision 不可用时应把 route 降级为 ocr（实际 {result.route}）"
    assert result.meta.get("route_downgraded") is True, result.meta
    assert result.meta.get("original_route") == "vision", result.meta
    assert result.vision_caption is None, "vision 不可用时不得把 OCR 文本写进 vision_caption"
    print("  ok test_vision_unavailable_downgrades_route_and_does_not_fake_caption")


# ─────────────────────────────────────────────────────────────────────────────
# 任务 B · 分类器：表格判 table、彩色图表 / UI 截图不判 table
# ─────────────────────────────────────────────────────────────────────────────


def _signals(**overrides) -> dict:
    base = {
        "width": 384, "height": 300, "inverted": False, "polarity_source": "page-ring",
        "h_lines": 0, "v_lines": 0,
        "text_bands": 6, "line_art_ratio": 0.9,
        "saturation": 0.0, "flat_blocks": 2, "chromatic_blocks": 0,
        "dominant_ratio": 0.8, "colorful_ratio": 0.0,
        "ocr_rows": 6, "ocr_cols": 9, "ocr_multi_cell_rows": 5,
        "ocr_alignment": 0.83, "ocr_col_stability": 0.50,
        "code_score": 0.0, "code_language": "text", "math_ratio": 0.0,
        "avg_line_chars": 5.0, "cjk_ratio": 0.4,
    }
    base.update(overrides)
    return base


def test_ruling_line_table_is_table() -> None:
    kind, _, reason = _decide(_signals(h_lines=6, v_lines=5), get_settings())
    assert kind == IMAGE_TYPE_TABLE, (kind, reason)
    print("  ok test_ruling_line_table_is_table")


def test_ocr_grid_table_is_table() -> None:
    """无框线但列稳定（列被多数行支撑）→ 判 table（t3/t4 阴影/透视后仍要能还原）."""
    kind, _, reason = _decide(
        _signals(h_lines=6, v_lines=0, ocr_alignment=0.83, ocr_col_stability=0.50),
        get_settings(),
    )
    assert kind == IMAGE_TYPE_TABLE, (kind, reason)
    print("  ok test_ocr_grid_table_is_table")


def test_colorful_chart_is_not_table() -> None:
    """彩色柱图：大面积彩色 + 统一背景 → chart，绝不判 table（real2_bar 的误报）."""
    kind, _, reason = _decide(
        _signals(
            h_lines=0, v_lines=0, saturation=0.11, colorful_ratio=0.18,
            dominant_ratio=0.67, ocr_rows=10, ocr_cols=17, ocr_multi_cell_rows=8,
            ocr_alignment=0.80, ocr_col_stability=0.13, text_bands=8,
            line_art_ratio=0.78,
        ),
        get_settings(),
    )
    assert kind == IMAGE_TYPE_CHART, (kind, reason)
    print("  ok test_colorful_chart_is_not_table")


def test_ui_screenshot_with_high_alignment_is_not_table() -> None:
    """UI 截图：OCR alignment 高但列稳定性低 → 绝不判 table（ui_documents/ui_quality 的误报）."""
    kind, _, reason = _decide(
        _signals(
            h_lines=0, v_lines=0, ocr_rows=11, ocr_cols=12, ocr_multi_cell_rows=8,
            ocr_alignment=0.73, ocr_col_stability=0.18, text_bands=0,
            line_art_ratio=0.96, dominant_ratio=0.93,
        ),
        get_settings(),
    )
    assert kind != IMAGE_TYPE_TABLE, (kind, reason)
    print("  ok test_ui_screenshot_with_high_alignment_is_not_table")


# ─────────────────────────────────────────────────────────────────────────────
# 执行（直接 python 运行本文件时）
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("文档照片表格表头修复回归测试（A/B/C/D）")
    print("\n[任务 A] 题注行不得占据表头位")
    test_caption_line_is_detected()
    test_caption_grid_not_header_and_not_lost()
    test_alignment_keeps_real_header_after_caption_strip()
    print("\n[任务 C] 退化表必须显式失败")
    test_ok_predicate_rejects_single_column()
    test_collapsed_alignment_table_fails()
    print("\n[任务 D] vision 不可用要可见")
    test_ocr_output_never_written_as_vision_caption()
    test_vision_engine_output_written_as_caption()
    test_keep_ocr_text_never_caption_and_clears_table()
    test_vision_unavailable_downgrades_route_and_does_not_fake_caption()
    print("\n[任务 B] 分类器")
    test_ruling_line_table_is_table()
    test_ocr_grid_table_is_table()
    test_colorful_chart_is_not_table()
    test_ui_screenshot_with_high_alignment_is_not_table()
    print("\nAll doc-photo header fix regression tests passed.")


if __name__ == "__main__":
    main()
