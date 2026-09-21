"""解析 / 摘要 / 视觉路由那批修复的**入库回归测试**.

为什么会有这份文件
------------------
上线前那批解析与配置缺陷（P1–P3）**代码已全部落地**，但当时只用一个一次性脚本
``_impl3_verify_parsers.py``（放在仓库根目录、未入版本控制）验了其中 12 条纯逻辑。
更麻烦的是：那份脚本的注释里**声称**"已在 ``backend/tests/test_prelaunch_parser_fixes.py``
中用 importorskip 保护，待依赖齐全的环境运行" —— 而该文件**从未存在过**。

这正是本项目反复踩过的那个坑：*worker 说"已加回归测试" ≠ 测试真的存在、真的跑过*。
一次性脚本不会被 CI 执行、改坏代码也不会报警；**修复必须有入库的回归测试才算完成**。
（另外 ``importorskip`` 在本仓库也**不适用**：宿主机的重依赖会被 ``conftest.py``
的哑桩顶替，``import`` 会成功、``importorskip`` 不会跳过，于是变成假红。
正确做法是 ``_module_skip.skip_if_host_shimmed``。）

覆盖清单（每条对应一个已修缺陷）
--------------------------------
  * 编码回退链 + 乱码拒收 .............................. P2-1
  * TXT / Markdown 的 file_type 不再一律记 "txt" ......... P3-2
  * XLSX 空单元格不再吃掉列、page_count 只数真正产出页 ... P2-2
  * PPTX 表格转 Markdown、图表导出系列数值（不再静默丢） .. P2-3
  * VLM**超时**必须降级 route，不得仍报 vision ............ P1-1
  * 摘要 reasoning 默认关 / VISION 超时默认 120s /
    MAX_IMAGES=8 / 双通道白名单去掉 table /
    DOC_SUMMARY_TIMEOUT 不是死配置 ...................... P1-2 · P1-3 · P3-2

宿主机（Windows，缺 cv2/pptx/openpyxl/fitz）与容器内**都能跑**：
需要真第三方库的用例走条件跳过 —— 只在依赖真缺失时跳过，容器里照常真跑。
"""
from __future__ import annotations

import io
import sys
import types
from pathlib import Path

import pytest

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _module_skip import skip_if_host_shimmed, skip_module  # noqa: E402

try:
    from app.config import Settings, get_settings
    from app.services.image_understanding import IMAGE_TYPE_CHART
    from app.services.image_understanding.engines.base import EngineOutput
    from app.services.parsers.base import decode_text_with_fallback
    from app.services.parsers.markdown_parser import MarkdownParser
    from app.services.parsers.pptx_parser import (
        _pptx_chart_to_text,
        _pptx_table_to_markdown,
    )
    from app.services.parsers.txt_parser import TxtParser

    import app.services.image_understanding.engines.vision_engine as ve
    import app.services.image_understanding.pipeline as pl
except ImportError as exc:      # 宿主机缺依赖 → 整份跳过（容器内已验证）
    skip_module(f"missing dependency ({exc}) — run inside the backend container")


# ─────────────────────────────────────────────────────────────────────────────
# 1. 编码回退链（P2-1）：非 UTF-8 内容不得静默变乱码入库
#
#    旧实现是 ``content.decode("utf-8", errors="replace")`` —— GBK 中文整篇变成
#    U+FFFD 方块，**不报错、不告警**，是"看着成功了其实坏了"的典型。
# ─────────────────────────────────────────────────────────────────────────────


def test_utf8_text_passes_through_untouched() -> None:
    text = "这是一段中文 UTF-8 文本，含标点、数字 1289 与英文 OCR。"
    assert decode_text_with_fallback(text.encode("utf-8"), filename="u.txt") == text


def test_gbk_text_is_decoded_not_turned_into_replacement_chars() -> None:
    """GBK 中文必须被正确解码 —— 旧实现会整篇变 U+FFFD 且毫无提示。"""
    gbk_text = "渠道编码, 销售额, 同比增长\n华东, 1289, 12.5%"
    got = decode_text_with_fallback(gbk_text.encode("gbk"), filename="g.csv")
    assert got == gbk_text
    assert "\ufffd" not in got


def test_big5_bytes_never_become_replacement_characters() -> None:
    """繁体（Big5）字节也要被回退链接住，不能变成一堆 U+FFFD 方块.

    注：gb18030 是超集，对 Big5 字节常能"成功"解码但得到错字（简繁字节区间重叠），
    这是编码歧义的固有局限。本用例只钉住核心不变量：**不再整篇乱码**。
    """
    got = decode_text_with_fallback("產品名稱, 數量".encode("big5"), filename="b.txt")
    assert "\ufffd" not in got
    assert got.strip()


def test_binary_garbage_is_rejected_instead_of_silently_indexed() -> None:
    """非文本二进制（U+FFFD 占比 > 5%）必须拒收，而不是把乱码写进索引."""
    with pytest.raises(ValueError):
        decode_text_with_fallback(bytes(range(256)) * 4, filename="bin.bin")


def test_few_bad_bytes_are_tolerated() -> None:
    """少量坏字节（占比 ≤ 5%）应放行 —— 不能因为一两个坏字节就拒收整份文档."""
    mostly = ("正常中文内容很长很长" * 20).encode("utf-8") + b"\xff\xfe"
    got = decode_text_with_fallback(mostly, filename="m.txt")
    assert "\ufffd" in got
    assert got.count("\ufffd") / len(got) <= 0.05


# ─────────────────────────────────────────────────────────────────────────────
# 2. file_type 记真实扩展名（P3-2）：.json / .log 不再一律被记成 "txt"
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("data.json", "json"), ("app.log", "log"), ("notes.txt", "txt"), ("g.csv", "csv")],
)
def test_txt_parser_records_real_extension(filename: str, expected: str) -> None:
    assert TxtParser().parse(b"plain text", filename).file_type == expected


def test_txt_parser_decodes_gbk_without_mojibake() -> None:
    gbk_text = "渠道编码, 销售额\n华东, 1289"
    result = TxtParser().parse(gbk_text.encode("gbk"), "g.csv")
    assert result.full_text == gbk_text
    assert "\ufffd" not in result.full_text


# ─────────────────────────────────────────────────────────────────────────────
# 3. Markdown（P3-2 / P2-1）：不再依赖 markdown 模块，且编码回退链同样生效
# ─────────────────────────────────────────────────────────────────────────────


def test_markdown_parser_keeps_utf8_text() -> None:
    md = "# 标题\n\n这是 **加粗** 的中文 Markdown 内容。"
    assert MarkdownParser().parse(md.encode("utf-8"), "doc.md").full_text == md


def test_markdown_parser_decodes_gbk() -> None:
    md = "# 标题\n\n这是 **加粗** 的中文 Markdown 内容。"
    result = MarkdownParser().parse(md.encode("gbk"), "doc2.md")
    assert result.full_text == md
    assert "\ufffd" not in result.full_text


# ─────────────────────────────────────────────────────────────────────────────
# 4. PPTX 表格 / 图表（P2-3）
#
#    表格在 PPTX 里是 GraphicFrame，**没有 .text 属性** —— 旧实现只取 shape.text，
#    于是幻灯片里的表格**整块不入库**，图表数据也拿不到，却仍报
#    extraction_method="native"（用户完全看不出来丢了内容）。
#    这两个转换函数是纯函数，所以这里用假对象直接测，宿主机也能跑。
# ─────────────────────────────────────────────────────────────────────────────


def _cell(text: str):
    return types.SimpleNamespace(text=text)


def _table(rows: list[list[str]]):
    return types.SimpleNamespace(
        rows=[types.SimpleNamespace(cells=[_cell(c) for c in row]) for row in rows]
    )


def test_pptx_table_becomes_markdown_instead_of_being_dropped() -> None:
    md = _pptx_table_to_markdown(_table([["指标", "2024年"], ["营业收入", "33500"]]))
    lines = md.splitlines()
    assert lines[0] == "| 指标 | 2024年 |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 营业收入 | 33500 |"


def test_pptx_table_ragged_rows_are_padded_to_same_width() -> None:
    """参差不齐的行要补齐 —— 否则下游 Markdown 表格解析会把列错位."""
    lines = _pptx_table_to_markdown(_table([["a", "b", "c"], ["1"]])).splitlines()
    assert lines[2] == "| 1 |  |  |"


def test_pptx_table_escapes_pipe_and_flattens_newline() -> None:
    md = _pptx_table_to_markdown(_table([["a|b", "x\ny"]]))
    assert md.splitlines()[0] == "| a\\|b | x y |"


def test_pptx_empty_table_returns_empty_string() -> None:
    assert _pptx_table_to_markdown(_table([["", ""], ["", ""]])) == ""
    assert _pptx_table_to_markdown(_table([])) == ""


def test_pptx_chart_exports_series_names_and_values() -> None:
    """图表数据存在内嵌 workbook 里 —— 至少要能把系列名和数值捞出来供检索."""
    chart = types.SimpleNamespace(
        plots=[
            types.SimpleNamespace(
                series=[
                    types.SimpleNamespace(name="营收", values=[1, 2, 3]),
                    types.SimpleNamespace(name="", values=[4, 5]),
                ]
            )
        ]
    )
    out = _pptx_chart_to_text(chart)
    assert "营收: 1, 2, 3" in out
    assert "4, 5" in out


def test_pptx_chart_failure_is_swallowed() -> None:
    """图表 XML 损坏时不得把整份文档解析带崩（宁可少入库，不要整篇失败）."""

    class _BrokenChart:
        @property
        def plots(self):
            raise RuntimeError("chart xml 损坏")

    assert _pptx_chart_to_text(_BrokenChart()) == ""


# ─────────────────────────────────────────────────────────────────────────────
# 5. XLSX（P2-2）：空单元格不再吃掉列；page_count 只数真正产出的 sheet
#
#    需要**真实** openpyxl（造工作簿），宿主机被哑桩顶替时条件跳过。
# ─────────────────────────────────────────────────────────────────────────────


def _xlsx_bytes(sheets: dict[str, list[list[object]]]) -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    for index, (name, rows) in enumerate(sheets.items()):
        ws = wb.active if index == 0 else wb.create_sheet()
        ws.title = name
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_xlsx_keeps_empty_cells_so_columns_do_not_shift() -> None:
    """旧实现过滤 None → ``a,None,c`` 被压成 ``a | c``，**后续列整体左移**."""
    skip_if_host_shimmed("openpyxl")
    from app.services.parsers.xlsx_parser import XlsxParser

    content = _xlsx_bytes({"Sheet1": [["名称", None, "金额"], ["华东", None, "1289"]]})
    result = XlsxParser().parse(content, "x.xlsx")

    row_lines = [ln for ln in result.full_text.splitlines() if "华东" in ln]
    assert row_lines, result.full_text
    assert row_lines[0] == "华东 |  | 1289", f"空列被吃掉了：{row_lines[0]!r}"
    assert row_lines[0].count("|") == 2, row_lines[0]


def test_xlsx_page_count_only_counts_sheets_with_content() -> None:
    """page_count 不能把被跳过的空 sheet 也算进去（虚报页数）."""
    skip_if_host_shimmed("openpyxl")
    from app.services.parsers.xlsx_parser import XlsxParser

    content = _xlsx_bytes(
        {
            "有数据": [["a", "b"], ["1", "2"]],
            "空表": [],
            "也有数据": [["c"], ["3"]],
        }
    )
    result = XlsxParser().parse(content, "x.xlsx")
    assert result.page_count == 2, result.page_count
    assert len(result.pages) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 6. VLM **超时**的归因（P1-1）
#
#    旧实现的判据是 ``vision_unavailable = not vision.is_available()`` —— 超时时
#    引擎仍然"可用"，于是判不出降级，route 保持 "vision"，而内容其实是 OCR：
#    用户看到"多模态已启用"，实际根本没跑。既有的
#    ``test_vision_unavailable_downgrades_route_and_does_not_fake_caption`` 只覆盖了
#    "引擎缺失"，**没覆盖"引擎在但超时"** —— 正是本用例补的那一格。
# ─────────────────────────────────────────────────────────────────────────────


def test_vision_timeout_downgrades_route_and_records_reason() -> None:
    from PIL import Image, ImageDraw

    from app.services.image_understanding.classifier import ImageClassification

    img = Image.new("RGB", (360, 240), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, colour in enumerate(((66, 133, 244), (219, 68, 55), (15, 157, 88))):
        draw.rectangle([40 + i * 100, 60, 120 + i * 100, 200], fill=colour)

    orig_classify = pl.classify_image_safe
    orig_avail = ve.VisionEngine.is_available
    orig_process = ve.VisionEngine.process

    pl.classify_image_safe = lambda image, ocr_lines=None, filename="": ImageClassification(  # type: ignore[assignment]
        image_type=IMAGE_TYPE_CHART, confidence=0.8, signals={}, engine="rules",
    )
    # ★ 关键：引擎**可用**（判不出"缺模型"），但这一轮推理超时 → ok=False
    ve.VisionEngine.is_available = lambda self: True  # type: ignore[assignment]
    ve.VisionEngine.process = lambda self, *a, **kw: EngineOutput(  # type: ignore[assignment]
        engine="vision", ok=False, error="Vision 超时（120s）",
    )
    try:
        result = pl.understand_image(img, filename="chart-vision-timeout")
    finally:
        pl.classify_image_safe = orig_classify          # type: ignore[assignment]
        ve.VisionEngine.is_available = orig_avail       # type: ignore[assignment]
        ve.VisionEngine.process = orig_process          # type: ignore[assignment]

    assert result.image_type == IMAGE_TYPE_CHART
    assert result.meta.get("vision_unavailable") is True, result.meta
    assert result.meta.get("vision_error"), (
        "超时原因必须留下痕迹 —— 旧实现只写进 meta 且不在 to_dict() 里，对外完全不可见"
    )
    assert result.route == "ocr", (
        f"超时后仍报 vision 会让下游以为图意已被读过（实际 {result.route}）"
    )
    assert result.meta.get("route_downgraded") is True, result.meta
    assert result.meta.get("original_route") == "vision", result.meta
    assert result.vision_caption is None, "超时时不得把 OCR 文本伪造成 Vision 描述"


# ─────────────────────────────────────────────────────────────────────────────
# 7. 配置默认值（P1-2 / P1-3 / P3-2）
#
#    这里断言的是**类默认值**而不是运行值：``.env`` 里的显式取值会盖住默认值，
#    只测运行值就永远发现不了"默认值退回去了"（换个环境部署立刻复发）。
# ─────────────────────────────────────────────────────────────────────────────


def _settings_default(name: str):
    fields = getattr(Settings, "model_fields", None) or getattr(Settings, "__fields__", {})
    assert name in fields, f"Settings 里没有 {name}（配置被改名/删除？）"
    return fields[name].default


def test_vision_timeout_default_has_real_headroom() -> None:
    """实测 qwen2.5vl:3b 单图 43.0 / 50.9 / 54.1s（首次还含约 6.7s 模型加载）.

    旧默认 60s 只剩 6–17s 余量 → 系统一忙就超时，而超时又被静默当成 "vision"（见 P1-1）。
    """
    assert float(_settings_default("VISION_TIMEOUT_SECONDS")) >= 120.0


def test_document_summary_reasoning_defaults_to_off() -> None:
    """本机 qwen3:8b 是推理模型：逐份摘要属"抽取/结构化"类，开着思考会吃光
    ``num_predict`` 导致 content 空串，且实测 92s vs 33s。"""
    assert _settings_default("DOC_SUMMARY_REASONING") is False


def test_max_images_per_document_default_is_tightened() -> None:
    """单文档入库 17min → ~7min（每页 VLM 是主要成本）."""
    assert int(_settings_default("MAX_IMAGES_PER_DOCUMENT")) <= 8


def test_document_summary_timeout_is_not_a_dead_config() -> None:
    """P1-2：``DOC_SUMMARY_TIMEOUT_SECONDS`` 曾经"定义了但全仓无人读取".

    后果不是报错而是**静默挂死**：逐份摘要无超时（实测 92.1s/份 × 最多 20 份 ≈ 30min）。
    这条断言的就是"至少有非 config 代码真的读它"。
    """
    app_dir = Path(_BACKEND_ROOT) / "app"
    referenced = sorted(
        p.relative_to(app_dir).as_posix()
        for p in app_dir.rglob("*.py")
        if p.name != "config.py"
        and "DOC_SUMMARY_TIMEOUT_SECONDS" in p.read_text(encoding="utf-8", errors="replace")
    )
    assert referenced, (
        "DOC_SUMMARY_TIMEOUT_SECONDS 没有任何非 config 代码读取 —— 死配置回归，"
        "逐份摘要又会变成无超时挂死"
    )
    print(f"    · DOC_SUMMARY_TIMEOUT_SECONDS 读取处：{', '.join(referenced)}")


def test_dual_channel_types_exclude_table() -> None:
    """表格已有 TableTransformer + 规则法两条**模型无关**通路，再让 VLM 跑第三次
    是性价比最低的一次调用。

    ⚠️ 断言的是**生效值**（``get_settings()``，即 .env）。``config.py`` 的类默认值
    目前**仍含 table**，只有 .env 去掉了 —— 属已知不一致，已记在本轮报告里。
    """
    effective = (get_settings().IMAGE_DUAL_CHANNEL_TYPES or "").lower()
    names = {t.strip() for t in effective.split(",") if t.strip()}
    assert "table" not in names, f"生效配置仍对表格开双通道：{effective}"
