"""
图片噪声处理 + OCR/VLM 双通道 + 产出质量校验 单元测试.

覆盖三层新能力：

**一、噪声处理 / 预处理（preprocess.py）**
    · ``estimate_noise`` —— Immerkær 估计能区分"干净导出图"与"有噪翻拍图"；
    · ``contrast_span`` —— **回归**：早期用 P5/P95 会把"黑字白纸"误判成
      零对比度（墨迹只占 1%~5% 像素，两个分位都落在白底上）；
    · ``border_trim`` —— 扫描黑边检测；
    · ``pick_mode`` —— 保守性：干净图一步都不做；
    · ``preprocess`` —— **双输出通道**：二值化只进 ocr_image，不污染给分类/
      Vision 用的 image（否则二值图的色彩统计全废，图表会被判成截图）。

**二、质量校验（quality.py）**
    · 代码语法：Python 走 ``ast.parse``（真解析器）、JSON 走 ``json.loads``、
      其余走**字符串感知**的括号配平（``print("(")`` 里的括号不算括号）；
    · OCR confidence：均值 / 最低行 / 低置信行占比，且**全部为 0 时判
      "未上报"而不是"全错"**（部分 OCR 后端不返回逐行置信度）；
    · VLM 幻觉：提示词泄漏、复读、围栏缺失、数字锚点。

**三、双通道融合（dual_channel.py）**
    · 三种策略（互补 / 视觉优先 / OCR 优先）各自的选择规则；
    · 代码类型用**语法校验**当裁判：谁解析得通就用谁；
    · ``FusionResult.confidence`` 必须是**引擎原始分**（质检折损由门控做
      唯一一次），否则会平方衰减把好产出压到阈值以下。

宿主机缺依赖时优雅跳过；推荐在 backend 容器内运行：
    docker exec rag_backend python tests/test_image_noise_quality.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

os.environ.setdefault("POSTGRES_PASSWORD", "test-placeholder")
_TMP_ROOT = tempfile.mkdtemp(prefix="rag_noise_test_")
os.environ.setdefault("IMAGE_STORAGE_DIR", os.path.join(_TMP_ROOT, "uploads"))

# 桩包：避免触发 app/services/__init__.py 的重依赖（asyncpg / paddle / docling）
if "app.services" not in sys.modules:
    _pkg = types.ModuleType("app.services")
    _pkg.__path__ = [str(Path(_BACKEND_ROOT) / "app" / "services")]
    sys.modules["app.services"] = _pkg

try:
    import importlib

    import numpy as np
    from PIL import Image, ImageDraw

    from app.services.image_understanding.confidence import (
        apply_quality,
        gate,
        validate,
    )
    from app.services.image_understanding.engines.base import EngineOutput
    from app.services.image_understanding.structured_content import IMAGE_TYPE_CODE

    # 注意：包 __init__ 里导出了同名函数 ``preprocess``，会把子模块名遮蔽掉，
    # 所以子模块必须用 importlib 按全名取（``from ... import preprocess as pp``
    # 拿到的是函数，不是模块）。
    dc = importlib.import_module("app.services.image_understanding.dual_channel")
    pl = importlib.import_module("app.services.image_understanding.pipeline")
    pp = importlib.import_module("app.services.image_understanding.preprocess")
    q = importlib.import_module("app.services.image_understanding.quality")
except ImportError as exc:  # 宿主机缺依赖 → 跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    # 不能用 sys.exit()：pytest 在收集阶段导入本模块，抛 SystemExit 会让整个
    # 会话 INTERNALERROR，同目录其它用例全部跑不了。
    skip_module(f"missing dependency ({exc}) — run inside the backend container")


# ── 造图工具 ─────────────────────────────────────────────────────────────────

def _line_art(width=400, height=300, ink=(0, 0, 0), bg=(255, 255, 255), lines=8):
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    for i in range(lines):
        y = 40 + i * max(8, (height - 80) // max(lines, 1))
        if y >= height - 20:
            break
        draw.line([(40, y), (width - 40, y)], fill=ink, width=2)
    return img


def _add_gaussian_noise(img: Image.Image, sigma: float, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    noisy = np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def _scanned_page(border_px: int = 12, width=420, height=320, gray_band: bool = False) -> Image.Image:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width - 1, border_px - 1], fill=(5, 5, 5))
    draw.rectangle([0, height - border_px, width - 1, height - 1], fill=(5, 5, 5))
    draw.rectangle([0, 0, border_px - 1, height - 1], fill=(5, 5, 5))
    draw.rectangle([width - border_px, 0, width - 1, height - 1], fill=(5, 5, 5))
    for i in range(6):
        draw.line([(60, 60 + i * 35), (width - 60, 60 + i * 35)], fill=(0, 0, 0), width=2)
    if gray_band:
        # 中间灰底块：用来证明 image 通道**没有**被二值化（180 不会被压成 0/255）
        draw.rectangle([70, 250, width - 70, 290], fill=(180, 180, 180))
    return img


class _Line:
    """假 OCRLine（只带质检需要的三个属性）."""

    def __init__(self, text: str, confidence: float = 0.9, box=None):
        self.text = text
        self.confidence = confidence
        self.box = box


def _channel(name: str, text: str, confidence: float, image_type: str,
             engine: str = "", ocr_lines=None):
    out = EngineOutput(text=text, confidence=confidence, engine=engine or name, ok=True)
    quality = q.verify_output(text, image_type, engine=out.engine, ocr_lines=ocr_lines)
    return dc.ChannelResult(name, out, quality)


# ─────────────────────────────────────────────────────────────────────────────
# 一、噪声处理 / 预处理
# ─────────────────────────────────────────────────────────────────────────────

def test_estimate_noise_separates_clean_from_noisy() -> None:
    clean = _line_art()
    noisy = _add_gaussian_noise(clean, 18)
    mild = _add_gaussian_noise(clean, 5, seed=3)

    s_clean = pp.estimate_noise(clean)
    s_mild = pp.estimate_noise(mild)
    s_noisy = pp.estimate_noise(noisy)

    assert s_clean < 1.5, f"干净线稿图不应被估出噪声：{s_clean}"
    assert s_mild < s_noisy, f"噪声强度应单调：{s_mild} vs {s_noisy}"
    assert s_noisy > pp.NOISE_SIGMA_THRESHOLD, f"强噪声应超过阈值：{s_noisy}"
    print(f"  ok test_estimate_noise_separates_clean_from_noisy ({s_clean}/{s_mild}/{s_noisy})")


def test_contrast_span_handles_sparse_ink() -> None:
    """
    回归：墨迹稀疏时百分位法会失效.

    一页正常文档的墨迹往往只占 1%~5% 像素，用 P5/P95 两个分位会**双双落在
    白底上**，把"黑字白纸"判成零对比度 —— 早期版本就是这么错的，导致所有
    正常文档都被拉去做对比度归一化。
    """
    dense = _line_art(lines=8)
    sparse = _line_art(width=400, height=300, lines=1)     # 墨迹 < 1%
    faded = _line_art(ink=(130, 130, 130), bg=(215, 215, 215))
    flat = Image.new("RGB", (200, 200), (128, 128, 128))

    assert pp.contrast_span(dense) > 0.9, pp.contrast_span(dense)
    assert pp.contrast_span(sparse) > 0.9, f"稀疏墨迹不该被判成零对比度：{pp.contrast_span(sparse)}"
    assert pp.contrast_span(faded) < pp.LOW_CONTRAST_SPAN, pp.contrast_span(faded)
    assert pp.contrast_span(flat) == 0.0, pp.contrast_span(flat)
    print("  ok test_contrast_span_handles_sparse_ink")


def test_border_trim_detects_scan_border() -> None:
    assert pp.border_trim(_scanned_page(border_px=12)) >= 10
    assert pp.border_trim(_line_art()) == 0, "无边框的图不该被裁"
    # 太小 / 太薄的图不做判断（防止把内容误裁）
    assert pp.border_trim(Image.new("RGB", (10, 10), (0, 0, 0))) == 0
    print("  ok test_border_trim_detects_scan_border")


def test_pick_mode_is_conservative_on_clean_images() -> None:
    """干净图必须**一步都不做** —— 无谓去噪会把笔画磨糊、让 OCR 更差."""
    mode, signals = pp.pick_mode(_line_art(), "photo")
    assert mode == "light", (mode, signals)
    assert signals["mode_reason"].startswith("type-default")

    result = pp.preprocess(_line_art(), mode=mode)
    assert result.applied == [], result.applied
    print("  ok test_pick_mode_is_conservative_on_clean_images")


def test_pick_mode_escalates_on_evidence() -> None:
    noisy_mode, noisy_sig = pp.pick_mode(_add_gaussian_noise(_line_art(), 18), "photo")
    assert noisy_mode in ("noisy", "scan"), (noisy_mode, noisy_sig)
    assert "noise" in noisy_sig["mode_reason"]

    faded_mode, faded_sig = pp.pick_mode(
        _line_art(ink=(130, 130, 130), bg=(215, 215, 215)), "photo"
    )
    assert faded_mode == "noisy" and "low-contrast" in faded_sig["mode_reason"]

    scan_mode, scan_sig = pp.pick_mode(_scanned_page(), "photo")
    assert scan_mode == "scan" and "border" in scan_sig["mode_reason"], scan_sig

    # 类型默认档更"强"时以类型为准（表格已判定要纠偏，不该被降成 light）
    table_mode, _ = pp.pick_mode(_line_art(), "table")
    assert table_mode == "geometric", table_mode
    print("  ok test_pick_mode_escalates_on_evidence")


def test_preprocess_binarize_goes_to_ocr_channel_only() -> None:
    """
    **关键不变量**：二值化只能进 ocr_image，绝不能替换 image.

    否则二值图的色彩统计全废（所有像素非黑即白），图表会被分类器判成截图，
    Vision 也拿不到配色信息 —— "想要更好的 OCR 就得毁掉分类"是最蠢的取舍。
    """
    result = pp.preprocess(_scanned_page(gray_band=True), mode="scan")
    assert "binarize" in result.applied, result.applied
    assert result.ocr_image is not None, "scan 档应产出二值化的 OCR 输入"
    assert result.ocr_input is result.ocr_image

    # image 通道不能被二值化：灰阶层次必须保留（二值化只剩 0/255 两级）
    levels_image = np.unique(np.asarray(result.image.convert("L"))).size
    levels_ocr = np.unique(np.asarray(result.ocr_image.convert("L"))).size
    assert levels_ocr <= 2, f"ocr 通道应是二值图，实际 {levels_ocr} 级灰度"
    assert levels_image > levels_ocr, (
        f"image 通道疑似被二值化（{levels_image} 级 vs ocr 通道 {levels_ocr} 级）"
    )
    assert result.image.size == result.ocr_image.size

    # 无二值化的档位：ocr_input 直接复用 image
    light = pp.preprocess(_line_art(), mode="light")
    assert light.ocr_image is None
    assert light.ocr_input is light.image
    print("  ok test_preprocess_binarize_goes_to_ocr_channel_only")


def test_preprocess_is_fault_tolerant() -> None:
    """任何一张坏图都不该让入库挂掉 —— 预处理必须返回原图而不是抛."""
    for img in (Image.new("RGB", (1, 1)), Image.new("L", (3, 300)),
                Image.new("RGB", (5, 5), (0, 0, 0))):
        for mode in ("light", "text", "geometric", "noisy", "scan"):
            result = pp.preprocess(img, mode=mode)
            assert result.image is not None
    print("  ok test_preprocess_is_fault_tolerant")


# ─────────────────────────────────────────────────────────────────────────────
# 二、质量校验
# ─────────────────────────────────────────────────────────────────────────────

def test_code_syntax_python_uses_real_parser() -> None:
    ok = q.check_code_syntax("```python\ndef add(a, b):\n    return a + b\n```")
    assert ok.passed and "ast.parse" in ok.checks, ok.to_dict()
    assert ok.language == "python"

    bad = q.check_code_syntax("```python\ndef add(a, b)\n    return a + b\n```")
    assert not bad.passed
    assert any("expected ':'" in e for e in bad.errors), bad.errors
    # 围栏里的语言标签优先于调用方猜
    assert q.check_code_syntax("```py\nx = 1\n```").language == "python"
    print("  ok test_code_syntax_python_uses_real_parser")


def test_code_syntax_json() -> None:
    assert q.check_code_syntax('```json\n{"a": 1, "b": [2, 3]}\n```').passed
    bad = q.check_code_syntax('```json\n{"a": 1,}\n```')
    assert not bad.passed and bad.language == "json"
    print("  ok test_code_syntax_json")


def test_code_syntax_balance_is_string_aware() -> None:
    """
    括号配平必须**跳过字符串与注释**.

    天真地数 ``code.count("(")`` 会把 ``print("(")`` 判成括号不配平 ——
    而代码截图里带括号的字符串非常常见，这种误判会直接把好产出打回人工复核。
    """
    cases = [
        ("```python\nprint(\"(\")\n```", True),
        ("```python\nx = \"it's ok\"\n```", True),
        ("```javascript\nconst s = '}'\nfunction f() { return s }\n```", True),
        ("```javascript\n// ( 未闭合的注释括号\nfunction f() { return 1 }\n```", True),
        ("```javascript\n/* ( */\nfunction f() { return 1 }\n```", True),
        ("```javascript\nfunction f() {\n  return 1;\n```", False),
        ("```javascript\nconst x = a + b);\n```", False),
    ]
    for code, expected in cases:
        report = q.check_code_syntax(code)
        assert report.passed is expected, (code, report.to_dict())
    print("  ok test_code_syntax_balance_is_string_aware")


def test_code_syntax_truncation_and_empty() -> None:
    trunc = q.check_code_syntax("```javascript\nconst x = a &&\n```")
    assert not trunc.passed and "truncation" in trunc.checks

    empty = q.check_code_syntax("")
    assert not empty.passed and empty.score == 0.0
    print("  ok test_code_syntax_truncation_and_empty")


def test_ocr_confidence_reports() -> None:
    good = q.assess_ocr_confidence([_Line("营收 1200", 0.95), _Line("利润 300", 0.92)])
    assert good.passed and good.reported and good.mean > 0.9

    bad = q.assess_ocr_confidence([_Line("营收", 0.4), _Line("利润", 0.3), _Line("毛利", 0.35)])
    assert not bad.passed and bad.low_ratio == 1.0 and bad.reasons

    none = q.assess_ocr_confidence([])
    assert not none.passed and none.lines == 0
    print("  ok test_ocr_confidence_reports")


def test_ocr_confidence_unreported_is_not_failure() -> None:
    """
    全部为 0 时必须判"未上报"，不能判"全错".

    部分 OCR 后端（某些 Tesseract 配置 / Paddle 的检测阶段）不返回逐行置信度。
    一刀切判失败会把整条链路误伤成"人工复核"，让本来正常的文档全部降级。
    """
    unreported = q.assess_ocr_confidence([_Line("营收 1200", 0.0), _Line("利润 300", 0.0)])
    assert unreported.reported is False
    assert unreported.passed is True
    assert "未上报" in unreported.reasons[0]
    print("  ok test_ocr_confidence_unreported_is_not_failure")


def test_vlm_prompt_leak_and_repetition() -> None:
    leak = q.check_vlm_output(
        "请提取：1) 图表类型 2) 横轴纵轴 3) 数据点。用中文输出，只输出内容本身。",
        "chart",
    )
    assert "prompt-leak" in leak.checks and leak.score < 1.0, leak.to_dict()

    repeated = "这是一段重复内容。\n" * 6
    rep = q.check_vlm_output(repeated, "chart")
    assert "repetition" in rep.checks and rep.score < 1.0

    assert not q.check_vlm_output("", "chart").ok
    print("  ok test_vlm_prompt_leak_and_repetition")


def test_vlm_code_output_checked_by_syntax() -> None:
    good = q.check_vlm_output("```python\ndef f():\n    return 1\n```", "code")
    assert good.ok and "code-syntax" in good.checks

    bad = q.check_vlm_output("```python\ndef f(\n    return 1\n```", "code")
    assert bad.score < good.score
    assert any("语法校验未通过" in r for r in bad.reasons), bad.reasons

    unfenced = q.check_vlm_output("def f():\n    return 1", "code")
    assert "code-fence" in unfenced.checks
    print("  ok test_vlm_code_output_checked_by_syntax")


def test_vlm_number_anchor_flags_hallucination() -> None:
    """VLM 报的数字若一个都不在 OCR 文本里 → 降分（软信号，不直接否决）."""
    suspicious = q.check_vlm_output(
        "数据点为 999、888、777、666，趋势上升。", "chart",
        ocr_text="营收 1200 万元 1500 万元 Q1 Q2",
    )
    assert "number-anchor" in suspicious.checks
    assert suspicious.meta["number_anchor"] == 0.0
    assert suspicious.score < 1.0

    consistent = q.check_vlm_output(
        "Q1 为 1200 万元，Q2 为 1500 万元。", "chart",
        ocr_text="营收 1200 万元 1500 万元 Q1 Q2",
    )
    assert consistent.meta["number_anchor"] == 1.0
    print("  ok test_vlm_number_anchor_flags_hallucination")


def test_verify_output_combines_ocr_and_code() -> None:
    report = q.verify_output(
        "```python\ndef f():\n    return 1\n```", "code", engine="vision",
        ocr_lines=[_Line("def f(", 0.3), _Line("return 1", 0.35)],
    )
    assert report.engine == "vision"
    assert report.ocr is not None and report.ocr.passed is False
    assert "ocr-confidence" in report.checks
    # OCR 差 → 打折但不否决（产出可能来自 VLM，OCR 差并不代表 VLM 也差）
    assert 0.0 < report.score < 1.0, report.to_dict()
    print("  ok test_verify_output_combines_ocr_and_code")


# ─────────────────────────────────────────────────────────────────────────────
# 三、双通道融合
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion_strategy_mapping() -> None:
    assert dc.strategy_for("table") == dc.STRATEGY_COMPLEMENTARY
    assert dc.strategy_for("code") == dc.STRATEGY_COMPLEMENTARY
    assert dc.strategy_for("formula") == dc.STRATEGY_COMPLEMENTARY
    assert dc.strategy_for("chart") == dc.STRATEGY_VISION_FIRST
    assert dc.strategy_for("diagram") == dc.STRATEGY_VISION_FIRST
    assert dc.strategy_for("screenshot") == dc.STRATEGY_VISION_FIRST
    assert dc.strategy_for("photo") == dc.STRATEGY_OCR_FIRST
    # 白名单默认不含 photo（省 VLM 的钱）
    assert "code" in dc.dual_channel_types()
    assert "photo" not in dc.dual_channel_types()
    print("  ok test_fusion_strategy_mapping")


def test_fusion_complementary_prefers_structure() -> None:
    """互补型：结构通道质检通过 → 用结构（表格 Markdown 比描述更接近原图）."""
    ocr = _channel(dc.CHANNEL_OCR, "| 指标 | 值 |\n| --- | --- |\n| 营收 | 100 |", 0.85, "table",
                   engine="table_parser:rules+cell-ocr")
    vlm = _channel(dc.CHANNEL_VLM, "表格列出了指标与对应的数值。", 0.75, "table", engine="vision")
    fused = dc.fuse_channels("table", [ocr, vlm])
    assert fused.chosen == dc.CHANNEL_OCR
    assert fused.text.startswith("| 指标 |")
    assert fused.strategy == dc.STRATEGY_COMPLEMENTARY
    print("  ok test_fusion_complementary_prefers_structure")


def test_fusion_complementary_switches_when_structure_fails() -> None:
    ocr = _channel(dc.CHANNEL_OCR, "乱码 一二三", 0.2, "formula")           # 质检不过
    vlm = _channel(dc.CHANNEL_VLM, "$$E = mc^2$$", 0.75, "formula", engine="vision")
    fused = dc.fuse_channels("formula", [ocr, vlm])
    assert fused.chosen == dc.CHANNEL_VLM
    assert any("结构通道未通过质检" in n for n in fused.notes), fused.notes
    print("  ok test_fusion_complementary_switches_when_structure_fails")


def test_fusion_code_uses_syntax_as_arbiter() -> None:
    """
    代码类型：**语法校验当裁判**.

    OCR 通道分更高但语法不成立，VLM 通道分更低但能通过 ast.parse ——
    这时必须选 VLM。这是"OCR 自信地读错"最典型的一种，也是双通道存在的
    核心理由。
    """
    broken = _channel(
        dc.CHANNEL_OCR,
        "```python\ndef f(\n    return 1\n```", 0.95, "code", engine="ocr+code-parser",
    )
    fixed = _channel(
        dc.CHANNEL_VLM,
        "```python\ndef f():\n    return 1\n```", 0.60, "code", engine="vision",
    )
    assert broken.rank > fixed.rank, (
        f"前提：坏通道的排序分更高（模拟 OCR 自信地错）"
        f" {broken.rank} vs {fixed.rank}"
    )

    fused = dc.fuse_channels("code", [broken, fixed])
    assert fused.chosen == dc.CHANNEL_VLM, fused.to_dict()
    assert any("语法校验" in n for n in fused.notes), fused.notes
    print("  ok test_fusion_code_uses_syntax_as_arbiter")


def test_fusion_vision_first_prefers_vlm_even_with_lower_rank() -> None:
    ocr = _channel(dc.CHANNEL_OCR, "季度 营收 1200 1500", 0.95, "chart")
    vlm = _channel(
        dc.CHANNEL_VLM,
        "柱状图。横轴为季度，纵轴为营收（万元）。Q1=1200，Q2=1500，呈上升趋势。",
        0.75, "chart", engine="vision",
    )
    # 前提：OCR 引擎自评更高（0.95 vs 0.75）—— 视觉优先策略照样选 VLM
    assert ocr.output.confidence > vlm.output.confidence

    fused = dc.fuse_channels("chart", [ocr, vlm])
    # 图表的信息在语义里，OCR 只能读到零散标注 —— 引擎自评分高也不该赢
    assert fused.chosen == dc.CHANNEL_VLM, fused.to_dict()
    print("  ok test_fusion_vision_first_prefers_vlm_even_with_lower_rank")


def test_fusion_vision_first_falls_back_to_ocr() -> None:
    ocr = _channel(
        dc.CHANNEL_OCR,
        "柱状图横轴为季度，纵轴为营收（万元）：Q1 为 1200，Q2 为 1500，Q3 为 1800，整体呈上升趋势。",
        0.9, "chart",
    )
    broken_vlm = dc.ChannelResult(
        dc.CHANNEL_VLM,
        EngineOutput(text="这是一段重复内容。\n" * 6, confidence=0.75, engine="vision", ok=True),
        q.QualityReport(ok=False, score=0.3, reasons=["复读率 100%"]),
    )
    fused = dc.fuse_channels("chart", [ocr, broken_vlm])
    assert fused.chosen == dc.CHANNEL_OCR
    assert any("VLM 产出未通过质检" in n for n in fused.notes), fused.notes
    print("  ok test_fusion_vision_first_falls_back_to_ocr")


def test_fusion_ocr_first_keeps_ocr_when_good() -> None:
    ocr = _channel(dc.CHANNEL_OCR, "会议纪要：预算 1200 万元，负责人张三。", 0.9, "photo")
    vlm = _channel(dc.CHANNEL_VLM, "画面中有文字与人物。", 0.75, "photo", engine="vision")
    fused = dc.fuse_channels("photo", [ocr, vlm])
    assert fused.chosen == dc.CHANNEL_OCR
    assert fused.strategy == dc.STRATEGY_OCR_FIRST

    # OCR 质量差 = 行置信度整体偏低（不是"文本短"——短文本本身不是错）
    weak_ocr = _channel(
        dc.CHANNEL_OCR, "abc", 0.2, "photo",
        ocr_lines=[_Line("abc", 0.25), _Line("d", 0.3)],
    )
    fused2 = dc.fuse_channels("photo", [weak_ocr, vlm])
    assert fused2.chosen == dc.CHANNEL_VLM, fused2.to_dict()
    print("  ok test_fusion_ocr_first_keeps_ocr_when_good")


def test_fusion_single_channel_passthrough() -> None:
    only = _channel(dc.CHANNEL_OCR, "| a | b |\n| --- | --- |\n| 1 | 2 |", 0.8, "table")
    fused = dc.fuse_channels("table", [only])
    assert fused.chosen == dc.CHANNEL_OCR and fused.text == only.text
    assert any("仅 ocr 通道" in n for n in fused.notes)

    empty = dc.fuse_channels("table", [])
    assert empty.text == "" and empty.chosen == "none"
    print("  ok test_fusion_single_channel_passthrough")


def test_fusion_confidence_is_raw_not_discounted() -> None:
    """
    ``FusionResult.confidence`` 必须是**引擎原始自评**，不能预先折损.

    管线会把它连同 quality 一起交给门控，由门控做唯一一次折损。若这里先折
    一次、门控再折一次，等于平方衰减 —— 0.9 × 0.7 × 0.7 = 0.44，好产出也
    会被压到 Accept 阈值以下，整条链路退化成"什么都要兜底"。
    """
    ocr = _channel(dc.CHANNEL_OCR, "| a | b |\n| --- | --- |\n| 1 | 2 |", 0.8, "table")
    fused = dc.fuse_channels("table", [ocr])
    assert fused.confidence == 0.8, fused.to_dict()
    assert fused.rank <= fused.confidence, "排序分（含折损）不应高于原始分"

    # 门控只折一次：0.8 × quality.score
    gated = gate(
        EngineOutput(text=fused.text, confidence=fused.confidence, engine="t", ok=True),
        "table", ocr.quality,
    )
    assert gated.confidence == round(0.8 * ocr.quality.score, 4), gated.to_dict()
    print("  ok test_fusion_confidence_is_raw_not_discounted")


def test_merge_captions_deduplicates() -> None:
    assert dc.merge_captions("A 描述", "B 描述") == "A 描述\n\nB 描述"
    assert dc.merge_captions("A 描述", "A 描述") == "A 描述"
    assert dc.merge_captions("A 描述更长一些", "A 描述") == "A 描述更长一些"
    assert dc.merge_captions("A", "") == "A"
    assert dc.merge_captions("", "B") == "B"
    print("  ok test_merge_captions_deduplicates")


# ─────────────────────────────────────────────────────────────────────────────
# 四、与门控联动
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_quality_discounts_linearly() -> None:
    assert apply_quality(0.9, None) == 0.9
    quality = q.QualityReport(ok=False, score=0.5)
    assert apply_quality(0.9, quality) == 0.45
    assert apply_quality(1.0, q.QualityReport(ok=True, score=1.0)) == 1.0
    print("  ok test_apply_quality_discounts_linearly")


def test_gate_pushes_confident_but_wrong_output_to_fallback() -> None:
    """
    门控的核心价值：把"引擎很自信但产出是错的"拉回兜底.

    没有质检时，OCR 对"读错但很自信"的输出照样给 0.9 → 直接 Accept。
    """
    broken_code = "```python\ndef f(\n    return 1\n```"
    out = EngineOutput(text=broken_code, confidence=0.9, engine="ocr+code-parser", ok=True)

    assert gate(out, "code").accepted is True, "无质检时会被错误接受（这正是问题）"

    quality = q.verify_output(broken_code, "code", engine=out.engine)
    verdict = gate(out, "code", quality)
    assert verdict.accepted is False, verdict
    assert verdict.decision == "fallback"
    print("  ok test_gate_pushes_confident_but_wrong_output_to_fallback")


def test_validate_code_uses_parser_not_heuristics() -> None:
    ok = validate("code", "```python\ndef f():\n    return 1\n```")
    assert ok.passed and ok.score >= 0.8

    broken = validate("code", "```python\ndef f(\n    return 1\n```")
    assert broken.passed is False, broken

    # 普通自然语言不该被当成"通过的代码"
    prose = validate("code", "这是一段普通的自然语言说明，没有任何代码结构。")
    assert prose.passed is False, prose
    print("  ok test_validate_code_uses_parser_not_heuristics")


# ─────────────────────────────────────────────────────────────────────────────
# 五、管线端到端（monkeypatch，不依赖真实 OCR / VLM）
# ─────────────────────────────────────────────────────────────────────────────

class _VisionStub:
    """假的 VisionEngine：可配置"是否可用"与"返回什么文本"."""

    available = True
    text = "```python\ndef f():\n    return 1\n```"
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def is_available(self) -> bool:
        return type(self).available

    def process(self, image, *, png_bytes=None, image_type="photo", role="primary", **kw):
        type(self).calls += 1
        return EngineOutput(text=type(self).text, confidence=0.75, engine="vision", ok=True)


def test_understand_image_dual_channel_end_to_end() -> None:
    """
    管线级验证：OCR 通道语法崩了 → 双通道融合选出 VLM 的版本.

    全程 monkeypatch，不依赖真实 OCR / VLM：
      · ``_run_ocr``    → 造一段"丢了括号"的代码行
      · ``classify_image_safe`` → 直接判成 code
      · ``_dispatch``   → 返回语法崩掉的代码（模拟 OCR 自信地读错）
      · ``VisionEngine``→ 返回语法正确的代码
    """
    from app.services.image_understanding.classifier import ImageClassification

    originals = {
        "run_ocr": pl._run_ocr,
        "classify": pl.classify_image_safe,
        "dispatch": pl._dispatch,
        "vision": pl.VisionEngine,
    }
    saved = (_VisionStub.available, _VisionStub.text, _VisionStub.calls)
    try:
        broken = "```python\ndef f(\n    return 1\n```"
        lines = [_Line("def f(", 0.9, (10, 10, 80, 20)), _Line("return 1", 0.9, (20, 30, 90, 40))]

        pl._run_ocr = lambda image: EngineOutput(  # type: ignore[assignment]
            text="def f( return 1", confidence=0.9, engine="tesseract", ok=True, lines=lines,
        )
        pl.classify_image_safe = lambda image, ocr_lines=None, filename="": ImageClassification(  # type: ignore[assignment]
            image_type="code", confidence=0.8, signals={"code_score": 0.7}, engine="rules",
        )
        pl._dispatch = lambda *a, **kw: EngineOutput(  # type: ignore[assignment]
            text=broken, confidence=0.90, engine="ocr+code-parser", ok=True, lines=lines,
        )
        pl.VisionEngine = _VisionStub  # type: ignore[assignment]

        _VisionStub.available = True
        _VisionStub.text = "```python\ndef f():\n    return 1\n```"
        _VisionStub.calls = 0

        result = pl.understand_image(_line_art(), page_number=1, filename="code.png")

        assert result.image_type == "code"
        assert result.fusion, "应记录双通道融合结论"
        assert result.fusion["strategy"] == dc.STRATEGY_COMPLEMENTARY
        assert result.fusion["chosen"] == dc.CHANNEL_VLM, result.fusion
        assert result.quality, "应记录质检结论"
        assert result.ocr_confidence, "应记录 OCR 置信度评估"
        assert result.preprocess_mode, "应记录预处理档位"
        assert result.structured_content and "def f():" in result.structured_content
        assert result.decision == "accept", result.decision
        assert _VisionStub.calls == 1, "通道 B 应恰好跑一次 VLM"
        print("  ok test_understand_image_dual_channel_end_to_end")
    finally:
        pl._run_ocr = originals["run_ocr"]              # type: ignore[assignment]
        pl.classify_image_safe = originals["classify"]  # type: ignore[assignment]
        pl._dispatch = originals["dispatch"]            # type: ignore[assignment]
        pl.VisionEngine = originals["vision"]           # type: ignore[assignment]
        _VisionStub.available, _VisionStub.text, _VisionStub.calls = saved


def test_understand_image_vision_route_does_not_call_vlm_twice() -> None:
    """
    chart/diagram/screenshot 的专用引擎**本身就是 Vision** —— 不能再调一次.

    否则每张图表都要付两次模型推理的钱，而结果完全一样。
    """
    from app.services.image_understanding.classifier import ImageClassification

    originals = {
        "run_ocr": pl._run_ocr,
        "classify": pl.classify_image_safe,
        "dispatch": pl._dispatch,
        "vision": pl.VisionEngine,
    }
    saved = (_VisionStub.available, _VisionStub.text, _VisionStub.calls)
    try:
        lines = [_Line("Q1", 0.9, (10, 10, 40, 20)), _Line("1200", 0.9, (50, 10, 90, 20))]
        pl._run_ocr = lambda image: EngineOutput(  # type: ignore[assignment]
            text="Q1 1200", confidence=0.9, engine="tesseract", ok=True, lines=lines,
        )
        pl.classify_image_safe = lambda image, ocr_lines=None, filename="": ImageClassification(  # type: ignore[assignment]
            image_type="chart", confidence=0.8, signals={}, engine="rules",
        )
        pl._dispatch = lambda *a, **kw: EngineOutput(  # type: ignore[assignment]
            text="柱状图，横轴为季度，纵轴为营收（万元）。Q1 为 1200，Q2 为 1500，整体呈上升趋势。",
            confidence=0.75, engine="vision", ok=True, lines=lines,
        )
        pl.VisionEngine = _VisionStub  # type: ignore[assignment]
        _VisionStub.available, _VisionStub.text, _VisionStub.calls = True, "irrelevant", 0

        result = pl.understand_image(_line_art(), page_number=1, filename="chart.png")

        assert _VisionStub.calls == 0, "Vision 路由不该再额外调用 VLM"
        assert result.fusion["chosen"] == dc.CHANNEL_VLM
        assert result.fusion["strategy"] == dc.STRATEGY_VISION_FIRST
        assert result.decision in ("accept", "fallback", "pass", "manual_review")
        print("  ok test_understand_image_vision_route_does_not_call_vlm_twice")
    finally:
        pl._run_ocr = originals["run_ocr"]              # type: ignore[assignment]
        pl.classify_image_safe = originals["classify"]  # type: ignore[assignment]
        pl._dispatch = originals["dispatch"]            # type: ignore[assignment]
        pl.VisionEngine = originals["vision"]           # type: ignore[assignment]
        _VisionStub.available, _VisionStub.text, _VisionStub.calls = saved


# ─────────────────────────────────────────────────────────────────────────────
# 六、元数据链路（Qdrant payload ↔ RetrievedChunk）
# ─────────────────────────────────────────────────────────────────────────────

def test_analysis_fields_roundtrip() -> None:
    """质检/融合报告必须能原样落 payload 再读回来（旧索引缺字段时安全降级）."""
    from app.services.retrieval_service import RetrievedChunk, _analysis_fields

    quality = q.verify_output(
        "```python\ndef f():\n    return 1\n```", "code", engine="vision",
        ocr_lines=[_Line("def f()", 0.9)],
    )
    fusion = dc.fuse_channels(
        "code",
        [_channel(dc.CHANNEL_OCR, "```python\nx=1\n```", 0.8, "code")],
    )

    fields = _analysis_fields({
        "analyze_quality": quality.to_dict(),
        "analyze_fusion": fusion.to_dict(),
    })
    chunk = RetrievedChunk(
        document_id="d", filename="a.pdf", page_number=1, chunk_index=0,
        text="```python\ndef f():\n    return 1\n```", score=0.9,
        content_type="image", image_id="d-p1-i1", **fields,
    )
    assert chunk.quality_ok is True
    assert 0.0 < chunk.quality_score <= 1.0
    assert chunk.ocr_confidence > 0.8
    assert chunk.fusion_strategy == dc.STRATEGY_COMPLEMENTARY
    assert chunk.fusion_chosen == dc.CHANNEL_OCR

    # 旧索引 / 脏数据 → 降级为"无警示"，绝不误报
    for payload in ({}, {"analyze_quality": "nope"}, {"analyze_fusion": 42}):
        legacy = RetrievedChunk(
            document_id="d", filename="a.pdf", page_number=1, chunk_index=1,
            text="x", score=0.5, **_analysis_fields(payload),
        )
        assert legacy.quality_ok is True
        assert legacy.quality_score == 1.0
        assert legacy.quality_reasons == []
        assert legacy.ocr_confidence == 0.0
        assert legacy.fusion_strategy is None
        assert legacy.fusion_chosen is None
    print("  ok test_analysis_fields_roundtrip")


def test_quality_warning_surfaces_reasons() -> None:
    """前端引用卡片要拿到**可读的原因**，而不是一个孤零零的分数."""
    from app.services.retrieval_service import RetrievedChunk

    quality = q.verify_output(
        "```python\ndef f(\n    return 1\n```", "code", engine="ocr+code-parser",
    )
    chunk = RetrievedChunk(
        document_id="d", filename="a.pdf", page_number=1, chunk_index=0,
        text="```python\ndef f(\n    return 1\n```", score=0.9,
        content_type="image", image_id="d-p1-i1",
        analyze_quality=quality.to_dict(),
    )
    assert chunk.quality_ok is False
    assert chunk.quality_reasons, "未通过质检时必须给出原因"
    assert any("语法" in r for r in chunk.quality_reasons), chunk.quality_reasons
    print("  ok test_quality_warning_surfaces_reasons")


# ─────────────────────────────────────────────────────────────────────────────
# 执行
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("图片噪声处理 / 双通道 / 质量校验 单元测试")

    print("\n[一] 噪声处理与预处理")
    test_estimate_noise_separates_clean_from_noisy()
    test_contrast_span_handles_sparse_ink()
    test_border_trim_detects_scan_border()
    test_pick_mode_is_conservative_on_clean_images()
    test_pick_mode_escalates_on_evidence()
    test_preprocess_binarize_goes_to_ocr_channel_only()
    test_preprocess_is_fault_tolerant()

    print("\n[二] 产出质量校验")
    test_code_syntax_python_uses_real_parser()
    test_code_syntax_json()
    test_code_syntax_balance_is_string_aware()
    test_code_syntax_truncation_and_empty()
    test_ocr_confidence_reports()
    test_ocr_confidence_unreported_is_not_failure()
    test_vlm_prompt_leak_and_repetition()
    test_vlm_code_output_checked_by_syntax()
    test_vlm_number_anchor_flags_hallucination()
    test_verify_output_combines_ocr_and_code()

    print("\n[三] OCR / VLM 双通道融合")
    test_fusion_strategy_mapping()
    test_fusion_complementary_prefers_structure()
    test_fusion_complementary_switches_when_structure_fails()
    test_fusion_code_uses_syntax_as_arbiter()
    test_fusion_vision_first_prefers_vlm_even_with_lower_rank()
    test_fusion_vision_first_falls_back_to_ocr()
    test_fusion_ocr_first_keeps_ocr_when_good()
    test_fusion_single_channel_passthrough()
    test_fusion_confidence_is_raw_not_discounted()
    test_merge_captions_deduplicates()

    print("\n[四] 与置信度门控联动")
    test_apply_quality_discounts_linearly()
    test_gate_pushes_confident_but_wrong_output_to_fallback()
    test_validate_code_uses_parser_not_heuristics()

    print("\n[五] 管线端到端")
    test_understand_image_dual_channel_end_to_end()
    test_understand_image_vision_route_does_not_call_vlm_twice()

    print("\n[六] 元数据链路")
    test_analysis_fields_roundtrip()
    test_quality_warning_surfaces_reasons()

    print("\nAll image noise / dual-channel / quality tests passed.")


if __name__ == "__main__":
    main()
