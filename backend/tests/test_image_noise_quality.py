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
    # ★ 必须同步父包属性：否则 pytest 的字符串式 monkeypatch 解析
    #   `app.services.x.y` 时 import 会成功，却在 getattr(app, "services")
    #   处抛 AttributeError —— 污染同一会话中后跑的测试（实测打断
    #   test_citation_open_recheck.py 的 10 个用例）。真实 app.services 被
    #   导入时会自动覆盖该属性，因此这里不引入额外持久污染。
    import app as _app_pkg

    _app_pkg.services = _pkg

try:
    import importlib

    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

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


# ── 翻拍件造图工具（D2/D3/D4 防回归用）───────────────────────────────────────

def _shade_gradient(img: Image.Image, strength: float = 0.45) -> Image.Image:
    """水平渐变阴影：左亮右暗（低频），模拟"一半被阴影压暗的翻拍页"."""
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    h, w = a.shape[:2]
    gain = np.linspace(1.0, strength, w, dtype=np.float64).reshape(1, w, 1)
    return Image.fromarray(np.clip(a * gain, 0, 255).astype(np.uint8))


def _defocus(img: Image.Image, radius: float = 2.5) -> Image.Image:
    """失焦/运动模糊（高频被抹平 → 噪声估计反而偏低，只有 re-blur 比值测得到）."""
    return img.filter(ImageFilter.GaussianBlur(radius))


def _rotate(img: Image.Image, degrees: float) -> Image.Image:
    """把"文档页"转歪 degrees 度（PIL 正角 = 逆时针），白底填充."""
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False,
                      fillcolor=(255, 255, 255))


def _chart_placeholder(width: int = 300, height: int = 200) -> Image.Image:
    """
    仿金标集 png_clean.png 的"彩色图例"：白底 + 深色标题条 + 4 个饱和色块.

    它的块均值极差（≈0.33-0.39）来自**内容**而不是光照，与阴影在
    illumination_range 上不可分 —— 是 colorful 门槛必须存在的活证据。
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 15, width - 20, 40], fill=(50, 50, 50))
    colors = [(66, 133, 244), (234, 67, 53), (52, 168, 83), (251, 188, 5)]
    bar_w = (width - 30 - 3 * 10) // 4
    for i, color in enumerate(colors):
        x0 = 15 + i * (bar_w + 10)
        draw.rectangle([x0, 85, x0 + bar_w, 175], fill=color)
    return img


class _SettingsStub:
    """给 understand_image 用的最小 Settings（隔离环境变量差异）."""

    IMAGE_PREPROCESS_GEOMETRIC = True
    MAX_IMAGE_OCR_CHARS = 20000


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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_if_host_shimmed

    # binarize 走 cv2.adaptiveThreshold；宿主机没装 opencv 时 app 侧
    # ``_binarize`` 是 ``except Exception: return image, None``（静默"不做"），
    # 于是 applied 里没有 'binarize' —— 这是环境缺库，不是"二值化污染了 image 通道"。
    skip_if_host_shimmed("cv2")
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


# ── 翻拍件信号（D2）与纠偏（D3）：新增信号的防回归 ───────────────────────────

def test_pick_mode_escalates_on_uneven_light() -> None:
    """
    D2：阴影翻拍页必须升 noisy 且原因含 uneven-light.

    阴影是**低频渐变**：全局 contrast_span 仍是 1.0（黑字/白纸两个极端都在）、
    噪声被平滑、只有分块均值的极差测得到"一半亮一半暗"。历史上这类图被判成
    light，一行清洗都不做 —— 而它恰恰是 CLAHE 收益最大的输入。
    """
    shadow = _shade_gradient(_line_art(ink=(30, 30, 30)), 0.45)
    assert pp.contrast_span(shadow) >= pp.LOW_CONTRAST_SPAN, (
        "前提：阴影不该被全局对比度抓到（否则走的是 low-contrast 分支）"
    )
    mode, sig = pp.pick_mode(shadow)
    assert mode == "noisy", (mode, sig.get("mode_reason"))
    assert "uneven-light" in sig["mode_reason"], sig["mode_reason"]
    # 判据在**纸面**口径上：阴影把白纸也压暗 → 纸面极差跟着涨（实测 0.41）
    assert sig["illumination_paper_range"] >= pp.ILLUMINATION_RANGE_THRESHOLD
    assert sig["illumination_gradient"] is True, (
        "前提：阴影是低频渐变，形态判据应成立"
    )
    assert sig["colorful_fraction"] < pp.COLORFUL_FRACTION_THRESHOLD, (
        "前提：阴影翻拍页是'纸+墨'，饱和色应接近 0"
    )

    # 清洗确实发生了：CLAHE（对比度归一化）进 applied
    result = pp.preprocess(shadow, mode=mode)
    assert "contrast" in result.applied, result.applied
    print("  ok test_pick_mode_escalates_on_uneven_light")


def test_pick_mode_escalates_on_defocus() -> None:
    """
    D2：失焦翻拍件必须升 noisy 且原因含 blur.

    模糊把高频噪声抹平 → 噪声估计**反而低于阈值**；对比度也不变（黑仍是黑、
    白仍是白）。唯一测得到的是清晰度：re-blur 比值（原图拉普拉斯方差 /
    高斯模糊后拉普拉斯方差）—— 锐利边缘"再糊一次会明显变糊"，已经糊的不会。

    造图用**粗笔画**页 + blur(2.5)：细笔画被 blur 抹没了会把 span 拉低、误走
    low-contrast 分支（实测 width=2 + blur(2.5) 的 span 只有 0.43）—— 这张图
    要验证的是"对比度正常但失焦"也能被抓住（实测 sharp=1.57、span=0.75）。
    """
    thick = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(thick)
    for i in range(8):
        y = 40 + i * 30
        draw.line([(40, y), (360, y)], fill=(0, 0, 0), width=6)
    blurred = _defocus(thick, 2.5)
    ratio = pp.sharpness_ratio(blurred)
    assert ratio < pp.SHARPNESS_MIN_RATIO, ratio
    assert pp.contrast_span(blurred) >= pp.LOW_CONTRAST_SPAN, (
        "前提：这张图的对比度正常（否则走的是 low-contrast 分支，测不到本信号）"
    )
    assert pp.estimate_noise(blurred) < pp.NOISE_SIGMA_THRESHOLD, (
        "前提：失焦件的噪声估计低于阈值（否则走的是 noise 分支）"
    )
    mode, sig = pp.pick_mode(blurred)
    assert mode == "noisy", (mode, sig.get("mode_reason"))
    assert "blur" in sig["mode_reason"], sig["mode_reason"]
    print("  ok test_pick_mode_escalates_on_defocus")


def test_colorful_chart_not_escalated_by_uneven_light() -> None:
    """
    **负例回归**：实心色块的图表不得因 illumination_range 被误升档.

    真实金标集里的彩色图例（png_clean.png，白底+深色标题条+4 个饱和色块）
    实测块均值极差 = 0.387 —— 纯**内容**就能超过任何拦得住阴影（实测可低到
    0.26）的阈值，二者在块均值上原理上不可分。colorful 门槛负责把它们让开：
    CLAHE 对彩色图要么无用要么有害，图表本就该保持 light。
    """
    chart = _chart_placeholder()
    # 前提：色块把**块均值**极差抬过阈值（这正是要防的误报源）。注意块均值口径
    # 连**纸面**口径都顶得动（色块整块填满时 90 分位测到的就是色块本身），
    # 所以 colorful 门槛是这里的唯一拦路口。
    assert max(pp.illumination_range(chart),
               pp.illumination_paper_range(chart)) >= pp.ILLUMINATION_RANGE_THRESHOLD, (
        f"前提：色块应把极差抬过阈值（mean={pp.illumination_range(chart)}、"
        f"paper={pp.illumination_paper_range(chart)}）"
    )
    assert pp.colorful_fraction(chart) >= pp.COLORFUL_FRACTION_THRESHOLD
    mode, sig = pp.pick_mode(chart)
    assert mode == "light", (mode, sig.get("mode_reason"))
    result = pp.preprocess(chart, mode=mode)
    assert result.applied == [], result.applied
    print("  ok test_colorful_chart_not_escalated_by_uneven_light")


def test_clean_pages_stay_light_and_untouched() -> None:
    """负例：干净文档页（不同墨量）必须保持 light 且一步清洗都不做."""
    for img in (_line_art(), _line_art(lines=4), _line_art(lines=12)):
        mode, sig = pp.pick_mode(img)
        assert mode == "light", (mode, sig.get("mode_reason"))
        result = pp.preprocess(img, mode=mode)
        assert result.applied == [], result.applied
    print("  ok test_clean_pages_stay_light_and_untouched")


def test_deskew_direction_and_trustworthiness() -> None:
    """
    D3：纠偏方向必须正确（历史缺陷：漏负号 → 倾斜被**加倍**），
    且照片纹理的噪声角度必须被"转正后峰度提升"关口拦下.

    判据是**旋转不变**的：真倾斜的文档页转正后行投影峰度大幅提升（实测 ≥5x）；
    方向错了（越纠越歪）或角度是噪声（照片纹理）都提升 ≈1x → 不许改图。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_if_host_shimmed

    skip_if_host_shimmed("cv2")

    # 1) 真倾斜文档页（t6 同款：-15°）：能纠、纠对、纠完显著变"齐"
    skewed = _rotate(_line_art(), -15)
    out, step, angle = pp._deskew(skewed.convert("RGB"))
    assert step == "deskew", (step, angle)
    assert abs(abs(angle) - 15) < 4, f"估出的角度应≈15°，实际 {angle}"
    before = pp._row_peak(np.asarray(skewed.convert("L")))
    after = pp._row_peak(np.asarray(out.convert("L")))
    assert after > before * 2, f"转正后行投影峰度应显著提升：{before:.2f} → {after:.2f}"

    # 2) 照片纹理：minAreaRect 会给出随机角度，但转正不会让行投影变尖 → 不许动
    rng = np.random.default_rng(7)
    photo = Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8))
    out_p, step_p, _ = pp._deskew(photo)
    assert step_p is None, "照片纹理不许被'纠偏'"
    assert out_p.size == photo.size
    print("  ok test_deskew_direction_and_trustworthiness")


# ── 管线接线（D1/D3 叠加 / D4 类型升级 + OCR 回滚）：端到端防回归 ────────────

def test_pipeline_geometric_overlay_keeps_noisy_clahe() -> None:
    """
    ① 的防回归：noisy 档做完 CLAHE 后，0b 纠偏必须**叠加**而不是覆盖.

    回修前的缺陷：纠偏用原图重跑 geometric 并整体替换 pre —— noisy 档已烘焙进
    像素的 CLAHE 被整段丢掉（geometric 档不做对比度归一化），而"有噪/发灰"的
    翻拍页恰恰最需要它。修后：喂 ``pre.image``（已清洗）重跑，applied 合并。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_if_host_shimmed
    from app.services.image_understanding.classifier import ImageClassification

    skip_if_host_shimmed("cv2")

    originals = {
        "run_ocr": pl._run_ocr,
        "classify": pl.classify_image_safe,
        "dispatch": pl._dispatch,
        "vision": pl.VisionEngine,
        "settings": pl.get_settings,
    }
    saved = (_VisionStub.available, _VisionStub.text, _VisionStub.calls)
    try:
        lines = [_Line("hello world", 0.9, (10, 10, 80, 20))]
        pl.get_settings = lambda: _SettingsStub()          # type: ignore[assignment]
        pl._run_ocr = lambda image: EngineOutput(          # type: ignore[assignment]
            text="hello world", confidence=0.9, engine="tesseract", ok=True, lines=lines,
        )
        # 判成 photo：preprocessing_mode_for("photo")=light，不会触发 D4 升级，
        # 保证本用例走的是 0b 内容侧纠偏（mode=noisy）
        pl.classify_image_safe = lambda image, ocr_lines=None, filename="": ImageClassification(  # type: ignore[assignment]
            image_type="photo", confidence=0.8, signals={}, engine="rules",
        )
        pl._dispatch = lambda *a, **kw: EngineOutput(      # type: ignore[assignment]
            text="照片中有文字 hello world", confidence=0.9, engine="tesseract", ok=True,
        )
        pl.VisionEngine = _VisionStub                      # type: ignore[assignment]
        _VisionStub.available, _VisionStub.text, _VisionStub.calls = False, "", 0

        # 有噪（σ=18 → noisy 档：去噪+CLAHE）+ 转歪 5°（0b 闸门应追加纠偏）
        img = _rotate(_add_gaussian_noise(_line_art(), 18), 5)
        result = pl.understand_image(img, page_number=1, filename="noisy-skew.png")

        assert result.preprocess_mode == "noisy", result.preprocess_mode
        assert "contrast" in result.preprocess, (
            f"① 回归：noisy 档的 CLAHE 被纠偏覆盖丢了 → {result.preprocess}"
        )
        assert "deskew" in result.preprocess, (
            f"D3 回归：纠偏没跑 → {result.preprocess}"
        )
        print("  ok test_pipeline_geometric_overlay_keeps_noisy_clahe")
    finally:
        pl._run_ocr = originals["run_ocr"]              # type: ignore[assignment]
        pl.classify_image_safe = originals["classify"]  # type: ignore[assignment]
        pl._dispatch = originals["dispatch"]            # type: ignore[assignment]
        pl.VisionEngine = originals["vision"]           # type: ignore[assignment]
        pl.get_settings = originals["settings"]         # type: ignore[assignment]
        _VisionStub.available, _VisionStub.text, _VisionStub.calls = saved


def _run_type_upgrade_scenario(monkey_ocr, expect_calls: int, upgrade_applied: bool):
    """
    D4 场景公共脚手架：干净但转歪的页 + 分类 table → 类型升级重跑 OCR.

    :param upgrade_applied: True = 重跑被采纳（preprocess_mode 应为 geometric、
        preprocess 含 deskew、VLM 拿到纠偏后的图）；False = 被拒（必须**什么都不
        变**，见缺陷 A 的防回归断言）。
    :return: ``{"result": ..., "png": Vision 侧收到的字节, "img": 原图}``
    """
    from app.services.image_understanding.classifier import ImageClassification

    originals = {
        "run_ocr": pl._run_ocr,
        "classify": pl.classify_image_safe,
        "dispatch": pl._dispatch,
        "vision": pl.VisionEngine,
        "settings": pl.get_settings,
    }
    saved = (_VisionStub.available, _VisionStub.text, _VisionStub.calls)
    calls = {"n": 0}
    captured: dict = {}
    try:
        stub = _SettingsStub()
        stub.IMAGE_PREPROCESS_GEOMETRIC = False   # 关掉 0b 内容侧纠偏，隔离出 D4 路径

        def fake_ocr(image):
            calls["n"] += 1
            return monkey_ocr(calls["n"])

        def fake_dispatch(image_type, image, ocr_out, png, **kwargs):
            # dispatch 是唯一必然拿到 png() 的入口 —— 借它记下 Vision 侧实际收到的字节
            captured["png"] = png()
            return EngineOutput(
                text="| 指标 | 值 |\n| --- | --- |\n| 营收 | 100 |",
                confidence=0.9, engine="table_parser:rules+cell-ocr", ok=True,
            )

        pl.get_settings = lambda: stub            # type: ignore[assignment]
        pl._run_ocr = fake_ocr                    # type: ignore[assignment]
        pl.classify_image_safe = lambda image, ocr_lines=None, filename="": ImageClassification(  # type: ignore[assignment]
            image_type="table", confidence=0.9, signals={}, engine="rules",
        )
        pl._dispatch = fake_dispatch              # type: ignore[assignment]
        pl.VisionEngine = _VisionStub             # type: ignore[assignment]
        _VisionStub.available, _VisionStub.text, _VisionStub.calls = False, "", 0

        # 干净（pick_mode=light）但转歪 5°：只有类型升级（table→geometric）会纠偏
        img = _rotate(_line_art(), 5)
        result = pl.understand_image(img, page_number=1, filename="skewed-table.png")
        assert calls["n"] == expect_calls, f"OCR 应恰好跑 {expect_calls} 次，实际 {calls['n']}"

        if upgrade_applied:
            assert result.preprocess_mode == "geometric", result.preprocess_mode
            assert "deskew" in result.preprocess, result.preprocess
        else:
            # 缺陷 A 的防回归：升级被拒必须**整体回滚**，不能留下半截中间态。
            # ① 档位 / 审计：不能报一个实际没用的 geometric
            assert result.preprocess_mode == "light", (
                f"被拒的升级仍写进了 preprocess_mode：{result.preprocess_mode}"
            )
            # ② 审计列表：被拒的那一步不许出现（否则排查时被误导）
            assert result.preprocess == [], (
                f"被拒的升级仍进审计：{result.preprocess}"
            )
            # ③ 图与 OCR 坐标同源：preprocess 为空 ⇒ work 仍是首轮那张图
            #    （配合 D1 闸门 ⇒ VLM 收到的必须是未经改动的首轮图）
        return {"result": result, "png": captured.get("png"), "img": img}
    finally:
        pl._run_ocr = originals["run_ocr"]              # type: ignore[assignment]
        pl.classify_image_safe = originals["classify"]  # type: ignore[assignment]
        pl._dispatch = originals["dispatch"]            # type: ignore[assignment]
        pl.VisionEngine = originals["vision"]           # type: ignore[assignment]
        pl.get_settings = originals["settings"]         # type: ignore[assignment]
        _VisionStub.available, _VisionStub.text, _VisionStub.calls = saved


def test_type_upgrade_retry_worse_keeps_first_ocr() -> None:
    """
    ② 的防回归：类型升级被拒时，**整升级**必须回滚（图 / 档位 / OCR 三者同源）.

    回修前的两层缺陷：
    · 第一层：升级档位后无条件用重跑结果替换 —— 引擎抖动/二值化把字吃掉时，
      首轮已经读到的文本被整段丢掉，"纠偏"反而把图读没了；
    · 第二层（缺陷 A）：只回滚了 OCR 文本，没回滚 pre/mode/work/ocr_input ——
      accept=False 时 work 是纠偏后的图而 ocr_out.lines 坐标来自纠偏前 →
      表格结构识别按行坐标定位时几何对不上；且 preprocess/preprocess_mode
      会把一个已被拒绝的升级照样报进审计。
    """
    first = "营收一千二百万元，利润三百万元，同比增长百分之十二"

    def worse(n: int) -> EngineOutput:
        if n == 1:
            return EngineOutput(text=first, confidence=0.92, engine="tesseract",
                                ok=True, lines=[_Line(first, 0.92)])
        return EngineOutput(text="", confidence=0.0, engine="tesseract", ok=False, lines=[])

    from app.services.image_understanding.imaging import encode_png_safe

    out = _run_type_upgrade_scenario(worse, expect_calls=2, upgrade_applied=False)
    result = out["result"]
    assert result.ocr_text == first, f"② 回归：首轮 OCR 文本被重跑清空 → {result.ocr_text!r}"
    rerun = result.preprocess_meta.get("ocr_rerun", {})
    assert rerun.get("decision") == "kept-first", rerun
    # A+B 叠加：什么清洗都没生效 ⇒ VLM 必须拿到**原图**字节（不是被判"读得更差"
    # 的那张纠偏图）
    assert out["png"] == encode_png_safe(out["img"]), (
        "缺陷 A/B 叠加回归：什么都没做却把改过的图送给了 Vision"
    )
    print("  ok test_type_upgrade_retry_worse_keeps_first_ocr")


def test_type_upgrade_retry_better_is_accepted() -> None:
    """D4 正常路径：重跑确实读得更好（更长且置信不降）→ **整升级**一起提交."""
    def better(n: int) -> EngineOutput:
        if n == 1:
            return EngineOutput(text="营收 1200", confidence=0.85, engine="tesseract",
                                ok=True, lines=[_Line("营收 1200", 0.85)])
        return EngineOutput(
            text="营收 1200 万元 利润 300 万元 同比增长 12%",
            confidence=0.93, engine="tesseract", ok=True,
            lines=[_Line("营收 1200 万元 利润 300 万元 同比增长 12%", 0.93)],
        )

    from app.services.image_understanding.imaging import encode_png_safe

    out = _run_type_upgrade_scenario(better, expect_calls=2, upgrade_applied=True)
    result = out["result"]
    expected = "营收 1200 万元 利润 300 万元 同比增长 12%"
    assert result.ocr_text == expected, result.ocr_text
    rerun = result.preprocess_meta.get("ocr_rerun", {})
    assert rerun.get("decision") == "accepted", rerun
    # 被采纳 ⇒ 图也跟着走（纠偏后的图必须同时到达 Vision）
    assert out["png"] != encode_png_safe(out["img"]), (
        "升级被采纳了，Vision 却还在看首轮那张没纠偏的图"
    )
    print("  ok test_type_upgrade_retry_better_is_accepted")


def test_vision_receives_cleaned_image_only_when_cleaned() -> None:
    """
    D1：做了清洗，送给多模态模型的**必须**是清洗产物；没做清洗，保持快路径.

    回修前的缺陷：``png()`` 恒复用调用方编码的原图字节 → 预处理产物到不了
    Vision，清洗等于白做。同时保留"没清洗就复用原字节"的行为 —— 干净图不多
    编码一次大图（评测基座余量只有 1.08x，不给干净图引入未度量的行为变化）。
    """
    from app.services.image_understanding.classifier import ImageClassification
    from app.services.image_understanding.imaging import encode_png_safe

    captured: dict = {}
    saved_cls = (_VisionStub.available, _VisionStub.text, _VisionStub.calls)

    class _CaptureVision(_VisionStub):
        def process(self, image, *, png_bytes=None, image_type="photo", role="primary", **kw):
            captured["png"] = png_bytes
            return EngineOutput(text="柱状图描述。", confidence=0.75, engine="vision", ok=True)

    originals = {
        "run_ocr": pl._run_ocr,
        "classify": pl.classify_image_safe,
        "dispatch": pl._dispatch,
        "vision": pl.VisionEngine,
        "settings": pl.get_settings,
    }
    try:
        pl.get_settings = lambda: _SettingsStub()          # type: ignore[assignment]
        pl._run_ocr = lambda image: EngineOutput(          # type: ignore[assignment]
            text="Q1 1200", confidence=0.9, engine="tesseract", ok=True,
            lines=[_Line("Q1 1200", 0.9)],
        )
        # 判成 chart → 专用引擎就是 Vision（dispatch 会以 png() 调用）
        pl.classify_image_safe = lambda image, ocr_lines=None, filename="": ImageClassification(  # type: ignore[assignment]
            image_type="chart", confidence=0.8, signals={}, engine="rules",
        )
        pl.VisionEngine = _CaptureVision                   # type: ignore[assignment]

        # (a) 阴影翻拍页：清洗确实做了（CLAHE）→ Vision 必须拿到清洗后的字节
        pl._dispatch = lambda *a, **kw: EngineOutput(      # type: ignore[assignment]
            text="", confidence=0.0, engine="vision", ok=False, error="stub",
        )
        shaded = _shade_gradient(_line_art(ink=(30, 30, 30)), 0.45)
        captured.clear()
        pl.understand_image(shaded, page_number=1, filename="shaded.png")
        assert captured.get("png") is not None, "Vision 应被调用并收到 png 字节"
        assert captured["png"] != encode_png_safe(shaded), (
            "D1 回归：Vision 收到的仍是原图字节 —— 清洗产物没有送达"
        )

        # (b) 干净图：一步没做 → 必须复用原图字节（不多编码、不引入行为变化）
        clean = _line_art()
        captured.clear()
        pl.understand_image(clean, page_number=1, filename="clean.png")
        assert captured.get("png") == encode_png_safe(clean), (
            "没做清洗的图不该被重新编码 —— 应复用调用方字节"
        )

        # (c) 小图只被 upscale：upscale **不在** _CLEANING_STEPS 里 → 同样必须
        #     复用原图字节。闸门没接上时（旧写法 `if pre.applied:`）这里会失败：
        #     一批原本干净的小图会被换成放大版送进 Vision，属于未度量的行为变化。
        small = Image.new("RGB", (120, 90), (255, 255, 255))
        ImageDraw.Draw(small).line([(10, 20), (110, 20)], fill=(0, 0, 0), width=2)
        ImageDraw.Draw(small).line([(10, 60), (110, 60)], fill=(0, 0, 0), width=2)
        captured.clear()
        small_result = pl.understand_image(small, page_number=1, filename="small.png")
        assert small_result.preprocess == ["upscale"], (
            f"前提：这张小图应只被放大，实际 {small_result.preprocess}"
        )
        assert captured.get("png") == encode_png_safe(small), (
            "闸门回归：仅 upscale 就换了 VLM 输入（给干净小图引入未度量的行为变化）"
        )
        print("  ok test_vision_receives_cleaned_image_only_when_cleaned")
    finally:
        pl._run_ocr = originals["run_ocr"]              # type: ignore[assignment]
        pl.classify_image_safe = originals["classify"]  # type: ignore[assignment]
        pl._dispatch = originals["dispatch"]            # type: ignore[assignment]
        pl.VisionEngine = originals["vision"]           # type: ignore[assignment]
        pl.get_settings = originals["settings"]         # type: ignore[assignment]
        _VisionStub.available, _VisionStub.text, _VisionStub.calls = saved_cls


def test_pick_mode_backward_compatible_signature() -> None:
    """``pick_mode(image[, image_type])`` 既有签名语义必须原样保留."""
    img = _line_art()

    mode1, sig1 = pp.pick_mode(img)                    # 不传类型 → 内容判定，默认 light
    assert mode1 == "light" and sig1["mode"] == "light", (mode1, sig1.get("mode_reason"))

    mode2, _ = pp.pick_mode(img, "table")              # 老签名：类型默认档可以抬高
    assert mode2 == "geometric", mode2

    mode3, _ = pp.pick_mode(img, image_type="photo")   # 关键字传参也要能用
    assert mode3 == "light", mode3

    # 类型默认档映射（D4 依赖它）：结构型 → geometric，图表/照片 → light
    for t in ("table", "code", "formula"):
        assert pp.preprocessing_mode_for(t) == "geometric", t
    for t in ("chart", "photo", "diagram", "screenshot"):
        assert pp.preprocessing_mode_for(t) == "light", t

    # 新信号与新阈值对调用方可见（meta 里带全，便于排查"为什么升档"）
    sigs = pp.noise_signals(img)
    for key in ("noise_sigma", "contrast_span", "border_px", "sharpness_ratio",
                "illumination_range", "illumination_paper_range",
                "illumination_gradient", "colorful_fraction"):
        assert key in sigs, key
    limits = pp.thresholds()
    for key in ("noise_sigma_threshold", "low_contrast_span", "sharpness_min_ratio",
                "illumination_range_threshold", "colorful_fraction_threshold",
                "illumination_gradient_tolerance", "illumination_min_paper_ratio",
                "deskew_min_peak_gain"):
        assert key in limits, key
    print("  ok test_pick_mode_backward_compatible_signature")


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
# QA 独立验证（闸门完整性 / 双向纠偏 / 真实页负例 / 升级回滚同源）
#
# 与工程师自带用例的区别：自带用例证明"改动按设计跑通"，这里专门去**证伪**
# 设计与实现（拼错的集合成员、只测了一个方向的纠偏、覆盖面不足的负例）。
# ─────────────────────────────────────────────────────────────────────────────

_QA_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _qa_load_font(size: int):
    """真实字体（抗锯齿正文）—— 造"导出页风格"负例的前提."""
    try:
        from PIL import ImageFont
    except Exception:      # noqa: BLE001
        return None
    for path in _QA_FONT_CANDIDATES:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
        except Exception:      # noqa: BLE001
            continue
    return None


def _qa_text_page(
    width: int = 1240, height: int = 1754, *, size: int = 23, rows: int = 20,
    top: int = 300, bottom: int = 700, margin: int = 140,
    ink=(25, 25, 25), bg=(255, 255, 255),
) -> Image.Image:
    """
    真实导出页风格：白底 + 页边距留白 + 抗锯齿正文墨迹（150dpi A4 量级）.

    与 ``_line_art``（8 条横贯线）的区别：真实排版是**行数可控 + 留白可变 +
    字形抗锯齿**，墨迹覆盖率与边缘统计都更接近入库期的原生导出页 —— 只有这类
    图才能检验新加的内容敏感信号（illumination/sharpness）会不会误报。
    """
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = _qa_load_font(size)
    step = (height - top - bottom) / max(rows, 1)
    body = "retrieval augmented generation pipeline evaluation report"
    for i in range(rows):
        y = top + i * step
        if y > height - bottom - step * 0.5:
            break
        if font is not None:
            draw.text((margin, y), body, fill=ink, font=font)
        else:
            draw.rectangle([margin, y, margin + 620, y + size], fill=ink)
    return img


def test_cleaning_steps_cover_every_real_applied_name() -> None:
    """
    D1 闸门的**完整性**（不是"闸门存在"，而是"闸门真的接得住"）.

    ``png()`` 用的是集合交集 ``_CLEANING_STEPS & set(pre.applied)``：拼错一个
    成员**不会报错**，只会静默失效 —— 清洗照做，Vision 却仍拿原图字节，且没人
    会发现。所以这里从两个方向夹逼：

        · 静态：集合里每个名字都必须是 preprocess.py 真实会返回的字面量；
        · 动态：真实跑出来的每个 step（除 upscale）都必须落在集合里，
          包括**无 OpenCV 的回退分支**（denoise-median / contrast-stretch
          最容易漏，因为它们只在容器没装 cv2 时才出现）。
    """
    import inspect

    src = inspect.getsource(pp)

    # 方向 1：拼错的成员不可能出现在源码里
    for name in pl._CLEANING_STEPS:
        assert f'"{name}"' in src or f"'{name}'" in src, (
            f"_CLEANING_STEPS 里的 {name!r} 在 preprocess.py 源码中查无此字面量 —— "
            "疑似拼错，D1 闸门对它会静默失效"
        )

    # 方向 2：把所有档位 + 两种依赖分支都跑一遍，收集真实 step 名
    noisy = _add_gaussian_noise(_line_art(), 20)
    scanned = _scanned_page()
    small = Image.new("RGB", (120, 90), (255, 255, 255))
    ImageDraw.Draw(small).line([(10, 30), (110, 30)], fill=(0, 0, 0), width=2)

    skewed = _rotate(noisy, 6)         # 有倾斜才会真的触发 deskew 分支
    observed: set[str] = set()
    for mode in ("light", "text", "geometric", "noisy", "scan"):
        observed.update(pp.preprocess(scanned, mode=mode).applied)
        observed.update(pp.preprocess(noisy, mode=mode).applied)
        observed.update(pp.preprocess(skewed, mode=mode).applied)
    observed.update(pp.preprocess(small, mode="light").applied)

    saved_cv2 = sys.modules.get("cv2")
    try:
        sys.modules["cv2"] = None      # import cv2 → ImportError → 走纯 numpy/PIL 回退
        for mode in ("noisy", "scan"):
            observed.update(pp.preprocess(noisy, mode=mode).applied)
    finally:
        if saved_cv2 is not None:
            sys.modules["cv2"] = saved_cv2
        else:
            sys.modules.pop("cv2", None)

    assert "upscale" in observed, f"前提：小图放大分支没被覆盖到（observed={observed}）"
    assert observed - {"upscale"}, f"前提：没有任何清洗步骤被触发（observed={observed}）"
    # 前提：deskew 这个**最关键**的结构性步骤必须被覆盖到 —— 没覆盖的话下面的
    # "每个 step 都在集合里"就是一句空话，而 deskew 恰好是最容易拼错的那个
    assert "deskew" in observed, (
        f"前提：deskew 分支没被触发，覆盖断言形同虚设（observed={sorted(observed)}）"
    )

    for step in observed:
        if step == "upscale":
            continue
        assert step in pl._CLEANING_STEPS, (
            f"preprocess 真的做了 {step!r}，但它不在 _CLEANING_STEPS 里 → "
            "清洗产物送不到 Vision（D1 闸门静默失效）"
        )
    # upscale 必须**刻意**被排除：短边 <200 的干净小图被放大后送 VLM 属于
    # 未度量的行为变化（评测基座余量仅 1.08x）
    assert "upscale" not in pl._CLEANING_STEPS
    print(f"  ok test_cleaning_steps_cover_every_real_applied_name (observed={sorted(observed)})")


def test_deskew_corrects_both_directions() -> None:
    """
    D3 符号缺陷的防回归：**两个方向**都要纠，而不是"改了图"就算过.

    只测一个方向（工程师用例是 -15°）时，"符号恰好对这一个方向正确、对另一个
    方向加倍"的实现照样通过 —— 历史缺陷正是漏了负号（5°→10°、15°→30°）。
    因此这里 +θ / -θ 各测一次，并用**残余角**（同一 minAreaRect 估计器重测）
    而不是"是否返回 deskew"来判定方向对不对。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_if_host_shimmed

    skip_if_host_shimmed("cv2")
    import cv2

    def residual(gray) -> float | None:
        """用与 _deskew 同一套估计器测残余倾角."""
        ink = (np.asarray(gray) < 160).astype(np.uint8)
        if int(ink.sum()) < 200:
            return None
        coords = np.column_stack(np.where(ink > 0))
        ang = float(cv2.minAreaRect(coords.astype(np.float32))[-1])
        if ang > 45:
            ang -= 90
        return round(ang, 2)

    page = _qa_text_page(rows=26, top=150, bottom=150)
    for degrees in (+15, -15):
        skewed = _rotate(page, degrees)
        before_angle = residual(np.asarray(skewed.convert("L")))
        before_peak = pp._row_peak(np.asarray(skewed.convert("L")))
        out, step, angle = pp._deskew(skewed.convert("RGB"))
        after_peak = pp._row_peak(np.asarray(out.convert("L")))
        after_angle = residual(np.asarray(out.convert("L")))

        assert step == "deskew", f"{degrees:+d}° 没被纠偏（step={step}）"
        assert after_peak > before_peak * 1.5, (
            f"{degrees:+d}° 转正后行投影没变尖（{before_peak:.2f}→{after_peak:.2f}）—— "
            "很可能纠反了方向（倾斜被加倍）"
        )
        assert after_angle is not None and abs(after_angle) < abs(before_angle) * 0.5, (
            f"{degrees:+d}° 残余倾角没降下来：{before_angle}° → {after_angle}°（越纠越歪）"
        )
        assert abs(abs(angle) - abs(degrees)) < 4, (
            f"{degrees:+d}° 估出的角度应≈{abs(degrees)}°，实际 {angle}"
        )
    print("  ok test_deskew_corrects_both_directions")


def test_realistic_export_pages_stay_light() -> None:
    """
    D2 误报方向（补充工程师负例的覆盖面）：**真实排版**的导出页必须留在 light.

    工程师的负例是"合成整页表格"（墨迹均匀铺满、无留白），而真实导出页的特征
    恰恰相反 —— **大片留白**、**页眉页脚**、**高 dpi 渲染**。这些都会改变
    块均值分布与边缘统计，是最可能顶穿新加的 illumination_range / sharpness_ratio
    的一类图。断言之外同时打印**距阈值的余量**：只看 pass/fail 会漏掉"勉强通过"
    的临界图（余量 <1.2x 就该警惕）。
    """
    cases = {
        "常规正文页@150dpi": _qa_text_page(rows=20, top=300, bottom=700),
        "常规正文页@300dpi": _qa_text_page(2480, 3508, size=46, rows=20, top=600,
                                          bottom=1400, margin=280),
        "稀疏正文 + 大片留白": _qa_text_page(rows=6, top=160, bottom=1000),
        "页眉页脚 + 中段留白": None,
        "双栏 + 宽页边距": None,
    }
    hf = _qa_text_page(rows=10, top=300, bottom=1000)
    draw = ImageDraw.Draw(hf)
    draw.line([(140, 250), (1100, 250)], fill=(90, 90, 90), width=2)
    draw.line([(140, 1500), (1100, 1500)], fill=(90, 90, 90), width=2)
    cases["页眉页脚 + 中段留白"] = hf

    two_col = _qa_text_page(rows=18, top=300, bottom=700, margin=120)
    d2 = ImageDraw.Draw(two_col)
    f2 = _qa_load_font(23)
    for i in range(18):
        y = 300 + i * (754 / 18)
        if f2 is not None:
            d2.text((660, y), "column two body text here", fill=(25, 25, 25), font=f2)
    cases["双栏 + 宽页边距"] = two_col

    limits = pp.thresholds()
    for name, img in cases.items():
        sig = pp.noise_signals(img)
        mode, picked = pp.pick_mode(img)
        margin_illum = (
            limits["illumination_range_threshold"] / sig["illumination_range"]
            if sig["illumination_range"] > 0 else float("inf")
        )
        margin_sharp = sig["sharpness_ratio"] / limits["sharpness_min_ratio"]
        margin_sigma = (
            limits["noise_sigma_threshold"] / sig["noise_sigma"]
            if sig["noise_sigma"] > 0 else float("inf")
        )
        print(
            f"    {name}: sigma={sig['noise_sigma']:.2f}(余量{margin_sigma:.1f}x) "
            f"illum={sig['illumination_range']:.3f}(余量{margin_illum:.1f}x) "
            f"sharp={sig['sharpness_ratio']:.1f}(余量{margin_sharp:.1f}x) -> {mode}"
        )
        assert mode == "light", (
            f"{name} 被误升到 {mode}（{picked.get('mode_reason')}）—— "
            f"signals={sig}"
        )
        # 留在 light 的干净页：一步清洗都不该做（否则 D1 会把清洗产物送进 VLM）
        assert pp.preprocess(img, mode=mode).applied == [], (
            f"{name} 档位是 light 却仍被改图：{pp.preprocess(img, mode=mode).applied}"
        )
    print("  ok test_realistic_export_pages_stay_light")


def test_achromatic_dark_block_does_not_escalate_clean_page() -> None:
    """
    D2 缺陷（本次改动引入的**新**误报）：**消色**深色块会顶穿 illumination_range.

    ``colorful_fraction`` 门槛只挡得住"饱和色块"（彩色图例，实测 colorful=0.64），
    挡不住**无饱和色的深色块** —— 深色标题条 / 深色表头行 / 深色代码区块的
    colorful_fraction ≈ 0.00，却同样把 4×4 分块的块均值极差抬到阴影量级。

    后果链（三件事叠加，正是这次改动自己声明要避免的）：
        illumination_range ≥ 0.18 → noisy 档
        → 施加 CLAHE（applied 含 "contrast"）
        → contrast ∈ _CLEANING_STEPS ⇒ D1 把**清洗后**的字节重新编码送 VLM
    于是"一张干净锐利的原生导出页"被换了输入给多模态模型 —— 而改动自己在
    _CLEANING_STEPS 的注释里写明了"评测基座余量只有 1.08 倍，不给干净图引入
    未度量的行为变化"。

    前提断言先钉死"这是新增误报"：旧判据（sigma / border / contrast_span）
    一条都不触发，升级**只可能**来自新的 uneven-light 规则。
    """
    limits = pp.thresholds()
    img = _qa_text_page(rows=20, top=300, bottom=700)
    draw = ImageDraw.Draw(img)
    draw.rectangle([140, 150, 1100, 260], fill=(45, 45, 45))   # 深色标题条：占页高 ~6%

    sig = pp.noise_signals(img)
    # ① 前提：这是一张干净、锐利的原生导出页（三个旧信号全都不该触发）
    assert sig["noise_sigma"] < limits["noise_sigma_threshold"], (
        f"前提失效：sigma={sig['noise_sigma']} 已越过旧阈值")
    assert sig["contrast_span"] >= limits["low_contrast_span"], (
        f"前提失效：span={sig['contrast_span']} 已越过旧阈值")
    assert sig["border_px"] < limits["border_trigger_px"], (
        f"前提失效：border={sig['border_px']} 已越过旧阈值")
    assert sig["sharpness_ratio"] >= limits["sharpness_min_ratio"], (
        f"前提失效：清晰度 {sig['sharpness_ratio']} 已越过新阈值，不是干净页")
    # ② 前提：深色块是**消色**的 → colorful 门槛救不了它（这正是缺陷所在）
    assert sig["colorful_fraction"] < limits["colorful_fraction_threshold"], (
        "前提失效：该深色块是饱和色，colorful 门槛本就该拦住它")

    mode, picked = pp.pick_mode(img)
    assert mode == "light", (
        f"D2 误报：干净锐利的导出页（sigma={sig['noise_sigma']}、"
        f"sharpness={sig['sharpness_ratio']}）仅因一块**消色**深色标题条就被升到 "
        f"{mode}（{picked.get('mode_reason')}）→ 会被施加 CLAHE，且因 "
        f"contrast ∈ _CLEANING_STEPS，清洗后的字节会被重新编码送进 VLM"
    )
    print("  ok test_achromatic_dark_block_does_not_escalate_clean_page")


def test_type_upgrade_rejected_preserves_homogeneity() -> None:
    """
    D4 同源性（强化版）：被拒的升级不能污染**非空**的 applied / 图 / OCR 坐标.

    工程师已有用例（``test_type_upgrade_retry_worse_keeps_first_ocr``）的前提是
    ``applied == []`` —— 那是**平凡情况**：它只能证明"没凭空多出步骤"，证明不了
    "被拒的步骤没有**追加**到已有步骤后面"。这里让首轮就已经有一步（upscale），
    再制造一次被拒的类型升级，断言四者同源：

        · 档位停在 light（不是 geometric）
        · applied 停在 ['upscale']（被拒的 deskew 不许追加进来）
        · ocr_text / ocr_line_count 保住首轮
        · 送进 dispatch 的 work 仍是首轮那张图（不是纠偏后的图）
    """
    from app.services.image_understanding.classifier import ImageClassification
    from app.services.image_understanding.imaging import encode_png_safe

    # 短边 <200 → light 档也会先做 upscale；再转歪 8° 让 geometric 升级真的有事可做
    small = Image.new("RGB", (160, 200), (255, 255, 255))
    draw = ImageDraw.Draw(small)
    for i in range(7):
        y = 25 + i * 25
        draw.line([(20, y), (140, y)], fill=(0, 0, 0), width=2)
    img = _rotate(small, 8)

    first = "营收一千二百万元，利润三百万元"
    captured: dict = {}
    saved_vision = (_VisionStub.available, _VisionStub.text, _VisionStub.calls)
    originals = {
        "run_ocr": pl._run_ocr, "classify": pl.classify_image_safe,
        "dispatch": pl._dispatch, "vision": pl.VisionEngine,
        "settings": pl.get_settings,
    }
    calls = {"n": 0}
    try:
        stub = _SettingsStub()
        stub.IMAGE_PREPROCESS_GEOMETRIC = False   # 关掉 0b，隔离出 D4 路径

        def fake_ocr(image):
            calls["n"] += 1
            if calls["n"] == 1:
                return EngineOutput(text=first, confidence=0.92, engine="tesseract",
                                    ok=True, lines=[_Line(first, 0.92)])
            return EngineOutput(text="", confidence=0.0, engine="tesseract",
                                ok=False, lines=[])

        def fake_dispatch(image_type, image, ocr_out, png, **kwargs):
            captured["work"] = image
            captured["png"] = png()
            return EngineOutput(text="| a | b |", confidence=0.9,
                                engine="table_parser:rules", ok=True)

        pl.get_settings = lambda: stub            # type: ignore[assignment]
        pl._run_ocr = fake_ocr                    # type: ignore[assignment]
        pl.classify_image_safe = lambda image, ocr_lines=None, filename="": ImageClassification(  # type: ignore[assignment]
            image_type="table", confidence=0.9, signals={}, engine="rules",
        )
        pl._dispatch = fake_dispatch              # type: ignore[assignment]
        pl.VisionEngine = _VisionStub             # type: ignore[assignment]
        _VisionStub.available, _VisionStub.text, _VisionStub.calls = False, "", 0

        result = pl.understand_image(img, page_number=1, filename="small-skewed-table.png")

        # 前提：升级确实被**尝试**过（否则下面全是平凡断言）
        assert calls["n"] == 2, f"OCR 应跑 2 次（首轮 + 升级重跑），实际 {calls['n']}"
        rerun = result.preprocess_meta.get("ocr_rerun", {})
        assert rerun.get("decision") == "kept-first", rerun

        # ① 档位
        assert result.preprocess_mode == "light", (
            f"被拒的升级仍写进了 preprocess_mode：{result.preprocess_mode}")
        # ② 审计列表：被拒的 deskew 不许追加到 ['upscale'] 后面
        assert result.preprocess == ["upscale"], (
            f"被拒的升级混进了审计列表：{result.preprocess}")
        # ③ OCR 文本与行数保住首轮
        assert result.ocr_text == first, f"首轮 OCR 文本被重跑清空：{result.ocr_text!r}"
        assert result.ocr_line_count == 1, result.ocr_line_count
        # ④ 图与 OCR 坐标同源：work 必须是首轮那张（只放大、未纠偏）的图
        expected_work = pp.preprocess(img, mode="light").image
        assert np.array_equal(
            np.asarray(captured["work"].convert("RGB")),
            np.asarray(expected_work.convert("RGB")),
        ), "被拒的升级改了送进引擎的图（work 与首轮 OCR 坐标不同源）"
        # ⑤ 叠加 D1：只做了 upscale（非 _CLEANING_STEPS）⇒ 闸门必须关着
        assert captured["png"] == encode_png_safe(img), (
            "仅 upscale 就把改过的图送给了 Vision")
        print("  ok test_type_upgrade_rejected_preserves_homogeneity")
    finally:
        pl._run_ocr = originals["run_ocr"]              # type: ignore[assignment]
        pl.classify_image_safe = originals["classify"]  # type: ignore[assignment]
        pl._dispatch = originals["dispatch"]            # type: ignore[assignment]
        pl.VisionEngine = originals["vision"]           # type: ignore[assignment]
        pl.get_settings = originals["settings"]         # type: ignore[assignment]
        _VisionStub.available, _VisionStub.text, _VisionStub.calls = saved_vision


def test_new_thresholds_are_wired_into_settings() -> None:
    """
    config.py 新增的 6 个阈值必须**真的被读到**，而不是写给人看的死配置.

    ``preprocess._cfg`` 走 ``getattr(get_settings(), name, 常量)``：名字对不上会
    静默退回模块常量，实测时"改 .env 不生效"。这里用一个改了值的 Settings 桩
    顶替 ``app.config.get_settings``，断言 ``thresholds()`` 读到的就是桩里的值。
    """
    import app.config as app_config

    names = (
        "IMAGE_SHARPNESS_MIN_RATIO", "IMAGE_SHARPNESS_BLUR_SIGMA",
        "IMAGE_ILLUMINATION_RANGE_THRESHOLD", "IMAGE_ILLUMINATION_GRID",
        "IMAGE_COLORFUL_FRACTION_THRESHOLD", "IMAGE_DESKEW_MIN_PEAK_GAIN",
    )
    from app.config import Settings

    fields = getattr(Settings, "model_fields", None) or getattr(Settings, "__fields__", {})
    for name in names:
        assert name in fields, f"Settings 未声明 {name}（config.py 的改动没落地）"

    class _AltSettings:
        IMAGE_NOISE_SIGMA_THRESHOLD = 4.0
        IMAGE_NOISE_SIGMA_STRONG = 18.0
        IMAGE_LOW_CONTRAST_SPAN = 0.55
        IMAGE_BORDER_TRIGGER_PX = 6
        IMAGE_BORDER_DARK_THEME_RATIO = 0.5
        IMAGE_SHARPNESS_MIN_RATIO = 111.0
        IMAGE_SHARPNESS_BLUR_SIGMA = 9.5
        IMAGE_ILLUMINATION_RANGE_THRESHOLD = 0.77
        IMAGE_ILLUMINATION_GRID = 4
        IMAGE_COLORFUL_FRACTION_THRESHOLD = 0.88
        IMAGE_DESKEW_MIN_PEAK_GAIN = 9.0

    saved = app_config.get_settings
    try:
        app_config.get_settings = lambda: _AltSettings()   # type: ignore[assignment]
        th = pp.thresholds()
        assert th["sharpness_min_ratio"] == 111.0, th
        assert th["illumination_range_threshold"] == 0.77, th
        assert th["colorful_fraction_threshold"] == 0.88, th
        assert th["deskew_min_peak_gain"] == 9.0, th
        assert pp._cfg("IMAGE_SHARPNESS_BLUR_SIGMA", 1.5) == 9.5
    finally:
        app_config.get_settings = saved                    # type: ignore[assignment]

    # 还原后必须回到真实配置（证明桩没有污染后续用例）
    assert pp.thresholds()["sharpness_min_ratio"] != 111.0
    print("  ok test_new_thresholds_are_wired_into_settings")


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
    test_pick_mode_escalates_on_uneven_light()
    test_pick_mode_escalates_on_defocus()
    test_colorful_chart_not_escalated_by_uneven_light()
    test_clean_pages_stay_light_and_untouched()
    test_deskew_direction_and_trustworthiness()
    test_pipeline_geometric_overlay_keeps_noisy_clahe()
    test_type_upgrade_retry_worse_keeps_first_ocr()
    test_type_upgrade_retry_better_is_accepted()
    test_vision_receives_cleaned_image_only_when_cleaned()
    test_pick_mode_backward_compatible_signature()

    print("\n[一b] QA 独立验证")
    test_cleaning_steps_cover_every_real_applied_name()
    test_deskew_corrects_both_directions()
    test_realistic_export_pages_stay_light()
    test_achromatic_dark_block_does_not_escalate_clean_page()
    test_type_upgrade_rejected_preserves_homogeneity()
    test_new_thresholds_are_wired_into_settings()

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
