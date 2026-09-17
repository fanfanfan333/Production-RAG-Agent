"""
图片预处理 + 噪声处理（流程图的 "Preprocessing" 环节）.

    Image → [Preprocessing] → Specialized OCR → confidence ...

预处理的原则是**保守**：只做"确定有帮助且不会引入新错误"的操作，每一步都要
能被单独关掉、单独审计。盲目二值化会把彩色图表、代码高亮、公式符号直接毁掉，
这比不做预处理更糟。

## 噪声为什么必须单独对待

入库期的图片来自三类来源，噪声特征完全不同：

    ┌──────────────┬────────────────────────────────────────────────────┐
    │ 原生导出      │ PDF/PPTX 里的矢量图转位图 —— 几乎无噪，动了反而糊   │
    │ 屏幕截图      │ 无传感器噪声，但可能有 JPEG 压缩块、缩放插值振铃    │
    │ 扫描 / 翻拍   │ 传感器噪点 + 不均匀光照 + 透视形变 + 摩尔纹        │
    └──────────────┴────────────────────────────────────────────────────┘

因此这里先**估计噪声**再决定动不动手：噪声低于阈值就一步都不做（原生导出图
占了知识库里的大多数），高于阈值才按强度施加去噪。这也是"噪声处理"和
"无脑去噪"的区别 —— 后者会让本来就干净的图变糊，OCR 反而更差。

## 档位（按"操作强度"递增）

    ┌────────────┬────────────────────────────────────────────────────────┐
    │ light      │ 放大（小图）+ 按需去噪。对所有类型安全，默认档            │
    │ text       │ light + 灰度 + 自适应二值化。只给 OCR 用，不替换原图      │
    │ geometric  │ light + 纠偏（去斜）。扫描件/拍照件才有明显收益            │
    │ noisy      │ light + **强度自适应**去噪 + 对比度归一化。噪点重时用      │
    │ scan       │ noisy + 去边框 + 摩尔纹抑制 + 纠偏 + 二值化（翻拍整页）    │
    └────────────┴────────────────────────────────────────────────────────┘

所有函数都容忍异常：预处理失败必须**返回原图**而不是抛，否则一张坏图能拖垮
整篇文档的入库。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.utils.logging import get_logger

logger = get_logger(__name__)

#: 短边小于该值就放大 —— 太小的图 OCR 基本是噪声
MIN_SHORT_SIDE = 200
#: 放大后的目标短边
TARGET_SHORT_SIDE = 640
#: 超过该角度才值得纠偏（小角度纠偏的插值损失 > 收益）
MIN_SKEW_DEGREES = 0.8
#: 拉普拉斯方差低于该值认为"糊/噪声大"，值得去噪（历史判据，保留兼容）
BLUR_VARIANCE_THRESHOLD = 60.0

#: 噪声标准差（0-255 灰度域）高于该值才值得去噪。经验值：
#:   原生导出 ≈ 0.5-1.5   截图 ≈ 2-4   扫描件 ≈ 5-12   翻拍 ≈ 10-25
NOISE_SIGMA_THRESHOLD = 4.0
#: 噪声很强时的参考上限（用于把 sigma 映射成去噪强度）
NOISE_SIGMA_STRONG = 18.0
#: 对比度跨度（5%-95% 分位差 / 255）低于该值认为"发灰"，值得归一化
LOW_CONTRAST_SPAN = 0.55
#: 边框检测：某条边超过该比例的像素"近黑"就认为存在扫描黑边
BORDER_DARK_RATIO = 0.60
#: "近黑"的灰度分界
BORDER_DARK_LEVEL = 60
#: 四条边都至少有这么多像素的黑边，才值得进 scan 档（避免误判深色主题截图）
BORDER_TRIGGER_PX = 6
#: 边框最多裁掉的比例（防止把内容误裁掉）
BORDER_MAX_TRIM = 0.06
#: 整幅图近黑像素占比超过该值 → 判定为**深色主题**而不是扫描黑边。
#: 深色背景的流程图 / 代码截图 / 暗色 UI 四条边天然近黑，与扫描仪盖板黑边在
#: "四边近黑"这一个信号上无法区分；唯一可靠的判别是看**内部**——扫描件内部
#: 是白纸（近黑占比通常 < 20%），深色主题内部同样发暗。
BORDER_DARK_THEME_RATIO = 0.5
#: 底色一侧之外，还压着多大比例的反色大面积区域时判定为"混合极性"。
#: 正常文档页的墨迹覆盖率通常在 20% 以内；一旦反向区域超过这个值，说明图上
#: 除了纸还有一整块反色面板（深色主题流程图 / 终端截图贴在白底页面里），
#: 二值化必定毁掉其中一侧 —— 此时直接跳过二值化。
MIXED_POLARITY_FRAC = 0.35


@dataclass
class PreprocessResult:
    """
    预处理产出：真正的图 + 做了哪些操作（可审计）.

    **两个输出通道**，这是本模块最重要的一个约定：

        image       给"看图"用的（分类 / Vision / 表格结构识别）—— 永远保持
                    彩色与灰度层次。二值化会毁掉图表配色与照片细节，绝不能
                    混进来。
        ocr_image   给"读字"用的 —— 在 image 之上可选再叠一次二值化。

    历史上只有一个输出，导致"想要更好的 OCR 就得把二值图交给分类器"，
    而二值图的色彩统计全废，图表会被判成截图。拆成两个通道后，两边都能
    拿到自己最合适的输入。
    """

    image: Image.Image
    applied: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    ocr_image: Image.Image | None = None

    @property
    def ocr_input(self) -> Image.Image:
        """喂 OCR 的图（没有二值化时就是 ``image``）."""
        return self.ocr_image if self.ocr_image is not None else self.image

    @property
    def changed(self) -> bool:
        return bool(self.applied)

    def to_dict(self) -> dict:
        return {"applied": list(self.applied), "meta": dict(self.meta)}


# ─────────────────────────────────────────────────────────────────────────────
# 噪声 / 质量信号（先量再动手）
# ─────────────────────────────────────────────────────────────────────────────

#: Immerkær (1996) 快速噪声方差估计用的 3×3 掩膜。
#: 掩膜对平滑区域响应为 0、对逐像素噪声响应为 ~4σ，因此
#:   σ ≈ sqrt(π/2) / (6·(W-2)·(H-2)) · Σ|conv|
#: 相比"拉普拉斯方差"，它不会被图像本身的高频内容（文字笔画、网格线）
#: 误判成噪声 —— 这是"干净的代码截图被当成噪点图去模糊"这类事故的根因。
_NOISE_MASK = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float64
)


def estimate_noise(image: Image.Image) -> float:
    """
    估计灰度噪声标准差（0-255 域）；无法估计时返回 0.0.

    取图片的**中心 90%** 区域来估计：扫描件的黑边、翻拍的桌面背景都会把
    估计值拉高，而那些区域恰恰是要被裁掉的。
    """
    try:
        gray = np.asarray(image.convert("L"), dtype=np.float64)
    except Exception:      # noqa: BLE001
        return 0.0
    h, w = gray.shape
    if h < 8 or w < 8:
        return 0.0

    pad_y, pad_x = int(h * 0.05), int(w * 0.05)
    core = gray[pad_y:h - pad_y, pad_x:w - pad_x]
    ch, cw = core.shape
    if ch < 4 or cw < 4:
        return 0.0

    # 用切片卷积（不依赖 scipy）：对每个 3×3 窗口取加权和
    windows = np.stack([
        core[i:i + ch - 2, j:j + cw - 2]
        for i in range(3) for j in range(3)
    ])
    response = np.tensordot(_NOISE_MASK.ravel(), windows, axes=([0], [0]))
    sigma = math.sqrt(math.pi / 2.0) / (6.0 * (ch - 2) * (cw - 2)) * float(
        np.abs(response).sum()
    )
    return round(float(sigma), 3)


def contrast_span(image: Image.Image) -> float:
    """
    对比度跨度 = (P99.5 - P0.5) / 255，范围 0~1.

    发灰的扫描件通常 < 0.45；正常文档 ≥ 0.6。用它决定要不要做对比度归一化。

    取 0.5 / 99.5 这种**极端分位**而不是 5/95：一页正常文档的墨迹往往只占
    1%~5% 的像素，5/95 分位会双双落在白底上，把"黑字白纸"误判成"零对比度"
    （第一版就是这么错的）。极端分位能踩到墨迹；只有当墨迹稀疏到连 0.5% 都
    不到时（纯白底 + 细线稿），才退化为极值差 —— 此时百分位法必然失效。
    """
    try:
        gray = np.asarray(image.convert("L"), dtype=np.float64)
    except Exception:      # noqa: BLE001
        return 1.0
    if gray.size == 0:
        return 1.0
    low, high = np.percentile(gray, [0.5, 99.5])
    if high - low < 1.0:
        low, high = float(gray.min()), float(gray.max())
    return round(float((high - low) / 255.0), 4)


def border_trim(image: Image.Image) -> int:
    """
    估计"扫描黑边"厚度（像素），无黑边返回 0.

    只看四条边：某条边上超过 ``BORDER_DARK_RATIO`` 的像素接近黑
    （< ``BORDER_DARK_LEVEL``），就认为这条边是扫描仪盖板留下的黑边。取四边
    最小值作为统一裁剪量 —— 单边裁剪会让画面偏心，视觉上更糟。

    深色主题（深色背景的流程图 / 代码截图 / 暗色 UI）**四边天然近黑**，与
    扫描黑边在这个信号上无法区分。因此先做一次"整幅近黑占比"检查：内部也
    发暗就是深色主题，返回 0，绝不进 scan 档去裁边 + 二值化。
    """
    try:
        gray = np.asarray(image.convert("L"), dtype=np.int16)
    except Exception:      # noqa: BLE001
        return 0
    h, w = gray.shape
    if h < 20 or w < 20:
        return 0

    dark_all = gray < BORDER_DARK_LEVEL
    if float(dark_all.mean()) >= _cfg(
        "IMAGE_BORDER_DARK_THEME_RATIO", BORDER_DARK_THEME_RATIO
    ):
        # 深色主题：这是背景色，不是扫描仪盖板 —— 裁掉等于把内容切掉一圈
        return 0

    limit = max(2, int(min(h, w) * BORDER_MAX_TRIM))
    dark = dark_all
    thickness: list[int] = []
    for side in ("top", "bottom", "left", "right"):
        run = 0
        for i in range(limit):
            band = (
                dark[i, :] if side == "top" else
                dark[h - 1 - i, :] if side == "bottom" else
                dark[:, i] if side == "left" else
                dark[:, w - 1 - i]
            )
            if band.size and float(band.mean()) >= BORDER_DARK_RATIO:
                run = i + 1
            else:
                break
        thickness.append(run)
    return int(min(thickness))


def noise_signals(image: Image.Image) -> dict:
    """一次算全部噪声相关信号（写进 meta，便于排查"为什么这张图被去噪了"）."""
    return {
        "noise_sigma": estimate_noise(image),
        "contrast_span": contrast_span(image),
        "border_px": border_trim(image),
    }


def _cfg(name: str, default: float) -> float:
    """
    读配置（缺失/非法时回退到模块常量）.

    阈值放在 config 里是为了"调参不用改代码"，但模块必须能在没有 Settings 的
    环境（单测、脚本）下独立跑，所以这里吞掉一切异常。
    """
    try:
        from app.config import get_settings

        return float(getattr(get_settings(), name, default))
    except Exception:      # noqa: BLE001
        return float(default)


def thresholds() -> dict:
    """当前生效的噪声/对比度阈值（写进日志便于复现一次判定）."""
    return {
        "noise_sigma_threshold": _cfg("IMAGE_NOISE_SIGMA_THRESHOLD", NOISE_SIGMA_THRESHOLD),
        "noise_sigma_strong": _cfg("IMAGE_NOISE_SIGMA_STRONG", NOISE_SIGMA_STRONG),
        "low_contrast_span": _cfg("IMAGE_LOW_CONTRAST_SPAN", LOW_CONTRAST_SPAN),
        "border_trigger_px": _cfg("IMAGE_BORDER_TRIGGER_PX", BORDER_TRIGGER_PX),
        "border_dark_theme_ratio": _cfg(
            "IMAGE_BORDER_DARK_THEME_RATIO", BORDER_DARK_THEME_RATIO
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(image: Image.Image, *, mode: str = "light") -> PreprocessResult:
    """
    按档位预处理.

    :param mode: ``light`` | ``text`` | ``geometric`` | ``noisy`` | ``scan``
    """
    applied: list[str] = []
    meta: dict = {}

    try:
        out = image.convert("RGB") if image.mode not in ("RGB", "L") else image
    except Exception:      # noqa: BLE001
        return PreprocessResult(image=image, meta={"error": "convert failed"})

    # 噪声/对比度信号对所有档位都算 —— 既用于决定"要不要动手"，也是审计依据。
    # 成本是一次缩略图像素遍历，相对于后面要跑的 OCR / VLM 可以忽略。
    signals = noise_signals(out)
    meta.update(signals)

    out, step, info = _upscale(out)
    if step:
        applied.append(step)
        meta.update(info)

    # ── 去噪：只有"确实有噪"才做 ────────────────────────────────────────────
    # noisy / scan 档位允许在噪声略低时也做（调用方已判定这是翻拍/扫描件），
    # light / text / geometric 则严格按阈值 —— 它们是"干净图"的默认路径。
    limit = thresholds()
    force_denoise = mode in ("noisy", "scan")
    if force_denoise or signals["noise_sigma"] >= limit["noise_sigma_threshold"]:
        out, step, info = _denoise_adaptive(out, signals["noise_sigma"])
        if step:
            applied.append(step)
            meta.update(info)

    # ── 对比度归一化：发灰的扫描件收益最大 ──────────────────────────────────
    if mode in ("noisy", "scan") or signals["contrast_span"] < limit["low_contrast_span"]:
        out, step = _normalize_contrast(out)
        if step:
            applied.append(step)

    # ── 去边框 / 摩尔纹抑制：只有 scan 档位做 ───────────────────────────────
    if mode == "scan":
        out, step, info = _trim_border(out, signals["border_px"])
        if step:
            applied.append(step)
            meta.update(info)
        # 摩尔纹抑制会降分辨率，只有"确实有高频噪声"时才值得付这个代价。
        # 干净的扫描页（sigma≈0）直接跳过 —— 白降 25% 分辨率没有任何收益。
        if signals["noise_sigma"] >= limit["noise_sigma_threshold"]:
            out, step = _suppress_moire(out)
            if step:
                applied.append(step)

    if mode in ("geometric", "scan"):
        out, step, angle = _deskew(out)
        if step:
            applied.append(step)
            meta["skew_degrees"] = angle

    # ── 二值化只进 OCR 通道，绝不替换 image ────────────────────────────────
    ocr_image: Image.Image | None = None
    if mode in ("text", "scan"):
        binary, step = _binarize(out)
        if step:
            applied.append(step)
            ocr_image = binary

    return PreprocessResult(image=out, applied=applied, meta=meta, ocr_image=ocr_image)


def preprocessing_mode_for(image_type: str) -> str:
    """
    图片类型 → 预处理档位（**不看图像内容**的类型侧默认值）.

    表格 / 代码 / 公式都是"文字为主"，纠偏与二值化收益明显；
    图表和照片则只做 light —— 二值化会把数据系列、渐变、照片细节毁掉。

    真正选档请用 :func:`pick_mode`：它在类型默认值之上再看噪声信号，
    把"翻拍/扫描"这类**内容特征**补进来。
    """
    from app.services.image_understanding.structured_content import (
        IMAGE_TYPE_CODE,
        IMAGE_TYPE_FORMULA,
        IMAGE_TYPE_TABLE,
    )

    if image_type in (IMAGE_TYPE_TABLE, IMAGE_TYPE_CODE, IMAGE_TYPE_FORMULA):
        return "geometric"
    return "light"


#: 档位强度排序（只用于"往上抬"时的比较，不用于往下压）
_MODE_RANK = {"light": 0, "geometric": 1, "noisy": 2, "scan": 3, "text": 1}


def pick_mode(image: Image.Image, image_type: str | None = None) -> tuple[str, dict]:
    """
    自动选档：类型默认值 + 噪声信号 → 最终档位.

    规则（保守，只在**有证据**时升级档位）：

        noise_sigma ≥ 强噪声   → scan     （翻拍/扫描：去噪+去边框+摩尔纹+纠偏）
        四条边都有黑边          → scan     （扫描仪盖板痕迹，必须裁掉）
        noise_sigma ≥ 阈值     → noisy    （有噪但结构干净：去噪+对比度）
        contrast_span 很低      → noisy    （发灰的扫描件，即使噪声不大）
        其余                    → 类型默认值

    返回 ``(mode, signals)``，signals 里带上 `mode_reason` 便于排查。
    """
    signals = noise_signals(image)
    default = preprocessing_mode_for(image_type) if image_type else "light"
    sigma = signals["noise_sigma"]
    span = signals["contrast_span"]
    border = signals["border_px"]
    limits = thresholds()
    signals["thresholds"] = limits

    if sigma >= limits["noise_sigma_strong"]:
        mode, reason = "scan", f"strong-noise({sigma})"
    elif border >= limits["border_trigger_px"]:
        # 黑边是"这是扫描件"的硬证据：噪声可能已被扫描软件压掉，但边框还在
        mode, reason = "scan", f"border({border}px)"
    elif sigma >= limits["noise_sigma_threshold"]:
        mode, reason = "noisy", f"noise({sigma})"
    elif span < limits["low_contrast_span"]:
        mode, reason = "noisy", f"low-contrast({span})"
    else:
        mode, reason = default, f"type-default({image_type or '?'})"

    # 类型默认档更"强"时以类型为准（例如表格已判定要纠偏，不该被降成 light）
    if _MODE_RANK.get(default, 0) > _MODE_RANK.get(mode, 0):
        mode, reason = default, f"type-override({image_type})"

    signals["mode"] = mode
    signals["mode_reason"] = reason
    return mode, signals


# ─────────────────────────────────────────────────────────────────────────────
# 各步操作（每一步都：失败返回原图 + 返回"做了什么"的标记）
# ─────────────────────────────────────────────────────────────────────────────


def _upscale(image: Image.Image) -> tuple[Image.Image, str | None, dict]:
    """短边太小就双三次放大（小字放大后 OCR 识别率显著提升）."""
    w, h = image.size
    short = min(w, h)
    if short <= 0 or short >= MIN_SHORT_SIDE:
        return image, None, {}
    scale = TARGET_SHORT_SIDE / short
    # 上限 6 倍，避免一张 20px 的图标被放大成天文数字
    scale = min(scale, 6.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    try:
        return image.resize(new_size, Image.BICUBIC), "upscale", {"scale": round(scale, 2)}
    except Exception:      # noqa: BLE001
        return image, None, {}


def _denoise_adaptive(
    image: Image.Image, sigma: float
) -> tuple[Image.Image, str | None, dict]:
    """
    按估计噪声强度自适应去噪.

    两个设计要点：

    1. **强度跟噪声走**：``fastNlMeansDenoising`` 的 h 参数就是"允许多大差异
       仍视为同一块"。h 固定 10 时，低噪图会被过度平滑（笔画变糊、细线消失），
       高噪图又去不干净。这里把 sigma 线性映射到 h ∈ [3, 18]。
    2. **彩色图走彩色路径**：对彩色图表只对亮度通道去噪，色度保留 —— 否则
       柱状图的色块边界会渗出。

    没有 OpenCV 时退化为 PIL 中值滤波（弱一些但零依赖）。
    """
    # 低于"有意义噪声"的下限就一步都不做：干净的扫描页也会被 pick_mode 判成
    # scan（因为检测到黑边），但它其实一点噪都没有 —— 硬去噪只会把笔画磨糊。
    limit = thresholds()
    if sigma < limit["noise_sigma_threshold"] * 0.4:
        return image, None, {}

    # h = 3 + (sigma - 阈值) 归一化后 × 15，再夹到 [3, 18]
    span = max(1.0, limit["noise_sigma_strong"] - limit["noise_sigma_threshold"])
    ratio = min(1.0, max(0.0, (sigma - NOISE_SIGMA_THRESHOLD) / span))
    h = round(3.0 + 15.0 * ratio, 2)
    info = {"denoise_h": h, "denoise_sigma": sigma}

    try:
        import cv2

        arr = np.array(image.convert("RGB"))
        if image.mode == "L" or _is_greyscale(arr):
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            out = cv2.fastNlMeansDenoising(gray, None, h, 7, 21)
            return Image.fromarray(out).convert("RGB"), "denoise", info
        # 彩色：只动亮度，色度原样保留
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.fastNlMeansDenoising(lab[:, :, 0], None, h, 7, 21)
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)), "denoise", info
    except Exception:      # noqa: BLE001
        pass

    try:
        from PIL import ImageFilter

        size = 3 if h < 8 else 5
        return (
            image.filter(ImageFilter.MedianFilter(size=size)),
            "denoise-median",
            info | {"filter": f"median(size={size})"},
        )
    except Exception:      # noqa: BLE001
        return image, None, {}


def _normalize_contrast(image: Image.Image) -> tuple[Image.Image, str | None]:
    """
    对比度归一化：CLAHE（限制对比度的自适应直方图均衡）.

    为什么用 CLAHE 而不是全局直方图均衡：全局均衡会把"大片白底"的文档
    拉成灰底，并把局部阴影一起放大。CLAHE 在**小块内**均衡并限制增益，
    对"一半被阴影压暗的翻拍页"这种不均匀光照最有效。
    """
    try:
        import cv2

        arr = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)), "contrast"
    except Exception:      # noqa: BLE001
        pass

    # 无 OpenCV：百分位拉伸（比直方图均衡保守）
    try:
        gray = np.asarray(image.convert("L"), dtype=np.float64)
        # 与 contrast_span 同口径的极端分位（墨迹稀疏时退化为极值）
        low, high = np.percentile(gray, [0.5, 99.5])
        if high - low < 1.0:
            low, high = float(gray.min()), float(gray.max())
        if high - low < 1:
            return image, None
        stretched = np.clip((gray - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
        return Image.fromarray(stretched).convert("RGB"), "contrast-stretch"
    except Exception:      # noqa: BLE001
        return image, None


def _trim_border(
    image: Image.Image, known_px: int
) -> tuple[Image.Image, str | None, dict]:
    """裁掉扫描黑边（``known_px`` 已由 :func:`border_trim` 估出，0 表示没有）."""
    if known_px <= 0:
        return image, None, {}
    w, h = image.size
    if w - 2 * known_px < 32 or h - 2 * known_px < 32:
        return image, None, {}
    try:
        box = (known_px, known_px, w - known_px, h - known_px)
        return image.crop(box), "trim-border", {"trim_px": known_px}
    except Exception:      # noqa: BLE001
        return image, None, {}


def _suppress_moire(image: Image.Image) -> tuple[Image.Image, str | None]:
    """
    抑制翻拍屏幕/印刷网点造成的摩尔纹.

    做法是"先轻度模糊再微下采样"：摩尔纹的频率高于文字笔画，一次
    ``高斯模糊 + 0.75 倍缩放`` 能压掉大部分周期性纹理，而文字仍然可读。
    代价是分辨率下降，因此只在 scan 档位（已判定是翻拍）使用。
    """
    try:
        from PIL import ImageFilter

        w, h = image.size
        if min(w, h) < 64:
            return image, None
        blurred = image.filter(ImageFilter.GaussianBlur(radius=0.8))
        target = (max(1, int(w * 0.75)), max(1, int(h * 0.75)))
        return blurred.resize(target, Image.LANCZOS), "moire-suppress"
    except Exception:      # noqa: BLE001
        return image, None


def _deskew(image: Image.Image) -> tuple[Image.Image, str | None, float]:
    """
    倾斜校正.

    用所有"墨点"的最小外接矩形角度估计倾斜。这个估计对文字块很准，对无文字的
    图会给出噪声角度 —— 因此低于 ``MIN_SKEW_DEGREES`` 一律不动。
    """
    try:
        import cv2

        gray = np.array(image.convert("L"))
        ink = (gray < 160).astype(np.uint8)
        if int(ink.sum()) < 200:
            return image, None, 0.0

        coords = np.column_stack(np.where(ink > 0))
        angle = float(cv2.minAreaRect(coords.astype(np.float32))[-1])
        if angle > 45:
            angle -= 90
        if abs(angle) < MIN_SKEW_DEGREES:
            return image, None, 0.0

        h, w = gray.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            np.array(image.convert("RGB")), matrix, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(rotated), "deskew", round(angle, 2)
    except Exception:      # noqa: BLE001
        return image, None, 0.0


def _binarize(image: Image.Image) -> tuple[Image.Image, str | None]:
    """
    自适应二值化 —— 只用于喂 OCR 的副本，不替换落盘原图.

    **必须先归正极性**：``adaptiveThreshold`` 的规则是"比邻域均值暗 → 判为
    前景(0)"。深底浅字的图（深色主题截图 / 暗色流程图 / 亮字终端）里整块
    深色背景都符合"比邻域均值暗"，于是背景与字形一起被判成前景，二值化后
    整幅图变成一片白 —— 内容被抹平（实测：深色流程图二值化后亮像素占比从
    42% 涨到 84%，图彻底没了）。

    所有 OCR 引擎（Tesseract LSTM / PaddleOCR）的识别模型都是在"深字浅底"
    上训练的，因此深底图先整体反色（255 - v）再二值化，得到的正是引擎想要的
    形态：浅底 + 深字。反色不改变字形，只是把"纸"和"字"换到模型熟悉的那一侧。

    **混合极性的图直接跳过二值化**：白底页面上嵌一块大面积深色图时，无论
    按哪一侧归正，另一侧都会被抹掉（按白底归正 → 深色图整块变白；按深底归正
    → 白底页边距整块变黑）。这种图二值化不可能同时保住两侧，因此宁可不做 ——
    与模块开头"只做确定有帮助的操作"的原则一致。混合极性在现实中很常见：
    深色主题流程图/终端窗口截图粘进白底文档就是这一类。

    返回 ``(image, None)`` 表示"这一步没做"，调用方据此不会把 binarize
    写进 ``applied``，审计时能看到"这张图为什么没二值化"。
    """
    try:
        import cv2

        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        polarity = _page_polarity(gray)
        if polarity == "mixed":
            return image, None
        if polarity == "dark":
            gray = cv2.bitwise_not(gray)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        return Image.fromarray(binary).convert("RGB"), "binarize"
    except Exception:      # noqa: BLE001
        return image, None


def _page_polarity(gray: np.ndarray) -> str:
    """
    这张（灰度）图的"纸"是哪一侧：``"light"`` | ``"dark"`` | ``"mixed"``.

    判据与 ``imaging.background_is_dark`` 同一套（看最外圈背景环，环太花就退回
    全图多数），额外再判"混合极性"：底色一侧之外还压着一大块**反向**的大面积
    区域（占比 ≥ :data:`MIXED_POLARITY_FRAC`），说明图上同时有纸和一块反色面板，
    二值化必然毁掉其中之一。

    入参已经是 ndarray，直接用切片取样，省掉一次 ravel→list 的拷贝 ——
    这也是这里没有直接复用 ``imaging.page_background`` 的原因（那个函数收的是
    分类器用的扁平像素列表）。判据本身只有一行，重复实现的风险远小于搬运像素。
    """
    from app.services.image_understanding.imaging import (
        BG_DARK_LEVEL,
        BG_PURITY,
        BG_RING_MAX_PX,
        BG_RING_RATIO,
    )

    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        return "light"
    band = max(1, min(BG_RING_MAX_PX, int(min(h, w) * BG_RING_RATIO) or 1))
    ring = np.concatenate([
        gray[:band, :].ravel(), gray[h - band:, :].ravel(),
        gray[:, :band].ravel(), gray[:, w - band:].ravel(),
    ])
    dark_frac = float((ring < BG_DARK_LEVEL).mean())
    if max(dark_frac, 1.0 - dark_frac) < BG_PURITY:
        # 背景环本身很花（满幅照片）→ 退回全图多数，与 imaging 保持一致
        dark_frac = float((gray < BG_DARK_LEVEL).mean())

    dark_paper = dark_frac > 0.5
    # 反色大块占比：浅底图上数"深色"块，深底图上数"亮色"块（阈值对称）
    opposite = (
        float((gray < BG_DARK_LEVEL).mean()) if not dark_paper
        else float((gray > 255 - BG_DARK_LEVEL).mean())
    )
    if opposite >= MIXED_POLARITY_FRAC:
        return "mixed"
    return "dark" if dark_paper else "light"


def _is_greyscale(arr: np.ndarray) -> bool:
    """整张图是否近似灰度（各通道几乎相同）—— 决定走单通道还是彩色去噪."""
    try:
        if arr.ndim != 3 or arr.shape[2] < 3:
            return True
        diff = np.abs(arr[:, :, 0].astype(np.int16) - arr[:, :, 2].astype(np.int16))
        return float(diff.mean()) < 2.0
    except Exception:      # noqa: BLE001
        return True


def page_polarity(image: Image.Image) -> str:
    """
    这张图的"纸"在哪一侧：``"light"`` | ``"dark"`` | ``"mixed"``.

    公开出来是给 **OCR 兜底**用的：OCR 引擎的识别模型都在"深字浅底"上训练，
    深底浅字（深色主题截图 / 终端 / 暗色流程图）直接喂进去会大量漏检 ——
    Tesseract 在这种图上通常直接返回空串。拿到极性后可以喂一张反色副本
    （见 :func:`invert_for_ocr`）再读一遍。
    """
    try:
        return _page_polarity(np.asarray(image.convert("L")))
    except Exception:      # noqa: BLE001
        return "light"


def invert_for_ocr(image: Image.Image) -> Image.Image:
    """把图翻成正极性（浅底 + 深字），专供"只认深字浅底"的 OCR 引擎."""
    try:
        flipped = 255 - np.asarray(image.convert("L"))
        return Image.fromarray(flipped.astype(np.uint8)).convert("RGB")
    except Exception:      # noqa: BLE001
        return image


__all__ = [
    "PreprocessResult",
    "preprocess",
    "preprocessing_mode_for",
    "pick_mode",
    "noise_signals",
    "thresholds",
    "estimate_noise",
    "contrast_span",
    "border_trim",
    "page_polarity",
    "invert_for_ocr",
    "MIN_SHORT_SIDE",
    "TARGET_SHORT_SIDE",
    "MIN_SKEW_DEGREES",
    "NOISE_SIGMA_THRESHOLD",
    "NOISE_SIGMA_STRONG",
    "LOW_CONTRAST_SPAN",
    "BORDER_TRIGGER_PX",
    "MIXED_POLARITY_FRAC",
]
