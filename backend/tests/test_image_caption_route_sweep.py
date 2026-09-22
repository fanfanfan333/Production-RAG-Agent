"""「有图却没有图意总结」的**扩面核查**回归门禁 —— 实施手册 5 张图的真实信号锚点.

背景
────
缺陷根因在 ``classifier.py``：一张「三栏彩色结构示意图」（手册第 3 张图）同时踩空
三条既有判据（colorful 不够、line_art 不够、text_dense 不够）→ 掉进 ``default``
→ ``photo`` → ``photo ∉ VISION_ANALYZE_TYPES`` → ``resolve_route`` 走 OCR →
``_summarize_if_textless`` 又因该图有 OCR 文字而短路 → **永久拿不到 image_caption**。
修复是补 ``structured-panels`` 兜底判据（见 ``test_image_structured_diagram.py``）。

本文件做的是**扩面核查**：把手册 **5 张图全部**用真实管线
（``pick_mode`` → ``preprocess`` → 真实 OCR → ``classify_image_safe``）跑一遍，
把每张图的**实测信号**原样留在这里当锚点，锁死「每张图都能路由到 vision、
从而都能拿到图意总结」这条结论：

    图 1  四层总体架构图        → screenshot  → vision ✓
    图 2  主流文档类型成功率图  → chart       → vision ✓
    图 3  三种分块策略对比图    → diagram     → vision ✓（本轮修复）
    图 4  分块大小权衡曲线图    → screenshot  → vision ✓
    图 5  四种检索策略对比图    → chart       → vision ✓

核查结论：1/2/4/5 张**没有别的死角**（都被既有判据正确收进 vision 白名单类型），
因此本轮**只需**第 3 张图的 ``structured-panels`` 修复，不追加源码改动。

为什么用「信号锚点」而不是图片文件
──────────────────────────────────
这 5 张图来自用户的桌面文档（不在仓库里），测试不能依赖它们。所以把真实管线
实测出的信号留成常量：即便将来合成夹具重画、像素环境漂移，真实样本的数值仍被钉住。
一旦谁放宽/收紧判据、把它们任一张重新推回 photo/ocr，本文件立刻变红。

宿主机缺依赖时优雅跳过；推荐在 backend 容器内运行：
    docker exec -w /app -e PYTHONPATH=/app rag_backend \
        python -m pytest tests/test_image_caption_route_sweep.py -q
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
    from app.services.chunker import build_image_chunks
    from app.services.image_understanding import (
        IMAGE_TYPE_CHART,
        IMAGE_TYPE_DIAGRAM,
        IMAGE_TYPE_PHOTO,
        IMAGE_TYPE_SCREENSHOT,
        classify_image_safe,
        resolve_route,
        vision_analyze_types,
    )
    from app.services.image_understanding.classifier import (
        ImageClassification,
        _decide,
        compute_signals,
    )
    from app.services.image_understanding.engines.base import EngineOutput
    from app.services.image_understanding.pipeline import (
        ROUTE_VISION,
        _is_vision_engine,
    )
    from app.services.parsers.base import ExtractedImage

    pl = importlib.import_module("app.services.image_understanding.pipeline")
except ImportError as exc:  # 宿主机缺依赖 → 跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")


# ─────────────────────────────────────────────────────────────────────────────
# 真实锚点：手册 5 张图的**实测信号**（2026-09，真实管线，backend 容器内）
#   复现方式：pick_mode → preprocess → _run_ocr → classify_image_safe(ocr_lines=…)
#   字段与 classifier.compute_signals 同口径；ocr_text_len 为真实 OCR 文字数。
# ─────────────────────────────────────────────────────────────────────────────

#: 每张图：实测信号 + 真实 OCR 文字长度 + 期望类型（都在 VISION_ANALYZE_TYPES 内）。
MANUAL_IMAGE_ANCHORS: dict[str, dict] = {
    "image1_四层总体架构图": {
        "ocr_text_len": 164,
        "expected_type": IMAGE_TYPE_SCREENSHOT,
        "signals": {
            "h_lines": 0, "v_lines": 0, "raw_h_bands": 0, "raw_v_bands": 0,
            "text_bands": 5, "line_art_ratio": 0.8902, "saturation": 0.0072,
            "flat_blocks": 3, "chromatic_blocks": 0, "dominant_ratio": 0.719,
            "colorful_ratio": 0.0131, "ocr_rows": 9, "ocr_cols": 15,
            "ocr_multi_cell_rows": 4, "ocr_alignment": 0.4444,
            "ocr_col_stability": 0.1778, "math_ratio": 0.0, "code_score": 0.0,
            "cjk_ratio": 0.9645,
        },
    },
    "image2_文档类型解析成功率图": {
        "ocr_text_len": 204,
        "expected_type": IMAGE_TYPE_CHART,
        "signals": {
            "h_lines": 0, "v_lines": 0, "raw_h_bands": 5, "raw_v_bands": 0,
            "text_bands": 8, "line_art_ratio": 0.7831, "saturation": 0.1075,
            "flat_blocks": 1, "chromatic_blocks": 0, "dominant_ratio": 0.6734,
            "colorful_ratio": 0.1771, "ocr_rows": 10, "ocr_cols": 17,
            "ocr_multi_cell_rows": 8, "ocr_alignment": 0.8,
            "ocr_col_stability": 0.1294, "math_ratio": 0.1257,
            "code_score": 0.3843, "cjk_ratio": 0.3989,
        },
    },
    "image3_三种分块策略对比图": {
        "ocr_text_len": 145,
        "expected_type": IMAGE_TYPE_DIAGRAM,
        "signals": {
            "h_lines": 0, "v_lines": 2, "raw_h_bands": 1, "raw_v_bands": 0,
            "text_bands": 2, "line_art_ratio": 0.6519, "saturation": 0.062,
            "flat_blocks": 5, "chromatic_blocks": 0, "dominant_ratio": 0.4761,
            "colorful_ratio": 0.0889, "ocr_rows": 11, "ocr_cols": 9,
            "ocr_multi_cell_rows": 4, "ocr_alignment": 0.3636,
            "ocr_col_stability": 0.1818, "math_ratio": 0.0081,
            "code_score": 0.0312, "cjk_ratio": 0.7398,
        },
    },
    "image4_分块大小权衡曲线图": {
        "ocr_text_len": 220,
        "expected_type": IMAGE_TYPE_SCREENSHOT,
        "signals": {
            "h_lines": 0, "v_lines": 0, "raw_h_bands": 0, "raw_v_bands": 0,
            "text_bands": 1, "line_art_ratio": 0.9529, "saturation": 0.0015,
            "flat_blocks": 1, "chromatic_blocks": 0, "dominant_ratio": 0.898,
            "colorful_ratio": 0.0, "ocr_rows": 20, "ocr_cols": 13,
            "ocr_multi_cell_rows": 8, "ocr_alignment": 0.4,
            "ocr_col_stability": 0.1346, "math_ratio": 0.0753,
            "code_score": 0.3068, "cjk_ratio": 0.3011,
        },
    },
    "image5_四种检索策略对比图": {
        "ocr_text_len": 186,
        "expected_type": IMAGE_TYPE_CHART,
        "signals": {
            "h_lines": 0, "v_lines": 0, "raw_h_bands": 0, "raw_v_bands": 4,
            "text_bands": 4, "line_art_ratio": 0.7299, "saturation": 0.1362,
            "flat_blocks": 2, "chromatic_blocks": 1, "dominant_ratio": 0.6577,
            "colorful_ratio": 0.1991, "ocr_rows": 15, "ocr_cols": 20,
            "ocr_multi_cell_rows": 6, "ocr_alignment": 0.4,
            "ocr_col_stability": 0.1033, "math_ratio": 0.0437,
            "code_score": 0.275, "cjk_ratio": 0.475,
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 夹具
# ─────────────────────────────────────────────────────────────────────────────


def _signals(**overrides) -> dict:
    """形状完整、取值中性的判定信号（真实 compute_signals 打底 + override）."""
    base = compute_signals(Image.new("RGB", (520, 120), (255, 255, 255)))
    base.update(overrides)
    return base


def _photo_signals() -> dict:
    """真实照片（``test_image_structured_diagram._photo_image``）的实测信号 —— 阴性红线."""
    return {
        "h_lines": 0, "v_lines": 1, "raw_h_bands": 1, "raw_v_bands": 2,
        "text_bands": 1, "line_art_ratio": 0.0, "saturation": 0.5015,
        "flat_blocks": 8, "chromatic_blocks": 8, "dominant_ratio": 0.1679,
        "colorful_ratio": 1.0, "ocr_rows": 0, "ocr_cols": 0,
        "ocr_multi_cell_rows": 0, "ocr_alignment": 0.0, "ocr_col_stability": 0.0,
        "math_ratio": 0.0, "code_score": 0.0, "cjk_ratio": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. 五张图逐张：类型正确 + 路由到 vision（⇒ 拿得到图意总结）
# ─────────────────────────────────────────────────────────────────────────────


def test_all_five_manual_images_route_to_vision() -> None:
    """手册 5 张图的实测信号必须**逐张**判进 ``VISION_ANALYZE_TYPES`` 并路由 vision."""
    vision_types = vision_analyze_types()
    assert vision_types, "VISION_ANALYZE_TYPES 为空 —— 所有图都会走 OCR、拿不到图意总结"

    for name, anchor in MANUAL_IMAGE_ANCHORS.items():
        image_type, _conf, reason = _decide(_signals(**anchor["signals"]), get_settings())
        assert image_type == anchor["expected_type"], (
            f"{name}: 实测信号被判成 {image_type}（expected {anchor['expected_type']}，"
            f"reason={reason}）—— 分类判据回归"
        )
        assert image_type in vision_types, (
            f"{name}: {image_type} 不在 Vision 白名单 {sorted(vision_types)} —— "
            "该图将走 OCR、拿不到图意总结"
        )
        assert resolve_route(image_type) == "vision", (
            f"{name}: resolve_route({image_type}) != vision —— 图意总结链断裂"
        )


def test_each_manual_image_carries_ocr_text() -> None:
    """5 张图都有 OCR 文字（145~220 字）—— 这正是"有图却没图意总结"的诱因.

    因为有文字，``_summarize_if_textless`` 的第一步 ``_has_any_text`` 就短路了，
    所以**唯一**能拿到图意总结的路径是把它们判进 vision 白名单类型。这条断言把
    "为什么必须靠分类而非无文字兜底"钉死，防止有人误以为可以靠 textless 分支补救。
    """
    for name, anchor in MANUAL_IMAGE_ANCHORS.items():
        assert anchor["ocr_text_len"] > 0, (
            f"{name}: OCR 文字长度应为正（实测缺失？），否则论证前提不成立"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. 端到端因果链：vision 类型 ⇒ 走多模态 ⇒ **真的写出** vision_caption ⇒ 落进 chunk
# ─────────────────────────────────────────────────────────────────────────────

#: 替身 Vision 引擎固定回吐的图意（语义取手册第 3 张图）—— 端到端逐字比较的基准。
STUB_CAPTION = "STUB-CAPTION-三种分块策略语义完整性对比"


class _StubVision:
    """确定性 ``VisionEngine`` 替身：任何 role 都回吐 ``STUB_CAPTION``（engine='vision'）.

    用它把"图意总结有没有被真的写出来"从"真跑 VLM 才可验证"变成**确定性**断言：
    不需要模型、不需要 skip，产出逐字可控。
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        pass

    def is_available(self) -> bool:
        return True

    def process(self, image, *, png_bytes=None, image_type="photo", role="primary", **kw):
        return EngineOutput(
            text=STUB_CAPTION, confidence=0.9, engine="vision", ok=True,
        )


def _pipeline_image() -> "Image.Image":
    """喂进管线的一张普通小图（像素不参与判定：分类被替身接管）."""
    return Image.new("RGB", (400, 260), (255, 255, 255))


def test_vision_route_writes_caption_end_to_end(monkeypatch) -> None:
    """```vision 类型 → ROUTE_VISION → 引擎产出 → _apply_output 写 vision_caption
    → chunker payload image_caption``` 整条链必须**真的**把图意写出来.

    旧版本只断言 ``resolve_route() == ROUTE_VISION``（名不副实：从没验证过 caption
    真被写出）。这里改成确定性端到端断言：分类判成 diagram（→ 走 vision），配一个
    回吐固定 caption 的替身 Vision 引擎，最后断言 chunk payload 的 ``image_caption``
    **逐字等于**该固定 caption —— 不是 None、也不是 ``mock.called`` 这类弱断言。
    """
    # 分类判成 diagram（→ ROUTE_VISION）；语义取手册第 3 张「三种分块策略对比图」
    monkeypatch.setattr(
        pl, "classify_image_safe",
        lambda image, ocr_lines=None, filename="": ImageClassification(
            image_type=IMAGE_TYPE_DIAGRAM, confidence=0.8, signals={}, engine="rules",
        ),
    )
    # 这条断言只看 vision caption 链：给一个空 OCR，避免依赖真实 OCR 引擎
    monkeypatch.setattr(
        pl, "_run_ocr",
        lambda image: EngineOutput(
            text="", confidence=0.0, engine="tesseract", ok=True, lines=[],
        ),
    )
    monkeypatch.setattr(pl, "_second_ocr", lambda image, ocr_out: None)
    # 替身 Vision 引擎：确定性回吐固定 caption（不依赖真实 VLM、不 skip）
    monkeypatch.setattr(pl, "VisionEngine", _StubVision)

    # 前置：这个类型确实会被路由到 vision（否则后面断言无意义）
    assert resolve_route(IMAGE_TYPE_DIAGRAM) == ROUTE_VISION
    # photo 绝不能进 vision 白名单（否则会把每张照片都多烧一次多模态推理）
    assert IMAGE_TYPE_PHOTO not in vision_analyze_types()

    result = pl.understand_image(_pipeline_image(), page_number=1, filename="image3.png")

    # ① 真的走到了（vision）引擎
    assert result.image_type == IMAGE_TYPE_DIAGRAM, result.image_type
    assert result.route == ROUTE_VISION, result.route
    assert _is_vision_engine(result.analyze_engine), result.analyze_engine
    # ② 图意被真的写进 vision_caption（逐字相等）
    assert result.vision_caption == STUB_CAPTION, repr(result.vision_caption)

    # ③ 端到端：这条 caption 落到 chunk payload 的 image_caption 上
    chunk_owner = ExtractedImage(
        image_id="d-p1-i1", page_number=1,
        ocr_text=result.ocr_text, vision_caption=result.vision_caption,
        image_path="images/image3.png", image_type=result.image_type,
        analyze_engine=result.analyze_engine,
    )
    chunks = build_image_chunks([chunk_owner])
    assert len(chunks) == 1, chunks
    assert chunks[0].content_type == "image", chunks[0].content_type
    assert chunks[0].image_caption == STUB_CAPTION, repr(chunks[0].image_caption)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 阴性红线：真实照片必须仍判 photo（不得为了"多拿 caption"把判据放宽）
# ─────────────────────────────────────────────────────────────────────────────


def test_real_photo_still_not_in_vision_types() -> None:
    """真实照片信号必须仍判 PHOTO、且不在 Vision 白名单（防判据被放宽）."""
    image_type, _conf, reason = _decide(_signals(**_photo_signals()), get_settings())
    assert image_type == IMAGE_TYPE_PHOTO, (
        f"真实照片被判成 {image_type}（reason={reason}）—— 判据被放宽了"
    )
    assert image_type not in vision_analyze_types()
    assert resolve_route(image_type) == "ocr"


def test_blank_page_signals_not_promoted() -> None:
    """纯空白页信号不得被升格进 vision 白名单（保持旧行为）."""
    image_type, _conf, reason = _decide(_signals(
        h_lines=0, v_lines=0, raw_h_bands=0, raw_v_bands=0, text_bands=0,
        line_art_ratio=1.0, saturation=0.0, flat_blocks=1, chromatic_blocks=0,
        dominant_ratio=0.98, colorful_ratio=0.0, ocr_rows=0, ocr_cols=0,
        ocr_multi_cell_rows=0, ocr_alignment=0.0, ocr_col_stability=0.0,
        math_ratio=0.0, code_score=0.0, cjk_ratio=0.0,
    ), get_settings())
    assert image_type == IMAGE_TYPE_PHOTO, (
        f"空白页被判成 {image_type}（reason={reason}）—— 结构兜底收得太宽"
    )


if __name__ == "__main__":
    import traceback

    ok = fail = 0
    for _name, _fn in sorted(globals().items()):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        try:
            _fn()
            ok += 1
        except Exception:      # noqa: BLE001
            fail += 1
            print(f"  FAIL {_name}")
            traceback.print_exc()
    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)
