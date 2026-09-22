"""
「彩色结构示意图」分类死角回归门禁 —— 修复「有图却没图意总结」的缺陷.

被验证的行为（改动前旧行为见 _decide 的最后一步 ``return IMAGE_TYPE_PHOTO,
0.4, "default"``）：

  实测缺陷：文档「企业知识库RAG系统建设实施手册.docx」第 3 张图（三栏并排对比
  示意图，每栏带蓝/青/紫色标题条 + 卡片面板 + 灰色文本条）—— 一张**结构示意图**
  —— 被判成了 ``photo``，于是：

      photo ∉ VISION_ANALYZE_TYPES(=chart,diagram,screenshot)
        → resolve_route() 走 ROUTE_OCR（专用引擎是 OCR，不是 Vision）
        → pipeline._apply_output() 只在 vision 引擎产出时才写 vision_caption
        → pipeline._summarize_if_textless() 第一步就是 `if _has_any_text: return`
          （这张图有 145 字 OCR 文本）
        → 最终 Qdrant payload ``image_caption=None``：图**永久没有图意总结**。

  根因：这张图同时踩空三条既有判据 ——
     · 图表要求 colorful_ratio ≥ 0.15，实测只有 0.0889；
     · 流程图/结构图要求 line_art_ratio ≥ 0.8，实测只有 0.6519；
     · 截图要求 line_art_ratio ≥ 0.75，同样够不上。
  于是它既够不上 chart、又够不上 diagram，掉进 photo。

修复：在 ``_decide`` 补一条「结构图兜底」判据（reason 标记 ``structured-panels``）：
统一背景（dominant_ratio 高）+ 确有结构（粗带/彩色块/直线/多行文字）+ 彩度不足
以判图表（colorful_ratio < 0.15），即认作 ``IMAGE_TYPE_DIAGRAM``，让它走多模态、
拿得到图意描述。

本文件锁死三件事：

  1. **阳性**：合成「彩色结构示意图」必须判 DIAGRAM（reason=structured-panels）；
  2. **真实锚点**：把 image 3 的**实测信号**直接喂给 ``_decide``，必须判 DIAGRAM
     —— 这样即便合成夹具的像素环境漂移，真实世界的那个失败样本也不会再回归；
  3. **阴性对照**：真实照片（``_photo_image``）必须仍判 PHOTO（防止把判据放宽）；
     另附两条护栏：①「纯空白页 / 纯色块不得被升格成 diagram」；
     ②「**纯文字翻拍页**不得被升格成 diagram」—— 第 6.5 步的"确有结构"**只认几何
     结构**（粗带 / 彩色描边块 / 细直线），**不**把 ``ocr_rows`` 当结构信号（纯文字页
     的正解是 OCR，不是 vision）；
  4. **端到端因果链**：判成 diagram ⇒ ``resolve_route()`` 返回 ROUTE_VISION
     ⇒「会走多模态 ⇒ 拿得到 caption」这条链被钉死。

宿主机缺依赖时优雅跳过（与既有单测同一约定）；推荐在 backend 容器内运行：
    docker exec -w /app -e PYTHONPATH=/app rag_backend \
        python -m pytest tests/test_image_structured_diagram.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    from PIL import Image, ImageDraw

    from app.config import get_settings
    from app.services.image_understanding import (
        IMAGE_TYPE_DIAGRAM,
        IMAGE_TYPE_PHOTO,
        classify_image_safe,
        resolve_route,
    )
except ImportError as exc:  # 宿主机缺依赖 → 跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")


# ── 夹具 ─────────────────────────────────────────────────────────────────────

def _photo_image():
    """照片化：连续渐变 + 噪声，无规则线条、无统一背景（必须仍判 photo）."""
    import random

    random.seed(11)
    img = Image.new("RGB", (520, 320))
    px = img.load()
    for y in range(320):
        for x in range(520):
            base = int(90 + 70 * (x / 520.0) + 40 * (y / 320.0))
            px[x, y] = (
                min(255, base + random.randint(-18, 18)),
                min(255, int(base * 0.72) + random.randint(-18, 18)),
                min(255, int(base * 0.5) + random.randint(-18, 18)),
            )
    return img


def _colored_structure_diagram():
    """
    合成的「三栏彩色结构示意图」：统一白底 + 彩色标题条 + 卡片面板 + 灰文本条.

    意图复现 image 3 的信号画像（line_art≈0.65、colorful≈0.09、dominant≈0.47）——
    一张彩度不足以判图表、线稿占比又不够高的**结构示意图**。
    """
    img = Image.new("RGB", (1050, 450), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    titles = [(70, 130, 220), (60, 200, 200), (150, 90, 210)]   # 蓝 / 青 / 紫
    for ci, colour in enumerate(titles):
        x0 = 40 + ci * 330
        x1 = x0 + 280
        draw.rectangle([x0, 60, x1, 110], fill=colour)          # 彩色标题条
        draw.rectangle([x0, 130, x1, 400], outline=(120, 120, 120), width=3)
        for ri in range(6):
            y = 160 + ri * 38
            draw.rectangle([x0 + 25, y, x1 - 25, y + 16], fill=(190, 190, 190))
    return img


class _TextLine:
    """真实 ``OCRLine`` 的极简替身（``_ocr_layout`` 只读 ``text`` 与 ``box``）."""

    def __init__(self, text: str, box: tuple[float, float, float, float]) -> None:
        self.text = text
        self.box = box


def _text_only_page_photo():
    """
    纯文字翻拍页：白底 + 右侧纸灰阴影（翻拍不均匀光照）+ 若干**短**黑文字行.

    刻意做到：dominant 高（大片白纸）、line_art 中等（0.55~0.8，不足以走第 6 步
    线稿判据）、colorful≈0（黑白灰）、**无**直线 / 粗带 / 彩色块 —— 即一张
    "有文字但没有任何几何结构"的页面。它的正解是 **OCR**（要文字），不是 vision
    （要图意），因此**不得**被判成 diagram。
    """
    width, height = 800, 1000
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # 右侧约 38% 覆盖纸灰阴影（灰度 215 < 近白阈值 235 → 压低 line_art，模拟翻拍）
    draw.rectangle([int(width * 0.62), 0, width, height], fill=(215, 215, 215))
    # 4 行**短**黑文字（宽度 ~300/800 → 不足以形成 raw_h_bands 的 ≥55% 贯穿带）
    for i in range(4):
        y = 120 + i * 70
        draw.rectangle([60, y, 360, y + 28], fill=(0, 0, 0))
    return img


def _blank_signals(**overrides) -> dict:
    """
    构造形状完整、取值中性的判定信号（与 test_image_pipeline._blank_signals 同法）.

    用一张纯白画布走**真实** compute_signals，形状永远与生产路径一致；再 update
    上 overrides。这样把真实锚点喂给 _decide 时不会因漏键而 KeyError。
    """
    from app.services.image_understanding.classifier import compute_signals

    base = compute_signals(Image.new("RGB", (520, 120), (255, 255, 255)))
    base.update(overrides)
    return base


#: image 3 的**实测信号**（真实管线 pick_mode→preprocess→_run_ocr→classify 复现，
#: 2026-09）。作为回归锚点原样留在这里：只要它还能判成 diagram，真实世界的那个
#: 「无图意总结」样本就不会再回归。
IMAGE3_SIGNALS = {
    "h_lines": 0, "v_lines": 2, "raw_h_bands": 1, "raw_v_bands": 0,
    "text_bands": 2, "line_art_ratio": 0.6519, "saturation": 0.062,
    "flat_blocks": 5, "chromatic_blocks": 0, "dominant_ratio": 0.4761,
    "colorful_ratio": 0.0889, "ocr_rows": 11, "ocr_cols": 9,
    "ocr_multi_cell_rows": 4, "ocr_alignment": 0.3636, "ocr_col_stability": 0.1818,
    "math_ratio": 0.0081, "code_score": 0.0312, "cjk_ratio": 0.7398,
}


# ── 1. 阳性：彩色结构示意图必须判 diagram ─────────────────────────────────────

def test_colored_structure_diagram_is_diagram_not_photo() -> None:
    """一张「彩色结构示意图」必须判 DIAGRAM，而不是掉进 photo（→无图意总结）."""
    result = classify_image_safe(_colored_structure_diagram(), filename="colored-diagram")

    assert result.image_type == IMAGE_TYPE_DIAGRAM, (
        f"彩色结构示意图被判成 {result.image_type}"
        f"（reason={result.signals.get('reason')}）—— photo 会走 OCR、永远拿不到图意总结"
    )
    assert result.signals.get("reason") == "structured-panels", result.signals
    assert 0.0 < result.confidence <= 1.0
    print("  ok test_colored_structure_diagram_is_diagram_not_photo")


# ── 2. 真实锚点：image 3 的实测信号必须判 diagram ──────────────────────────────

def test_real_image3_signals_classify_as_diagram() -> None:
    """
    直接喂 image 3 的**实测信号**——不改判据就必须判 diagram.

    这条独立于合成夹具的像素环境：即使将来合成图重画得不再踩中判据，真实样本
    的数值仍被钉住。修复前这条会落到 ``photo/default``。
    """
    from app.services.image_understanding.classifier import _decide

    image_type, confidence, reason = _decide(_blank_signals(**IMAGE3_SIGNALS), get_settings())

    assert image_type == IMAGE_TYPE_DIAGRAM, (
        f"image 3 实测信号被判成 {image_type}（reason={reason}）—— 这正是缺陷复现"
    )
    assert reason == "structured-panels", reason
    assert 0.0 < confidence <= 1.0
    print("  ok test_real_image3_signals_classify_as_diagram")


# ── 3. 阴性对照：真实照片必须仍判 photo（红线）───────────────────────────────

def test_real_photo_still_classified_as_photo() -> None:
    """真实照片夹具必须仍判 PHOTO（防止把「结构图兜底」判据放宽）."""
    from app.services.image_understanding.classifier import _decide

    # 端到端：走真实像素管线
    result = classify_image_safe(_photo_image(), filename="photo")
    assert result.image_type == IMAGE_TYPE_PHOTO, (
        f"真实照片被判成了 {result.image_type}（reason={result.signals.get('reason')}）"
        "—— 结构图兜底判据放得太宽了"
    )
    assert resolve_route(result.image_type) == "ocr"

    # 信号级：把 _photo_image 的实测信号喂给 _decide，锁死"三条判据都落在门外"
    photo_signals = {
        "h_lines": 0, "v_lines": 1, "raw_h_bands": 1, "raw_v_bands": 2,
        "text_bands": 1, "line_art_ratio": 0.0, "saturation": 0.5015,
        "flat_blocks": 8, "chromatic_blocks": 8, "dominant_ratio": 0.1679,
        "colorful_ratio": 1.0, "ocr_rows": 0, "ocr_cols": 0,
        "ocr_multi_cell_rows": 0, "ocr_alignment": 0.0, "ocr_col_stability": 0.0,
        "math_ratio": 0.0, "code_score": 0.0, "cjk_ratio": 0.0,
    }
    image_type, _, reason = _decide(_blank_signals(**photo_signals), get_settings())
    assert image_type == IMAGE_TYPE_PHOTO, (
        f"照片信号被判成 {image_type}（reason={reason}）—— dominant 低 / line_art≈0 "
        "/ colorful 极高，三条判据都应把它挡在门外"
    )
    print("  ok test_real_photo_still_classified_as_photo")


def test_blank_and_solid_page_not_promoted_to_diagram() -> None:
    """
    纯空白页 / 纯色块**不得**凭「背景统一」被升格成 diagram.

    结构图兜底要求「确有结构」（粗带 / 彩色块 / 直线 / 多行文字之一）；这条正则
    保证空图仍然落到 photo（保持旧行为），避免兜底判据把"什么都没有"也收编。
    """
    from app.services.image_understanding.classifier import (
        _decide,
        compute_signals,
    )

    for img in (
        Image.new("RGB", (520, 200), (255, 255, 255)),   # 纯白
        Image.new("RGB", (520, 200), (128, 128, 128)),   # 纯灰
    ):
        image_type, _, reason = _decide(compute_signals(img), get_settings())
        assert image_type != IMAGE_TYPE_DIAGRAM, (
            f"无结构内容被判成 diagram（reason={reason}）—— 兜底判据收得太宽"
        )
    print("  ok test_blank_and_solid_page_not_promoted_to_diagram")


# ── 3b. 阴性红线：纯文字翻拍页不得被升格成 diagram（本次收紧的证据）──────────

def test_text_only_page_not_promoted_to_diagram() -> None:
    """
    纯文字翻拍页**不得**凭"统一背景 + 彩度低 + OCR 多行"被升格成 diagram.

    第 6.5 步的"确有结构"**只认几何结构**（粗带 / 彩色描边块 / 细直线）。历史实现
    把 ``ocr_rows >= 2`` 也算作结构信号 —— 于是一张"白底黑字 + 纸灰阴影、无线无色带"
    的**纯文字页**满足"统一背景 + 彩度低 + OCR 多行"，被误升格成 diagram（走 vision
    拿"图意"）。而纯文字页的正解是 **OCR**（要文字），不是 vision —— 这类误升格既是
    语义错误，也白烧一次多模态推理。

    端到端（真跑像素 + stub OCR 行，不依赖真实 OCR 服务）：
    ``classify_image_safe`` 必须判 **photo**、reason 不是 ``structured-panels``。

    红证明：把 ``or ocr_rows >= 2`` 加回第 6.5 步，这条**必须失败**
    （该图会被判成 diagram / structured-panels）。
    """
    ocr_lines = [
        _TextLine(f"第{i + 1}行正文内容", (60.0, 120.0 + i * 70, 330.0, 148.0 + i * 70))
        for i in range(4)
    ]
    result = classify_image_safe(
        _text_only_page_photo(), ocr_lines=ocr_lines, filename="text-only-page",
    )

    assert result.image_type == IMAGE_TYPE_PHOTO, (
        f"纯文字翻拍页被判成 {result.image_type}"
        f"（reason={result.signals.get('reason')}）—— 它应走 OCR，不该被升格成 diagram"
    )
    assert result.signals.get("reason") != "structured-panels", (
        f"纯文字页命中了 structured-panels（reason={result.signals.get('reason')}）"
        "—— 第 6.5 步把 OCR 行数当成了结构信号"
    )
    assert resolve_route(result.image_type) == "ocr", (
        f"纯文字页路由应为 ocr，实际 {resolve_route(result.image_type)}"
    )
    print("  ok test_text_only_page_not_promoted_to_diagram")


# ── 4. 端到端因果链：diagram ⇒ resolve_route() == vision ──────────────────────

def test_diagram_route_is_vision_end_to_end() -> None:
    """
    「判成 diagram ⇒ 走多模态 ⇒ 拿得到 caption」这条因果链必须成立.

    这正是缺陷的本质：photo 走 OCR、拿不到图意总结；把它归到 diagram 之后，
    ``resolve_route`` 才会返回 vision，``_apply_output`` 才会写 ``vision_caption``
    （进而落成 payload ``image_caption``）。任一环断了，缺陷就换个形式复现。
    """
    from app.services.image_understanding.classifier import vision_analyze_types

    # diagram 必须在 Vision 白名单里（VISION_ANALYZE_TYPES 默认含 diagram）
    assert IMAGE_TYPE_DIAGRAM in vision_analyze_types(), sorted(vision_analyze_types())
    assert IMAGE_TYPE_PHOTO not in vision_analyze_types(), "photo 不应被塞进 Vision 白名单"
    assert resolve_route(IMAGE_TYPE_DIAGRAM) == "vision"

    # 端到端：合成示意图 → diagram → vision
    result = classify_image_safe(_colored_structure_diagram(), filename="colored-diagram")
    assert resolve_route(result.image_type) == "vision", (
        f"{result.image_type!r} 未路由到 vision，图意总结链断裂"
    )
    print("  ok test_diagram_route_is_vision_end_to_end")


if __name__ == "__main__":
    import traceback

    ok = fail = 0
    for _name, _fn in sorted(globals().items()):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        try:
            _fn()
            ok += 1
        except Exception:
            fail += 1
            print(f"  FAIL {_name}")
            traceback.print_exc()
    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)
