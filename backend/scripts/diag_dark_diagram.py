"""
诊断：深色底流程示意图为什么提取失败.

逐层打印信号 —— 分类 → 预处理 → OCR → 置信度评估，定位断点。
用法：docker exec -u root -e HOME=/tmp rag_backend sh -c "cd /app && python /tmp/diag_dark_diagram.py <png路径>"
"""

from __future__ import annotations

import sys

from PIL import Image


def main(path: str) -> int:
    img = Image.open(path)
    print(f"=== 原图 ===")
    print(f"  size={img.size}  mode={img.mode}")

    # ── 1. 分类信号 ────────────────────────────────────────────────────────
    from app.services.image_understanding.classifier import (
        classify_image,
        compute_signals,
    )

    signals = compute_signals(img)
    print("\n=== 分类信号 ===")
    for k, v in signals.items():
        print(f"  {k:22s} = {v}")

    cls = classify_image(img, filename=path)
    print(f"\n=== 分类结果 ===")
    print(f"  image_type = {cls.image_type}  conf={cls.confidence:.2f}  reason={cls.signals.get('reason')}")

    # 该类型是否需要走 Vision
    from app.services.image_understanding.classifier import vision_analyze_types

    print(f"  VISION_ANALYZE_TYPES = {vision_analyze_types()}")
    print(f"  将走 Vision? {cls.image_type in vision_analyze_types()}")

    # ── 2. 预处理 ──────────────────────────────────────────────────────────
    from app.services.image_understanding import preprocess

    fn = getattr(preprocess, "preprocess_image", None) or getattr(preprocess, "preprocess", None)
    print(f"\n=== 预处理 ===")
    pp = None
    if fn is None:
        print("  未找到预处理入口，列出模块函数：")
        print("  ", [n for n in dir(preprocess) if not n.startswith('_')])
    else:
        try:
            pp = fn(img)
            print(f"  applied = {pp.applied}")
            print(f"  meta    = {pp.meta}")
            print(f"  ocr_input size = {pp.ocr_input.size}")
        except Exception as exc:  # noqa: BLE001
            print(f"  预处理抛异常: {type(exc).__name__}: {exc}")

    # ── 3. OCR（对原图 vs 预处理后）─────────────────────────────────────────
    from app.services.ocr import get_ocr_service

    svc = get_ocr_service()
    print(f"\n=== OCR ===")

    def run_ocr(label: str, im):
        try:
            text, engine, lines = svc.extract_text_with_lines(im)
            print(f"  [{label}] engine={engine} lines={len(lines)}")
            print(f"  [{label}] text={text!r}")
            for ln in lines[:12]:
                print(f"      box={ln.box} conf={ln.confidence:.2f} text={ln.text!r}")
            return lines
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] OCR 抛异常: {type(exc).__name__}: {exc}")
            return []

    lines_raw = run_ocr("原图", img.convert("RGB"))
    if pp is not None:
        lines_pp = run_ocr("预处理", pp.ocr_input.convert("RGB"))
    else:
        lines_pp = []

    # ── 4. 置信度评估 ──────────────────────────────────────────────────────
    from app.services.image_understanding.quality import assess_ocr_confidence

    print(f"\n=== 置信度评估 ===")
    for label, ls in (("原图", lines_raw), ("预处理", lines_pp)):
        rep = assess_ocr_confidence(ls)
        print(f"  [{label}] lines={rep.lines} mean={rep.mean} min={rep.minimum} "
              f"low_ratio={rep.low_ratio} reported={rep.reported} passed={rep.passed}")
        print(f"  [{label}] reasons={rep.reasons}")

    # ── 5. 完整管线 ────────────────────────────────────────────────────────
    print(f"\n=== 完整理解管线 ===")
    try:
        from app.services.image_understanding import understand_image  # type: ignore

        res = understand_image(img, filename=path)
        print(f"  result type = {type(res).__name__}")
        print(f"  {res}")
    except Exception as exc:  # noqa: BLE001
        print(f"  管线入口调用失败: {type(exc).__name__}: {exc}")
        import app.services.image_understanding as iu

        print("  可用入口：", [n for n in dir(iu) if not n.startswith('_')])

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dark_diagram.png"))
