"""诊断：深色主题截图在本管线各环节的判定结果（临时脚本）.

用法（容器内）：
    python scripts/diag_dark_shot.py /tmp/shot.png
"""

from __future__ import annotations

import sys

from PIL import Image

from app.services.image_understanding import classifier
import importlib

preprocess = importlib.import_module("app.services.image_understanding.preprocess")
from app.services.image_understanding.imaging import (
    INK_THRESHOLD,
    INVERTED_THRESHOLD,
    line_art_ratio,
    to_gray_pixels,
)
from app.services.image_understanding.structured_content import IMAGE_TYPE_LABELS

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shot.png"
img = Image.open(path)
print("=" * 74)
print(f"file        : {path}")
print(f"mode/size   : {img.mode} {img.size}")
w, h = img.size
print(f"short side  : {min(w, h)}  (MIN_SHORT_SIDE={preprocess.MIN_SHORT_SIDE})")

# ── 灰度分布 ─────────────────────────────────────────────────────────────────
gray = img.convert("L")
pixels = list(gray.getdata())
total = len(pixels)
hist = {}
for v in pixels:
    band = v // 32
    hist[band] = hist.get(band, 0) + 1
print("\n灰度直方图（每 32 灰阶一档, 0=最暗）:")
for band in range(8):
    n = hist.get(band, 0)
    bar = "#" * int(60 * n / total)
    label = f"{band*32:>3}-{band*32+31:>3}"
    print(f"  {label} | {n/total*100:6.2f}% {bar}")

dark = sum(1 for v in pixels if v < INK_THRESHOLD)
light = sum(1 for v in pixels if v > INVERTED_THRESHOLD)
print(f"\ndark(<{INK_THRESHOLD})  = {dark/total*100:6.2f}%")
print(f"light(>{INVERTED_THRESHOLD}) = {light/total*100:6.2f}%")
print(f"mid   = {(total-dark-light)/total*100:6.2f}%")
print(f"-> inverted(深底浅字)? {dark > total*0.5}")

lp, lw, lh = to_gray_pixels(img, 256)
print(f"\nline_art_ratio(256px) = {line_art_ratio(lp)}")

# ── 分类信号 ─────────────────────────────────────────────────────────────────
sig = classifier.compute_signals(img)
print("\n--- compute_signals ---")
for k in sorted(sig):
    print(f"  {k:22s} = {sig[k]}")

cls = classifier.classify_image_safe(img, filename=path)
print(f"\nclassify -> {cls.image_type} ({IMAGE_TYPE_LABELS.get(cls.image_type)})"
      f" conf={cls.confidence:.2f} engine={cls.engine}")
print(f"  reason = {cls.signals.get('reason')}")

# ── 预处理档位 ───────────────────────────────────────────────────────────────
mode, msig = preprocess.pick_mode(img, cls.image_type)
print(f"\npick_mode -> {mode}  reason={msig['mode_reason']}")
print(f"  noise_sigma  = {msig['noise_sigma']}")
print(f"  contrast_span= {msig['contrast_span']}")
print(f"  border_px    = {msig['border_px']}")
print(f"  thresholds   = {msig['thresholds']}")

res = preprocess.preprocess(img, mode=mode)
print(f"\npreprocess(applied) = {res.applied}")
print(f"  meta = {res.meta}")
if res.ocr_image is not None:
    print(f"  ocr_image size = {res.ocr_image.size}")
else:
    print("  ocr_image = None（本档位不做二值化，OCR 直接用彩色/灰度原图）")

# 各档位产出的 OCR 输入做对比：深色底在二值化后会发生什么
print("\n--- 各档位 ocr_input 的灰度均值（255=全白）---")
for m in ("light", "text", "geometric", "noisy", "scan"):
    r = preprocess.preprocess(img, mode=m)
    g = list(r.ocr_input.convert("L").getdata())
    inv = sum(1 for v in g if v > INVERTED_THRESHOLD) / len(g)
    print(f"  {m:10s} applied={str(r.applied):62s} mean={sum(g)/len(g):6.1f} light%={inv*100:5.1f}")
print("=" * 74)
