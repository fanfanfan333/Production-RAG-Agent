"""对比 tesseract 在不同形态下的输出（容器内运行）."""

import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.services.image_understanding.engines.registry import get_registry
from app.services.image_understanding.preprocess import invert_for_ocr, page_polarity

img = Image.open(sys.argv[1]).convert("RGB")
W, H = img.size
print("polarity:", page_polarity(img))
inv = invert_for_ocr(img)
eng = get_registry().get("tesseract")

variants = {
    "orig 1x": img,
    "invert_for_ocr 1x": inv,
    "invert 2x": inv.resize((W * 2, H * 2), Image.LANCZOS),
    "invert 3x": inv.resize((W * 3, H * 3), Image.LANCZOS),
    "invert 4x": inv.resize((W * 4, H * 4), Image.LANCZOS),
}
for name, im in variants.items():
    out = eng.process(im)
    print(f"{name:22s} ok={out.ok} text={(out.text or '').strip()!r}")

# 旋转 + 放大
for angle in (90, 270):
    im = inv.resize((W * 3, H * 3), Image.LANCZOS).rotate(
        angle, expand=True, fillcolor=(255, 255, 255))
    out = eng.process(im)
    print(f"invert 3x rot{angle}: {(out.text or '').strip()!r}")
