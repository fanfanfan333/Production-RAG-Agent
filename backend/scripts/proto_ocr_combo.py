"""最终组合验证：work + 反色 + 放大 + 旋转 OCR（容器内运行）."""

import sys

from PIL import Image

from app.services.image_understanding.engines.registry import get_registry
from app.services.image_understanding.preprocess import (
    invert_for_ocr,
    page_polarity,
    pick_mode,
    preprocess,
)

img = Image.open(sys.argv[1]).convert("RGB")
mode, _ = pick_mode(img, "diagram")
work = preprocess(img, mode=mode).image
pol = page_polarity(work)
base = invert_for_ocr(work) if pol != "light" else work
W, H = base.size
print(f"mode={mode} work={work.size} polarity={pol}")

eng = get_registry().get("tesseract")

for scale in (1, 2, 3):
    im = base.resize((W * scale, H * scale), Image.LANCZOS) if scale > 1 else base
    for angle in (0, 90, 270):
        v = im if angle == 0 else im.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        out = eng.process(v)
        toks = [t for t in (out.text or "").split() if t.strip()]
        print(f"scale={scale}x @{angle:3d}°: {toks}")
