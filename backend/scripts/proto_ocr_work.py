"""用管线 preprocess 的产物测多角度 OCR（容器内运行）."""

import sys

from PIL import Image

from app.services.image_understanding.engines.registry import get_registry
from app.services.image_understanding.preprocess import pick_mode, preprocess

img = Image.open(sys.argv[1]).convert("RGB")
mode, _sig = pick_mode(img, "diagram")
pp = preprocess(img, mode=mode)
print("mode:", mode)
work = pp.image
print("preprocess:", pp.applied, "work size:", work.size)

eng = get_registry().get("tesseract")
for angle in (0, 90, 270):
    im = work if angle == 0 else work.rotate(angle, expand=True)
    out = eng.process(im)
    toks = [t for t in (out.text or "").split() if t.strip()]
    print(f"ocr @{angle:3d}°: {toks}")
    if angle == 0 and out.lines:
        for ln in out.lines:
            print("   line:", ln.text, ln.box, round(ln.confidence, 2))
