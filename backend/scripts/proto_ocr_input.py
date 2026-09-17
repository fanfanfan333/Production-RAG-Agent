"""验证 ocr_input 通道的多角度/放大 OCR（容器内运行）."""

import sys

from PIL import Image

from app.services.image_understanding.engines.registry import get_registry
from app.services.image_understanding.preprocess import pick_mode, preprocess

img = Image.open(sys.argv[1]).convert("RGB")
mode, _ = pick_mode(img)
pre = preprocess(img, mode=mode)
print(f"mode={mode} applied={pre.applied} work={pre.image.size} ocr_input={pre.ocr_input.size}")

eng = get_registry().get("tesseract")
o = eng.process(pre.ocr_input)
print("main ocr_input:", repr((o.text or "").strip()), "lines:", len(o.lines or []))
for ln in (o.lines or []):
    print("   line:", repr(ln.text), ln.box, round(ln.confidence, 2))

W, H = pre.ocr_input.size
for scale in (2, 3, 4):
    big = pre.ocr_input.resize((W * scale, H * scale), Image.LANCZOS)
    for angle in (0, 90, 270):
        v = big if angle == 0 else big.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        out = eng.process(v)
        toks = [t for t in (out.text or "").split() if t.strip()]
        if toks:
            print(f"scale={scale}x @{angle:3d}°: {toks}")
