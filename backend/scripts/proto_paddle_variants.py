"""PaddleOCR 在放大/旋转/反色下的产出（容器内运行）."""
import sys
from PIL import Image
from app.services.image_understanding.engines.registry import get_registry
from app.services.image_understanding.preprocess import invert_for_ocr, page_polarity, pick_mode, preprocess

img = Image.open(sys.argv[1]).convert("RGB")
mode, _ = pick_mode(img)
pre = preprocess(img, mode=mode)
base = pre.image
pol = page_polarity(base)
print("polarity:", pol)
eng = get_registry().get("paddleocr")

variants = {"base": base}
if pol != "light":
    variants["inverted"] = invert_for_ocr(base)
W, H = base.size
for s in (2, 3, 4):
    variants[f"base_{s}x"] = base.resize((W*s, H*s), Image.LANCZOS)
    if pol != "light":
        variants[f"inv_{s}x"] = invert_for_ocr(base).resize((W*s, H*s), Image.LANCZOS)

for name, im in variants.items():
    out = eng.process(im)
    print(f"{name:10s}: {(out.text or '').strip()!r}")
    for ang in (90, 270):
        r = im.rotate(ang, expand=True, fillcolor=(255, 255, 255))
        o2 = eng.process(r)
        t = (o2.text or '').strip()
        if t:
            print(f"   rot{ang}: {t!r}")
