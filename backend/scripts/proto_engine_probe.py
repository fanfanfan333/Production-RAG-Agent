"""探测引擎可用性与实际产出（容器内运行）."""
import sys
from PIL import Image
from app.services.image_understanding.engines.registry import get_registry
from app.services.image_understanding.preprocess import pick_mode, preprocess

img = Image.open(sys.argv[1]).convert("RGB")
mode, _ = pick_mode(img)
pre = preprocess(img, mode=mode)
reg = get_registry()
for name in ("paddleocr", "tesseract"):
    e = reg.get(name)
    if e is None:
        print(name, "-> None"); continue
    print(name, "available:", e.is_available())
    if e.is_available():
        for label, im in (("work", pre.image), ("ocr_input", pre.ocr_input)):
            out = e.process(im)
            print(f"   {label}: ok={out.ok} conf={round(out.confidence,2)} text={(out.text or '').strip()!r}")
