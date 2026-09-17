"""诊断：Vision 产出与门控拒绝原因（容器内运行）."""
import sys
from PIL import Image
from app.services.image_understanding import understand_image
from app.services.image_understanding.engines.vision_engine import VisionEngine
from app.services.image_understanding.imaging import encode_png_safe
from app.services.image_understanding.quality import verify_output
from app.services.image_understanding.confidence import gate

img = Image.open(sys.argv[1]).convert("RGB")
png = encode_png_safe(img)

v = VisionEngine()
print("vision available:", v.is_available())
out = v.process(img, png_bytes=png, image_type="diagram", role="primary")
print("ok:", out.ok, "conf:", out.confidence, "engine:", out.engine)
print("VISION TEXT:", repr(out.text))

quality = verify_output(out.text, "diagram", engine=out.engine, ocr_lines=[], ocr_text="output\nFan\ninput")
print("quality:", quality.to_dict())
verdict = gate(out, "diagram", quality)
print("gate:", verdict.accepted, verdict.confidence, verdict.reason)

res = understand_image(img, page_number=1, filename="flow.png", png_bytes=png)
print("result.meta:", res.meta)
print("result.decision:", res.decision, "engine:", res.analyze_engine)
print("result.vision_caption:", repr(res.vision_caption))
print("result.quality:", res.quality)
print("result.fusion:", res.fusion)
