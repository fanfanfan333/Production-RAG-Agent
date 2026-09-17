"""端到端诊断：整条 understand_image 管线在这张深色截图上的真实产出（临时脚本）."""

from __future__ import annotations

import json
import sys

from PIL import Image

from app.config import get_settings
from app.services.image_understanding.engines.registry import get_registry
from app.services.image_understanding.pipeline import understand_image

s = get_settings()
print("=" * 74)
print("引擎可用性")
for name in ("paddleocr", "tesseract", "paddleocr-formula", "table-transformer", "ocr+code-parser"):
    e = get_registry().get(name)
    print(f"  {name:22s} exists={e is not None} available={e.is_available() if e else '-'}")

from app.services.image_understanding.engines.vision_engine import VisionEngine

v = VisionEngine()
print(f"  vision                 available={v.is_available()}")

for k in (
    "VISION_ANALYZE_TYPES", "IMAGE_DUAL_CHANNEL_TYPES", "IMAGE_CLASSIFIER_ENGINE",
    "IMAGE_CLASSIFICATION_ENABLED", "ENABLE_IMAGE_OCR", "VISION_ENGINE",
    "VISION_MODEL", "OLLAMA_BASE_URL", "MAX_IMAGE_OCR_CHARS",
):
    print(f"  settings.{k:26s} = {getattr(s, k, '<缺失>')}")

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shot.png"
img = Image.open(path).convert("RGB")

r = understand_image(img, page_number=1, filename=path)
print("\n" + "=" * 74)
print("understand_image 结果")
print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))

print("\n--- 最终落库内容 ---")
print("structured_content =", repr(r.structured_content)[:600])
print("vision_caption     =", repr(r.vision_caption)[:600])
print("ocr_text           =", repr(r.ocr_text)[:600])
if r.table is not None:
    print("\n--- TableStructure ---")
    print(json.dumps(r.table.to_dict(), ensure_ascii=False, indent=2)[:2500])
print("=" * 74)
