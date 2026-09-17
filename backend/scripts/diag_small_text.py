"""量一下箭头上的极小标注到底能不能读出来（放大倍数 × 引擎）."""

from __future__ import annotations

from PIL import Image

from app.services.image_understanding.engines.registry import get_registry

img = Image.open("/tmp/shot.png").convert("RGB")
w, h = img.size
print(f"原图 {w}x{h}")

# 裁出中间那条"箭头走廊"（两个标注都在这条带里）
crop = img.crop((int(w * 0.33), int(h * 0.08), int(w * 0.62), int(h * 0.92)))
crop.save("/tmp/crop_arrows_1x.png")
cw, ch = crop.size
print(f"箭头走廊裁切 = {cw}x{ch}")

reg = get_registry()
for scale in (1, 2, 4, 8, 12):
    if scale == 1:
        im = crop
    else:
        im = crop.resize((cw * scale, ch * scale), Image.LANCZOS)
    if scale in (4, 12):
        im.save(f"/tmp/crop_arrows_{scale}x.png")
    row = [f"{scale:>2}x mean_in_ocr={sum(im.convert('L').getdata())/ (im.width*im.height):6.1f}"]
    for name in ("paddleocr", "tesseract"):
        e = reg.get(name)
        if e is None or not e.is_available():
            continue
        out = e.process(im)
        txt = (out.text or "").strip().replace("\n", " | ")
        row.append(f"{name}: conf={out.confidence:.2f} {txt[:60]!r}")
    print("  ".join(row))

# 标注字高估计：在走廊里找"墨"（亮像素）连通行
import numpy as np

g = np.asarray(crop.convert("L"))
ink = g > 120
rows = ink.sum(axis=1)
groups, start = [], None
for y, v in enumerate(rows):
    if v > 0 and start is None:
        start = y
    elif v == 0 and start is not None:
        groups.append((start, y - 1, y - start))
        start = None
if start is not None:
    groups.append((start, len(rows) - 1, len(rows) - start))
print("\n走廊内每段'有墨'区域的高度（像素）:")
for y0, y1, hh in groups:
    print(f"  y={y0:>3}..{y1:<3} 高={hh}px")
print("\n整幅图尺寸:", (w, h), "—— 其中高度 <8px 的文字段即为亚可读字号")
