"""定位误判来源：h_lines / v_lines 到底落在图像的哪些位置（临时脚本）."""

from __future__ import annotations

import sys

from PIL import Image

from app.services.image_understanding.imaging import (
    INK_THRESHOLD,
    INVERTED_THRESHOLD,
    bands,
    ink_profiles,
    to_gray_pixels,
)

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shot.png"
img = Image.open(path).convert("RGB")

pixels, w, h = to_gray_pixels(img, 256)
row_ink, col_ink, inverted = ink_profiles(pixels, w, h)
print(f"size={w}x{h} inverted={inverted}")

# 逐行灰度均值，用来肉眼定位内容分区
gray = img.convert("L").resize((w, h), Image.BILINEAR)
rows_mean = []
for y in range(h):
    band = list(gray.crop((0, y, w, y + 1)).getdata())
    rows_mean.append(sum(band) / len(band))

print("\n--- 行剖面（每 8 行采样）---")
print(" y   mean(0-255)  row_ink  ink%    bar")
for y in range(0, h, 8):
    m = rows_mean[y]
    ri = row_ink[y]
    bar = "#" * int(50 * ri / w)
    print(f"{y:4d} {m:9.1f} {ri:9d} {ri/w*100:6.1f}% {bar}")

hb = bands(row_ink, w, 0.55, 2)
vb = bands(col_ink, h, 0.55, 2)
tb = bands(row_ink, w, 0.10, 2)
print(f"\nh_lines(ratio=0.55) = {len(hb)} -> y 区间 {hb}")
print(f"v_lines(ratio=0.55) = {len(vb)} -> x 区间 {vb}")
print(f"text_bands(ratio=0.10) = {len(tb)} -> {tb}")

# 这 2 条竖线所在列的灰度剖面
print("\n--- 竖线所在列的灰度均值 ---")
for x0, x1 in vb:
    xc = (x0 + x1) // 2
    col = list(gray.crop((xc, 0, xc + 1, h)).getdata())
    print(f"  x={xc} mean={sum(col)/len(col):.1f}  ink={col_ink[xc]}/{h} ({col_ink[xc]/h*100:.1f}%)")

# 仅用"四边背景"估极性的对比
px = list(img.convert("L").getdata())
W, H = img.size
edge = []
for y in range(H):
    for x in range(W):
        if x < 3 or x >= W - 3 or y < 3 or y >= H - 3:
            edge.append(px[y * W + x])
print(f"\n全图 dark 占比      = {sum(1 for v in px if v < INK_THRESHOLD)/len(px)*100:.1f}%"
      f"  -> inverted={sum(1 for v in px if v < INK_THRESHOLD) > len(px)*0.5}")
print(f"四边背景 dark 占比  = {sum(1 for v in edge if v < INK_THRESHOLD)/len(edge)*100:.1f}%"
      f"  -> inverted={sum(1 for v in edge if v < INK_THRESHOLD) > len(edge)*0.5}")
print(f"四边背景 mean       = {sum(edge)/len(edge):.1f}")
