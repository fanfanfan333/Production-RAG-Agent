"""原型：深色流程图的结构检测 + 多角度 OCR（容器内运行，验证可行性）."""

import sys

import cv2
import numpy as np
from PIL import Image

from app.services.image_understanding.engines.registry import get_registry
from app.services.image_understanding.imaging import background_is_dark, to_gray_pixels

img = Image.open(sys.argv[1]).convert("RGB")
W, H = img.size
gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

px, w, h = to_gray_pixels(img, 256)
dark = background_is_dark(px, w, h)
print(f"size={W}x{H} dark_bg={dark}")

norm = cv2.bitwise_not(gray) if dark else gray
# 前景 = 深（亮底深字形态下）
binary = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)

# ── 节点框：轮廓 → 多边形逼近 → 矩形/圆角矩形 ──────────────────────────────
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img_area = W * H
nodes = []
for c in contours:
    area = cv2.contourArea(c)
    if area < img_area * 0.01 or area > img_area * 0.9:
        continue
    x, y, bw, bh = cv2.boundingRect(c)
    extent = area / float(bw * bh)
    aspect = bw / max(bh, 1)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if extent > 0.5 and 0.2 < aspect < 5.0:
        nodes.append((x, y, bw, bh, len(approx)))
print(f"node boxes: {len(nodes)}")
for n in nodes:
    print("  box:", n)

# ── 连接线：去掉节点区域后，细长形态学成分 ──────────────────────────────────
mask = binary.copy()
for (x, y, bw, bh, _v) in nodes:
    cv2.rectangle(mask, (x, y), (x + bw, y + bh), 0, -1)
horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, W // 12), 1)))
vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, H // 12))))
n_h, _ = cv2.connectedComponents(horiz)
n_v, _ = cv2.connectedComponents(vert)
print(f"connectors: horiz={n_h - 1} vert={n_v - 1}")

# ── 多角度 OCR：0/90/270 抓竖排标注 ─────────────────────────────────────────
eng = get_registry().get("tesseract")

# 极性归一 + 3x 放大（箭头旁 6px 小字不放大读不出来）
base = img
if dark:
    base = Image.fromarray(cv2.bitwise_not(np.array(img)))
scale = 3
base = base.resize((W * scale, H * scale), Image.LANCZOS)

for angle in (0, 90, 270):
    im = base if angle == 0 else base.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    out = eng.process(im)
    toks = [t for t in (out.text or "").split() if t.strip()]
    print(f"ocr @{angle:3d}°: {toks}")
