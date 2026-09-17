"""标定回归用例的夹具与断言值（临时脚本）."""

from __future__ import annotations

from PIL import Image, ImageDraw

from app.services.image_understanding import classifier
import importlib

pp = importlib.import_module("app.services.image_understanding.preprocess")
from app.services.image_understanding.imaging import (
    background_is_dark,
    count_bands,
    ink_profiles,
    to_gray_pixels,
)


def mixed_polarity_flowchart():
    """复刻用户那张截图：白底页面上嵌一块深色流程图面板（混合极性）."""
    img = Image.new("RGB", (289, 313), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # 页眉注释（黑字白底）
    for i in range(11):
        d.line([(14 + i * 13, 6), (14 + i * 13, 20)], fill=(40, 40, 40), width=2)
    # 深色流程图面板
    d.rectangle([20, 34, 200, 285], fill=(10, 10, 10))
    # 三个节点框（浅色描边）
    for (x0, y0, x1, y1, c) in (
        (40, 62, 170, 102, (150, 150, 120)),
        (32, 150, 190, 195, (140, 140, 140)),
        (40, 248, 170, 288, (150, 150, 120)),
    ):
        d.rectangle([x0, y0, x1, y1], outline=c, width=3)
    # 两条带箭头的连线（浅色细线）
    d.line([(111, 103), (111, 149)], fill=(150, 150, 150), width=2)
    d.line([(111, 196), (111, 247)], fill=(150, 150, 150), width=2)
    return img


def dark_theme_screenshot():
    """深色主题铺满整幅图（真·深底浅字）."""
    img = Image.new("RGB", (320, 220), (18, 18, 20))
    d = ImageDraw.Draw(img)
    for row in range(8):
        for i in range(12):
            d.line([(16 + i * 22, 20 + row * 24), (16 + i * 22, 36 + row * 24)],
                   fill=(200, 200, 205), width=2)
    return img


def light_document():
    """白底黑字文档页."""
    img = Image.new("RGB", (320, 220), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for row in range(8):
        for i in range(12):
            d.line([(16 + i * 22, 20 + row * 24), (16 + i * 22, 36 + row * 24)],
                   fill=(30, 30, 30), width=2)
    return img


def dark_theme_text_page():
    """深底浅字（用于验证二值化不再把内容抹平）."""
    img = Image.new("RGB", (400, 180), (12, 12, 14))
    d = ImageDraw.Draw(img)
    for row in range(5):
        for i in range(10):
            d.line([(20 + i * 36, 20 + row * 30), (20 + i * 36, 40 + row * 30)],
                   fill=(230, 230, 235), width=4)
    return img


for name, img in (
    ("mixed_polarity_flowchart", mixed_polarity_flowchart()),
    ("dark_theme_screenshot", dark_theme_screenshot()),
    ("light_document", light_document()),
    ("dark_theme_text_page", dark_theme_text_page()),
):
    px, w, h = to_gray_pixels(img, 256)
    inv, src = background_is_dark(px, w, h)
    row_ink, col_ink, _ = ink_profiles(px, w, h, inverted=inv)
    cls = classifier.classify_image_safe(img, filename=name)
    print(f"\n### {name}  {img.size}")
    print(f"  background_is_dark = {inv}  ({src})")
    print(f"  h_lines={count_bands(row_ink, w, 0.55, 2)} v_lines={count_bands(col_ink, h, 0.55, 2)}")
    print(f"  classify -> {cls.image_type} conf={cls.confidence:.2f} reason={cls.signals.get('reason')}")
    print(f"  polarity = {pp.page_polarity(img)}")

print("\n### 二值化通道对比")
for name, img in (("dark_theme_text_page", dark_theme_text_page()),
                  ("mixed_polarity_flowchart", mixed_polarity_flowchart())):
    for mode in ("text", "scan"):
        r = pp.preprocess(img, mode=mode)
        ocr = r.ocr_input
        vals = list(ocr.convert("L").getdata())
        fg = sum(1 for v in vals if v < 128) / len(vals)
        print(f"  {name:26s} mode={mode:5s} applied={str(r.applied):40s} "
              f"ocr_image={'None' if r.ocr_image is None else 'set'} "
              f"mean={sum(vals)/len(vals):6.1f} foreground={fg*100:5.2f}%")
