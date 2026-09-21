# -*- coding: utf-8 -*-
"""
生成「文档照片里的表格」合成夹具（QA 独立验证用）.

为什么需要这个夹具
------------------

线上知识库里真实存在的图片素材（``/app/uploads/*/images/page_*.png``）经逐张
目视确认**全部是图表/流程图/示意图，没有一张是"表格照片"**。因此要验证
「文档照片里的表格表头会不会被还原」，必须自造一张可复现的合成图。

本脚本产出一组**受控**的表格图片：同一张表格，分别施加不同程度的"翻拍退化"，
再分别单独改变某一个变量，从而把"到底哪一步把表头弄丢"定位到具体机制。

夹具清单（每个都写进 ``groundtruth.json``，含表头行原文）
--------------------------------------------------------

    1. t0_clean_fullpage    表格几乎铺满整页（框线跨 ~85% 画幅）—— 正对照
    2. t1_clean_titled      同表放进"有标题 + 大页边距"的版面（表格跨 ~45% 画幅）
    3. t2_photo_jpeg        在 t1 基础上做 JPEG 低质量压缩 + 轻模糊（手机出图）
    4. t3_photo_shadow      再叠加不均匀光照（纸面阴影）
    5. t4_photo_perspective 再叠加轻微透视（翻拍梯形）
    6. t5_skew5             干净表旋转 5°
    7. t6_skew15            干净表旋转 15°（反证：倾斜过大）
    8. t7_hlines_only       只有横向框线、没有竖向框线（反证：分类判据）
    9. t8_header_filled     表头行单独填色（反证：表头与数据行异色）

用法
----

    # 容器内（推荐：管线依赖都在容器里）
    docker cp make_doc_photo_fixture.py rag_backend:/tmp/
    docker exec -w /app -e PYTHONPATH=/app rag_backend \
        python /tmp/make_doc_photo_fixture.py --out /tmp/fixtures --font /tmp/fonts/msyh.ttc

    # 没有 CJK 字体时自动退回 ASCII 表头（仍可复现，只是不还原中文）

字体按 ``--font`` → 候选列表 的顺序查找；全部找不到时用 PIL 默认位图字体并
把表头降级为 ASCII（写进 groundtruth.json 的 ``font_mode``）。
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# ── 默认候选字体（Windows 宿主机 → 常见 Linux → 兜底 DejaVu）──────────────────
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# CJK 字体文件名特征（用于判断能否渲染中文表头）
_CJK_HINTS = ("msyh", "simhei", "simsun", "noto", "wqy", "uming", "ukai", "cjk")

# ── 表格内容（表头行 = groundtruth 的断言对象）────────────────────────────────
_HEADER_CN = ["指标", "2023年", "2024年", "同比"]
_ROWS_CN = [
    ["营业收入", "30240", "33500", "+10.8%"],
    ["净利润", "2180", "2460", "+12.8%"],
    ["毛利率", "24.1%", "25.0%", "+0.9pp"],
    ["研发投入", "1520", "1810", "+19.1%"],
]
_HEADER_ASCII = ["Metric", "2023", "2024", "YoY"]
_ROWS_ASCII = [
    ["Revenue", "30240", "33500", "+10.8%"],
    ["NetProfit", "2180", "2460", "+12.8%"],
    ["GrossMargin", "24.1%", "25.0%", "+0.9pp"],
    ["RandD", "1520", "1810", "+19.1%"],
]

_LINE = (25, 25, 25)          # 框线颜色（近黑）
_HEADER_FILL = (222, 231, 246)  # 表头填色（浅蓝，与数据行异色）
_PAGE_BG = (255, 255, 255)


# ─────────────────────────────────────────────────────────────────────────────
# 字体
# ─────────────────────────────────────────────────────────────────────────────


def resolve_font_path(explicit: str | None) -> str | None:
    """按 ``explicit`` → 候选列表 查找第一个存在的字体文件."""
    if explicit and Path(explicit).exists():
        return explicit
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def load_font(size: int, font_path: str | None) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:  # noqa: BLE001
            pass
    try:
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def _is_cjk_capable(font_path: str | None) -> bool:
    if not font_path:
        return False
    name = Path(font_path).name.lower()
    return any(hint in name for hint in _CJK_HINTS)


def _center(draw: ImageDraw.ImageDraw, box, text, font, fill=(17, 17, 17)) -> None:
    """把 *text* 居中画进 *box*（box = (x0, y0, x1, y1)）."""
    x0, y0, x1, y1 = box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        (x0 + (x1 - x0 - (right - left)) / 2 - left,
         y0 + (y1 - y0 - (bottom - top)) / 2 - top),
        text, font=font, fill=fill,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 表格绘制
# ─────────────────────────────────────────────────────────────────────────────


def _draw_table(
    img: Image.Image,
    box: tuple[int, int, int, int],
    header: list[str],
    rows: list[list[str]],
    *,
    font_path: str | None,
    draw_vertical: bool = True,
    header_fill: tuple[int, int, int] | None = None,
    title: str = "",
    title_font_path: str | None = None,
) -> None:
    """在 *box* 内画一张带表头的表格（首行即表头行）."""
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    ncols = len(header)
    nrows = len(rows) + 1

    # 首列宽一些（中文指标名较长），其余等宽
    weights = [2.2] + [1.0] * (ncols - 1)
    total = sum(weights)
    edges = [x0]
    acc = 0.0
    for weight in weights:
        acc += weight
        edges.append(x0 + (x1 - x0) * acc / total)

    row_h = (y1 - y0) / nrows
    row_edges = [y0 + i * row_h for i in range(nrows + 1)]

    cell_font = load_font(max(14, int(row_h * 0.42)), font_path)

    if title:
        title_font = load_font(max(18, int(row_h * 0.55)), title_font_path or font_path)
        _center(draw, (x0, y0 - int(row_h * 0.95), x1, y0), title, title_font, (30, 30, 30))

    # 表头底色（与数据行异色，用于反证"表头靠底色区分"）
    if header_fill is not None:
        draw.rectangle([x0, y0, x1, row_edges[1]], fill=header_fill)

    # 横线
    for y in row_edges:
        draw.line([x0, y, x1, y], fill=_LINE, width=2)
    # 竖线
    if draw_vertical:
        for x in edges:
            draw.line([x, y0, x, y1], fill=_LINE, width=2)

    # 单元格文字
    all_rows = [header] + rows
    for r, row in enumerate(all_rows):
        for c, text in enumerate(row):
            _center(
                draw,
                (edges[c] + 4, row_edges[r], edges[c + 1] - 4, row_edges[r + 1]),
                text, cell_font,
            )


def render_clean_fullpage(header, rows, font_path) -> Image.Image:
    """正对照：表格几乎铺满整页（框线跨 ~85% 画幅）—— 期望被判成 table."""
    img = Image.new("RGB", (1000, 760), _PAGE_BG)
    _draw_table(img, (70, 60, 930, 700), header, rows, font_path=font_path)
    return img


def render_clean_titled(header, rows, font_path) -> Image.Image:
    """常见版面：有标题 + 大页边距，表格只占 ~45% 画幅高度."""
    img = Image.new("RGB", (1000, 900), _PAGE_BG)
    _draw_table(
        img, (170, 330, 830, 620), header, rows,
        font_path=font_path, title="表 3　2024 年度核心经营指标",
    )
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 翻拍退化（每一步都可单独开关，便于把变量隔离开）
# ─────────────────────────────────────────────────────────────────────────────


def degrade_jpeg(img: Image.Image, quality: int = 32) -> Image.Image:
    """JPEG 低质量压缩：制造块效应与色度损失（手机直出/微信转发典型）."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_blur(img: Image.Image, radius: float = 1.2) -> Image.Image:
    """轻微高斯模糊：模拟对焦不实/手抖."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def degrade_shadow(img: Image.Image) -> Image.Image:
    """不均匀光照：左亮右暗的线性渐变相乘（模拟纸面阴影/顶光）."""
    width, height = img.size
    grad = Image.new("L", (width, height))
    gd = ImageDraw.Draw(grad)
    for x in range(width):
        # 235 → 155 的横向衰减（越往右越暗）
        value = int(235 - 80 * (x / max(width - 1, 1)))
        gd.line([x, 0, x, height], fill=value, width=1)
    shade = Image.merge("RGB", (grad, grad, grad))
    # 逐通道相乘 = 把纸面按渐变压暗（0-255 × 0-255 / 255）
    return ImageChops.multiply(img, shade)


def _find_perspective_coeffs(dst_quad, src_quad):
    """解出把 dst→src 的透视系数（PIL Image.transform 约定）."""

    def _solve(mat):
        # 8x8 线性方程，用高斯消元（避免引入 numpy 依赖）
        n = 8
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
            mat[col], mat[pivot] = mat[pivot], mat[col]
            pv = mat[col][col] or 1e-12
            for r in range(col + 1, n):
                factor = mat[r][col] / pv
                if factor:
                    for c in range(col, n + 1):
                        mat[r][c] -= factor * mat[col][c]
        out = [0.0] * n
        for r in range(n - 1, -1, -1):
            acc = mat[r][n] - sum(mat[r][c] * out[c] for c in range(r + 1, n))
            out[r] = acc / (mat[r][r] or 1e-12)
        return out

    dx0, dy0 = dst_quad[0]
    dx1, dy1 = dst_quad[1]
    dx2, dy2 = dst_quad[2]
    dx3, dy3 = dst_quad[3]
    sx0, sy0 = src_quad[0]
    sx1, sy1 = src_quad[1]
    sx2, sy2 = src_quad[2]
    sx3, sy3 = src_quad[3]

    matrix = []
    for (dx, dy, sx, sy) in (
        (dx0, dy0, sx0, sy0), (dx1, dy1, sx1, sy1),
        (dx2, dy2, sx2, sy2), (dx3, dy3, sx3, sy3),
    ):
        matrix.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy, sx])
        matrix.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy, sy])
    return _solve(matrix)


def degrade_perspective(img: Image.Image, amount: float = 0.06) -> Image.Image:
    """轻微透视（翻拍梯形）：上边略窄、下边略宽."""
    width, height = img.size
    inset = width * amount
    src = [(0, 0), (width, 0), (width, height), (0, height)]
    dst = [(inset, 0), (width - inset, 0), (width, height), (0, height)]
    coeffs = _find_perspective_coeffs(src, dst)
    return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def degrade_rotate(img: Image.Image, degrees: float) -> Image.Image:
    """旋转（白底填充），模拟扫描/翻拍歪斜."""
    return img.rotate(degrees, resample=Image.BICUBIC, expand=True, fillcolor=_PAGE_BG)


# ─────────────────────────────────────────────────────────────────────────────
# 组装
# ─────────────────────────────────────────────────────────────────────────────


def build_fixtures(out_dir: Path, font_path: str | None) -> list[dict]:
    """产出全部夹具，返回 groundtruth 记录列表."""
    if _is_cjk_capable(font_path):
        header, rows, font_mode = _HEADER_CN, _ROWS_CN, "cjk"
    else:
        header, rows, font_mode = _HEADER_ASCII, _ROWS_ASCII, "ascii"

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    def emit(name: str, image: Image.Image, *, variant: str, degrade: list[str]) -> None:
        path = out_dir / f"{name}.png"
        image.convert("RGB").save(path, format="PNG")
        records.append({
            "name": name,
            "file": str(path),
            "variant": variant,
            "degrade": degrade,
            "header": list(header),
            "rows": [list(r) for r in rows],
            "size": list(image.size),
            "font_mode": font_mode,
        })

    # 1) 正对照：表格铺满整页
    emit("t0_clean_fullpage", render_clean_fullpage(header, rows, font_path),
         variant="clean-fullpage", degrade=[])
    # 2) 干净但有标题/页边距（表格只占 ~45% 画幅高）
    titled = render_clean_titled(header, rows, font_path)
    emit("t1_clean_titled", titled, variant="clean-titled", degrade=[])
    # 3) + JPEG 压缩 + 模糊
    jpeg = degrade_blur(degrade_jpeg(titled), 1.0)
    emit("t2_photo_jpeg", jpeg, variant="photo-jpeg", degrade=["jpeg", "blur"])
    # 4) + 阴影
    shadow = degrade_shadow(jpeg)
    emit("t3_photo_shadow", shadow, variant="photo-shadow", degrade=["jpeg", "blur", "shadow"])
    # 5) + 透视
    persp = degrade_perspective(shadow, amount=0.06)
    emit("t4_photo_perspective", persp,
         variant="photo-perspective", degrade=["jpeg", "blur", "shadow", "perspective"])
    # 6/7) 倾斜
    emit("t5_skew5", degrade_rotate(titled, 5.0), variant="skew5", degrade=["rotate5"])
    emit("t6_skew15", degrade_rotate(titled, 15.0), variant="skew15", degrade=["rotate15"])
    # 8) 只有横向框线
    hl = Image.new("RGB", (1000, 900), _PAGE_BG)
    _draw_table(hl, (170, 330, 830, 620), header, rows,
                font_path=font_path, draw_vertical=False,
                title="表 4　只有横线的表格")
    emit("t7_hlines_only", hl, variant="hlines-only", degrade=["no-vertical-rules"])
    # 9) 表头行单独填色
    hf = Image.new("RGB", (1000, 900), _PAGE_BG)
    _draw_table(hf, (170, 330, 830, 620), header, rows,
                font_path=font_path, header_fill=_HEADER_FILL,
                title="表 5　表头带底色的表格")
    emit("t8_header_filled", hf, variant="header-filled", degrade=["header-fill"])

    gt = out_dir / "groundtruth.json"
    gt.write_text(json.dumps({
        "generated_by": "make_doc_photo_fixture.py",
        "font_path": font_path,
        "font_mode": font_mode,
        "note": "header 为表头行原文（断言 header_present 的基准）",
        "fixtures": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="生成文档照片表格夹具")
    parser.add_argument("--out", default="/tmp/fixtures", help="输出目录")
    parser.add_argument("--font", default=None, help="显式字体文件路径")
    args = parser.parse_args()

    font_path = resolve_font_path(args.font)
    records = build_fixtures(Path(args.out), font_path)
    print(f"输出目录：{args.out}")
    print(f"字体：{font_path}（mode={'cjk' if _is_cjk_capable(font_path) else 'ascii'}）")
    for rec in records:
        print(f"  {rec['name']:<24} {rec['size']}  header={rec['header']}")


if __name__ == "__main__":
    main()
