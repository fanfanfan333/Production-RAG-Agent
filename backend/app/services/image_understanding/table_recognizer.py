"""
Table Structure Recognition（表格结构识别）.

设计稿第二层：

    Table Image → Table Parser → 结构化表格

输入是一张被判为"表格"的图片，输出是**可直接入库的 Markdown 表格**：

    | 指标 | 数值 |
    |---|---|
    | 准确率 |95%|
    | 召回率 |92%|

这正是设计稿里 ``{"type":"table","page":1,"content":"| 指标 | 数值 |..."}``
的 content 部分。为什么一定要还原成表格而不是"把 OCR 文本拼起来"？
    · 表格的语义在**单元格与行列关系**里，"准确率 95% 召回率 92%" 这样的
      平铺文本丢掉了"哪个值属于哪个指标"，LLM 极容易读错；
    · Markdown 表格既能被 BM25 按关键词召回，也能被向量检索按语义召回；
    · 同一个 chunker 已经会识别 Markdown 表格并标 content_type="table"，
      因此**图片里的表格和正文里的表格在检索层完全同构**。

还原策略（从最可靠到最兜底）：
    1. 框线法     —— 检测图片中的横/竖 ruling line，用线的位置切行列网格；
    2. OCR 对齐法 —— 无边框表格：按文字行的 y 聚行、x 位置聚列；
    3. 文本兜底   —— 连坐标都没有时，用多空格/制表符切分猜列。

任一步失败都返回 ``ok=False``，由调用方退回"OCR 纯文本"，绝不产出半成品表格。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from app.config import get_settings
from app.services.image_understanding.imaging import (
    bands,
    ink_profiles,
    to_gray_pixels,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 线条检测的分析分辨率（长边），兼顾精度与纯 Python 遍历的开销
_ANALYZE_MAX_SIDE = 512
# 一条"线"占横/纵向的比例阈值（比分类器宽松，因为表可能只占画面一部分）
_RULE_RATIO = 0.35
# 允许的断裂像素，用来把虚线/抗锯齿合并成一条线
_RULE_MAX_GAP = 2


@dataclass
class TableStructure:
    """表格结构识别的结果."""

    markdown: str = ""
    rows: int = 0
    cols: int = 0
    header: list[str] = field(default_factory=list)
    engine: str = "table_parser"
    method: str = "none"          # rules | ocr-alignment | text-split | failed
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.markdown.strip()) and self.rows >= 1 and self.cols >= 1

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "method": self.method,
            "engine": self.engine,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 框线检测
# ─────────────────────────────────────────────────────────────────────────────


def detect_rules(img: Image.Image) -> tuple[list[float], list[float]]:
    """
    检测横 / 竖框线的中心坐标（**原图像素**坐标系）.

    返回 ``(row_positions, col_positions)``：
        row_positions —— 每条横线的 y 中心（表格的行分隔）
        col_positions —— 每条竖线的 x 中心（表格的列分隔）

    墨迹投影走 imaging 模块，因此**深色底表格**（深色主题截图里的表）也能
    正确检测：投影函数会先判断底色极性再决定哪一侧算墨。
    """
    pixels, sw, sh = to_gray_pixels(img, _ANALYZE_MAX_SIDE)
    row_ink, col_ink, _inverted = ink_profiles(pixels, sw, sh)

    width, height = img.size
    sx = width / float(sw)
    sy = height / float(sh)

    row_positions = [
        (start + end) / 2.0 * sy + sy / 2.0
        for start, end in bands(row_ink, sw, _RULE_RATIO, _RULE_MAX_GAP)
    ]
    col_positions = [
        (start + end) / 2.0 * sx + sx / 2.0
        for start, end in bands(col_ink, sh, _RULE_RATIO, _RULE_MAX_GAP)
    ]
    return row_positions, col_positions


# ─────────────────────────────────────────────────────────────────────────────
# OCR 版面 → 行列
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def width(self) -> float:
        return max(1.0, self.x1 - self.x0)


def _usable_lines(lines: list) -> list[_Line]:
    """抽出带坐标的 OCR 行（无坐标的直接丢弃，走文本兜底路径）."""
    out: list[_Line] = []
    for item in lines or []:
        text = (getattr(item, "text", "") or "").strip()
        box = getattr(item, "box", None)
        if not text or not box:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        out.append(_Line(text=text, x0=x0, y0=y0, x1=x1, y1=y1))
    return out


def _cluster_rows(lines: list[_Line]) -> list[list[_Line]]:
    """按 y 中心把文字行聚成表格行（同一行的多个单元格会聚到一起）."""
    ordered = sorted(lines, key=lambda l: (l.cy, l.x0))
    heights = sorted(l.height for l in ordered)
    median_h = heights[len(heights) // 2] or 1.0
    tol = median_h * 0.7

    rows: list[list[_Line]] = []
    centers: list[float] = []
    for line in ordered:
        placed = False
        for index, row in enumerate(rows):
            if abs(line.cy - centers[index]) <= tol:
                row.append(line)
                # 更新该行的参考中心（取平均，避免被首个元素带偏）
                centers[index] = sum(item.cy for item in row) / len(row)
                placed = True
                break
        if not placed:
            rows.append([line])
            centers.append(line.cy)
    for row in rows:
        row.sort(key=lambda l: l.x0)
    return rows


def _row_cells_by_rules(row: list[_Line], boundaries: list[float]) -> list[str]:
    """
    用竖线位置把一行的文字切进单元格.

    *boundaries* 是竖线的 x 中心（升序）。一个文字行落在哪两个竖线之间，
    就属于哪一列 —— 比"按 x 聚类"稳定得多，因为列边界是**真实画出来的**。
    """
    cells: list[list[str]] = [[] for _ in range(len(boundaries) + 1)]
    for line in row:
        index = 0
        for position in boundaries:
            if line.cx > position:
                index += 1
            else:
                break
        cells[index].append(line.text)
    return [" ".join(parts).strip() for parts in cells]


def _row_cells_by_alignment(row: list[_Line], column_centers: list[float]) -> list[str]:
    """无框线时：按全局列中心做最近邻分配."""
    cells: list[list[str]] = [[] for _ in range(len(column_centers))]
    for line in row:
        best = 0
        best_distance = None
        for index, center in enumerate(column_centers):
            distance = abs(line.cx - center)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best = index
        cells[best].append(line.text)
    return [" ".join(parts).strip() for parts in cells]


def _column_centers_from_lines(lines: list[_Line]) -> list[float]:
    """
    无框线表格的列中心推断.

    做法：把所有行的第一个单元格左边界当作"列锚点"收集起来聚类。表格通常
    左对齐，因此每行第一个文字块的 x 起点就是列起点的一个观测值。
    """
    anchors = sorted(
        l.x0 for l in lines if l.x1 - l.x0 < 10_000
    )
    if not anchors:
        return []
    widths = sorted(l.width for l in lines)
    median_w = widths[len(widths) // 2] or 40.0
    tol = max(8.0, median_w * 0.12)

    groups: list[list[float]] = []
    for value in anchors:
        if groups and value - groups[-1][-1] <= tol:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


# ─────────────────────────────────────────────────────────────────────────────
# Markdown 渲染
# ─────────────────────────────────────────────────────────────────────────────


def _escape(cell: str) -> str:
    """单元格文本转义：竖线会破坏 Markdown 表格结构."""
    return cell.replace("|", "\\|").replace("\n", " ").strip()


def to_markdown(grid: list[list[str]]) -> str:
    """二维单元格 → Markdown 表格（首行作表头）."""
    if not grid or not grid[0]:
        return ""
    width = max(len(row) for row in grid)
    normalized = [row + [""] * (width - len(row)) for row in grid]
    header = [_escape(cell) for cell in normalized[0]]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def _trim_grid(grid: list[list[str]]) -> list[list[str]]:
    """去掉全空的行与列（OCR 噪声常在边缘留下空列）."""
    kept_rows = [row for row in grid if any(cell.strip() for cell in row)]
    if not kept_rows:
        return []
    width = max(len(row) for row in kept_rows)
    padded = [row + [""] * (width - len(row)) for row in kept_rows]
    keep_cols = [
        index for index in range(width)
        if any(row[index].strip() for row in padded)
    ]
    return [[row[index] for index in keep_cols] for row in padded]


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────


def recognize_table(
    img: Image.Image,
    *,
    lines: list | None = None,
    page_number: int = 1,
    ocr_fn=None,
) -> TableStructure:
    """
    把一张表格图片还原成 Markdown 表格.

    *lines* 是带坐标的 OCR 行（``OCRLine``）。调用方通常已经做过一次 OCR，
    这里直接复用，避免重复推理。

    *ocr_fn* 是"对一小块图做 OCR"的可调用对象（``f(pil_image) -> str``）。
    提供它时会启用**逐单元格 OCR** —— 这是表格还原质量最高的一条路径，
    也是设计稿"Table Parser"的本意：先用框线切出网格，再逐格识别内容。
    逐格识别的三大好处：

        1. 每个单元格是干净的小图，OCR 不受相邻列文字干扰；
        2. 完全不依赖 OCR 引擎是否返回行坐标（Tesseract 的坐标质量较差）；
        3. 小字放大后再识别，弱引擎的准确率显著提升。

    还原顺序（最可靠 → 最兜底）：
        1. 框线 + 逐单元格 OCR
        2. 框线 + OCR 行坐标分配
        3. OCR 行/列对齐（无边框表格）
        4. 空白切分文本兜底
    """
    settings = get_settings()
    min_rows = getattr(settings, "TABLE_IMAGE_MIN_ROWS", 2)
    min_cols = getattr(settings, "TABLE_IMAGE_MIN_COLS", 2)
    max_rows = getattr(settings, "TABLE_IMAGE_MAX_ROWS", 200)
    max_cols = getattr(settings, "TABLE_IMAGE_MAX_COLS", 30)
    max_cells = getattr(settings, "TABLE_IMAGE_MAX_OCR_CELLS", 240)

    usable = _usable_lines(lines or [])

    # ── 框线检测（先做，两条最可靠的路径都依赖它）──────────────────────────
    try:
        row_positions, col_positions = detect_rules(img)
    except Exception as exc:      # noqa: BLE001
        logger.warning("Rule detection failed: %s", exc)
        row_positions, col_positions = [], []

    # ── 路径 1：框线 + 逐单元格 OCR（Table Parser 主线）────────────────────
    if ocr_fn is not None and len(row_positions) >= 2 and len(col_positions) >= 1:
        structure = _from_rules_with_cell_ocr(
            img, row_positions, col_positions, ocr_fn,
            min_rows=min_rows, min_cols=min_cols,
            max_rows=max_rows, max_cols=max_cols, max_cells=max_cells,
            page_number=page_number,
        )
        if structure.ok:
            return structure

    # ── 路径 2：框线 + OCR 行坐标分配 ──────────────────────────────────────
    if usable and (row_positions or col_positions):
        structure = _from_rules(
            usable, row_positions, col_positions,
            min_rows=min_rows, min_cols=min_cols,
            max_rows=max_rows, max_cols=max_cols,
            page_number=page_number,
        )
        if structure.ok:
            return structure
        logger.info(
            "Rule-based table parse insufficient (%s) — falling back to OCR alignment",
            structure.meta.get("reason"),
        )

    # ── 路径 3：无坐标时的文本兜底（放在对齐法之前：没有坐标就没法对齐）────
    if not usable:
        return _from_text_lines(lines or [], page_number=page_number)

    # ── 路径 4：OCR 行/列对齐 ─────────────────────────────────────────────
    structure = _from_alignment(
        usable, min_rows=min_rows, min_cols=min_cols,
        max_rows=max_rows, max_cols=max_cols, page_number=page_number,
    )
    if structure.ok:
        return structure

    fallback = _from_text_lines(lines or [], page_number=page_number)
    if fallback.ok:
        fallback.meta["degraded_from"] = "ocr-alignment"
        return fallback

    return TableStructure(
        method="failed",
        meta={
            "reason": "no-grid-recovered",
            "ocr_lines": len(usable),
            "h_rules": len(row_positions),
            "v_rules": len(col_positions),
        },
    )


def _cell_crop(
    img: Image.Image,
    x0: float, y0: float, x1: float, y1: float,
) -> Image.Image | None:
    """
    裁出一个单元格，并做两件对 OCR 很关键的事：内缩（避开框线）+ 放大（小字）。

    框线本身会被 OCR 误读成 `|`、`-` 之类的字符，内缩是必要的；文档里的表格
    字号通常只有 10–14px，放大 2–3 倍能显著提升识别率。
    """
    width, height = img.size
    left = max(0, int(x0))
    top = max(0, int(y0))
    right = min(width, int(x1))
    bottom = min(height, int(y1))
    if right - left < 4 or bottom - top < 4:
        return None

    inset_x = max(2, int((right - left) * 0.06))
    inset_y = max(2, int((bottom - top) * 0.10))
    left += inset_x
    right -= inset_x
    top += inset_y
    bottom -= inset_y
    if right - left < 3 or bottom - top < 3:
        return None

    crop = img.crop((left, top, right, bottom)).convert("RGB")
    cell_h = crop.size[1]
    if cell_h < 36:
        factor = min(3.0, 36.0 / max(cell_h, 1))
        crop = crop.resize(
            (max(1, int(crop.size[0] * factor)), max(1, int(crop.size[1] * factor))),
            Image.LANCZOS,
        )
    return crop


def _from_rules_with_cell_ocr(
    img: Image.Image,
    row_positions: list[float],
    col_positions: list[float],
    ocr_fn,
    *,
    min_rows: int,
    min_cols: int,
    max_rows: int,
    max_cols: int,
    max_cells: int,
    page_number: int,
) -> TableStructure:
    """框线切网格 → 逐格裁剪 OCR → Markdown（设计稿 Table Parser 主线）."""
    cols_edges = sorted(col_positions)
    if len(cols_edges) < 1:
        return TableStructure(method="failed", meta={"reason": "no-vertical-rules"})

    # 只取**相邻框线之间**的区间：表格内容一定画在框线之内，框线之外是页边距。
    # 若把首尾也当成一格，会凭空多出一整圈空白行列（表现为 6x5 而真实是 4x3）。
    col_bounds = list(zip(cols_edges, cols_edges[1:]))
    row_bounds = list(zip(sorted(row_positions), sorted(row_positions)[1:]))

    cols = len(col_bounds)
    rows = len(row_bounds)
    if cols < min_cols:
        return TableStructure(method="failed", meta={"reason": f"cols={cols}<{min_cols}"})
    if cols > max_cols:
        return TableStructure(method="failed", meta={"reason": f"cols={cols}>{max_cols}"})
    if rows < min_rows:
        return TableStructure(method="failed", meta={"reason": f"rows={rows}<{min_rows}"})
    if rows > max_rows:
        row_bounds = row_bounds[:max_rows]
        rows = max_rows
    if rows * cols > max_cells:
        return TableStructure(
            method="failed",
            meta={"reason": f"cells={rows * cols}>{max_cells}"},
        )

    grid: list[list[str]] = []
    recognized = 0
    for top, bottom in row_bounds:
        cells: list[str] = []
        for left, right in col_bounds:
            crop = _cell_crop(img, left, top, right, bottom)
            text = ""
            if crop is not None:
                try:
                    text = (ocr_fn(crop) or "").strip()
                except Exception as exc:      # noqa: BLE001
                    logger.warning("Cell OCR failed: %s", exc)
            text = " ".join(text.split())
            if text:
                recognized += 1
            cells.append(text)
        grid.append(cells)

    if recognized == 0:
        return TableStructure(method="failed", meta={"reason": "cells-all-empty"})

    # 注意：这里**不做** _trim_grid。网格是框线画出来的，结构本身就是可信的
    # —— 某个单元格 OCR 失败不代表那一行/列不存在。如果按"空就删"的规则裁剪，
    # 一列中文全部识别失败时整列会消失、行列错位，产出比没有更糟。
    # 只有"到底识别出多少内容"才决定是否算成功。
    if len(grid) < min_rows:
        return TableStructure(method="failed", meta={"reason": f"rows={len(grid)}<{min_rows}"})
    filled = sum(1 for row in grid for cell in row if cell.strip())
    if filled < max(min_rows * min_cols, 3):
        return TableStructure(
            method="failed", meta={"reason": f"cells-filled={filled}"}
        )

    grid = grid[:max_rows]
    return TableStructure(
        markdown=to_markdown(grid),
        rows=len(grid),
        cols=max(len(r) for r in grid),
        header=grid[0],
        method="rules+cell-ocr",
        meta={
            "page": page_number,
            "h_rules": len(row_positions),
            "v_rules": len(col_positions),
            "cells_recognized": recognized,
            "cells_filled": filled,
        },
    )


def _from_rules(
    lines: list[_Line],
    row_positions: list[float],
    col_positions: list[float],
    *,
    min_rows: int,
    min_cols: int,
    max_rows: int,
    max_cols: int,
    page_number: int,
) -> TableStructure:
    """框线法：用真实画出的线切网格."""
    if len(col_positions) < 1:
        return TableStructure(method="failed", meta={"reason": "no-vertical-rules"})

    boundaries = sorted(col_positions)
    cols = len(boundaries) + 1
    if cols < min_cols:
        return TableStructure(
            method="failed", meta={"reason": f"cols={cols}<{min_cols}"}
        )
    if cols > max_cols:
        return TableStructure(
            method="failed", meta={"reason": f"cols={cols}>{max_cols}"}
        )

    # 行：有横线就按横线切，没有就退回 y 聚类
    if len(row_positions) >= 2:
        sorted_rows = sorted(row_positions)
        row_bands: list[tuple[float, float]] = []
        edges = [float("-inf")] + sorted_rows + [float("inf")]
        for index in range(len(edges) - 1):
            row_bands.append((edges[index], edges[index + 1]))
        grid: list[list[str]] = []
        for low, high in row_bands:
            members = [l for l in lines if low < l.cy <= high]
            if not members:
                continue
            members.sort(key=lambda l: l.x0)
            grid.append(_row_cells_by_rules(members, boundaries))
        method = "rules"
    else:
        grid = [
            _row_cells_by_rules(row, boundaries)
            for row in _cluster_rows(lines)
        ]
        method = "rules+ocr-rows"

    grid = _trim_grid(grid)
    if len(grid) < min_rows:
        return TableStructure(
            method="failed", meta={"reason": f"rows={len(grid)}<{min_rows}"}
        )
    grid = grid[:max_rows]

    markdown = to_markdown(grid)
    return TableStructure(
        markdown=markdown,
        rows=len(grid),
        cols=max(len(r) for r in grid),
        header=grid[0],
        method=method,
        meta={"page": page_number, "h_rules": len(row_positions),
              "v_rules": len(col_positions)},
    )


def _from_alignment(
    lines: list[_Line],
    *,
    min_rows: int,
    min_cols: int,
    max_rows: int,
    max_cols: int,
    page_number: int,
) -> TableStructure:
    """OCR 对齐法：无边框表格（或框线没检测到）时按文字对齐切列."""
    rows = _cluster_rows(lines)
    if len(rows) < min_rows:
        return TableStructure(
            method="failed", meta={"reason": f"rows={len(rows)}<{min_rows}"}
        )

    # 列锚点来自"每行的起始 x"，比用全部文字块聚类更稳
    anchors: list[float] = []
    for row in rows:
        if row:
            anchors.append(row[0].x0)
            # 一行里有多个明显分开的文字块（gap 较大）也算一个新列的起点
            for previous, current in zip(row, row[1:]):
                if current.x0 - previous.x1 > max(6.0, previous.height * 0.5):
                    anchors.append(current.x0)

    column_centers = _column_centers_from_lines(
        [_Line(text="", x0=a, y0=0, x1=a + 1, y1=1) for a in anchors]
    )
    if len(column_centers) < min_cols:
        return TableStructure(
            method="failed",
            meta={"reason": f"cols={len(column_centers)}<{min_cols}"},
        )
    column_centers = column_centers[:max_cols]

    grid = [_row_cells_by_alignment(row, column_centers) for row in rows]
    grid = _trim_grid(grid)
    if len(grid) < min_rows:
        return TableStructure(
            method="failed", meta={"reason": f"rows={len(grid)}<{min_rows}"}
        )
    grid = grid[:max_rows]

    # 大多数行只有一个单元格 → 这更像段落而不是表格
    multi = sum(1 for row in grid if sum(1 for c in row if c.strip()) >= 2)
    if multi < max(1, len(grid) // 2):
        return TableStructure(
            method="failed", meta={"reason": f"multi-cell rows={multi}/{len(grid)}"}
        )

    return TableStructure(
        markdown=to_markdown(grid),
        rows=len(grid),
        cols=max(len(r) for r in grid),
        header=grid[0],
        method="ocr-alignment",
        meta={"page": page_number, "multi_cell_rows": multi},
    )


def _from_text_lines(lines: list, *, page_number: int) -> TableStructure:
    """
    文本兜底：用 2 个以上空格 / 制表符切列.

    只有在完全没有坐标信息时才会走到这里（例如纯 Tesseract 文本输出）。
    """
    raw: list[str] = []
    for item in lines or []:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, str):
            text = item
        text = (text or "").strip()
        if text:
            raw.append(text)
    if len(raw) < 2:
        return TableStructure(method="failed", meta={"reason": "text-too-short"})

    rows: list[list[str]] = []
    for line in raw:
        if "\t" in line:
            cells = [c.strip() for c in line.split("\t")]
        else:
            # 2 个以上空格视为列分隔；单个空格保留（英文单词 / 中文词组）
            cells = [c.strip() for c in line.split("  ") if c.strip()]
        rows.append(cells)

    widths = [len(r) for r in rows]
    best = max(set(widths), key=widths.count)
    if best < 2:
        return TableStructure(method="failed", meta={"reason": "single-column-text"})

    grid = _trim_grid(rows)
    if not grid:
        return TableStructure(method="failed", meta={"reason": "empty-text-grid"})

    return TableStructure(
        markdown=to_markdown(grid),
        rows=len(grid),
        cols=max(len(r) for r in grid),
        header=grid[0],
        method="text-split",
        meta={"page": page_number, "degraded": "no-ocr-coordinates"},
    )


def recognize_table_safe(img: Image.Image, **kwargs) -> TableStructure:
    """永不抛出的入口 —— 表格识别失败必须让图片退化为普通图片，而不是丢图."""
    try:
        return recognize_table(img, **kwargs)
    except Exception as exc:      # noqa: BLE001
        logger.warning("Table structure recognition failed: %s", exc)
        return TableStructure(method="failed", meta={"error": str(exc)})


__all__ = [
    "TableStructure",
    "recognize_table",
    "recognize_table_safe",
    "detect_rules",
    "to_markdown",
]
