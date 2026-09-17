"""
Table Transformer 引擎（表格结构识别 —— 研究型方案）.

对应设计稿"表格结构研究型方案 → Table Transformer"。用微软
``microsoft/table-transformer-structure-recognition`` 检测**行 / 列 / 表头**，
把检测到的行框与列框做笛卡尔组合得到单元格，再对每个单元格单独 OCR。

和纯规则表格识别（框线投影）的分工：

    ┌──────────────┬──────────────────────────────┬─────────────────────┐
    │ 方案          │ 强项                          │ 弱项                 │
    ├──────────────┼──────────────────────────────┼─────────────────────┤
    │ 规则（框线）   │ 有线表格、快、零依赖            │ 无边框/花边框会散架   │
    │ Table Trans. │ 无线表格、跨行跨列、表头语义     │ 需模型、慢、占内存    │
    └──────────────┴──────────────────────────────┴─────────────────────┘

两者不是替代关系：规则先跑（便宜），置信度低时再让 Table Transformer 复核，
正好落在流程图的 "Specialized OCR → confidence 低 → Fallback" 上。
"""

from __future__ import annotations

import threading

from PIL import Image

from app.services.image_understanding.engines.base import (
    EngineKind,
    EngineOutput,
    ImageEngine,
)
from app.services.image_understanding.table_recognizer import to_markdown
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: HF 上的结构识别模型（microsoft 官方）
MODEL_ID = "microsoft/table-transformer-structure-recognition"

#: 模型输出标签（顺序与官方 config 一致）
LABELS = {
    0: "table",
    1: "table column",
    2: "table row",
    3: "table column header",
    4: "table row header",
    5: "table projected row header",
    6: "table spanning cell",
    7: "no object",
}

_lock = threading.Lock()
_model = None
_processor = None


class TableTransformerEngine(ImageEngine):
    """行/列检测 + 单元格 OCR → Markdown 表格."""

    name = "table-transformer"
    kind = EngineKind.TABLE

    def is_available(self) -> bool:
        """
        轻量探测：只查 transformers 是否提供该类，**不下载、不加载权重**.

        真正加载在首次 ``process`` 时惰性发生 —— 否则一个没配模型的部署会在
        启动期卡在几百 MB 的下载上。
        """
        try:
            from transformers import TableTransformerForObjectDetection  # noqa: F401  # noqa: PLC0415

            return True
        except Exception:      # noqa: BLE001
            return False

    def unavailable_reason(self) -> str:
        if self.is_available():
            return ""
        return "transformers 未提供 TableTransformerForObjectDetection"

    def _load(self):
        global _model, _processor
        if _model is not None:
            return _model, _processor
        with _lock:
            if _model is None:
                from transformers import (  # noqa: PLC0415
                    AutoImageProcessor,
                    TableTransformerForObjectDetection,
                )

                _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
                _model = TableTransformerForObjectDetection.from_pretrained(MODEL_ID)
                _model.eval()
                logger.info("Table Transformer loaded: %s", MODEL_ID)
        return _model, _processor

    def process(self, image: Image.Image, *, ocr_fn=None, threshold: float = 0.6, **kwargs) -> EngineOutput:
        """
        检测结构 → 单元格 OCR → Markdown.

        *ocr_fn*：``callable(PIL.Image) -> str``，单元格级 OCR。不传则退化为
        空单元格（仍能返回"有几行几列"的结构，只是没字）。
        """
        try:
            import torch  # noqa: PLC0415

            model, processor = self._load()
        except Exception as exc:      # noqa: BLE001
            logger.warning("Table Transformer unavailable: %s", exc)
            return EngineOutput(engine=self.name, ok=False, error=str(exc))

        try:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            result = processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes
            )[0]
        except Exception as exc:      # noqa: BLE001
            logger.warning("Table Transformer inference failed: %s", exc)
            return EngineOutput(engine=self.name, ok=False, error=str(exc))

        rows, cols, header_rows, scores = [], [], set(), []
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
            name = LABELS.get(int(label), "")
            x0, y0, x1, y1 = (float(v) for v in box.tolist())
            scores.append(float(score))
            if name in ("table row", "table column header", "table projected row header", "table row header"):
                rows.append((y0, y1, x0, x1))
                if name in ("table column header", "table projected row header"):
                    header_rows.add(round(y0))
            elif name == "table column":
                cols.append((x0, x1, y0, y1))

        rows.sort(key=lambda r: r[0])
        cols.sort(key=lambda c: c[0])

        # Table Transformer 会把同一物理行同时标成 "table column header" 和
        # "table row"，导致表头在 rows 里出现两次（输出重复表头行）。
        # 按 y 区间重叠去重：保留先出现的（已排序，header 框与 row 框 y 相近）。
        rows = _dedupe_rows(rows)

        if len(rows) < 2 or len(cols) < 2:
            return EngineOutput(
                engine=self.name, ok=False,
                error=f"结构不足 (rows={len(rows)}, cols={len(cols)})",
                meta={"rows": len(rows), "cols": len(cols)},
            )

        grid: list[list[str]] = []
        for (ry0, ry1, _, _) in rows:
            row_cells: list[str] = []
            for (cx0, cx1, _, _) in cols:
                cell = image.crop((
                    max(0, int(cx0)), max(0, int(ry0)),
                    min(image.width, int(cx1)), min(image.height, int(ry1)),
                ))
                if cell.width < 3 or cell.height < 3:
                    row_cells.append("")
                    continue
                row_cells.append(_cell_text(cell, ocr_fn))
            grid.append(row_cells)

        markdown = to_markdown(grid)
        if not markdown:
            return EngineOutput(engine=self.name, ok=False, error="Markdown 为空")

        # 置信度 = 检测分数均值 × 结构完整度（非空格占比）
        mean_score = sum(scores) / len(scores) if scores else 0.0
        total = sum(len(r) for r in grid) or 1
        filled = sum(1 for r in grid for c in r if c.strip())
        confidence = round(mean_score * (0.4 + 0.6 * filled / total), 4)

        return EngineOutput(
            text=markdown,
            confidence=confidence,
            engine=self.name,
            ok=True,
            meta={
                "rows": len(rows),
                "cols": len(cols),
                "mean_score": round(mean_score, 4),
                "filled_ratio": round(filled / total, 4),
                "model": MODEL_ID,
            },
        )


def _dedupe_rows(rows: list[tuple]) -> list[tuple]:
    """合并 y 区间重叠的行框（Table Transformer 的表头行会被重复标出）."""
    if len(rows) <= 1:
        return rows
    out: list[tuple] = []
    for row in rows:
        y0, y1 = row[0], row[1]
        merged = False
        for prev in out:
            py0, py1 = prev[0], prev[1]
            overlap = min(y1, py1) - max(y0, py0)
            span = max(y1 - y0, py1 - py0, 1e-6)
            if overlap / span > 0.5:      # 重叠超过一半视为同一行
                merged = True
                break
        if not merged:
            out.append(row)
    return out


def _cell_text(cell: Image.Image, ocr_fn) -> str:
    """
    单元格 OCR.

    小单元格先放大再识别 —— 12px 高的字直接喂 OCR 基本是噪声，
    3 倍双三次插值后识别率明显改善（与规则路径同一套预处理思路）。
    """
    if ocr_fn is None:
        return ""
    try:
        if cell.height < 32:
            scale = max(2, min(4, 32 // max(cell.height, 1)))
            cell = cell.resize((cell.width * scale, cell.height * scale), Image.BICUBIC)
        return (ocr_fn(cell) or "").replace("\n", " ").strip()
    except Exception:      # noqa: BLE001
        return ""


__all__ = ["TableTransformerEngine", "MODEL_ID", "LABELS"]
