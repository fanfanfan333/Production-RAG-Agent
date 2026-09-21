# -*- coding: utf-8 -*-
"""
独立实测：「文档照片里的表格表头」到底会不会被还原出来.

核心问题（用户指定）
--------------------

    把一张"文档照片"喂进图片理解管线，如果它里面是一张**带表头的表格**，
    最终还原出的 Markdown 表格里，**表头行的文字是否出现**？

本脚本对每个夹具 / 真实图片，逐张给出：

    · 分类结果（image_type / 关键信号 h_lines·v_lines·…）
    · OCR 引擎与行数
    · **直调** ``recognize_table`` 的结果（method / rows / cols / header / markdown）
    · **端到端** ``understand_image`` 的结果（route / analyze_engine / decision /
      structured_content / vision 是否降级）
    · 核心断言 ``header_present``：还原出的 Markdown 表格**首行（表头行）**
      是否包含 groundtruth 给定的表头文字，并把**表头行原文**作为证据贴出

同时给出多模态通道是否**静默降级**的证据（``OLLAMA_VISION_MODEL`` 为空时
``VisionEngine.is_available()`` 为 False，chart/diagram/screenshot 会退成纯 OCR）。

用法
----

    docker exec -w /app -e PYTHONPATH=/app rag_backend \
        python /tmp/measure_doc_photo_header.py \
            --fixtures /tmp/fixtures/groundtruth.json \
            --extra-json /tmp/extra_inputs.json \
            --out /tmp/doc_photo_header_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

from app.services.image_understanding.classifier import (
    classify_image_safe,
    vision_analyze_types,
)
from app.services.image_understanding.engines.vision_engine import VisionEngine
from app.services.image_understanding.pipeline import (
    _cell_ocr_fn,
    _run_ocr,
    resolve_route,
    understand_image,
)
from app.services.image_understanding.table_recognizer import recognize_table_safe


# ─────────────────────────────────────────────────────────────────────────────
# 表头断言
# ─────────────────────────────────────────────────────────────────────────────


def _norm(text: str) -> str:
    """归一化：去掉所有空白、统一全角竖线，便于子串比较."""
    return re.sub(r"\s+", "", (text or "")).replace("｜", "|")


def _markdown_rows(markdown: str) -> list[str]:
    """抽出 Markdown 里所有表格行（以 ``|`` 开头的行）."""
    return [
        line.strip()
        for line in (markdown or "").splitlines()
        if line.strip().startswith("|")
    ]


def check_header(gt_header: list[str], markdown: str) -> dict:
    """
    表头断言.

    * ``header_present``：还原表格的**首行**是否包含**全部**表头文字（最严格）.
    * ``header_present_anywhere``：表头文字是否在整段 Markdown 里出现过
      （用于区分"结构没还原但文字还在"与"文字也丢了"）.
    """
    tokens = [h for h in (gt_header or []) if _norm(h)]
    rows = _markdown_rows(markdown)
    header_line = rows[0] if rows else ""
    norm_line = _norm(header_line)
    norm_all = _norm(markdown)

    found_line = [h for h in tokens if _norm(h) in norm_line]
    found_all = [h for h in tokens if _norm(h) in norm_all]
    return {
        "header_line_raw": header_line,
        "header_tokens_total": len(tokens),
        "header_tokens_in_header_line": found_line,
        "header_tokens_in_markdown": found_all,
        "header_present": bool(tokens) and len(found_line) == len(tokens),
        "header_present_anywhere": bool(tokens) and len(found_all) == len(tokens),
        "markdown_row_count": len(rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 单张实测
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_KEYS = [
    "h_lines", "v_lines", "text_bands", "line_art_ratio", "saturation",
    "flat_blocks", "chromatic_blocks", "dominant_ratio", "colorful_ratio",
    "ocr_rows", "ocr_cols", "ocr_alignment", "inverted", "polarity_source",
]


def measure_one(name: str, path: str, gt_header: list[str]) -> dict:
    img = Image.open(path).convert("RGB")

    cls = classify_image_safe(img, filename=name)
    signals = cls.signals or {}

    # 直调表格识别（复用一次 OCR 的行坐标，与生产 ``_table_primary`` 一致）
    ocr_out = _run_ocr(img)
    structure = recognize_table_safe(
        img, lines=ocr_out.lines, page_number=1, ocr_fn=_cell_ocr_fn(),
    )

    # 端到端管线
    und = understand_image(img, filename=name, page_number=1)

    rec = {
        "name": name,
        "file": path,
        "size": list(img.size),
        "gt_header": list(gt_header),
        # 分类
        "classify_type": cls.image_type,
        "classify_conf": cls.confidence,
        "classify_reason": signals.get("reason"),
        "signals": {k: signals.get(k) for k in _SIGNAL_KEYS},
        "route_from_type": resolve_route(cls.image_type),
        # OCR
        "ocr_engine": ocr_out.engine,
        "ocr_ok": bool(ocr_out.ok),
        "ocr_lines": len(ocr_out.lines or []),
        "ocr_text_head": (ocr_out.text or "")[:160],
        # 直调 recognize_table
        "table_method": structure.method,
        "table_rows": structure.rows,
        "table_cols": structure.cols,
        "table_header": list(structure.header),
        "table_markdown": structure.markdown,
        # 端到端 understand_image
        "u_image_type": und.image_type,
        "u_route": und.route,
        "u_analyze_engine": und.analyze_engine,
        "u_decision": und.decision,
        "u_confidence": round(float(und.confidence), 4),
        "u_manual_review": und.manual_review,
        "u_structured_content": und.structured_content,
        "u_vision_caption": und.vision_caption,
        "u_vision_unavailable": und.meta.get("vision_unavailable"),
        "u_meta_keys": sorted(und.meta.keys()),
        "u_quality_reasons": (und.quality or {}).get("reasons"),
    }
    # 核心断言
    rec.update(check_header(gt_header, structure.markdown))
    # 端到端结构化内容里的表头（table 路由时才有 structured_content）
    rec["u_header_present"] = (
        check_header(gt_header, und.structured_content or "")["header_present"]
        if (gt_header and und.image_type == "table")
        else None
    )
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────


def _load_extra(extra_json: str | None) -> list[tuple[str, str, list[str]]]:
    if not extra_json or not Path(extra_json).exists():
        return []
    data = json.loads(Path(extra_json).read_text(encoding="utf-8"))
    out = []
    for item in data:
        out.append((item["name"], item["file"], item.get("header") or []))
    return out


def _vision_probe() -> dict:
    """VLM 通道可用性证据."""
    vision = VisionEngine()
    import os
    return {
        "OLLAMA_VISION_MODEL": os.environ.get("OLLAMA_VISION_MODEL", ""),
        "VISION_ENABLED": os.environ.get("VISION_ENABLED", ""),
        "VISION_ANALYZE_TYPES": os.environ.get("VISION_ANALYZE_TYPES", ""),
        "vision_analyze_types_parsed": sorted(vision_analyze_types()),
        "VisionEngine.is_available()": vision.is_available(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="实测文档照片表格表头还原")
    parser.add_argument("--fixtures", default="/tmp/fixtures/groundtruth.json")
    parser.add_argument("--extra-json", default=None)
    parser.add_argument("--out", default="/tmp/doc_photo_header_report.json")
    args = parser.parse_args()

    targets: list[tuple[str, str, list[str]]] = []
    if Path(args.fixtures).exists():
        gt = json.loads(Path(args.fixtures).read_text(encoding="utf-8"))
        for rec in gt["fixtures"]:
            targets.append((rec["name"], rec["file"], rec["header"]))
    targets += _load_extra(args.extra_json)

    results = [_vision_probe()]
    for name, path, header in targets:
        try:
            results.append(measure_one(name, path, header))
        except Exception as exc:  # noqa: BLE001
            results.append({"name": name, "file": path, "error": repr(exc)})

    payload = {
        "vision_probe": results[0],
        "fixtures": [r for r in results[1:]],
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"报告：{args.out}（{len(results) - 1} 张）")


if __name__ == "__main__":
    main()
