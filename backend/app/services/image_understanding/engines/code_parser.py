"""
代码截图引擎（OCR + Code Parser）.

对应设计稿"代码截图 → OCR + Code Parser"。这一路的关键不在 OCR —— 通用
OCR 会把代码当自然语言处理，丢掉**缩进**（Python 里缩进就是语法！），并把
连续符号切碎。所以本引擎做两件事：

1. **缩进重建**：用每行的左边界 x 坐标 + 估计字符宽度，反推出该行前面的
   空格数。OCR 输出的是"文本 + 版面坐标"，缩进信息其实一直都在，只是通用
   路径把它扔了。
2. **语言判定 + 围栏**：按关键字/符号特征猜语言，套上 ``` 围栏，让下游
   （RAG 检索、Word 生成、LLM 上下文）拿到的是**可直接用的代码块**而不是
   一坨散行。

引擎不自己造 OCR：文本行由外部传入（管线里就是那次"只跑一次"的 OCR），
没有传入时才退回调用通用 OCR 引擎。
"""

from __future__ import annotations

import re
import statistics

from PIL import Image

from app.services.image_understanding.engines.base import (
    EngineKind,
    EngineOutput,
    ImageEngine,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: 语言特征表 —— 命中最多者胜出。顺序无关，靠计分。
_LANGUAGE_PATTERNS: list[tuple[str, list[str]]] = [
    ("python", [r"\bdef\s+\w+\s*\(", r"\bimport\s+\w+", r"\bclass\s+\w+.*:",
                r"\bself\b", r"\breturn\b", r"\bprint\s*\(", r":\s*$", r"\belif\b"]),
    ("javascript", [r"\bfunction\s+\w+", r"\bconst\s+\w+\s*=", r"\blet\s+\w+\s*=",
                    r"=>", r"\bconsole\.log\b", r"\bexport\s+(default\s+)?",
                    r"\basync\b", r"\bawait\b"]),
    ("typescript", [r":\s*(string|number|boolean|void)\b", r"\binterface\s+\w+",
                    r"\btype\s+\w+\s*=", r"\benum\s+\w+"]),
    ("java", [r"\bpublic\s+(static\s+)?(void|class|int|String)\b", r"\bprivate\b",
              r"\bnew\s+\w+\s*\(", r"\bSystem\.out\.println"]),
    ("c_cpp", [r"#include\s*<", r"\bint\s+main\s*\(", r"\bprintf\s*\(", r"std::",
               r"\bstruct\s+\w+", r"\bnullptr\b"]),
    ("go", [r"\bfunc\s+\w+\s*\(", r"\bpackage\s+\w+", r"\bfmt\.", r":=", r"\bdefer\b"]),
    ("rust", [r"\bfn\s+\w+\s*\(", r"\blet\s+mut\b", r"\bimpl\b", r"\bpub\s+fn\b",
              r"println!"]),
    ("sql", [r"\bSELECT\b.*\bFROM\b", r"\bINSERT\s+INTO\b", r"\bUPDATE\b.*\bSET\b",
             r"\bCREATE\s+TABLE\b", r"\bWHERE\b"]),
    ("shell", [r"^\s*\$\s", r"\bsudo\b", r"#!/bin/(ba)?sh", r"\bexport\s+\w+=",
               r"\bchmod\b", r"\bpip\s+install\b"]),
    ("html", [r"<html", r"</\w+>", r"<div\b", r"<script\b", r"<!DOCTYPE"]),
    ("json", [r'^\s*[\{\[]', r'"[^"]+"\s*:', r"^\s*[\}\]]\s*$"]),
    ("yaml", [r"^\s*\w+:\s*$", r"^\s*-\s+\w+", r"^\s{2,}\w+:\s+\S"]),
]

_COMPILED = [(lang, [re.compile(p, re.MULTILINE) for p in pats]) for lang, pats in _LANGUAGE_PATTERNS]

#: 判定"这是代码而不是自然语言"的信号
_CODE_SYMBOLS = set("{}[]()<>;=+-*/%&|!~^:,.$#@\\\"'")


def detect_language(lines: list[str]) -> tuple[str, float]:
    """
    猜语言，返回 (语言, 置信度).

    置信度 = 命中特征数归一化。全都不命中就返回 ``("text", 低分)`` ——
    说明这段其实更像自然语言，调用方据此走 OCR 路径而不是代码路径。
    """
    blob = "\n".join(lines)
    if not blob.strip():
        return "text", 0.0

    best_lang, best_hits = "text", 0
    for lang, pats in _COMPILED:
        hits = sum(1 for p in pats if p.search(blob))
        if hits > best_hits:
            best_lang, best_hits = lang, hits

    if best_hits == 0:
        return "text", 0.0
    return best_lang, min(1.0, best_hits / 3.0)


def code_likeness(lines: list[str]) -> float:
    """
    一段文本"像不像代码"（0~1）——分类器用它把代码截图从界面截图里分出来.

    三个信号：符号密度、缩进行占比、行尾标点（``; { } )``）占比。自然语言
    的符号密度通常低于 8%，代码普遍在 15% 以上。
    """
    joined = [l for l in lines if l.strip()]
    if not joined:
        return 0.0

    total_chars = sum(len(l) for l in joined) or 1
    symbol_chars = sum(1 for l in joined for ch in l if ch in _CODE_SYMBOLS)
    symbol_density = symbol_chars / total_chars

    indented = sum(1 for l in joined if l[:1] in (" ", "\t"))
    indent_ratio = indented / len(joined)

    tail = sum(1 for l in joined if l.rstrip().endswith((";", "{", "}", ")", ":", ",")))
    tail_ratio = tail / len(joined)

    _, lang_conf = detect_language(joined)

    score = (
        0.40 * min(1.0, symbol_density / 0.20)
        + 0.20 * indent_ratio
        + 0.20 * tail_ratio
        + 0.20 * lang_conf
    )
    return round(min(1.0, score), 4)


class CodeParserEngine(ImageEngine):
    """OCR 文本行 → 还原缩进 → 加围栏的可读代码块."""

    name = "ocr+code-parser"
    kind = EngineKind.CODE

    def is_available(self) -> bool:
        # 纯 Python 实现，无外部依赖
        return True

    def process(self, image: Image.Image, *, ocr_lines=None, **kwargs) -> EngineOutput:
        """
        *ocr_lines*：带坐标的 OCRLine 列表（管线里复用那次唯一 OCR）。

        没有传就自己跑一次通用 OCR —— 保持引擎可独立调用。
        """
        lines = ocr_lines
        if not lines:
            from app.services.image_understanding.engines.paddle_engines import PaddleOCREngine

            out = PaddleOCREngine().process(image)
            if not out.ok:
                return EngineOutput(engine=self.name, ok=False, error="OCR 不可用")
            lines = out.lines

        texts = [l.text for l in lines if (getattr(l, "text", "") or "").strip()]
        if not texts:
            return EngineOutput(engine=self.name, ok=False, error="无文本行")

        language, lang_conf = detect_language(texts)
        body = rebuild_indentation(lines)

        markdown = f"```{language}\n{body}\n```" if language != "text" else body

        # 置信度：OCR 质量 × "像代码"的程度
        ocr_conf = _mean_conf(lines)
        confidence = round(0.5 * ocr_conf + 0.5 * code_likeness(texts), 4)

        return EngineOutput(
            text=markdown,
            confidence=confidence,
            engine=self.name,
            ok=True,
            lines=lines,
            meta={"language": language, "language_confidence": lang_conf, "lines": len(texts)},
        )


def rebuild_indentation(lines) -> str:
    """
    用行左边界反推缩进.

    步骤：取所有行的最小 x0 作为左边距基准；用"字符宽度中位数"把像素偏移换算
    成空格数（1 级缩进 ≈ 4 空格 / 1 个 tab）。没有坐标信息时退化为按原样拼接。
    """
    valid = [l for l in lines if (getattr(l, "text", "") or "").strip()]
    if not valid:
        return ""
    boxes = [getattr(l, "box", None) for l in valid]
    if not all(boxes):
        return "\n".join(l.text for l in valid)

    widths: list[float] = []
    for l, box in zip(valid, boxes):
        n = len(l.text)
        if n:
            w = (box[2] - box[0]) / n
            if 0 < w < 200:
                widths.append(w)
    char_w = statistics.median(widths) if widths else 8.0

    left = min(b[0] for b in boxes if b)
    out: list[str] = []
    for l, box in zip(valid, boxes):
        pad = max(0, int(round((box[0] - left) / char_w)))
        # 每 4 个空格算一级缩进，避免 OCR 抖动造成 1~2 个空格的噪声
        pad = (pad // 4) * 4
        out.append(" " * pad + l.text.rstrip())
    return "\n".join(out)


def _mean_conf(lines) -> float:
    vals = [float(getattr(l, "confidence", 0.0) or 0.0) for l in lines]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


__all__ = [
    "CodeParserEngine",
    "detect_language",
    "code_likeness",
    "rebuild_indentation",
]
