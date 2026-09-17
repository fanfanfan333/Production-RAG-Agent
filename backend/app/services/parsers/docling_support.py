"""
Docling 接入辅助（PDF / DOCX 基础解析）.

把"要不要用 Docling、失败怎么办"收敛到一个地方，让 pdf_parser / docx_parser
只写一行调用。

**为什么 PDF 默认不用 Docling**：现有 PyMuPDF 路径做了两件 Docling 路径替代
不了的事 ——① 逐页图片抽取并把页码归属到 chunk（检索引用要显示"第 3 页"）；
② 逐页"文字太少就整页 OCR"的回退。Docling 是**整篇**转换，套进来会丢掉逐页
归属。因此：

    DOCX  → Docling 直接可用（DOCX 正文本来就没有分页，零回归）
    PDF   → 需要时用 DOCLING_PDF_ENABLED 显式打开（默认关）

需要 Docling 的版面模型（复杂多栏、扫描件、公式/表格密集）时，把
``DOCLING_PDF_ENABLED`` 打开即可；代价是正文只归到第 1 页。
"""

from __future__ import annotations

import re

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Docling 输出的 Markdown 表格分隔行是紧凑形式（|------|------|），而本系统
# 其余路径（_table_to_markdown / table_recognizer）统一输出 | --- | --- |。
# 下游 chunker 的检测、Document Agent 的还原、以及测试断言都按 | --- | 约定，
# 因此这里把 Docling 的表格归一化到同一种格式，保证全链路一致。
_DOCLING_SEP_RE = re.compile(
    r"^\s*\|(\s*:?-+:?\s*\|)+\s*$"
)


def _normalize_docling_table(text: str) -> str:
    """把 Docling 输出的紧凑表格分隔行（|------|）统一成 | --- |."""
    lines = text.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        # 前一行是表头、本行是分隔行时归一化
        if (
            _DOCLING_SEP_RE.match(line)
            and i > 0
            and _DOCLING_SEP_RE.match(lines[i - 1]) is None
            and lines[i - 1].strip().startswith("|")
        ):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            out.append("| " + " | ".join("---" for _ in cells) + " |")
        else:
            out.append(line)
    return "\n".join(out)


def docling_text(content: bytes, filename: str) -> str | None:
    """
    用 Docling 解析文档正文，返回 Markdown；不可用/失败返回 None.

    **绝不抛异常**：Docling 是"锦上添花"的路径，它挂了必须让调用方安静地回到
    原生解析器，而不是让整个上传失败。
    """
    try:
        from app.config import get_settings

        if not getattr(get_settings(), "DOCLING_ENABLED", True):
            return None
    except Exception:      # noqa: BLE001
        return None

    try:
        from app.services.image_understanding.engines.docling_engine import (
            DoclingEngine,
        )

        engine = DoclingEngine()
        if not engine.is_available():
            return None
        result = engine.parse(content, filename)
        text = (result.full_text or "").strip()
        if not text:
            return None
        text = _normalize_docling_table(text)
        logger.info(
            "Docling parsed '%s': %d chars (was falling back to native otherwise)",
            filename, len(text),
        )
        return text
    except Exception as exc:      # noqa: BLE001
        logger.warning("Docling parse failed for '%s': %s — using native parser", filename, exc)
        return None


def pdf_docling_enabled() -> bool:
    """PDF 是否走 Docling（默认关，见模块 docstring 的取舍说明）."""
    try:
        from app.config import get_settings

        return bool(getattr(get_settings(), "DOCLING_PDF_ENABLED", False))
    except Exception:      # noqa: BLE001
        return False


__all__ = ["docling_text", "pdf_docling_enabled"]
