"""
Document Agent（最终效果：Word 写入 / Word 插入图片）.

把一次检索的结果落成一份 **可直接交付的 Word 文档**：

    retrieve/grade → multimodal_context 的 chunks
        ├── text  chunk  → 正文段落（保留 heading 层级）
        ├── table chunk  → 真正的 Word 表格（Markdown 表格还原为表格对象）
        └── image chunk  → 从 image_path 取**原始图片**插入文档，附图片说明

配套能力：
- `GET /documents/generated/{filename}` 下载入口；
- 文档末尾自动附「参考来源」清单（文件名 + 页码 + 相关度），可追溯。

设计取舍
────────
- 生成失败绝不抛出到对话主链路：Agent 节点捕获异常后回退成"纯文本总结"，
  用户至少能拿到答案，而不是一个 500。
- 目录穿越防护：文件名只由服务端生成（时间戳 + 随机后缀），不接受用户输入
  作为路径。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.services.storage import resolve_image_path
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 页面内容区宽度（python-docx 默认模板 A4 + 1 英寸页边距 ≈ 6.5 英寸）
_IMAGE_WIDTH_INCHES = 5.5

# Markdown 表格行：| a | b |
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
# 对齐行：| --- | :--: |
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


@dataclass
class GeneratedDocument:
    """一次 Document Agent 生成的产物元信息."""

    filename: str                  # 磁盘文件名（ASCII 安全）
    download_url: str              # 前端下载地址
    title: str
    section_count: int = 0
    table_count: int = 0
    image_count: int = 0
    char_count: int = 0
    size_bytes: int = 0
    sources: list[dict] = field(default_factory=list)
    error: str | None = None

    def as_event_payload(self) -> dict:
        """SSE `document` 事件负载."""
        return {
            "filename": self.filename,
            "title": self.title,
            "download_url": self.download_url,
            "section_count": self.section_count,
            "table_count": self.table_count,
            "image_count": self.image_count,
            "char_count": self.char_count,
            "size_bytes": self.size_bytes,
            "error": self.error,
        }


# ── 输出目录 ──────────────────────────────────────────────────────────────────

def output_dir() -> Path:
    root = Path(get_settings().DOCUMENT_OUTPUT_DIR)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_generated_file(filename: str) -> Path | None:
    """
    Resolve a generated-document filename to an absolute path (traversal-safe).

    Only plain file names are accepted — no separators, no parent references.
    """
    if not filename:
        return None
    if "/" in filename or "\\" in filename or ".." in filename:
        return None
    candidate = (output_dir() / filename).resolve()
    base = output_dir().resolve()
    if candidate.parent != base:
        return None
    return candidate if candidate.is_file() else None


# ── Markdown 表格 → Word 表格 ─────────────────────────────────────────────────

def _split_cells(line: str) -> list[str]:
    m = _MD_TABLE_ROW_RE.match(line)
    if not m:
        return []
    return [c.strip() for c in m.group(1).split("|")]


def _extract_markdown_table(text: str) -> tuple[list[str], list[list[str]], str]:
    """
    抽出文本里的第一段 Markdown 表格。

    Returns:
        (header, rows, remaining_text) —— 没找到表格时 header/rows 为空。
    """
    lines = text.splitlines()
    start = None
    block: list[str] = []
    for i, line in enumerate(lines):
        if _MD_TABLE_ROW_RE.match(line):
            start = i
            block.append(line)
        elif start is not None:
            break

    if start is None or len(block) < 2:
        return [], [], text

    header = _split_cells(block[0])
    body_rows: list[list[str]] = []
    for line in block[1:]:
        if _MD_TABLE_SEP_RE.match(line):
            continue
        cells = _split_cells(line)
        if cells:
            body_rows.append(cells)

    if not header or not body_rows:
        return [], [], text

    remaining = "\n".join(lines[:start] + lines[start + len(block):]).strip()
    return header, body_rows, remaining


def _add_markdown_table(doc, text: str, max_rows: int) -> int:
    """
    把 *text* 中的 Markdown 表格写成真正的 Word 表格.

    Returns:
        写入的表格数量（0 或 1）
    """
    header, rows, remaining = _extract_markdown_table(text)
    if not header:
        return 0

    if remaining:
        for para in remaining.split("\n\n"):
            para = para.strip()
            if para:
                doc.add_paragraph(para)

    rows = rows[:max_rows]
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    for i, cell_text in enumerate(header):
        table.rows[0].cells[i].text = cell_text
    for row in rows:
        cells = table.add_row().cells
        for i in range(len(header)):
            cells[i].text = row[i] if i < len(row) else ""
    doc.add_paragraph("")
    return 1


# ── 生成主流程 ────────────────────────────────────────────────────────────────

def _safe_title(query: str) -> str:
    title = (query or "").strip().replace("\n", " ")
    if len(title) > 60:
        title = title[:60] + "…"
    return title or "知识库生成文档"


def _build_docx(
    query: str,
    chunks: list,
    *,
    title: str | None = None,
) -> tuple[object, GeneratedDocument]:
    """
    用 python-docx 组装文档。返回 (docx Document, 统计信息).

    图片插入失败（文件缺失等）不影响整体生成 —— 只记录跳过。
    """
    from docx import Document
    from docx.shared import Inches, Pt

    settings = get_settings()
    max_sources = max(int(settings.DOCUMENT_AGENT_MAX_SOURCES), 1)
    max_images = max(int(settings.DOCUMENT_AGENT_MAX_IMAGES), 0)
    max_table_rows = max(int(settings.DOCUMENT_AGENT_MAX_TABLE_ROWS), 1)

    doc = Document()

    # 标题
    doc_title = title or _safe_title(query)
    doc.add_heading(doc_title, level=0)

    meta = doc.add_paragraph()
    meta_run = meta.add_run(
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　|　"
        f"来源片段：{len(chunks)}"
    )
    meta_run.font.size = Pt(9)

    section_count = 0
    table_count = 0
    image_count = 0
    char_count = 0
    sources: list[dict] = []

    text_chunks = [c for c in chunks if getattr(c, "content_type", "text") != "image"]
    image_chunks = [c for c in chunks if getattr(c, "content_type", "text") == "image"]

    # ── 正文：文本 / 表格 ────────────────────────────────────────────────────
    for chunk in text_chunks[:max_sources]:
        section_count += 1
        src_label = f"{chunk.filename}（第 {chunk.page_number} 页）"
        heading = getattr(chunk, "heading", None) or f"来源 {section_count}：{src_label}"
        doc.add_heading(str(heading)[:120], level=1)

        body = getattr(chunk, "parent_text", None) or chunk.text or ""

        if getattr(chunk, "content_type", "text") == "table":
            used = _add_markdown_table(doc, body, max_table_rows)
            if used:
                table_count += used
            else:
                doc.add_paragraph(body)
        else:
            for para in body.split("\n\n"):
                para = para.strip()
                if para:
                    doc.add_paragraph(para)

        char_count += len(body)
        sources.append(
            {
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content_type": getattr(chunk, "content_type", "text"),
                "score": round(float(getattr(chunk, "score", 0.0)), 4),
            }
        )

    # ── 插图：图片对象的原始图片（部分2 "返回原始图片" 的文档化落地）──────────
    if image_chunks and max_images > 0:
        doc.add_page_break()
        doc.add_heading("相关图片", level=1)
        for chunk in image_chunks[:max_images]:
            path = resolve_image_path(chunk.document_id, getattr(chunk, "image_path", None))
            if path is None:
                continue
            try:
                doc.add_picture(str(path), width=Inches(_IMAGE_WIDTH_INCHES))
            except Exception as exc:      # noqa: BLE001
                logger.warning("Failed to insert image %s: %s", path, exc)
                continue
            caption = (
                getattr(chunk, "image_caption", None)
                or (getattr(chunk, "text", "") or "").strip().splitlines()[0][:120]
                or "图片"
            )
            cap = doc.add_paragraph(f"图 {image_count + 1}：{caption}")
            cap.runs[0].font.size = Pt(9)
            image_count += 1
            sources.append(
                {
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "content_type": "image",
                    "score": round(float(getattr(chunk, "score", 0.0)), 4),
                }
            )

    # ── 参考来源清单 ─────────────────────────────────────────────────────────
    if sources:
        doc.add_page_break()
        doc.add_heading("参考来源", level=1)
        for i, s in enumerate(sources, start=1):
            kind = {"image": "图片", "table": "表格"}.get(s["content_type"], "正文")
            doc.add_paragraph(
                f"[{i}] {s['filename']} · 第 {s['page_number']} 页 · {kind} "
                f"· 相关度 {s['score']:.2f}",
                style=None,
            )

    info = GeneratedDocument(
        filename="",              # 落盘时填充
        download_url="",
        title=doc_title,
        section_count=section_count,
        table_count=table_count,
        image_count=image_count,
        char_count=char_count,
        sources=sources,
    )
    return doc, info


def generate_document(
    query: str,
    chunks: list,
    *,
    title: str | None = None,
) -> GeneratedDocument:
    """
    生成一份 Word 文档并落盘，返回可下载的产物信息.

    失败时返回带 ``error`` 的 GeneratedDocument（不抛异常）——调用方
    （master graph 节点）据此决定回退为纯文本回答。
    """
    settings = get_settings()
    try:
        doc, info = _build_docx(query, chunks, title=title)
    except Exception as exc:      # noqa: BLE001
        logger.exception("Document Agent failed to build docx: %s", exc)
        return GeneratedDocument(
            filename="", download_url="", title=title or _safe_title(query),
            error=str(exc),
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rag_doc_{stamp}_{uuid.uuid4().hex[:6]}.docx"
    try:
        target = output_dir() / filename
        doc.save(str(target))
        size = target.stat().st_size
    except Exception as exc:      # noqa: BLE001
        logger.exception("Document Agent failed to save docx: %s", exc)
        return GeneratedDocument(
            filename="", download_url="", title=info.title, error=str(exc),
        )

    info.filename = filename
    info.download_url = f"{settings.DOCUMENT_DOWNLOAD_PREFIX}/{filename}"
    info.size_bytes = size

    logger.info(
        "Document Agent generated '%s' (%d sections, %d tables, %d images, %d bytes)",
        filename, info.section_count, info.table_count, info.image_count, size,
    )
    return info


def build_fallback_summary(query: str, info: GeneratedDocument) -> str:
    """生成文档失败时的可读回退答案."""
    return (
        f"文档生成未能完成（{info.error or '未知原因'}）。\n\n"
        f"已基于检索到的 {len(info.sources)} 条来源整理出内容，"
        f"你可以先查看下方引用来源，或稍后重试生成。"
    )


__all__ = [
    "GeneratedDocument",
    "generate_document",
    "resolve_generated_file",
    "output_dir",
    "build_fallback_summary",
]
