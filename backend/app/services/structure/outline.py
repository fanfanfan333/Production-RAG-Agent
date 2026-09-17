"""
从 Markdown 还原结构节点（章节树 + 元素类型）.

为什么需要一个通用的 Markdown 结构分析器
────────────────────────────────────────
四个 provider 产出的 Markdown 细节各不相同（MinerU 的标题靠 ``text_level``、
Marker 直接输出 ``## ``、Docling 的标题来自 ``section_header`` label、原生
PyMuPDF 的 ``get_text("markdown")`` 按字号猜）。如果让每个 provider 各写一套
节点抽取，下游拿到的 ``section_path`` 就会因引擎而异 —— 而 ``section_path``
会进父子分块（父块边界）、元数据大纲、以及引用卡片上的"出自 3.2 节"。

所以约定：**provider 只负责 Markdown 与页码**，结构节点一律由本模块从 Markdown
统一还原。好处是换引擎不改变下游结构语义，且没有 `##` 的非 Markdown 输出
（如纯文本 PDF）也能走同一条路。

代码围栏内的 ``# 注释`` **不是标题** —— 这里的围栏感知与分块器的
``_parse_semantic_blocks`` 保持同一套规则，否则同一份文档在"结构树"与
"分块"两处会得出不同的章节划分。
"""

from __future__ import annotations

from app.services.structure.model import (
    NodeKind,
    PageMark,
    StructureNode,
)

# 与 chunker 的 _FENCE_RE 保持一致：``` 或 ~~~，可带语言标签
_FENCE_CHARS = ("```", "~~~")

_HEADING_RE_LINE = None  # 运行时构造，避免重复编译


def _page_for(offset: int, page_marks: list[PageMark]) -> int:
    """字符偏移 → 页码（二分）；无页码信息时返回 1."""
    if not page_marks:
        return 1
    if offset < page_marks[0].char_start:
        return page_marks[0].page_number
    lo, hi = 0, len(page_marks)
    while lo < hi:
        mid = (lo + hi) // 2
        if page_marks[mid].char_start <= offset:
            lo = mid + 1
        else:
            hi = mid
    return page_marks[max(0, lo - 1)].page_number


def _fence_marker(line: str) -> str | None:
    """该行是否是一个代码围栏（返回围栏标记，否则 None）."""
    stripped = line.strip()
    for marker in _FENCE_CHARS:
        if stripped.startswith(marker):
            return marker
    return None


def _classify_block(text: str) -> str:
    """判定一个非标题块的类型（与分块器/图片分类的词汇表对齐）."""
    stripped = text.strip()
    if not stripped:
        return NodeKind.PARAGRAPH
    if stripped.startswith("```") or stripped.startswith("~~~"):
        return NodeKind.CODE
    if stripped.startswith("$$") or stripped.startswith("\\["):
        return NodeKind.FORMULA
    if stripped.startswith("![") or stripped.startswith("<!-- image"):
        return NodeKind.FIGURE

    # 表格：连续两行以上以 | 开头/结尾
    table_run = 0
    for line in stripped.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|") and len(s) > 1:
            table_run += 1
            if table_run >= 2:
                return NodeKind.TABLE
        else:
            table_run = 0

    # 列表：第一行以 - / * / + / 数字. 开头
    first = stripped.splitlines()[0].strip()
    if first.startswith(("- ", "* ", "+ ")):
        return NodeKind.LIST
    import re as _re
    if _re.match(r"^\d+[.)]\s", first):
        return NodeKind.LIST
    return NodeKind.PARAGRAPH


def build_nodes_from_markdown(
    markdown: str,
    page_marks: list[PageMark] | None = None,
    *,
    max_nodes: int = 5000,
) -> list[StructureNode]:
    """
    把 Markdown 还原成结构节点列表（含 ``section_path`` 祖先标题链）.

    ``section_path`` 的维护方式：遇到 level=L 的标题时，先把栈里 level >= L 的
    标题弹出，压入当前标题，栈内即为其祖先链。这样 "3.2 核算方法" 的
    ``section_path`` 就是 ("第 3 章 财务情况", "3.2 核算方法")，与人的阅读直觉一致。
    """
    marks = page_marks or []
    nodes: list[StructureNode] = []

    stack: list[tuple[int, str]] = []      # (level, title)
    in_fence = False
    fence_marker: str | None = None

    # 逐行游标（保留原始偏移，不要用 splitlines 丢偏移）
    lines: list[tuple[int, str]] = []
    pos = 0
    for raw_line in markdown.split("\n"):
        lines.append((pos, raw_line))
        pos += len(raw_line) + 1

    block_start: int | None = None
    block_lines: list[str] = []

    def flush_block(end_offset: int) -> None:
        nonlocal block_start, block_lines
        if block_start is None or not block_lines:
            block_start = None
            block_lines = []
            return
        text = "\n".join(block_lines)
        if text.strip():
            nodes.append(StructureNode(
                kind=_classify_block(text),
                level=0,
                title="",
                char_start=block_start,
                char_end=max(block_start, end_offset),
                page_number=_page_for(block_start, marks),
                section_path=tuple(t for _, t in stack),
            ))
        block_start = None
        block_lines = []

    for offset, line in lines:
        if len(nodes) >= max_nodes:
            break

        marker = _fence_marker(line)
        if marker:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif fence_marker and line.strip().startswith(fence_marker):
                in_fence = False
                fence_marker = None
            if block_start is None:
                block_start = offset
            block_lines.append(line)
            continue

        if in_fence:
            block_lines.append(line)
            continue

        # ── 标题 ──────────────────────────────────────────────────────────────
        if line.lstrip().startswith("#"):
            stripped = line.lstrip()
            hashes = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[hashes:].strip()
            if 1 <= hashes <= 6 and title:
                flush_block(offset)
                while stack and stack[-1][0] >= hashes:
                    stack.pop()
                stack.append((hashes, title))
                nodes.append(StructureNode(
                    kind=NodeKind.HEADING,
                    level=hashes,
                    title=title,
                    char_start=offset,
                    char_end=offset + len(line),
                    page_number=_page_for(offset, marks),
                    section_path=tuple(t for _, t in stack),
                ))
                continue

        # ── 空行 = 块边界 ─────────────────────────────────────────────────────
        if not line.strip():
            flush_block(offset)
            continue

        if block_start is None:
            block_start = offset
        block_lines.append(line)

    flush_block(len(markdown))
    return nodes


def section_path_for_offset(
    offset: int,
    nodes: list[StructureNode],
) -> tuple[str, ...]:
    """
    给定字符偏移，返回其所属章节路径（取最后一个"起点 <= offset 的标题"的路径）.

    分块器用它给每个 chunk 打上"出自哪一节"；同父去重、元数据大纲也复用。
    """
    best: tuple[str, ...] = ()
    best_start = -1
    for node in nodes:
        if not node.is_heading:
            continue
        if node.char_start <= offset and node.char_start >= best_start:
            best = node.section_path
            best_start = node.char_start
    return best
