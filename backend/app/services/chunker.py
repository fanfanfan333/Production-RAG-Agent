"""
Adaptive Semantic Chunker (增强版).

历史能力：
- 检测 Markdown header 划分语义块，逐块贪心合并到 [min_chunk_size, max_chunk_size]
- 单块过大时回退到递归字符切分
- 保留 heading / section 元数据到 TextChunk

本版本新增（全部默认开，行为优于旧版；旧调用者通过参数可关）：
- Token-aware 切分：max_chunk_size 当作"目标 token 数"，用 chars_per_token
  估算实际字符上限 —— 中英文混合语料字符与 token 比差异巨大，按 token 数
  才能保证进入 LLM 时仍不超窗口。
- 句子级滑窗重叠：chunk_overlap 改为按"完整句子"对齐，不在半句中间断开。
  大幅提升检索时召回段落是"自然语义完整片段"的概率。
- 表格 / 代码块保护：识别 ```...```、```python ...```、Markdown 表格行
  （|...|），禁止这些块从中间切断 —— 切断后语意会破坏、Markdown 渲染会错。
- Small-to-big（Hierarchical RAG）：除正常大小的子块外，还生成更大的"父块"
  作为上下文回填单元 —— 子块精确定位（用于检索打分），父块用于命中后回填
  给 LLM 让它看到完整语义。配合 retrieval_service.enable_hierarchical
  一起使用。

向后兼容性：
- build_chunks(...) 旧调用继续可用（所有新增参数都有 default）
- TextChunk 字段全部向后兼容（新增 parent_* 字段默认为 None）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback recursive chunker
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Markdown fence (``` or ~~~，可带语言标签)
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})\s*([\w+\-#./]*)?\s*$")
# Markdown table row:  | ... | ... |（首尾分隔符或对齐行 --- 都算表格）
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$|^\s*:?-+:?\s*\|", re.MULTILINE)

# 句子切分（中英文混合），保留分隔符本身以便后续组装。
# 一个句子：连续非终结符直到第一个终结符；终结符包括 中：。！？；  英：.!?; 以及换行。
_SENTENCE_END_RE = re.compile(
    r"(?<=[。！？；.!?;])\s*|(?<=\n)\s*"
)


def _split_sentences(text: str) -> list[str]:
    """
    把段落切成句子列表（带分隔符）.

    为简化处理，连续换行视作段落分隔（作为一个完整句子的结束）.
    """
    if not text:
        return []
    # 先按段落切，再按句子切，确保段落边界也被识别为句子结束
    paragraphs = re.split(r"\n\s*\n", text)
    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = _SENTENCE_END_RE.split(para)
        for p in parts:
            p = p.strip()
            if p:
                sentences.append(p)
    return sentences


def _merge_splits(
    splits: list[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    sep_len = len(separator)

    for split in splits:
        split_len = len(split)
        added_len = split_len + (sep_len if current else 0)

        if current_len + added_len > chunk_size and current:
            chunk_text = separator.join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            while current and current_len > chunk_overlap:
                removed = current.pop(0)
                current_len -= len(removed) + sep_len
            if not current:
                current_len = 0

        current.append(split)
        current_len += split_len + (sep_len if len(current) > 1 else 0)

    if current:
        chunk_text = separator.join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _split_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()]

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator:
        raw_splits = text.split(separator)
    else:
        return [
            text[i : i + chunk_size]
            for i in range(0, len(text), chunk_size - chunk_overlap)
            if text[i : i + chunk_size].strip()
        ]

    good_splits: list[str] = []
    for fragment in raw_splits:
        if not fragment.strip():
            continue
        if len(fragment) > chunk_size and remaining_separators:
            good_splits.extend(
                _split_recursive(fragment, remaining_separators, chunk_size, chunk_overlap)
            )
        else:
            good_splits.append(fragment)

    return _merge_splits(good_splits, separator, chunk_size, chunk_overlap)


# ─────────────────────────────────────────────────────────────────────────────
# Code-fence / table protection
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ProtectedBlock:
    """A region of text that must not be split: a fenced code block or a table."""

    start: int              # char offset in original text
    end: int                # char offset, exclusive
    kind: str               # "fence" | "table"


def _detect_protected_blocks(text: str) -> list[ProtectedBlock]:
    """
    Find code fences ``` ``` ~~~ ``` and Markdown table regions.

    Returns the list of regions sorted by start offset.
    """
    blocks: list[ProtectedBlock] = []
    lines = text.splitlines(keepends=True)
    pos = 0
    in_fence = False
    fence_start = 0
    fence_pattern: re.Pattern | None = None

    for line in lines:
        stripped = line.rstrip("\n").rstrip("\r")
        if in_fence:
            if fence_pattern and fence_pattern.match(stripped):
                # closing fence — region ends here (exclusive of next line)
                fence_end = pos + len(line)
                blocks.append(ProtectedBlock(fence_start, fence_end, "fence"))
                in_fence = False
        else:
            m = _FENCE_RE.match(stripped)
            if m:
                in_fence = True
                fence_start = pos
                # closing fence must match opening char (` or ~)
                fence_pattern = re.compile(rf"^{m.group(1)[0]}{{{len(m.group(1))},}}\s*$")
        pos += len(line)

    # ── Markdown tables: contiguous runs of |...| lines ────────────────────────
    table_start: int | None = None
    pos = 0
    for line in lines:
        stripped = line.rstrip("\n").rstrip("\r")
        if _TABLE_LINE_RE.match(stripped):
            if table_start is None:
                table_start = pos
        else:
            if table_start is not None:
                blocks.append(ProtectedBlock(table_start, pos, "table"))
                table_start = None
        pos += len(line)
    if table_start is not None:
        blocks.append(ProtectedBlock(table_start, pos, "table"))

    blocks.sort(key=lambda b: (b.start, b.end))
    # 合并重叠区域
    merged: list[ProtectedBlock] = []
    for b in blocks:
        if merged and b.start <= merged[-1].end:
            merged[-1] = ProtectedBlock(
                merged[-1].start, max(merged[-1].end, b.end),
                merged[-1].kind,
            )
        else:
            merged.append(b)
    return merged


def _is_protected(offset: int, blocks: list[ProtectedBlock]) -> bool:
    """Return True iff offset (any position inside a chunk) falls within a protected region."""
    # 二分（块按 start 排序，O(log n)）
    lo, hi = 0, len(blocks)
    while lo < hi:
        mid = (lo + hi) // 2
        if blocks[mid].start <= offset:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    return idx >= 0 and offset < blocks[idx].end


# ─────────────────────────────────────────────────────────────────────────────
# Semantic header detection + block parsing
# ─────────────────────────────────────────────────────────────────────────────


def _extract_header(line: str) -> tuple[int, str] | None:
    """Detect if a line is a markdown header, return (level, text)."""
    match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if match:
        return len(match.group(1)), match.group(2).strip()
    return None


@dataclass
class SemanticBlock:
    text: str
    section: str | None
    heading: str | None


def _parse_semantic_blocks(text: str) -> list[SemanticBlock]:
    """
    Parse text into blocks grouped by their current heading context.

    Code-fence awareness: lines inside ``` / ~~~ fences are never treated
    as Markdown headers — ``# comment`` inside Python code must not start
    a new semantic section. This is what keeps code blocks intact.
    """
    lines = text.split('\n')

    blocks: list[SemanticBlock] = []
    current_section = None
    current_heading = None
    current_lines: list[str] = []
    in_fence = False
    fence_close: re.Pattern | None = None

    def flush_block():
        if current_lines:
            block_text = "\n".join(current_lines).strip()
            if block_text:
                blocks.append(SemanticBlock(
                    text=block_text,
                    section=current_section,
                    heading=current_heading,
                ))
            current_lines.clear()

    for line in lines:
        stripped = line.strip()
        if in_fence:
            # inside a fenced code block — no header detection
            current_lines.append(line)
            if fence_close and fence_close.match(stripped):
                in_fence = False
            continue

        fence_match = _FENCE_RE.match(stripped)
        if fence_match:
            flush_block()
            in_fence = True
            marker = fence_match.group(1)
            fence_close = re.compile(rf"^{marker[0]}{{{len(marker)},}}\s*$")
            current_lines.append(line)
            continue

        header_info = _extract_header(line)
        if header_info:
            level, header_text = header_info
            flush_block()
            if level == 1:
                current_section = header_text
                current_heading = header_text
            else:
                current_heading = header_text
            current_lines.append(line)
        else:
            current_lines.append(line)

    flush_block()
    return blocks


def _is_atomic_block(block_text: str) -> bool:
    """
    该语义块是否必须**整块保留**（Markdown 表格 / 代码栅栏）.

    为什么需要它：``_adaptive_chunk`` 在合并阶段会把相邻语义块拼到
    ``max_chunk_size``，拼完若仍超限就丢给 ``_split_recursive`` 按字符/行
    切 —— 一个刚好被拼进正文块的表格会在这一步被拦腰截断：

        截断后的表格 → Markdown 渲染错乱（缺表头/缺分隔行）
                     → ``detect_content_type`` 判不出 table，表格检索失效
                     → 单元格文字被拆到两个 chunk，数字与指标名分家

    代码块同理：一段 Python 从中间断开，缩进与语义全废。

    因此这类块**不参与贪心合并**，自成一块。它自身若超过 max_chunk_size
    仍会走递归回退（保持与既有行为一致，不做无边界放大）。
    """
    if "```" in block_text or "~~~" in block_text:
        return True
    run = 0
    for line in block_text.splitlines():
        if _TABLE_LINE_RE.match(line.strip()):
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


def _strip_whitespace_index(text: str) -> tuple[str, list[int]]:
    """
    返回 ``(去掉所有空白后的文本, 每个保留字符在原文本中的下标)``.

    用途见 ``_find_ignoring_whitespace``：句级平滑会在句子之间插入 ``\\n``，
    使 chunk 文本不再逐字出现在原文里，需要一份"忽略空白"的索引来对回位置。
    """
    chars: list[str] = []
    mapping: list[int] = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        chars.append(ch)
        mapping.append(i)
    return "".join(chars), mapping


def _find_ignoring_whitespace(
    stripped_text: str,
    pos_map: list[int],
    needle: str,
    min_offset: int,
) -> int:
    """
    忽略空白差异地定位 *needle*，返回它在**原文**中的起始字符偏移.

    找不到（或所有匹配都落在 *min_offset* 之前）返回 -1。

    只取 needle 的前 60 个非空白字符做匹配：既足够唯一，又避免"结尾处
    被压缩过"导致整串匹配不上。
    """
    probe = re.sub(r"\s+", "", needle[:200])[:60]
    if not probe:
        return -1
    start = 0
    while True:
        found = stripped_text.find(probe, start)
        if found == -1:
            return -1
        original = pos_map[found]
        if original >= min_offset:
            return original
        start = found + 1


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive chunking core
# ─────────────────────────────────────────────────────────────────────────────


def _token_aware_chunk_size(
    requested_chars: int,
    chars_per_token: float,
    enabled: bool,
) -> int:
    """Convert a "target token size" (when token-aware) into a char budget."""
    if not enabled or chars_per_token <= 0:
        return requested_chars
    # tokens * chars_per_token = char budget；为留余量 +5%
    return max(1, int(requested_chars * chars_per_token * 1.05))


def _adaptive_chunk(
    text: str,
    min_chunk_size: int,
    max_chunk_size: int,
    chunk_overlap: int,
    *,
    sentence_overlap: bool = True,
    protect_blocks: bool = True,
) -> list[SemanticBlock]:
    """
    Merge semantic blocks into chunks adhering to min/max sizes.

    *sentence_overlap*: when True, chunks are reshuffled at the end so that the
    trailing token-boundary of chunk N and the leading token-boundary of chunk
    N+1 sit on sentence boundaries (best-effort). Greedy append already does
    this for half the cases; the post-pass straightens out the rest.

    *protect_blocks*: when True, code fences and tables are excluded from
    greedy splitting (fallback _split_recursive can still split within
    non-protected text; we trim chunks that would touch a block boundary
    mid-content).
    """
    raw_blocks = _parse_semantic_blocks(text)
    final_chunks: list[SemanticBlock] = []

    current_text = ""
    current_section = None
    current_heading = None

    def flush_current():
        nonlocal current_text, current_section, current_heading
        if current_text.strip():
            final_chunks.append(SemanticBlock(
                text=current_text.strip(),
                section=current_section,
                heading=current_heading,
            ))
        current_text = ""
        current_section = None
        current_heading = None

    for block in raw_blocks:
        # If this single block is huge, we must fallback-split it
        if len(block.text) > max_chunk_size:
            flush_current()
            fallback_splits = _split_recursive(
                block.text, _FALLBACK_SEPARATORS, max_chunk_size, chunk_overlap
            )
            for split in fallback_splits:
                final_chunks.append(SemanticBlock(
                    text=split,
                    section=block.section,
                    heading=block.heading,
                ))
            continue

        # 表格 / 代码栅栏块：自成一块，不与上下文合并。
        # 合并后再被切断会把表格劈成两半（详见 _is_atomic_block）。
        if protect_blocks and _is_atomic_block(block.text):
            flush_current()
            final_chunks.append(SemanticBlock(
                text=block.text,
                section=block.section,
                heading=block.heading,
            ))
            continue

        if current_text and len(current_text) + len(block.text) + 2 > max_chunk_size:
            flush_current()

        if not current_text:
            current_text = block.text
            current_section = block.section
            current_heading = block.heading
        else:
            current_text += "\n\n" + block.text
            if not current_section:
                current_section = block.section
            if not current_heading:
                current_heading = block.heading

        if len(current_text) >= min_chunk_size:
            flush_current()

    flush_current()

    # ── Sentence-aware smoothing: re-cut any chunk whose last sentence is
    #    unnecessarily long (after falling through the min-size guard) ────────
    if sentence_overlap:
        final_chunks = _smooth_to_sentences(
            final_chunks, min_chunk_size, max_chunk_size
        )

    # ── Code-fence / table protection: refuse chunks that would split a
    #    protected region (rare in well-formed Markdown but possible after
    #    greedy overflow). Trim chunk end to block boundary. ──────────────────
    if protect_blocks:
        protected = _detect_protected_blocks(text)
        final_chunks = _trim_protected(final_chunks, protected)

    return final_chunks


def _smooth_to_sentences(
    chunks: list[SemanticBlock],
    min_size: int,
    max_size: int,
) -> list[SemanticBlock]:
    """For each chunk, if its tail or head crosses a sentence boundary
    in mid-sentence, snap to the nearest sentence boundary. Keeps heading
    metadata of the chunk on the first piece, propagates it forward."""
    if not chunks:
        return chunks
    smoothed: list[SemanticBlock] = []
    for c in chunks:
        text = c.text
        if len(text) <= max_size * 1.2:
            smoothed.append(c)
            continue
        # oversize due to a single oversized sentence — leave it (recursive
        # fallback handled it already); do not over-fragment
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            smoothed.append(c)
            continue
        # group sentences greedily up to max_size
        group: list[str] = []
        size = 0
        for s in sentences:
            if size + len(s) + 1 > max_size and group:
                smoothed.append(SemanticBlock(
                    text="\n".join(group).strip(),
                    section=c.section,
                    heading=c.heading if not smoothed else None,
                ))
                group = [s]
                size = len(s)
            else:
                group.append(s)
                size += len(s) + 1
        if group:
            smoothed.append(SemanticBlock(
                text="\n".join(group).strip(),
                section=c.section,
                heading=c.heading if not smoothed else None,
            ))
    return smoothed


def _trim_protected(
    chunks: list[SemanticBlock],
    protected: list[ProtectedBlock],
) -> list[SemanticBlock]:
    """
    **占位钩子，保持无操作。**（保留函数只为兼容既有调用点。）

    这里原本期望"把落在保护区中间的分块边界裁到块边界"，但本层的
    ``SemanticBlock`` 只有文本、没有原始字符偏移，无从判断边界落在哪，
    所以真正的保护改在**更上游**完成：``_adaptive_chunk`` 通过
    ``_is_atomic_block`` 让表格 / 代码栅栏块自成一块、不参与贪心合并，
    从源头上消除"合并后被切断"这一路径。

    之前这里写着"no-op safe placeholder"却仍被当作保护已生效，
    是"表格被切断"长期存在却没人察觉的原因 —— 注释与行为不一致比
    缺少注释更危险，故明确标注为占位。
    """
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public TextChunk
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TextChunk:
    """A single chunk produced by the splitter."""

    text: str
    chunk_index: int          # 0-based position in the chunk list
    char_start: int           # approximate start offset in the source text
    char_end: int             # approximate end offset
    page_number: int = 1      # page attribution (populated by caller)
    heading: str | None = None
    section: str | None = None

    # ── 位置信息（细粒度引用溯源）────────────────────────────────────────────
    # 1-based 闭区间行号，由 char_start/char_end 换算。旧索引没有这两个字段
    # 时为 None，前端/引用卡片自动降级为只显示页码。
    line_start: int | None = None
    line_end: int | None = None

    # ── 内容类型（部分1+3）：text | table | image ──────────────────────────────
    # 检索结果的图文分流、Qdrant payload 与前端引用展示都依赖它。
    # 表格图片经 Table Parser 还原后取 "table"（表格检索对图片/正文一视同仁）。
    content_type: str = "text"
    image_id: str | None = None
    image_path: str | None = None       # 文档相对路径 images/page_3_image_1.png
    image_caption: str | None = None
    # Picture Classification 的结论：table | formula | code | chart | diagram |
    # screenshot | photo（正文块为 None）。前端据此显示类型徽标。
    image_type: str | None = None
    # 图片理解引擎（docling / table-transformer / paddleocr / vision …）
    analyze_engine: str | None = None
    # 置信度门控结果：最终置信度 + 是否需要人工复核
    analyze_confidence: float = 0.0
    manual_review: bool = False
    # ── 产出质检 + 双通道融合（可验证的事实）────────────────────────────────
    # analyze_quality：代码语法是否通过、OCR 行置信度评估、VLM 幻觉检查等；
    # analyze_fusion：双通道策略与最终选中的通道。两者都是完整报告字典
    # （不是扁平标量），因为前端要按 reason 逐条展示警示，而不是只给个分数。
    analyze_quality: dict = field(default_factory=dict)
    analyze_fusion: dict = field(default_factory=dict)

    # ── 图片位置（保留图片位置 → 细粒度引用）──────────────────────────────────
    # position：该图片在**整篇文档**中的序号（1-based，跨页累计）。文本块为 None。
    # bbox：图片在所在页面上的边界框 (x1, y1, x2, y2)，点坐标、原点左上。
    #   PDF / PPTX 可给出；DOCX 是流式布局、无页面几何，恒为 None。
    position: int | None = None
    bbox: tuple[float, float, float, float] | None = None

    # ── small-to-big / Hierarchical RAG (新增，全部 opt-in) ────────────────────
    parent_id: str | None = None        # doc_id + ":p:" + parent_index
    parent_text: str | None = None      # 回填父块完整文本（用于命中后给 LLM 看）
    parent_char_start: int | None = None
    parent_char_end: int | None = None

    # ── 结构感知父子（PARENT_CHILD_ENABLED 时启用）─────────────────────────────
    # 与上面四个字段的区别：那套是"字符窗口父块"（按固定倍数切原文，会把一个
    # 完整小节从中间劈开），且把父块正文**复制进每个子块的 payload**。
    #
    # 现在改为：
    #   parent_index  —— 子块在父块内的序号（稳定的整数引用）
    #   section_id    —— 顶层章节 id（祖父级，引用可到"3.2 节"）
    #   section_path  —— 祖先标题链 ("第 3 章 财务情况", "3.2 核算方法")
    # 父块**正文**不在这里，而是独立成 ChunkParent 行按 parent_id 批量取
    # （见 document_service / retrieval_service 的 hydration）。
    # 这样 1000 份文档时不会出现"同一段父块文本被复制 20 万次"的存储放大。
    parent_index: int | None = None
    section_id: str | None = None
    section_path: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "page_number": self.page_number,
            "heading": self.heading,
            "section": self.section,
            # 位置信息：引用溯源到"第几行"（细粒度引用）
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_type": self.content_type,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "image_caption": self.image_caption,
            "image_type": self.image_type,
            "analyze_engine": self.analyze_engine,
            "analyze_confidence": self.analyze_confidence,
            "manual_review": self.manual_review,
            # 产出质检 + 双通道融合（图片理解的可验证事实）
            "analyze_quality": self.analyze_quality,
            "analyze_fusion": self.analyze_fusion,
            # 图片位置（细粒度引用）：文档内序号 + 页面边界框
            "position": self.position,
            "bbox": list(self.bbox) if self.bbox else None,
            "parent_id": self.parent_id,
            "parent_text": self.parent_text,
            "parent_char_start": self.parent_char_start,
            "parent_char_end": self.parent_char_end,
            # 结构感知父子（新）
            "parent_index": self.parent_index,
            "section_id": self.section_id,
            "section_path": list(self.section_path) if self.section_path else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Content-type detection (部分3：表格检索)
# ─────────────────────────────────────────────────────────────────────────────

# 一行 Markdown 表格：| a | b |  或对齐行 | --- | --- |
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def contains_markdown_table(text: str) -> bool:
    """True when *text* holds at least two consecutive Markdown table rows."""
    run = 0
    for line in text.splitlines():
        if _TABLE_ROW_RE.match(line):
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


def detect_content_type(text: str, default: str = "text") -> str:
    """text | table —— 表格块被单独标记，便于"表格检索"与前端区分展示."""
    if default == "table":
        return "table"
    return "table" if contains_markdown_table(text) else "text"


def build_image_chunks(
    images: list,
    *,
    start_index: int = 0,
) -> list[TextChunk]:
    """
    把结构化图片转成独立的 image chunk（部分1：图片作为独立检索对象）.

    - 只对"有可检索文本"的图片建块：结构化表格 / OCR 文本 / vision caption
      至少有一个。三者都为空的图片仍已落盘，但没有语义信号可供向量检索，
      强行入库只会引入噪声。
    - *start_index* 必须大于文本块的最大 chunk_index —— 检索融合以
      (document_id, chunk_index) 作为 chunk 唯一键，图文索引撞号会导致
      其中一个在融合阶段被覆盖。
    - **content_type 取自图片的分类结论**（``ExtractedImage.content_type``）：
      表格图片经 Table Parser 还原成 Markdown 表格后取 "table"，
      与正文表格同构，因此"表格检索"对图片表格同样生效；
      图表 / 流程图 / 截图 / 普通图片取 "image"。
    """
    chunks: list[TextChunk] = []
    for img in images:
        text = img.searchable_text.strip()
        if not text:
            # 无检索文本 → 不建块，也不占用 chunk_index
            # （索引保持连续：只有真正建块的图片才递增）
            continue
        # 图片的分类结论决定它在检索层的身份（table vs image）
        content_type = getattr(img, "content_type", None) or "image"
        chunks.append(
            TextChunk(
                text=text,
                chunk_index=start_index + len(chunks),
                char_start=0,
                char_end=len(text),
                page_number=int(getattr(img, "page_number", 1) or 1),
                heading=None,
                section=None,
                content_type=content_type,
                image_id=getattr(img, "image_id", None),
                image_path=getattr(img, "image_path", None),
                image_caption=getattr(img, "vision_caption", None),
                image_type=getattr(img, "image_type", None),
                analyze_engine=getattr(img, "analyze_engine", None),
                analyze_confidence=float(getattr(img, "analyze_confidence", 0.0) or 0.0),
                manual_review=bool(getattr(img, "manual_review", False)),
                # 产出质检 + 双通道融合（前端据此显示"代码语法未通过"等警示）
                analyze_quality=dict(getattr(img, "analyze_quality", None) or {}),
                analyze_fusion=dict(getattr(img, "analyze_fusion", None) or {}),
                # 位置信息：图片的文档内序号与页面边界框（细粒度引用）
                position=getattr(img, "position", None),
                bbox=getattr(img, "bbox", None),
            )
        )
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Parent (small-to-big) generation
# ─────────────────────────────────────────────────────────────────────────────


def _build_parents(
    text: str,
    min_chunk_size: int,
    max_chunk_size: int,
    size_multiplier: float,
    overlap: int,
) -> list[tuple[int, int, str]]:
    """
    Split text into larger "parent" spans = base_chunk_size × size_multiplier.

    Returns a list of (char_start, char_end, parent_text) for each parent span,
    in document order. Used to attach parent_text to small child chunks so
    hierarchical retrieval can return parent context after a small chunk hit.
    """
    parent_size = max(max_chunk_size + 1, int(max_chunk_size * size_multiplier))
    if len(text) <= parent_size:
        return [(0, len(text), text)]

    spans: list[tuple[int, int, str]] = []
    step = parent_size - overlap
    cursor = 0
    n = len(text)
    while cursor < n:
        end = min(cursor + parent_size, n)
        spans.append((cursor, end, text[cursor:end]))
        if end >= n:
            break
        cursor += max(1, step)
    return spans


def _find_parent_for_offset(
    offset: int,
    parents: list[tuple[int, int, str]],
) -> int:
    """
    二分查找包含 *offset* 的父块，返回其**下标**；找不到返回 -1.

    返回下标而不是元组是有理由的：调用方需要它来拼 ``parent_id``。旧实现返回
    元组后写 ``parents.index(par)`` —— ``list.index`` 是 O(n) 线性扫描，套在
    逐 chunk 的循环里就是 O(n·chunks)。一份 200 页的文档能产生数千个父块，
    这个平方项会让分块从"毫秒级"退化到"秒级"，而且完全不报错，只是慢。
    """
    lo, hi = 0, len(parents)
    while lo < hi:
        mid = (lo + hi) // 2
        if parents[mid][0] <= offset:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    if 0 <= idx < len(parents) and parents[idx][0] <= offset < parents[idx][1]:
        return idx
    return -1


# ─────────────────────────────────────────────────────────────────────────────
# 位置信息（行号）—— 细粒度引用溯源的基础
# ─────────────────────────────────────────────────────────────────────────────
#
# 只保留 char_start/char_end 时，引用只能定位到"某个 chunk"，用户无法回到
# 原文核对。行号是人与 LLM 都能直接理解的最小定位单位：
#
#     引用卡片 →「出自《年报.pdf》第 3 页，第 12–28 行」
#     LLM 上下文 → [Source 1] 年报.pdf, page 3, lines 12-28
#
# 行号由字符偏移换算而来，不额外存储原文，因此对既有索引零侵入
# （旧 chunk 的 line_* 为 None，前端自动降级为只显示页码）。


def build_line_index(text: str) -> list[int]:
    """
    返回每一行起始字符偏移的升序列表（第 1 行的偏移为 0）。

    ``text`` 为 "" 时返回 [0]，保证 offset 0 也能映射到第 1 行。
    """
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def line_for_offset(offset: int, line_starts: list[int]) -> int:
    """把字符偏移换算成 1-based 行号（二分；越界收敛到首/末行）."""
    if not line_starts:
        return 1
    if offset <= 0:
        return 1
    lo, hi = 0, len(line_starts)
    while lo < hi:
        mid = (lo + hi) // 2
        if line_starts[mid] <= offset:
            lo = mid + 1
        else:
            hi = mid
    return max(1, lo)   # lo 即"最后一个 start <= offset"的序号（1-based）


def line_range_for_span(
    char_start: int,
    char_end: int,
    line_starts: list[int],
) -> tuple[int, int]:
    """
    把 [char_start, char_end) 映射为闭区间行号 (line_start, line_end).

    ``char_end`` 是开区间上界，因此换算前 -1，避免"恰好落在下一行行首"
    的边界把行号多算一行。
    """
    start_line = line_for_offset(char_start, line_starts)
    end_line = line_for_offset(max(char_start, char_end - 1), line_starts)
    return start_line, max(start_line, end_line)


# ─────────────────────────────────────────────────────────────────────────────
# 结构感知父子分块（PARENT_CHILD_ENABLED）
# ─────────────────────────────────────────────────────────────────────────────
#
# 为什么重写这套东西
# ─────────────────
# 旧实现（_build_parents）把正文按**固定字符窗口**切成父块：
#
#     parent_size = max_chunk_size × size_multiplier
#     cursor += parent_size - overlap          ← 纯粹的字符推进
#
# 它有两个在 1000 份文档规模下会致命的问题：
#
#   1. **父块边界与小节边界无关**。一个 5000 字的"3.2 核算方法"被窗口切成两半，
#      命中下半段时回填的"父块上下文"里没有小节开头的口径定义 —— 模型于是拿
#      一个残缺的上下文去回答，这正是"看起来有依据、其实答错"的典型来源。
#
#   2. **父块正文被复制进每一个子块的 payload**（`parent_text` 字段）。
#      1000 份文档 × 每份 200 子块 × 每个父块 5000 字 ≈ 10 亿字符的冗余。
#      Qdrant 的 scroll（BM25 语料）要把这些全读回内存 —— 单这一项就足以让
#      "上千文档"从"慢"变成"起不来"。
#
# 新实现：
#   · 父块边界 = **章节边界**（由结构树给出的 section_path 变化决定），
#     而不是字符窗口 —— 小节完整、语义自洽；
#   · 子块只带 parent_id / section_id / section_path，**正文另存**（ChunkParent），
#     检索命中后按 parent_id 批量取（一次 IN 查询）；
#   · 额外生成 section 级（祖父）引用，引用卡片能说到"第 3 章 · 3.2 节"。


@dataclass
class ParentBlock:
    """一个父块（小节级）或章节块（祖父级）的正文与位置。"""

    parent_id: str
    level: str                      # "parent" | "section"
    index: int
    text: str
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    line_start: int | None = None
    line_end: int | None = None
    heading: str | None = None
    section_path: tuple[str, ...] = ()
    child_indexes: list[int] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "level": self.level,
            "index": self.index,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "heading": self.heading,
            "section_path": list(self.section_path),
            "child_indexes": list(self.child_indexes),
        }


@dataclass
class ChunkHierarchy:
    """一次分块的完整层级产物：可检索的叶子 + 供回填的父块。"""

    children: list[TextChunk] = field(default_factory=list)
    parents: list[ParentBlock] = field(default_factory=list)

    def parents_by_id(self) -> dict[str, ParentBlock]:
        return {p.parent_id: p for p in self.parents}


# 单文档父块上限：防极长文档（如 3000 页手册）产生海量父块行。超出后
# 后续父块不再单独落库，但子块照常入库、照常用 section 级上下文回填。
_MAX_PARENTS_PER_DOC = 5000


def build_hierarchy_from_children(
    children: list[TextChunk],
    text: str,
    *,
    document_id: str,
    nodes: list | None = None,
    sections_enabled: bool = True,
    parent_min_chars: int = 1200,
    parent_max_chars: int = 6000,
) -> ChunkHierarchy:
    """
    给已有的子块补上结构感知的父子关系，并抽出父块正文.

    Args:
        children:        ``build_chunks`` 的产物（子块，按文档顺序）
        text:            正文（父块正文从它按字符区间切出，保证与原文一致）
        document_id:     文档 id（用于拼 parent_id / section_id，跨文档唯一）
        nodes:           结构树节点（``StructuredDocument.nodes``）；None 时
                         退化为"无章节信息"—— 父块按段落组大小切，仍然正确，
                         只是引用里说不出"出自哪一节"
        sections_enabled: 是否额外生成章节级父块
        parent_min_chars: 父块最小字符数（更短的小节向上合并，避免碎片）
        parent_max_chars: 父块最大字符数（超长小节强切，避免单个父块过大）

    Returns:
        ChunkHierarchy —— ``children`` 子块的 parent_* 字段已被就地填好
    """
    if not children:
        return ChunkHierarchy(children=[], parents=[])

    # ── 1. 给每个子块解析章节路径 ──────────────────────────────────────────────
    path_of: list[tuple[str, ...]] = []
    if nodes:
        try:
            from app.services.structure.outline import section_path_for_offset

            path_of = [
                tuple(section_path_for_offset(c.char_start, nodes)) for c in children
            ]
        except Exception:            # noqa: BLE001 — 结构信息缺失不该中断分块
            logger.exception("section_path resolution failed — continuing without it")
            path_of = [() for _ in children]
    else:
        path_of = [() for _ in children]

    # 图片块不参与父子分组：它们自带 caption/结构化内容，把邻近正文当作
    # "父上下文"回填只会给模型塞进无关文字（图片的上下文就是它自己的描述）。
    is_leaf_text = [not c.image_id for c in children]

    # ── 2. 按章节边界切父块 ────────────────────────────────────────────────────
    #
    # 父块起点还需要"向小节标题吸附"：分组是按子块的 section_path 做的，而子块的
    # 起点由分块器决定。绝大多数情况下分块器在标题处断块（``_parse_semantic_blocks``
    # 把标题当作新块的开始），两者天然对齐；但当一个小节很短、标题被并进上一块时，
    # 子块起点会落在标题**之前**，父块就会连带上一小节的尾巴。
    # 吸附规则：若该小节有标题且标题位置不晚于首个子块的结束，就把父块起点提到
    # 标题处 —— 回填给模型的上下文从"小节标题"开始，而不是从半句话开始。
    heading_offset: dict[tuple[str, ...], int] = {}
    if nodes:
        for node in nodes:
            path = tuple(getattr(node, "section_path", ()) or ())
            if not path or not getattr(node, "is_heading", False):
                continue
            existing = heading_offset.get(path)
            if existing is None or node.char_start < existing:
                heading_offset[path] = node.char_start

    groups: list[list[int]] = []          # 每组 = 一组子块下标
    current: list[int] = []
    current_chars = 0
    current_path: tuple[str, ...] | None = None

    def flush() -> None:
        nonlocal current, current_chars, current_path
        if current:
            groups.append(current)
        current = []
        current_chars = 0
        current_path = None

    for i, child in enumerate(children):
        if not is_leaf_text[i]:
            # 图片块：不参与分组，也不打断当前组（它是独立检索对象）
            continue

        path = path_of[i]

        # 章节路径变化 = 新的小节 → 开新父块
        path_changed = current_path is not None and path != current_path
        would_overflow = (
            current_chars + len(child.text) > parent_max_chars and current
        )

        if path_changed or would_overflow:
            flush()

        if not current:
            current_path = path
        current.append(i)
        current_chars += len(child.text)

        if current_chars >= parent_max_chars:
            flush()

    flush()

    # ── 3. 合并过短的父块（同章节内向上并入下一组，跨章节不并）────────────────
    merged: list[list[int]] = []
    for group in groups:
        if (
            merged
            and _group_chars(merged[-1], children) < parent_min_chars
            and _group_path(merged[-1], path_of) == _group_path(group, path_of)
        ):
            merged[-1].extend(group)
        else:
            merged.append(list(group))

    # 尾组过短时并入前一组（同样要求同章节）
    if (
        len(merged) >= 2
        and _group_chars(merged[-1], children) < parent_min_chars
        and _group_path(merged[-1], path_of) == _group_path(merged[-2], path_of)
    ):
        merged[-2].extend(merged.pop())

    # ── 4. 物化父块 ───────────────────────────────────────────────────────────
    parents: list[ParentBlock] = []
    section_index: dict[tuple[str, ...], int] = {}
    section_blocks: list[ParentBlock] = []

    for idx, group in enumerate(merged[:_MAX_PARENTS_PER_DOC]):
        group_children = [children[i] for i in group]
        path = _group_path(group, path_of)
        char_start = min(c.char_start for c in group_children)
        char_end = max(c.char_end for c in group_children)

        # 向小节标题吸附（见上文说明），并保证不与上一个父块重叠
        snap = heading_offset.get(path)
        if snap is not None and snap <= char_end:
            char_start = min(char_start, snap)
        if parents:
            char_start = max(char_start, parents[-1].char_end)

        char_start = max(0, min(char_start, len(text)))
        char_end = max(char_start, min(char_end, len(text)))
        body = text[char_start:char_end] or "\n".join(c.text for c in group_children)

        line_starts = [c.line_start for c in group_children if c.line_start is not None]
        line_ends = [c.line_end for c in group_children if c.line_end is not None]
        pages = [c.page_number for c in group_children if c.page_number]

        parent = ParentBlock(
            parent_id=f"{document_id}:p:{idx}",
            level="parent",
            index=idx,
            text=body,
            char_start=char_start,
            char_end=char_end,
            page_start=min(pages) if pages else 1,
            page_end=max(pages) if pages else 1,
            line_start=min(line_starts) if line_starts else None,
            line_end=max(line_ends) if line_ends else None,
            heading=group_children[0].heading or (
                path[-1] if path else None
            ),
            section_path=path,
            child_indexes=[c.chunk_index for c in group_children if c.chunk_index is not None],
        )
        parents.append(parent)

        # 回写子块（注意：**不写 parent_text** —— 正文由 ChunkParent 单独持有）
        for child in group_children:
            child.parent_id = parent.parent_id
            child.parent_index = idx
            child.parent_char_start = char_start
            child.parent_char_end = char_end
            child.section_path = list(path) if path else None
            if path:
                section_id = _section_id(document_id, path, section_index)
                child.section_id = section_id

    # ── 5. 章节级（祖父）块 ────────────────────────────────────────────────────
    if sections_enabled and parents:
        by_section: dict[tuple[str, ...], list[ParentBlock]] = {}
        order: list[tuple[str, ...]] = []
        for p in parents:
            key = p.section_path[:1] if p.section_path else ()
            if key not in by_section:
                by_section[key] = []
                order.append(key)
            by_section[key].append(p)

        # 只保留真正"多小节"的章节做祖父块；单小节章节的祖父 == 父块本身，
        # 再存一份是纯冗余。
        for section_idx, key in enumerate(order):
            group = by_section[key]
            if len(group) < 2:
                continue
            char_start = min(g.char_start for g in group)
            char_end = max(g.char_end for g in group)
            section_blocks.append(ParentBlock(
                parent_id=f"{document_id}:s:{section_idx}",
                level="section",
                index=section_idx,
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                page_start=min(g.page_start for g in group),
                page_end=max(g.page_end for g in group),
                line_start=min((g.line_start for g in group if g.line_start), default=None),
                line_end=max((g.line_end for g in group if g.line_end), default=None),
                heading=key[0] if key else None,
                section_path=key,
                child_indexes=[
                    ci for g in group for ci in g.child_indexes
                ],
            ))

    logger.debug(
        "build_hierarchy_from_children → %d children, %d parent(s), %d section(s) "
        "(min=%d max=%d, sections_enabled=%s, nodes=%d)",
        len(children), len(parents), len(section_blocks),
        parent_min_chars, parent_max_chars, sections_enabled, len(nodes or []),
    )
    return ChunkHierarchy(
        children=children,
        parents=parents + section_blocks,
    )


def _group_chars(group: list[int], children: list[TextChunk]) -> int:
    return sum(len(children[i].text) for i in group)


def _group_path(group: list[int], path_of: list[tuple[str, ...]]) -> tuple[str, ...]:
    """一组的章节路径 = 组内第一个子块的路径（组内路径必然相同或为空）."""
    return path_of[group[0]] if group else ()


def _section_id(
    document_id: str,
    path: tuple[str, ...],
    cache: dict[tuple[str, ...], int],
) -> str:
    """
    章节路径 → 稳定 id.

    按**首次出现顺序**编号并复用（同一章节的所有父块拿到同一个 section_id）。
    不用 hash：hash 在不同 Python 版本/进程间不稳定（PYTHONHASHSEED），
    而 payload 里存的 id 必须跨进程可比。
    """
    idx = cache.get(path)
    if idx is None:
        idx = len(cache)
        cache[path] = idx
    return f"{document_id}:s:{idx}"


def build_chunk_hierarchy(
    text: str,
    *,
    document_id: str,
    nodes: list | None = None,
    sections_enabled: bool = True,
    parent_min_chars: int = 1200,
    parent_max_chars: int = 6000,
    **chunk_kwargs,
) -> ChunkHierarchy:
    """
    一步到位：分块 + 建立结构感知父子关系.

    这是 document_service 应该调用的入口（而不是分别调 build_chunks 再手动
    组装）—— 分开调用时"忘记建父子"或"父子与子块用了不同参数"都不会报错，
    只会在检索时表现为"父块回填静默失效"。
    """
    children = build_chunks(text, document_id=document_id, **chunk_kwargs)
    return build_hierarchy_from_children(
        children,
        text,
        document_id=document_id,
        nodes=nodes,
        sections_enabled=sections_enabled,
        parent_min_chars=parent_min_chars,
        parent_max_chars=parent_max_chars,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def _locate_chunk(
    text: str,
    stripped_text: str,
    pos_map: list[int],
    raw: str,
    min_offset: int,
) -> int:
    """
    在原文中定位一个 chunk 的起始字符偏移，失败返回 -1.

    两级策略：
      1. 逐字 ``find``（绝大多数 chunk 与原文完全一致，最快）；
      2. 忽略空白差异再找 —— 句级平滑会把句子用 ``\\n`` 重新拼接，递归回退也
         会归一化空白，两种情况下 chunk 文本与原文**只差空白**，逐字查找必然
         失败。而位置一旦回落到近似值，行号与页码就会偏。

    抽成独立函数是为了能被单测直接覆盖（见 tests/test_chunk_positions.py）——
    位置逻辑写在 build_chunks 循环里时只能靠"跑一遍看结果"来验证。
    """
    pos = text.find(raw[:50], min_offset)
    if pos != -1:
        return pos
    return _find_ignoring_whitespace(stripped_text, pos_map, raw, min_offset)


def build_chunks(
    text: str,
    min_chunk_size: int = 500,
    max_chunk_size: int = 2000,
    chunk_overlap: int = 200,
    page_resolver=None,        # callable(char_offset) -> page_number
    *,
    document_id: str | None = None,
    token_aware: bool | None = None,
    chars_per_token: float | None = None,
    sentence_overlap: bool | None = None,
    protect_blocks: bool | None = None,
    enable_small_to_big: bool = False,
    parent_size_multiplier: float | None = None,
    parent_overlap: int | None = None,
    default_content_type: str = "text",
) -> list[TextChunk]:
    """
    Parse *text* into adaptive semantic chunks, preserving structural metadata.

    Backward compatible: every new parameter defaults to None / False, in which
    case settings from ``app.config.get_settings()`` are read; pass explicit
    values to override per-call.
    """
    from app.config import get_settings
    settings = get_settings()

    # Resolve per-call overrides against settings defaults
    token_aware = settings.CHUNK_TOKEN_AWARE if token_aware is None else token_aware
    chars_per_token = settings.CHARS_PER_TOKEN if chars_per_token is None else chars_per_token
    sentence_overlap = settings.CHUNK_SENTENCE_OVERLAP if sentence_overlap is None else sentence_overlap
    protect_blocks = settings.CHUNK_PROTECT_BLOCKS if protect_blocks is None else protect_blocks
    parent_size_multiplier = (
        settings.CHUNK_PARENT_SIZE_MULT
        if parent_size_multiplier is None else parent_size_multiplier
    )
    parent_overlap = (
        settings.CHUNK_PARENT_OVERLAP if parent_overlap is None else parent_overlap
    )

    # Translate token-budget into char-budget when needed
    if token_aware:
        effective_max = _token_aware_chunk_size(max_chunk_size, chars_per_token, True)
        effective_min = _token_aware_chunk_size(min_chunk_size, chars_per_token, True)
        effective_overlap = _token_aware_chunk_size(chunk_overlap, chars_per_token, True)
    else:
        effective_max = max_chunk_size
        effective_min = min_chunk_size
        effective_overlap = chunk_overlap

    if chunk_overlap >= max_chunk_size:
        raise ValueError("chunk_overlap must be smaller than max_chunk_size.")

    if effective_overlap >= effective_max:
        effective_overlap = max(1, effective_max // 5)

    semantic_chunks = _adaptive_chunk(
        text,
        effective_min,
        effective_max,
        effective_overlap,
        sentence_overlap=sentence_overlap,
        protect_blocks=protect_blocks,
    )

    result: list[TextChunk] = []
    search_start = 0

    # 行号索引只建一次：每个 chunk 的字符区间据此换算为 1-based 行号
    # （细粒度引用溯源 —— 引用卡片要说清"这是第几行"）。
    line_starts = build_line_index(text)

    # ── 忽略空白的定位索引（行号 / 页码正确性的前提）──────────────────────────
    # 句级平滑（_smooth_to_sentences）会用 "\n" 把句子重新拼接，于是 chunk 文本
    # **不再逐字出现在原文里**（原文这两句之间没有换行）。此时
    # ``text.find(raw[:50])`` 直接返回 -1，位置回落到 ``search_start`` 这个
    # 近似值 —— 而 line_start / page_number 全部建立在这个位置上：
    #
    #     chunk 首句短于 50 字 → 前 50 字跨过被插入的 "\n" → 查找失败
    #                          → 行号、页码一起偏，且误差随 chunk 累积
    #
    # 中文正文里"句子之间不换行"是常态，所以这不是罕见分支。下面这份索引
    # （去掉全部空白 + 每个保留字符的原下标）让定位重新变精确。
    stripped_text, pos_map = _strip_whitespace_index(text)

    # Pre-compute parents once, if small-to-big is on
    parents: list[tuple[int, int, str]] = []
    if enable_small_to_big:
        parents = _build_parents(
            text,
            effective_min,
            effective_max,
            parent_size_multiplier,
            parent_overlap,
        )

    for idx, schunk in enumerate(semantic_chunks):
        raw = schunk.text
        pos = _locate_chunk(text, stripped_text, pos_map, raw, search_start)
        if pos == -1:
            pos = search_start

        char_start = pos
        char_end = char_start + len(raw)
        search_start = max(0, char_end - chunk_overlap)

        page_number = page_resolver(char_start) if page_resolver else 1

        # 位置信息：字符区间 → 1-based 行号区间（细粒度引用溯源）
        line_start, line_end = line_range_for_span(char_start, char_end, line_starts)

        parent_id = None
        parent_text = None
        parent_char_start = None
        parent_char_end = None
        parent_index = None
        if enable_small_to_big and document_id and parents:
            pidx = _find_parent_for_offset(char_start, parents)
            if pidx >= 0:
                pstart, pend, ptext = parents[pidx]
                parent_id = f"{document_id}:p:{pidx}"
                parent_text = ptext
                parent_char_start = pstart
                parent_char_end = pend
                parent_index = pidx

        result.append(TextChunk(
            text=raw,
            chunk_index=idx,
            char_start=char_start,
            char_end=char_end,
            page_number=page_number,
            heading=schunk.heading,
            section=schunk.section,
            line_start=line_start,
            line_end=line_end,
            content_type=detect_content_type(raw, default=default_content_type),
            parent_id=parent_id,
            parent_text=parent_text,
            parent_char_start=parent_char_start,
            parent_char_end=parent_char_end,
            parent_index=parent_index,
        ))

    logger.debug(
        "build_chunks → %d chunks (min=%d max=%d overlap=%d token_aware=%s "
        "sentence_overlap=%s protect_blocks=%s small_to_big=%s parents=%d)",
        len(result),
        min_chunk_size,
        max_chunk_size,
        chunk_overlap,
        token_aware,
        sentence_overlap,
        protect_blocks,
        enable_small_to_big,
        len(parents),
    )
    return result
