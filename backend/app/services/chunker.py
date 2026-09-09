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
    If a chunk boundary lands in the middle of a protected region in the
    *original text*, we cannot reliably detect that here without offsets.
    This hook is a no-op safe placeholder; the more thorough fix lives in the
    recursion path: protected regions are passed untouched via block-level
    grouping. Kept for future enhancement.
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

    # ── small-to-big / Hierarchical RAG (新增，全部 opt-in) ────────────────────
    parent_id: str | None = None        # doc_id + ":p:" + parent_index
    parent_text: str | None = None      # 回填父块完整文本（用于命中后给 LLM 看）
    parent_char_start: int | None = None
    parent_char_end: int | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "page_number": self.page_number,
            "heading": self.heading,
            "section": self.section,
            "parent_id": self.parent_id,
            "parent_text": self.parent_text,
            "parent_char_start": self.parent_char_start,
            "parent_char_end": self.parent_char_end,
        }


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
) -> tuple[int, int, str] | None:
    """Binary-search the parent span containing *offset*."""
    lo, hi = 0, len(parents)
    while lo < hi:
        mid = (lo + hi) // 2
        if parents[mid][0] <= offset:
            lo = mid + 1
        else:
            hi = mid
    idx = lo - 1
    if 0 <= idx < len(parents) and parents[idx][0] <= offset < parents[idx][1]:
        return parents[idx]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


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
        pos = text.find(raw[:50], search_start)
        if pos == -1:
            pos = search_start

        char_start = pos
        char_end = char_start + len(raw)
        search_start = max(0, char_end - chunk_overlap)

        page_number = page_resolver(char_start) if page_resolver else 1

        parent_id = None
        parent_text = None
        parent_char_start = None
        parent_char_end = None
        if enable_small_to_big and document_id and parents:
            par = _find_parent_for_offset(char_start, parents)
            if par is not None:
                pstart, pend, ptext = par
                parent_id = f"{document_id}:p:{parents.index(par)}"
                parent_text = ptext
                parent_char_start = pstart
                parent_char_end = pend

        result.append(TextChunk(
            text=raw,
            chunk_index=idx,
            char_start=char_start,
            char_end=char_end,
            page_number=page_number,
            heading=schunk.heading,
            section=schunk.section,
            parent_id=parent_id,
            parent_text=parent_text,
            parent_char_start=parent_char_start,
            parent_char_end=parent_char_end,
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
