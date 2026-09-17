"""
分块位置定位单测（细粒度引用的地基）.

为什么值得单独测
────────────────
引用卡片上的"第 12-28 行""第 3 页"全部由 chunk 的字符区间换算而来：

    char_start  →  line_range_for_span(...)  →  line_start / line_end
                →  page_resolver(char_start) →  page_number

而 ``char_start`` 是在**原文里定位 chunk 文本**得到的。句级平滑
（``_smooth_to_sentences``）会先用 "\\n" 把句子重新拼接，于是 chunk 文本不再
逐字出现在原文中（中文正文的句子之间通常没有换行）—— 逐字 ``find`` 必然失败。

旧实现失败后直接回落到 ``search_start``（上一块的结束位置附近）这个**近似值**，
于是行号与页码一起偏，且误差随 chunk 累积。本用例锁定"必须精确定位"这一性质：
命中位置处的原文去掉空白后，必须与 chunk 开头逐字一致。

纯标准库；通过 importlib 按文件路径加载，宿主机可直接运行：
    python backend/tests/test_chunk_positions.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "app" / "services" / "chunker.py"
)
_spec = importlib.util.spec_from_file_location("app.services.chunker", _MODULE_PATH)
_chunker = importlib.util.module_from_spec(_spec)
sys.modules["app.services.chunker"] = _chunker
_spec.loader.exec_module(_chunker)

SemanticBlock = _chunker.SemanticBlock
_adaptive_chunk = _chunker._adaptive_chunk
_find_ignoring_whitespace = _chunker._find_ignoring_whitespace
_locate_chunk = _chunker._locate_chunk
_smooth_to_sentences = _chunker._smooth_to_sentences
_strip_whitespace_index = _chunker._strip_whitespace_index
build_line_index = _chunker.build_line_index
line_range_for_span = _chunker.line_range_for_span

_WS = re.compile(r"\s+")


def _nows(value: str) -> str:
    return _WS.sub("", value)


# ── 1. 定位器本体 ────────────────────────────────────────────────────────────

def test_locate_chunk_exact_hit() -> None:
    """与原文完全一致的 chunk 走逐字查找（快路径）."""
    text = "第一行内容\n第二行内容\n第三行内容"
    stripped, pos_map = _strip_whitespace_index(text)
    # "第一行内容" 占 0-4，下标 5 是换行，所以第二行从 6 开始
    assert _locate_chunk(text, stripped, pos_map, "第二行内容", 0) == 6
    # 起点之后的第一个匹配，不能被更早的同名片段抢走
    # "A\n目标\nB\n目标" → 第二个"目标"从下标 7 开始
    text2 = "A\n目标\nB\n目标"
    stripped2, pos_map2 = _strip_whitespace_index(text2)
    assert _locate_chunk(text2, stripped2, pos_map2, "目标", 3) == 7
    print("  ok test_locate_chunk_exact_hit")


def test_locate_chunk_ignores_whitespace_differences() -> None:
    """chunk 与原文只差空白时，必须仍然精确定位（而不是回落到近似值）."""
    original = "第一句话。第二句话。第三句话。"
    smoothed = "第一句话。\n第二句话。\n第三句话。"
    stripped, pos_map = _strip_whitespace_index(original)

    assert original.find(smoothed[:50], 0) == -1, "前提：逐字查找应当失败"
    pos = _locate_chunk(original, stripped, pos_map, smoothed, 0)
    assert pos == 0, f"应定位到原文起点，实际 {pos}"
    assert _nows(original[pos:pos + 20]) == _nows(smoothed[:20])
    print("  ok test_locate_chunk_ignores_whitespace_differences")


def test_locate_chunk_respects_min_offset() -> None:
    """同一段文本在文档里重复出现时，不能定位到 min_offset 之前的那一次."""
    text = "重复句子。\n中间内容\n重复句子。"
    stripped, pos_map = _strip_whitespace_index(text)
    first = _locate_chunk(text, stripped, pos_map, "重复句子。", 0)
    second = _locate_chunk(text, stripped, pos_map, "重复句子。", first + 1)
    assert first == 0
    assert second == 11, f"第二次应定位到后一处，实际 {second}"
    print("  ok test_locate_chunk_respects_min_offset")


def test_locate_chunk_returns_minus_one_when_absent() -> None:
    original = "只有这一段内容。"
    stripped, pos_map = _strip_whitespace_index(original)
    assert _locate_chunk(original, stripped, pos_map, "完全不存在的句子。", 0) == -1
    print("  ok test_locate_chunk_returns_minus_one_when_absent")


# ── 2. 真实平滑路径：位置不漂移 ──────────────────────────────────────────────

def test_smoothed_chunks_are_located_exactly() -> None:
    """
    从句级平滑的产物出发，逐个 chunk 验证定位精确 —— 这正是旧实现出错的地方.

    夹具刻意构造成：首句短于 50 字，于是 chunk 的前 50 字会跨过平滑插入的
    "\\n"，逐字查找失败。
    """
    s1 = "第一句话只有二十来个字。"
    s2 = "第二句话同样不长但足以把前五十个字符撑过句子边界。"
    s3 = "第三句作为收尾也要写得足够长一点才好。"
    original = s1 + s2 + s3
    # max_size 取到"刚好能装下前两句"（平滑内部按 len(s)+1 累计，故 +2），
    # 于是平滑一定会把 s1/s2 拼成一块、并与 s3 分开
    max_size = len(s1) + len(s2) + 2

    assert len(s1) < 50, "夹具前提：首句必须短于 50 字"
    assert len(original) > max_size * 1.2, "夹具前提：必须触发平滑（超 max*1.2）"

    smoothed = _smooth_to_sentences(
        [SemanticBlock(text=original, section="S", heading="H")], 1, max_size
    )
    cross_sentence = [c for c in smoothed if "\n" in c.text[:50]]
    assert cross_sentence, "夹具必须至少产出一个'跨句'chunk，否则测不到问题"

    stripped, pos_map = _strip_whitespace_index(original)
    line_starts = build_line_index(original)

    search_start = 0
    for chunk in smoothed:
        pos = _locate_chunk(original, stripped, pos_map, chunk.text, search_start)
        assert pos != -1, f"chunk 必须能定位：{chunk.text[:40]!r}"

        # 核心断言：定位处去掉空白后，必须与 chunk 开头逐字一致
        window = _nows(original[pos:pos + len(chunk.text) + 20])
        assert window.startswith(_nows(chunk.text)[:40]), (
            f"定位漂移：chunk 开头 {_nows(chunk.text)[:40]!r} "
            f"≠ 原文定位处 {window[:40]!r}"
        )

        # 行号必须真的框住这个 chunk 的首行
        line_start, line_end = line_range_for_span(
            pos, pos + len(chunk.text), line_starts
        )
        assert line_start >= 1 and line_end >= line_start

        search_start = max(0, pos + len(chunk.text) - 1)

    # 反向对照：没有忽略空白兜底时，跨句 chunk 一定定位失败
    naive_failures = sum(
        1
        for chunk in cross_sentence
        if original.find(chunk.text[:50], 0) == -1
    )
    assert naive_failures == len(cross_sentence), (
        "这些 chunk 本应触发旧实现的定位失败 —— 用例前提不成立"
    )
    print("  ok test_smoothed_chunks_are_located_exactly")


def test_adaptive_chunk_positions_stay_inside_original() -> None:
    """端到端：自适应切分产物逐个定位，位置必须单调推进且不越界."""
    paragraphs = [
        "第一段落用来填充内容，句子一。句子二，稍微长一点。句子三收尾。",
        "第二段落同样用于填充，句子一。句子二，稍微长一点。句子三收尾。",
        "第三段落继续填充内容，句子一。句子二，稍微长一点。句子三收尾。",
    ]
    original = "\n\n".join(paragraphs)
    chunks = _adaptive_chunk(
        original, min_chunk_size=60, max_chunk_size=200, chunk_overlap=20
    )
    assert chunks, "夹具应当切出至少一个 chunk"

    stripped, pos_map = _strip_whitespace_index(original)
    line_starts = build_line_index(original)
    search_start = 0
    for chunk in chunks:
        pos = _locate_chunk(original, stripped, pos_map, chunk.text, search_start)
        assert pos != -1, f"切分产物必须能在原文中定位：{chunk.text[:40]!r}"
        line_start, line_end = line_range_for_span(
            pos, pos + len(chunk.text), line_starts
        )
        assert 1 <= line_start <= line_end <= len(original.split("\n"))
        search_start = max(0, pos + len(chunk.text) - 20)
    print("  ok test_adaptive_chunk_positions_stay_inside_original")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nAll chunk-position tests passed.")
