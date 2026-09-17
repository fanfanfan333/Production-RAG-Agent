"""
切片位置信息（行号）单元测试 —— 细粒度引用溯源的基础.

覆盖：
  - build_line_index / line_for_offset / line_range_for_span 三个纯函数
  - build_chunks 产出的每个 chunk 都带正确的 1-based 行号区间
  - TextChunk.to_dict() 输出包含 line_start / line_end
  - 表格/代码块等结构被整体保留时，行号仍然连续可用

注意：build_chunks 会读取 settings（token-aware 等），因此需要
POSTGRES_PASSWORD（Settings 无默认值）。这里显式设一个测试值，
不依赖 backend/.env 是否存在。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# Settings 必填项，避免因缺 .env 而失败（只影响本测试进程）
os.environ.setdefault("POSTGRES_PASSWORD", "test-only")

# 桩包：避免触发 app/services/__init__.py 的重依赖
if "app.services" not in sys.modules:
    _pkg = types.ModuleType("app.services")
    _pkg.__path__ = [str(_BACKEND_ROOT / "app" / "services")]
    sys.modules["app.services"] = _pkg

_MOD_PATH = _BACKEND_ROOT / "app" / "services" / "chunker.py"
_spec = importlib.util.spec_from_file_location("app.services.chunker", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["app.services.chunker"] = _mod
_spec.loader.exec_module(_mod)

build_chunks = _mod.build_chunks
build_line_index = _mod.build_line_index
line_for_offset = _mod.line_for_offset
line_range_for_span = _mod.line_range_for_span


# ── 纯函数 ───────────────────────────────────────────────────────────────────

def test_build_line_index():
    text = "abc\ndef\n\nghi"
    starts = build_line_index(text)
    assert starts == [0, 4, 8, 9], starts
    assert build_line_index("") == [0]
    print("[OK] test_build_line_index")


def test_line_for_offset():
    starts = build_line_index("abc\ndef\nghi")   # 行首偏移 0, 4, 8
    assert line_for_offset(0, starts) == 1
    assert line_for_offset(3, starts) == 1        # 第一行最后一个字符
    assert line_for_offset(4, starts) == 2        # 第二行行首
    assert line_for_offset(7, starts) == 2
    assert line_for_offset(8, starts) == 3
    assert line_for_offset(999, starts) == 3      # 越界收敛到末行
    assert line_for_offset(-5, starts) == 1
    print("[OK] test_line_for_offset")


def test_line_range_for_span():
    starts = build_line_index("abc\ndef\nghi")    # 0-2 / 4-6 / 8-10
    assert line_range_for_span(0, 3, starts) == (1, 1)
    assert line_range_for_span(0, 7, starts) == (1, 2)
    assert line_range_for_span(4, 11, starts) == (2, 3)
    # char_end 落在下一行行首时，不应把行号多算一行
    assert line_range_for_span(0, 4, starts) == (1, 1)
    print("[OK] test_line_range_for_span")


# ── build_chunks 集成 ────────────────────────────────────────────────────────

def test_chunks_carry_line_numbers():
    text = "\n".join(f"第 {i} 行：这是用于测试位置信息的内容，长度足够形成语义块。" for i in range(1, 41))
    chunks = build_chunks(text, min_chunk_size=100, max_chunk_size=300, chunk_overlap=20)
    assert chunks, "应至少产出一个 chunk"
    for c in chunks:
        assert c.line_start is not None, c
        assert c.line_end is not None, c
        assert 1 <= c.line_start <= c.line_end <= 40, (c.line_start, c.line_end)
    # 行号随 chunk 顺序单调不减
    starts = [c.line_start for c in chunks]
    assert starts == sorted(starts), starts
    print("[OK] test_chunks_carry_line_numbers")


def test_first_chunk_starts_at_line_one():
    text = "第一行内容。\n第二行内容。\n第三行内容。" * 20
    chunks = build_chunks(text, min_chunk_size=50, max_chunk_size=200, chunk_overlap=10)
    assert chunks[0].line_start == 1
    print("[OK] test_first_chunk_starts_at_line_one")


def test_to_dict_exposes_lines():
    text = "内容行。" * 60
    chunks = build_chunks(text, min_chunk_size=50, max_chunk_size=200, chunk_overlap=10)
    d = chunks[0].to_dict()
    assert "line_start" in d and "line_end" in d
    assert d["line_start"] == chunks[0].line_start
    print("[OK] test_to_dict_exposes_lines")


def test_line_numbers_survive_table_protection():
    """表格块被整体保留时，行号仍然可用（不会出现 None 或倒挂）."""
    table = "\n".join(["| 指标 | 数值 |", "| --- | --- |", "| 营收 | 1200 |", "| 利润 | 300 |"])
    text = "前言段落，用于测试位置信息。" * 5 + "\n\n" + table + "\n\n" + "结语段落。" * 10
    chunks = build_chunks(text, min_chunk_size=80, max_chunk_size=400, chunk_overlap=20)
    assert chunks
    for c in chunks:
        assert c.line_start is not None and c.line_end is not None
        assert c.line_start <= c.line_end
    print("[OK] test_line_numbers_survive_table_protection")


def test_single_line_document():
    chunks = build_chunks("只有一行内容", min_chunk_size=5, max_chunk_size=100, chunk_overlap=1)
    assert chunks
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 1
    print("[OK] test_single_line_document")


if __name__ == "__main__":
    tests = [
        test_build_line_index,
        test_line_for_offset,
        test_line_range_for_span,
        test_chunks_carry_line_numbers,
        test_first_chunk_starts_at_line_one,
        test_to_dict_exposes_lines,
        test_line_numbers_survive_table_protection,
        test_single_line_document,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} position-tracking tests passed.")
