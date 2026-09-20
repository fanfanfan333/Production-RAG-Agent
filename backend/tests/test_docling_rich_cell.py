"""回归测试：Docling 富文本表头（`<!-- rich cell -->`）不再吞掉整行表头.

背景
────
真实管线 ``document_service → parse_structure → DoclingProvider`` 用
``_docling_markdown(node)`` 逐元素取 Markdown。对 ``TableItem`` 若不传 ``doc``，
Docling 会走旧分支，把 Word 里被判成 ``RichTableCell`` 的**表头行**导出成字面量
``<!-- rich cell -->`` —— 用户看到"表头整行消失、文字跑到表格下方当游离段落"。

修复：``_docling_markdown(node, doc)`` 仅对 ``TableItem`` 传 ``doc``；并在
``_sanitize_docling_markdown`` 里加**防复发护栏**（兜底清掉残留令牌 + 计数告警 +
表格分隔行归一化）。

本测试分两层：
  * 确定性单元测试：护栏行为、TableItem/PictureItem 的 doc 传递限制；
  * 集成测试（docling 与实际示例文档存在时）：真实 DOCX 的占位符计数与表头恢复。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from app.config import get_settings
from app.services.structure import providers

RICH = "<!-- rich cell -->"

HANDBOOK = Path(
    "/app/uploads/cf33b1db5679d/47354b6a-b269-42e0-8235-3791c2cac705/original.docx"
)
_DOCLING = importlib.util.find_spec("docling") is not None


# ── 护栏（防复发）：把字面量令牌挡在入库之外 ──────────────────────────────────

def test_guardrail_removes_rich_cell_token_but_keeps_data_rows(caplog):
    import logging

    raw = "| A | B |\n| " + RICH + " | " + RICH + " |\n| 1 | 2 |"
    with caplog.at_level(logging.WARNING, logger="app.services.structure.providers"):
        out = providers._sanitize_docling_markdown(raw, "synthetic.docx")
    assert RICH not in out
    assert "| A | B |" in out
    assert "| 1 | 2 |" in out
    # 护栏必须**可观测**：计数写进 warning 日志（不是静默吞掉）
    assert any("2" in r.getMessage() and "rich cell" in r.getMessage() for r in caplog.records)


def test_guardrail_drops_all_placeholder_line_without_ragged_columns():
    # 整行只剩占位符 → 整行删除（宁可少一行，也不留 `|  |  |` 空列错位）
    raw = "| " + RICH + " | " + RICH + " |\n| x | y |"
    out = providers._sanitize_docling_markdown(raw, "synthetic.docx")
    assert RICH not in out
    assert out.strip() == "| x | y |"


def test_guardrail_normalizes_compact_table_separator():
    raw = "| A | B |\n|-----|-----|\n| 1 | 2 |"
    out = providers._sanitize_docling_markdown(raw, "synthetic.docx")
    assert "| --- | --- |" in out


# ── doc 传递限制：只有 TableItem 能收到 doc（保护图片占位通道）───────────────

def test_table_item_is_passed_doc(monkeypatch):
    seen: dict = {}

    class StubTable:
        def export_to_markdown(self, *args):
            seen["args"] = args
            return "| a | b |\n| --- | --- |"

    monkeypatch.setattr(providers, "_is_docling_table_item", lambda _n: True)
    out = providers._docling_markdown(StubTable(), doc=object())
    assert len(seen["args"]) == 1, "TableItem 必须收到 doc"
    assert out.startswith("| a | b |")


def test_non_table_item_is_not_passed_doc():
    # PictureItem 语义：绝不传 doc，保持 `<!-- image -->` 通道的既有行为
    class PictureItem:
        def export_to_markdown(self, *args):
            assert args == (), "非 TableItem 必须不带 doc 调用（保护图片通道）"
            return "pic"

    assert providers._docling_markdown(PictureItem(), doc=object()) == "pic"


# ── 集成：真实示例文档的表头恢复 ──────────────────────────────────────────────

@pytest.mark.skipif(not _DOCLING, reason="docling not installed")
def test_real_docx_rich_cell_header_restored():
    if not HANDBOOK.is_file():
        pytest.skip(f"fixture not present: {HANDBOOK}")

    provider = providers.DoclingProvider(get_settings())
    sd = provider.parse(HANDBOOK.read_bytes(), HANDBOOK.name)
    assert sd is not None, "DoclingProvider 应能解析该 DOCX"
    md = sd.markdown

    # 表头占位符必须被彻底清除
    assert md.count(RICH) == 0, f"仍有 {md.count(RICH)} 个 rich cell 占位符"
    # 原表头文字必须回来
    assert "对比项" in md, "表头文字丢失"
    # 表格数量不减（示例文档有 14 张表）
    sep_lines = sum(
        1
        for ln in md.splitlines()
        if ln.strip().startswith("|") and set(ln.strip()) <= set("|-: ")
    )
    assert sep_lines >= 14, f"表格数减少：{sep_lines}"
    # 图片占位通道不被破坏（docling 2.126 下该路径本就不产出 `<!-- image -->`；
    # 此处断言其与修复前一致 = 0，防未来回归）
    assert md.count("<!-- image -->") == 0
