"""Merged cells must not shift financial values into the wrong columns."""

import pytest

from app.services.structure.providers import html_table_to_markdown


def test_rowspan_preserves_row_labels_and_numeric_columns():
    html = """<table><tr><th>部门</th><th>指标</th><th>2024</th></tr>
    <tr><td rowspan='2'>研发</td><td>收入</td><td>33500</td></tr>
    <tr><td>利润</td><td>2460</td></tr></table>"""
    assert "| 研发 | 利润 | 2460 |" in html_table_to_markdown(html)


def test_colspan_and_entities_preserve_header_and_text():
    html = """<table><tr><th>指标</th><th colspan=2>年度</th></tr>
    <tr><td>A&amp;B<br>收入</td><td>100</td><td>200</td></tr></table>"""
    md = html_table_to_markdown(html)
    assert "| 指标 | 年度 | 年度 |" in md
    assert "| A&B 收入 | 100 | 200 |" in md


def test_rowspan_zero_extends_to_remaining_rows():
    html = """<table><tr><td rowspan=0>部门</td><td>2023</td></tr>
    <tr><td>2024</td></tr></table>"""
    assert "| 部门 | 2024 |" in html_table_to_markdown(html)


def test_excessive_colspan_is_rejected():
    with pytest.raises(ValueError, match="span"):
        html_table_to_markdown('<table><tr><td colspan="100000000">x</td></tr></table>')


def test_plain_table_escapes_pipe_and_keeps_surrounding_text():
    md = html_table_to_markdown('before<table><tr><td>a|b</td><td>c</td></tr></table>after')
    assert md == "before| a\\|b | c |\n| --- | --- |after"
