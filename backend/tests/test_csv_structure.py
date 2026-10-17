from app.services.parsers.csv_parser import CsvParser
from app.services.nodes.citation_verifier import verify_citations


def test_csv_keeps_year_columns_identifiers_and_empty_cells():
    data = '项目,年度,预算（万元）,备注\n0007,2024,4321,\n'
    result = CsvParser().parse(data.encode('utf-8-sig'), 'test.csv')
    assert '| 0007 | 2024 | 4321 |  |' in result.full_text
    assert '| --- | --- | --- | --- |' in result.full_text
    source = dict(text=result.full_text)
    assert verify_citations('项目0007在2024年预算为4321万元 [Source 1]。', [source]).verdicts[0].passed
    assert not verify_citations('项目0007在2023年预算为4321万元 [Source 1]。', [source]).verdicts[0].dates_consistent


def test_csv_quoted_commas_and_multiline_cells_keep_one_row():
    data = '项目,说明\nA,"comma, value\nand pipe|value"\n'
    result = CsvParser().parse(data.encode(), 'test.csv')
    assert 'comma, value<br>and pipe\\|value' in result.full_text
    assert len(result.full_text.splitlines()) == 3
