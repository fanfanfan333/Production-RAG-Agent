import importlib
from types import SimpleNamespace

import pytest
from PIL import Image

from app.services.image_understanding.engines.base import EngineOutput
from app.services.image_understanding.table_recognizer import TableStructure
from app.services.image_understanding.table_recognizer import _from_rules_with_cell_ocr

pl = importlib.import_module("app.services.image_understanding.pipeline")


@pytest.mark.parametrize("method,expected", [("rules+cell-ocr", "table_parser:rules+cell-ocr"), ("ocr-alignment", "table-transformer")])
def test_reliable_grid_precedes_model_but_alignment_does_not(monkeypatch, method, expected):
    calls = []

    def process(*args, **kwargs):
        calls.append("model")
        return EngineOutput(text="model table", confidence=0.9, engine="table-transformer", ok=True,
                            meta={"rows": 2, "cols": 3})

    engine = SimpleNamespace(is_available=lambda: True, process=process)
    structure = TableStructure(markdown="rule table", rows=2, cols=4, method=method)
    monkeypatch.setattr(pl, "get_settings", lambda: SimpleNamespace(TABLE_TRANSFORMER_ENABLED=True))
    monkeypatch.setattr(pl, "get_registry", lambda: SimpleNamespace(get=lambda name: engine))
    monkeypatch.setattr(pl, "_cell_ocr_fn", lambda: None)
    monkeypatch.setattr(pl, "recognize_table_safe", lambda *a, **k: structure)
    result = SimpleNamespace(meta={}, table=None)
    out = pl._table_primary(Image.new("RGB", (30, 30)), EngineOutput(), 1, result)
    assert out.engine == expected
    assert calls == ([] if method.startswith("rules") else ["model"])
    assert result.table.cols == (4 if method.startswith("rules") else 3)


def test_sparse_cell_ocr_grid_is_rejected():
    values = iter(["指标", "2023", "2024", "33500"] + [""] * 8)
    structure = _from_rules_with_cell_ocr(
        Image.new("RGB", (300, 200)), [0, 50, 100, 150, 200], [0, 100, 200, 300],
        lambda image: next(values), min_rows=2, min_cols=2, max_rows=200,
        max_cols=30, max_cells=240, page_number=1,
    )
    assert not structure.ok
    assert structure.meta["reason"] == "cells-filled=4"


def test_vision_rescue_replaces_stale_table_shape():
    result = SimpleNamespace(image_type="table", structured_content=None,
                             table=TableStructure(rows=7, cols=5), meta={})
    text = "表 3\n| 指标 | 2023 | 2024 | 同比 |\n|---|---|---|---|\n| 收入 | 100 | 200 | +100% |"
    pl._apply_output(result, EngineOutput(text=text, engine="vision", ok=True))
    assert result.table.cols == 4
    assert result.table.rows == 2
    assert result.table.header == ["指标", "2023", "2024", "同比"]
    assert result.table.markdown == result.structured_content == text
