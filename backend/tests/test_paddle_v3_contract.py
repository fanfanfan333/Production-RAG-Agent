from types import SimpleNamespace

import pytest

from app.utils import paddle_env


def test_paddle_v3_keeps_coordinates_scores_and_text():
    page = {"rec_texts": ["Revenue", "33500"], "rec_scores": [.9, .95], "rec_polys": [[(0, 0), (10, 10)], [(20, 0), (30, 10)]]}
    result = paddle_env.ocr_result(SimpleNamespace(predict=lambda image: iter([page])), object())
    assert result == [[[[ (0, 0), (10, 10)], ("Revenue", .9)], [[(20, 0), (30, 10)], ("33500", .95)]]]
    page["rec_scores"] = [.9]
    with pytest.raises(ValueError, match="mismatched"):
        paddle_env.ocr_result(SimpleNamespace(predict=lambda image: [page]), object())


def test_paddle_v3_constructor_preserves_safe_cpu_options(monkeypatch):
    monkeypatch.setattr(paddle_env, "import_paddleocr", lambda: SimpleNamespace(__version__="3.7.0", PaddleOCR=lambda **kwargs: kwargs))
    result = paddle_env.build_paddle_ocr()
    assert result["ocr_version"] == "PP-OCRv3"
    assert result["device"] == "cpu" and result["enable_mkldnn"] is False
    assert result["use_doc_unwarping"] is False


@pytest.mark.parametrize("block", [
    {"block_bbox": [1, 2, 30, 40], "block_label": "table", "block_content": "33500"},
    SimpleNamespace(bbox=[1, 2, 30, 40], label="table", content="33500"),
])
def test_structure_v3_predict_preserves_regions(monkeypatch, block):
    from PIL import Image
    from app.services.image_understanding.engines.paddle_engines import PPStructureEngine
    engine = PPStructureEngine()
    monkeypatch.setattr(engine, "_instance", lambda: SimpleNamespace(predict=lambda image: iter([{"parsing_res_list": [block]}])))
    result = engine.process(Image.new("RGB", (50, 50)))
    assert result.ok and result.meta["regions"] == [{"type": "table", "bbox": [1, 2, 30, 40], "text": "33500"}]


def test_formula_v3_predict_preserves_latex(monkeypatch):
    from PIL import Image
    from app.services.image_understanding.engines.paddle_engines import PaddleFormulaEngine
    engine = PaddleFormulaEngine()
    monkeypatch.setattr(engine, "_instance", lambda: SimpleNamespace(predict=lambda image: iter([{"rec_formula": "x^2"}])))
    result = engine.process(Image.new("RGB", (50, 50)))
    assert result.ok and result.text == "$x^2$"
