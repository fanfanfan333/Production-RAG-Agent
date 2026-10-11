import importlib
import sys
from types import SimpleNamespace


def test_rapidocr_uses_writable_cache_and_converter_is_reused(monkeypatch, tmp_path):
    engine = importlib.import_module("app.services.image_understanding.engines.docling_engine")
    converters = []

    def converter(**kwargs):
        converters.append(kwargs)
        return SimpleNamespace(**kwargs)

    options = lambda **kwargs: SimpleNamespace(**kwargs)
    rapidocr_rec = SimpleNamespace(DEFAULT_MODEL_PATH=None)
    monkeypatch.setitem(sys.modules, "rapidocr", SimpleNamespace(ch_ppocr_rec=SimpleNamespace(main=rapidocr_rec)))
    monkeypatch.setitem(sys.modules, "rapidocr.ch_ppocr_rec", SimpleNamespace(main=rapidocr_rec))
    monkeypatch.setitem(sys.modules, "rapidocr.ch_ppocr_rec.main", rapidocr_rec)
    monkeypatch.setitem(sys.modules, "docling.document_converter", SimpleNamespace(DocumentConverter=converter, PdfFormatOption=options))
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", SimpleNamespace(InputFormat=SimpleNamespace(PDF="pdf")))
    monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", SimpleNamespace(PdfPipelineOptions=options, RapidOcrOptions=options))
    monkeypatch.setattr(engine, "_converter", None)
    monkeypatch.setattr(engine, "_apply_pipeline_hash_shim", lambda: None)
    monkeypatch.setattr(engine, "get_settings", lambda: SimpleNamespace(PADDLE_CACHE_DIR=str(tmp_path)))
    instance = engine.DoclingEngine()
    first = instance._get_converter()
    assert instance._get_converter() is first
    assert len(converters) == 1
    ocr = first.format_options["pdf"].pipeline_options.ocr_options
    assert ocr.backend == "torch"
    assert ocr.rapidocr_params["Global.model_root_dir"] == tmp_path / "rapidocr"
    assert rapidocr_rec.DEFAULT_MODEL_PATH == tmp_path / "rapidocr"
    assert rapidocr_rec.DEFAULT_MODEL_PATH.is_dir()
