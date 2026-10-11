import pytest
import asyncio

from app.services.image_understanding.dual_channel import ChannelResult, fuse_channels
from app.services.image_understanding.engines.base import EngineOutput
from app.services.image_understanding.quality import QualityReport
from app.services.parsers.base import ExtractedImage


@pytest.mark.parametrize("left,right,conflict", [("+0.9pp", "+0.9pr", True), ("+10.8%", "10.8%", True), ("33500", "", True), ("33500", "33500", False)])
def test_table_agreement_requires_matching_numbers_and_units(left, right, conflict):
    sides = [ChannelResult(name, EngineOutput(text=f"| change | {value} |", confidence=0.9, ok=True), QualityReport(ok=True, score=1.0)) for name, value in (("ocr", left), ("vlm", right))]
    result = fuse_channels("table", sides)
    assert result.numeric_conflict is conflict
    assert result.agreed is not conflict


def test_conflicting_table_warns_answer_and_summary_consumers():
    image = ExtractedImage(image_id="test", page_number=1, structured_content="| change | +0.9pr |", analyze_fusion={"numeric_conflict": True})
    assert "不能据此确定数值" in image.searchable_text
    assert "+0.9pr" in image.searchable_text


def test_identical_numbers_in_different_columns_are_not_agreement():
    sides = [ChannelResult(name, EngineOutput(text=text, confidence=.9, ok=True), QualityReport(ok=True, score=1)) for name, text in (("ocr", "| Revenue | 30240 | 33500 |"), ("vlm", "| Revenue | 33500 | 30240 |"))]
    assert fuse_channels("table", sides).numeric_conflict


def test_review_warning_survives_context_compression(monkeypatch):
    from app.config import Settings
    from app.services.nodes import context_builder as plain
    from app.services.nodes import multimodal_context_node as multimodal
    from app.services.retrieval_service import RetrievedChunk

    settings = Settings(_env_file=None, POSTGRES_PASSWORD="test-only", CONTEXT_COMPRESSION_ENABLED=True, CONTEXT_PER_SOURCE_CHARS=100)
    monkeypatch.setattr(plain, "get_settings", lambda: settings)
    monkeypatch.setattr(multimodal, "get_settings", lambda: settings)
    monkeypatch.setattr(plain, "compress_text", lambda *args: "compressed table")
    monkeypatch.setattr(multimodal, "compress_text", lambda *args: "compressed table")
    chunk = RetrievedChunk(document_id="d", filename="table.pdf", page_number=1, chunk_index=0, text="table " * 2000, score=.9, content_type="table", manual_review=True, analyze_fusion={"numeric_conflict": True})
    for built in (plain.build_context([chunk], query="table"), asyncio.run(multimodal.build_multimodal_context([chunk], "table", enable_images=False))):
        assert "不能据此确定数值" in built.context
        assert "compressed table" in built.context
        assert built.sources[0]["manual_review"]


def test_summary_warning_is_visible_even_when_model_omits_it(monkeypatch):
    from app.services import master_graph as graph

    async def answer(*args):
        return "核心主题：经营数据汇总。"

    monkeypatch.setattr(graph, "build_summary_llm", lambda **kwargs: object())
    monkeypatch.setattr(graph, "_stream_llm_answer", answer)
    emitted = []
    monkeypatch.setattr(graph, "emit_text", emitted.append)
    result = asyncio.run(graph._summarize_node({"query": "总结", "digests": [{"filename": "table.pdf", "warnings": ["表格数字存在分歧，请核对原文"]}]}))
    assert "核对原文" in result["answer"]
    assert any("核对原文" in text for text in emitted)


@pytest.mark.parametrize("review", [True, False])
def test_qa_warning_is_visible_even_when_model_omits_it(monkeypatch, review):
    from app.services import master_graph as graph

    async def answer(*args):
        return "营业收入为 33500。"

    monkeypatch.setattr(graph, "_build_streaming_llm", lambda settings: object())
    monkeypatch.setattr(graph, "_stream_llm_answer", answer)
    emitted = []
    monkeypatch.setattr(graph, "emit_text", emitted.append)
    result = asyncio.run(graph._generate_node({"query": "营业收入", "history_messages": [], "multimodal_context": "visible table", "sources": [{"manual_review": review}]}))
    assert ("核对原文" in result["answer"]) is review
    assert bool(emitted) is review


@pytest.mark.parametrize("review", [True, False])
def test_legacy_stream_does_not_drop_deterministic_review_notice(review):
    from types import SimpleNamespace
    from app.services.rag_graph import _stream_graph_events

    class Graph:
        async def astream_events(self, *args, **kwargs):
            yield {"event": "on_chat_model_stream", "data": {"chunk": SimpleNamespace(content="33500", additional_kwargs={})}}
            yield {"event": "on_chain_end", "name": "generate", "data": {"output": {"answer": "33500" + ("\n\n> 图片识别结果需复核，请核对原文。" if review else "")}}}

    async def collect():
        return [event async for event in _stream_graph_events(Graph(), {"conversation_id": "test"}, sources_node="retrieve")]

    events = asyncio.run(collect())
    assert ("核对原文" in "".join(e["content"] for e in events if e["type"] == "chunk")) is review
    assert sum(e["type"] == "chunk" for e in events) == (2 if review else 1)
    assert events[-1]["type"] == "done"


@pytest.mark.parametrize("review", [True, False])
def test_legacy_qa_warning_is_visible_even_when_model_omits_it(monkeypatch, review):
    from app.services import rag_graph as graph
    from app.services.retrieval_service import RetrievedChunk

    async def answer(*args):
        return "营业收入为 33500。"

    monkeypatch.setattr(graph, "_build_llm", lambda settings: object())
    monkeypatch.setattr(graph, "_stream_llm_answer", answer)
    chunk = RetrievedChunk(document_id="d", filename="table.pdf", page_number=1, chunk_index=0, text="33500", score=.9, manual_review=review)
    result = asyncio.run(graph._generate_node({"query": "营业收入", "history_messages": [], "chunks": [chunk]}))
    assert ("核对原文" in result["answer"]) is review


def test_summary_generation_has_a_finite_token_budget(monkeypatch):
    from app.services.nodes import document_summary_node as node
    monkeypatch.setattr(node, "ChatOllama", lambda **kwargs: kwargs)
    llm = node.build_summary_llm(reasoning=False)
    assert llm["num_predict"] == 1024 and llm["reasoning"] is False
