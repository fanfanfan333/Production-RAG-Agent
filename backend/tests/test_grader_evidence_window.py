import asyncio
from types import SimpleNamespace

from app.services.nodes.context_builder import compress_text
from app.services.graders import retrieval_grader as grader


def test_english_compression_keeps_relevant_tail_and_decimal():
    body = "An unrelated introduction about infrastructure and maintenance. " * 25
    body += "The evaluation generated 125 test questions per dataset. The measured ratio was 12.5%."
    result = compress_text(body, "How many test questions per dataset?", 600)
    assert "125 test questions" in result
    assert "12.5%" in result
    assert len(result) <= 600


def test_grader_receives_relevant_tail_within_its_existing_budget(monkeypatch):
    seen = []

    class LLM:
        def __init__(self, **kwargs): pass

        async def ainvoke(self, messages):
            seen.append(messages[-1].content)
            return SimpleNamespace(content='{"verdicts":["relevant"]}')

    monkeypatch.setattr(grader, "ChatOllama", LLM)
    chunk = SimpleNamespace(filename="paper.pdf", page_number=1, score=.9,
        text="Unrelated details about infrastructure. " * 30 + "We generated 125 test questions per dataset.")
    good, _, _ = asyncio.run(grader.grade_retrieval("How many test questions per dataset?", [chunk]))
    assert good and "125 test questions" in seen[0]


def test_compression_keeps_paragraph_subject_with_its_numeric_sentence():
    intro = 'Unrelated infrastructure and maintenance. ' * 30
    paragraph = 'News articles. The corpus is divided into 3197 text chunks.'
    result = compress_text(intro + '\n\n' + paragraph, 'News chunks', 100)
    assert paragraph in result
    assert len(result) <= 100


def test_page_boundaries_prevent_cross_page_evidence_attribution():
    from app.services.chunker import build_chunks
    first = '## Conditions\n\nSeveral indexing conditions are compared.'
    second = 'Graph indexing took 281 minutes for the Podcast dataset.'
    text = first + '\n\n' + second
    boundary = len(first) + 2
    chunks = build_chunks(text, min_chunk_size=500, max_chunk_size=2000, chunk_overlap=20,
        token_aware=False, page_boundaries=[0, boundary],
        page_resolver=lambda offset: 2 if offset < boundary else 3)
    assert len(chunks) == 2
    assert chunks[0].page_number == 2
    assert chunks[1].page_number == 3 and '281 minutes' in chunks[1].text
    assert chunks[1].char_start >= boundary
