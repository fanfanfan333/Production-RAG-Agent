import asyncio
import json
from types import SimpleNamespace

import langchain_ollama

from app.services import query_transform as rewrite


def test_bilingual_retry_has_separate_cache_and_preserves_original(monkeypatch):
    query = "GraphRAG论文每个数据集生成了多少个测试问题？"
    calls = []

    class LLM:
        def __init__(self, **kwargs): pass

        async def ainvoke(self, messages):
            prompt = messages[0].content
            calls.append(prompt)
            text = "How many test questions per dataset?" if "追加到原问题后" in prompt else query
            return SimpleNamespace(content=json.dumps(dict(rewritten=text, variants=[], subqueries=[], hyde="")))

    monkeypatch.setattr(langchain_ollama, "ChatOllama", LLM)
    rewrite.clear_rewrite_cache()

    async def check():
        assert (await rewrite.rewrite_query(query)).rewritten == query
        result = await rewrite.rewrite_query_bilingual(query)
        assert result.rewritten.startswith(query) and "test questions" in result.rewritten
        assert (await rewrite.rewrite_query(query)).rewritten == query
        assert (await rewrite.rewrite_query_bilingual(query)).source == "cache"

    try:
        asyncio.run(check())
        assert len(calls) == 2
    finally:
        rewrite.clear_rewrite_cache()


def test_only_failed_cross_language_retrieval_uses_translation(monkeypatch):
    from app.services import master_graph as graph
    calls = []

    async def ordinary(query, history_messages=None):
        calls.append("ordinary")
        return rewrite.RewriteResult(rewritten=query)

    async def bilingual(query, history_messages=None):
        calls.append("bilingual")
        return rewrite.RewriteResult(rewritten=query + " test questions")

    monkeypatch.setattr(graph, "rewrite_query", ordinary)
    monkeypatch.setattr(graph, "rewrite_query_bilingual", bilingual)
    state = dict(query="论文生成多少测试问题", history_messages=[], chunks=[SimpleNamespace(text="English research document " * 20)])
    asyncio.run(graph._rewrite_node(state))
    asyncio.run(graph._rewrite_node(dict(state, retry_count=1)))
    asyncio.run(graph._rewrite_node(dict(state, retry_count=1, chunks=[SimpleNamespace(text="中文资料" * 30)])))
    assert calls == ["ordinary", "bilingual", "ordinary"]


def test_translation_cannot_invent_a_numeric_answer(monkeypatch):
    query = "GraphRAG论文每个数据集生成了多少个测试问题？"

    async def invoke(messages):
        return SimpleNamespace(content=json.dumps(dict(rewritten="GraphRAG generated 125 test questions per dataset", variants=[], subqueries=[], hyde="")))

    monkeypatch.setattr(langchain_ollama, "ChatOllama", lambda **kwargs: SimpleNamespace(ainvoke=invoke))
    rewrite.clear_rewrite_cache()
    try:
        assert asyncio.run(rewrite.rewrite_query_bilingual(query)).rewritten == query
    finally:
        rewrite.clear_rewrite_cache()


def test_cross_language_alignment_uses_existing_retry_without_relaxing_gate(monkeypatch):
    from app.services import master_graph as graph

    async def relevant(**kwargs):
        return True, [True], "llm"

    monkeypatch.setattr(graph, "grade_retrieval", relevant)
    state = dict(query="论文每个数据集生成多少测试问题", chunks=[SimpleNamespace(text="English dataset evaluation questions " * 10, score=.9)])
    result = asyncio.run(graph._grade_node(state))
    assert result["grade_good"] is False and result["retry_count"] == 1
    assert result["grade_reason"] == "cross_language_alignment"
    assert graph._route_after_evidence(dict(evidence={"passed": False})) == "refuse"


def test_aligned_cross_language_query_does_not_repeat_retry(monkeypatch):
    from app.services import master_graph as graph
    async def relevant(**kwargs): return True, [True], 'llm'
    monkeypatch.setattr(graph, 'grade_retrieval', relevant)
    state = dict(query='论文每个数据集生成多少测试问题', retry_count=1,
                 chunks=[SimpleNamespace(text='English dataset evaluation questions ' * 10, score=.9)])
    assert asyncio.run(graph._grade_node(state))['grade_good']
