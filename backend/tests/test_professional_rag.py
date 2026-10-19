import asyncio
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from app.services import graph_retrieval as graph
from app.services.training_data import TrainingDraft, export_examples
from app.services.retrieval_service import RetrievedChunk


def edge(left, right, left_type="organization", right_type="organization"):
    return graph.GraphEdge(left, left_type, right, right_type, "投资",
                           RetrievedChunk("doc", "test.txt", 1, 0, f"{left}投资{right}", .5), f"{left}投资{right}")


def test_graph_expansion_is_two_hops_and_type_sensitive():
    edges = [edge("甲", "乙"), edge("乙", "丙"), edge("丙", "丁"), edge("乙", "地点", "person", "location")]
    assert graph.expand("甲有哪些间接关联", edges) == edges[:2]
    assert len(graph.expand("甲", [edge("甲", str(i)) for i in range(100)])) == 50


def test_graph_bridge_context_requires_a_visible_typed_first_hop():
    from app.services.reranker import ranking_texts
    from app.services.retrieval_service import _merge_graph_candidates
    first, bridge, wrong_type = edge("甲", "乙"), edge("乙", "丙"), edge("乙", "丁", "person")
    for index, item in enumerate((first, bridge, wrong_type)):
        item.chunk.document_id = str(index)
    contexts = graph.bridge_contexts("甲的间接关联", [first, bridge, wrong_type])
    assert contexts == {("1", 0): [("0", 0, "甲投资乙")]}
    bridge.chunk._graph_context = contexts[("1", 0)]
    merged = {(c.document_id, c.chunk_index): c for c in (first.chunk, bridge.chunk)}
    picks = _merge_graph_candidates([], merged, set(merged), 12, graph_chunks=[bridge.chunk])
    assert ranking_texts(picks) == ["甲投资乙", "甲投资乙\n乙投资丙"]
    assert ranking_texts([bridge.chunk]) == ["乙投资丙"]
    assert bridge.chunk.text == "乙投资丙", "ranking context must not become cited source text"


def test_graph_quotes_and_predicate_must_be_grounded():
    item = dict(subject="甲", subject_type="organization", predicate="投资", object="乙", object_type="organization", evidence="甲投资乙")
    assert graph.validated_relations([item], "甲投资乙") == [item]
    assert graph.validated_relations([{**item, "evidence": "甲投资丙"}], "甲投资乙") == []
    assert graph.validated_relations([{**item, "predicate": "收购"}], "甲投资乙") == []
    with pytest.raises(ValueError):
        graph.validated_relations({}, "甲投资乙")


def test_acl_denial_precedes_source_fetch(monkeypatch):
    async def views(*args):
        return {"doc": {}}, {"doc": True}
    monkeypatch.setattr(graph, "load_view_indexes", views)
    monkeypatch.setattr(graph, "allows", lambda *args: SimpleNamespace(allowed=False))
    monkeypatch.setattr(graph, "get_qdrant_client", lambda: pytest.fail("denied source was fetched"))
    assert asyncio.run(graph.source_chunks([("doc", 0, "point")], SimpleNamespace(predicate=lambda: None))) == {}
    with pytest.raises(graph.GraphPermissionError):
        asyncio.run(graph.source_chunks([], None))


def example(index=0, **changes):
    data = dict(prompt=f"项目{index}收入是多少？", evidence=[dict(source_id=f"synthetic:doc{index}", text="收入100万元", location="第1页")],
                chosen="收入100万元。[Source 1]", rejected="收入200万元。", source_group=f"doc{index}",
                source_kind="synthetic", license="self-authored")
    data.update(changes)
    return SimpleNamespace(**data, status="approved", reviewed_by=uuid.uuid4())


def test_training_formats_split_sources_and_reject_unreviewed_data():
    rows = [example(i) for i in range(30)]
    result = export_examples(rows, "dpo")
    assert result["ready_for_training"]
    assert set(result["train"][0]) == {"prompt", "chosen", "rejected"}
    train_sources = {s for record in result["manifest"] if record["split"] == "train" for s in record["sources"]}
    validation_sources = {s for record in result["manifest"] if record["split"] == "validation" for s in record["sources"]}
    assert not train_sources & validation_sources
    row = example(31)
    row.status = "pending"
    assert len(export_examples(rows + [row], "sft")["manifest"]) == 30
    assert export_examples(rows, "sft")["train"][0]["messages"][-1]["role"] == "assistant"


def test_shared_evidence_cannot_cross_splits_and_answers_version_the_dataset():
    shared = [dict(source_id="synthetic:shared", text="收入100万元", location="第1页")]
    rows = [example(i, evidence=shared) for i in range(8)]
    result = export_examples(rows, "sft")
    assert len({record["split"] for record in result["manifest"]}) == 1
    assert not result["ready_for_training"]
    changed = export_examples([example(0, chosen="收入为100万元。[Source 1]")], "sft")
    assert changed["dataset_version"] != export_examples([example(0)], "sft")["dataset_version"]
    with pytest.raises(ValueError, match="冲突"):
        export_examples([example(0), example(0, chosen="收入为200万元")], "sft")


@pytest.mark.parametrize("changes", [dict(chosen="联系手机号13800138000"), dict(source_kind="private"),
    dict(rejected="收入100万元。[Source 1]"), dict(source_kind="public", license="CC0",
    evidence=[dict(source_id="https://192.168.1.1/doc", text="收入100万元", location="第1页")])])
def test_private_invalid_or_identical_preferences_rejected(changes):
    with pytest.raises(ValueError):
        TrainingDraft.model_validate(vars(example(**changes)))


def test_feedback_rejects_foreign_conversations_and_forged_answers(monkeypatch):
    from app.api import feedback
    owner = uuid.uuid4()
    conversation = SimpleNamespace(id=uuid.uuid4(), owner_id=uuid.uuid4(), tenant_id="tenant")
    class Session:
        async def get(self, *args):
            return conversation
        async def scalar(self, *args):
            return None
    @asynccontextmanager
    async def session():
        yield Session()
    monkeypatch.setattr(feedback, "get_db_session", session)
    monkeypatch.setattr(feedback, "home_tenant_id", lambda user: "tenant")
    body = feedback.FeedbackRequest(rating="up", question="真问题", answer="伪造答案", conversation_id=conversation.id)
    with pytest.raises(feedback.HTTPException) as error:
        asyncio.run(feedback.submit_feedback(body, SimpleNamespace(id=owner)))
    assert error.value.status_code == 404
    conversation.owner_id = owner
    with pytest.raises(feedback.HTTPException) as error:
        asyncio.run(feedback.submit_feedback(body, SimpleNamespace(id=owner)))
    assert error.value.status_code == 409


def test_offline_training_checks_tampering_without_loading_models():
    from scripts.train_rag_offline import validate_bundle
    bundle = export_examples([example(i) for i in range(30)], "dpo")
    assert validate_bundle(bundle) == "dpo"
    bundle["train"][0]["chosen"][0]["content"] = "tampered"
    with pytest.raises(ValueError, match="重新审核"):
        validate_bundle(bundle)


def test_fixture_golden_has_120_grounded_cases():
    from scripts.make_professional_golden import make_golden, verify
    golden = make_golden()
    verify(golden)
    assert len(golden["cases"]) == 120


@pytest.mark.parametrize("quote", ["甲未投资乙", "甲考虑投资乙", "甲投资丙，乙投资丁", "乙投资甲"])
def test_uncertain_mismatched_or_reversed_relations_rejected(quote):
    item = dict(subject="甲", subject_type="organization", predicate="投资", object="乙", object_type="organization", evidence=quote)
    assert graph.validated_relations([item], quote) == []


def test_entity_prefix_does_not_seed_other_entities():
    edges = [edge("企业甲1", "企业乙1"), edge("企业甲10", "企业乙10")]
    assert graph.expand("企业甲10投资谁", edges) == edges[1:]


def test_export_version_is_order_independent_and_url_aliases_share_split():
    rows = [example(i) for i in range(30)]
    assert export_examples(rows, "sft") == export_examples(rows[::-1], "sft")
    aliases = [example(i, source_kind="public", license="CC0", evidence=[dict(source_id=url, text="收入100万元", location="正文")])
               for i, url in enumerate(["https://example.org/doc#section", "https://example.org/doc?utm_source=test"])]
    assert len({r["split"] for r in export_examples(aliases, "sft")["manifest"]}) == 1


def test_training_rejects_truncation_and_sft_uses_completion():
    from scripts.train_rag_offline import training_samples
    bundle = export_examples([example(i) for i in range(30)], "sft")
    tokenizer = SimpleNamespace(apply_chat_template=lambda sequence, **kwargs: list(range(100 * len(sequence))))
    assert training_samples(bundle, tokenizer)["train"][0]["completion"][0]["role"] == "assistant"
    with pytest.raises(ValueError, match="截断"):
        training_samples(bundle, tokenizer, max_length=50)


def test_professional_report_counts_all_multihop_sources_and_errors():
    from scripts.evaluate_professional import score, aggregate
    case = dict(id="multi", category="multihop", expect_sources=["a.md", "b.md"], answer="100万元")
    events = [dict(type="sources", sources=[dict(filename="a.md")]), dict(type="chunk", content="100 万元"),
              dict(type="citation_check", total=2, passed=1), dict(type="stage_timing", stage="retrieve", duration_ms=100), dict(type="done")]
    result = score(case, events, 1000, 200)
    assert result["recall@5"] == .5 and result["numeric_unit_match"]
    assert aggregate([result])["citation_support_rate"] == .5
    assert not score(case, events + [dict(type="error", message="truncated")], 1000, 200)["answer_literal_match"]


def test_ingestion_cancel_waits_for_checkpoint_and_records_retryable_failure(monkeypatch):
    from app.services import document_service as ingestion
    doc_id = uuid.uuid4()
    saved = []
    async def run(*args, **kwargs):
        assert ingestion.cancel_ingestion(doc_id)
        await ingestion._update_document(doc_id, current_stage="next")
    class Session:
        async def execute(self, statement):
            saved.append(statement.compile().params)
    @asynccontextmanager
    async def session(): yield Session()
    monkeypatch.setattr(ingestion, "get_db_session", session)
    monkeypatch.setattr(ingestion, "_run_ingestion", run)
    async def check():
        task = asyncio.create_task(ingestion._ingestion_runner(doc_id, None, "test.md", b"", None, None, None))
        ingestion._ingestion_tasks[doc_id] = task
        await task
    asyncio.run(check())
    assert saved[-1]["current_stage"] == "cancelled"
    assert doc_id not in ingestion._cancelled_ingestions


def test_graph_report_gates_reject_regression_and_incomparable_scope():
    from scripts.compare_professional_reports import compare
    from copy import deepcopy
    baseline = dict(gold="fixed", collection_id="kb", account_scope={"id": "u"}, condition="warm", strategy="hybrid",
        cases=[dict(id="a", category="multihop", answer_literal_match=False), dict(id="b", category="fact", answer_literal_match=True)],
        summary={"retrieval_p95_ms": 100, "recall@5": 1, "citation_support_rate": 1, "errors": 0})
    result = deepcopy(baseline); result["strategy"] = "hybrid_graph"; result["cases"][0]["answer_literal_match"] = True
    assert compare(baseline, result)["multihop_gate"]
    assert not compare(baseline, result)["complete_dataset_gate"]
    assert not compare(baseline, result)["provenance_gate"]
    assert not compare(baseline, result)["scoring_gate"]
    baseline["scoring_version"] = result["scoring_version"] = "professional_v3_no_mixed_refusal"
    assert compare(baseline, result)["scoring_gate"]
    assert not compare(baseline, result)["warmup_gate"]
    assert not compare(baseline, result)["latency_gate"]
    for report in (baseline, result):
        report["warmup_policy"] = "per_case"
        report["warmups"] = [dict(id=row["id"], completed=True, error=None) for row in report["cases"]]
    assert compare(baseline, result)["warmup_gate"]
    result["summary"]["retrieval_p95_ms"] = 110
    assert compare(baseline, result)["latency_gate"]
    result["warmups"][-1]["error"] = "failed"
    assert not compare(baseline, result)["warmup_gate"]
    result["warmups"][-1]["error"] = None
    assert not compare(baseline, result)["evidence_chain_gate"]
    result["cases"][0]["citation_source_recall"] = 1
    result["cases"][1]["citation_source_recall"] = 1
    assert compare(baseline, result)["evidence_chain_gate"]
    conflict = dict(id="conflict", category="negative", answer_literal_match=True, **{"recall@5": 1})
    baseline["cases"].append(conflict.copy())
    result["cases"].append(conflict.copy())
    assert not compare(baseline, result)["evidence_chain_gate"]
    result["cases"][-1]["citation_source_recall"] = 1
    assert compare(baseline, result)["evidence_chain_gate"]
    result["summary"].update(citation_coverage=1, citation_support_rate=.97, **{"recall@5": .94})
    assert not compare(baseline, result)["citation_gate"]
    assert not compare(baseline, result)["recall_gate"]
    result["summary"].update(citation_support_rate=.98, **{"recall@5": .95})
    assert compare(baseline, result)["citation_gate"]
    assert compare(baseline, result)["recall_gate"]
    result["cases"][1]["answer_literal_match"] = False
    assert not compare(baseline, result)["ordinary_no_regression"]
    result["account_scope"] = {"id": "other"}
    with pytest.raises(ValueError, match="不可比较"): compare(baseline, result)


def test_training_rejects_blank_answers_and_conflicting_source_licenses():
    with pytest.raises(ValueError, match="空白"):
        TrainingDraft.model_validate(vars(example(chosen="   ")))
    evidence = [dict(source_id="https://example.org/doc", text="收入100万元", location="正文")]
    rows = [example(i, source_kind="public", license=license, evidence=evidence) for i, license in enumerate(("CC0", "CC-BY-4.0"))]
    with pytest.raises(ValueError, match="许可不一致"):
        export_examples(rows, "sft")


def test_graph_oversized_source_fails_before_inference(monkeypatch):
    monkeypatch.setattr(graph, "get_settings", lambda: SimpleNamespace(chat_num_ctx=2048))
    monkeypatch.setattr(graph.httpx, "AsyncClient", lambda **kwargs: pytest.fail("oversized prompt reached model"))
    with pytest.raises(graph.GraphExtractionError, match="上下文"):
        asyncio.run(graph.extract("原文" * 2000))


@pytest.mark.parametrize("result", [dict(done=True, done_reason="length", response='{"relations":[]}'),
    dict(done=False, response='{"relations":[]}'), dict(done=True, done_reason="stop", response="[]")])
def test_graph_incomplete_or_invalid_model_output_is_not_success(monkeypatch, result):
    settings = SimpleNamespace(chat_num_ctx=8192, OLLAMA_BASE_URL="http://localhost:11434", OLLAMA_MODEL="local",
                               OLLAMA_KEEP_ALIVE_SECONDS=0, OLLAMA_NUM_GPU=0)
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs):
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: result)
    monkeypatch.setattr(graph, "get_settings", lambda: settings)
    monkeypatch.setattr(graph.httpx, "AsyncClient", lambda **kwargs: Client())
    with pytest.raises(graph.GraphExtractionError): asyncio.run(graph.extract("甲投资乙"))


def test_complex_golden_has_scanned_and_merged_tables_and_unambiguous_chart_year():
    from scripts.make_complex_professional import make_complex_golden
    golden = make_complex_golden()
    assert len(golden["cases"]) == 120
    assert sum(name.endswith(".docx") for name in golden["documents"]) == 2
    assert sum(name.endswith(".pdf") for name in golden["documents"]) == 3
    assert all("2025" in case["query"] for case in golden["cases"] if case["category"] == "image")
    assert all(case["human_review"] == "pending_business_review" for case in golden["cases"])


@pytest.mark.parametrize("rate_limited", [False, True])
def test_acceptance_upload_is_manifest_scoped_and_checkpointed(monkeypatch, tmp_path, rate_limited):
    import json
    import sys
    from scripts import prepare_professional_acceptance as runner
    (tmp_path / "documents").mkdir()
    (tmp_path / "documents/test.md").write_text("test", encoding="utf-8")
    (tmp_path / "gold.json").write_text(json.dumps({"documents": {"test.md": "test"}}), encoding="utf-8")
    calls, collection = [], {}
    def response(data): return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: data)
    class Client:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, path, **kwargs):
            if path == "/auth/me": return response(dict(id="owner", role="admin", tenant_id="default"))
            if path == "/kb/collections": return response(dict(collections=[collection]))
            assert path == "/documents"
            return response(dict(documents=[dict(document_id="doc", status="completed", chunk_count=1, page_count=1)]))
        def post(self, path, **kwargs):
            calls.append(path)
            if path == "/kb/collections":
                collection.update(id="kb", name=kwargs["json"]["name"]); return response(collection)
            assert path == "/upload" and kwargs["data"]["collection_id"] == "kb"
            assert kwargs["files"]["files"][0] == "test.md"
            assert kwargs["files"]["files"][1].read() == b"test", "retry must rewind the uploaded file"
            if rate_limited and calls.count("/upload") == 1:
                return SimpleNamespace(status_code=429, headers={"Retry-After": "1"})
            return response(dict(documents=[dict(document_id="doc")]))
    monkeypatch.setattr(runner.httpx, "Client", lambda **kwargs: Client())
    readings = iter([6] + [3] * 20)
    monkeypatch.setattr(runner, "available_gb", lambda: next(readings))
    monkeypatch.setattr(runner, "wait_rate_limit", lambda response: int(response.headers["Retry-After"]))
    monkeypatch.setenv("RAG_EVAL_TOKEN", "test-token")
    monkeypatch.setattr(sys, "argv", ["runner", str(tmp_path), "--phase", "upload"])
    runner.main()
    state = json.loads((tmp_path / "run-state.json").read_text(encoding="utf-8"))
    assert state["upload_completed"] and state["documents"]["test.md"]["id"] == "doc"
    expected = ["/kb/collections", "/upload"] + (["/upload"] if rate_limited else [])
    assert calls == expected
    monkeypatch.setattr(runner, "available_gb", lambda: .5)
    with pytest.raises(SystemExit): runner.main()
    assert calls == expected, "low memory must not start a new upload"


def test_paired_acceptance_prewarm_is_not_counted_as_a_gold_case(monkeypatch, tmp_path):
    import json
    import sys
    import httpx
    from scripts import evaluate_professional as runner
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(dict(name="fixture", truth_origin="synthetic", cases=[dict(id="fact-01", category="fact", query="query", answer="value", expect_sources=["doc.md"])])), encoding="utf-8")
    calls, removed = [], []
    class Client:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, path): return httpx.Response(200, request=httpx.Request("GET", "http://local" + path), json=dict(id="owner"))
        def build_request(self, method, path, **kwargs): return httpx.Request(method, "http://local" + path, **kwargs)
        def send(self, request, **kwargs):
            calls.append(json.loads(request.content)["retrieval_strategy"])
            events = [dict(type="thinking", conversation_id=f"owned-{len(calls)}"), dict(type="sources", sources=[dict(filename="doc.md", page_number=1)]), dict(type="chunk", content="value"), dict(type="done")]
            return httpx.Response(200, request=request, text="\n".join("data: " + json.dumps(event) for event in events))
        def delete(self, path): removed.append(path); return SimpleNamespace(status_code=204)
    monkeypatch.setattr(runner.httpx, "Client", lambda **kwargs: Client())
    readings = iter([6] + [3] * 30)
    monkeypatch.setattr(runner, "available_gb", lambda: next(readings))
    monkeypatch.setenv("RAG_EVAL_TOKEN", "unit-token")
    monkeypatch.setattr(sys, "argv", ["runner", "--gold", str(gold), "--out", str(tmp_path / "paired.json"), "--collection", "kb", "--condition", "warm", "--config-version", "unit", "--limit", "1", "--paired"])
    runner.main()
    assert calls == ["hybrid", "hybrid", "hybrid_graph", "hybrid_graph"]
    for strategy in ("hybrid", "hybrid_graph"):
        report = json.loads((tmp_path / f"paired-{strategy}.json").read_text(encoding="utf-8"))
        assert report["summary"]["cases"] == 1 and report["warmup"]["completed"]
        assert report["session_start_available_gb"] == 6
    assert len(removed) == 4 and all("/conversations/owned-" in path for path in removed)


def test_rate_limit_wait_checks_runtime_memory_and_preserves_server_delay(monkeypatch):
    from scripts import evaluate_professional as runner
    waits = []
    monkeypatch.setattr(runner, "available_gb", lambda: 3)
    monkeypatch.setattr(runner.time, "sleep", waits.append)
    response = SimpleNamespace(headers={"Retry-After": "2"})
    assert runner.wait_rate_limit(response) == 2 and waits == [1, 1]
    monkeypatch.setattr(runner, "available_gb", lambda: .5)
    with pytest.raises(RuntimeError, match="低于1GB"): runner.wait_rate_limit(response)


def test_memory_guard_covers_wait_before_response_headers(monkeypatch):
    import threading
    from scripts import evaluate_professional as runner
    closed = threading.Event()
    monkeypatch.setattr(runner, "available_gb", lambda: .5)
    client = SimpleNamespace(close=closed.set)
    with pytest.raises(RuntimeError, match="低于1GB"):
        with runner.memory_guard(client):
            assert closed.wait(3), "guard must close connections before a response exists"


def test_negative_and_conflict_cases_do_not_accept_keyword_only_answers():
    from scripts.evaluate_professional import score
    missing = dict(id="negative-01", category="negative", query="不存在设备保修几年", answer=None, expect_sources=[], expect_refusal=True)
    def events(answer): return [dict(type="chunk", content=answer), dict(type="done")]
    assert not score(missing, events("无法完全确认，但保修三年。"), 1, 1)["answer_literal_match"]
    assert score(missing, events("没有相关资料，无法确认保修期限。"), 1, 1)["answer_literal_match"]
    conflict = dict(id="negative-20", category="negative", answer="材料冲突，无法确定", expect_sources=["a", "b"], expect_conflict_notice=True,
        source_locations=[dict(filename="a", evidence="预算109万元"), dict(filename="b", evidence="预算209万元")])
    assert not score(conflict, events("没有冲突，最终预算为209万元。"), 1, 1)["answer_literal_match"]
    assert not score(conflict, events("资料冲突，预算209万元。"), 1, 1)["answer_literal_match"]
    assert score(conflict, events("两份材料分别写109万元和209万元，无法确定，应核实来源。"), 1, 1)["answer_literal_match"]


def test_graph_candidate_pool_is_bounded_and_cannot_restore_denied_chunks():
    from app.services.retrieval_service import _merge_graph_candidates, _chunk_key
    ordinary = [RetrievedChunk("ordinary", "a", 1, i, "text", .5) for i in range(12)]
    graph_chunks = [RetrievedChunk("graph", "b", 1, i, "evidence", 0) for i in range(10)]
    merged = {_chunk_key(c): c for c in ordinary + graph_chunks}
    graph_keys = {_chunk_key(c) for c in graph_chunks} | {("denied", 0)}
    picks = _merge_graph_candidates(ordinary, merged, graph_keys, 12)
    assert len(picks) == 12 and picks[:8] == ordinary[:8]
    assert sum(c.document_id == "graph" for c in picks) == 4
    assert not any(c.document_id == "denied" for c in picks)
    assert _merge_graph_candidates(ordinary, merged, {("denied", 0)}, 12) == ordinary


def test_master_graph_uses_one_scoped_retrieval_and_filters_evidence_paths(monkeypatch):
    from app.services import master_graph as master
    from app.services import reranker
    chunk = RetrievedChunk("visible", "source.md", 1, 0, "甲投资乙", .9)
    calls = []
    async def candidates(*args):
        return [chunk], [dict(document_id="visible", chunk_index=0), dict(document_id="dropped", chunk_index=1)]
    async def retrieve(**kwargs):
        calls.append(kwargs)
        assert kwargs["graph_chunks"] == [chunk] and kwargs["scope"] == "scope"
        return [chunk]
    async def duplicate_rerank(*args, **kwargs):
        pytest.fail("master graph must not perform a second reranking")
    monkeypatch.setattr(master, "get_settings", lambda: SimpleNamespace(GRAPH_RETRIEVAL_ENABLED=True, RERANKER_ENABLED=True))
    monkeypatch.setattr(graph, "graph_candidates", candidates)
    monkeypatch.setattr(master, "retrieve_chunks", retrieve)
    monkeypatch.setattr(reranker, "rerank_chunks", duplicate_rerank)
    result = asyncio.run(master._retrieve_node(dict(query="甲的关联", top_k=5, user_scope="scope", retrieval_strategy="hybrid_graph")))
    assert len(calls) == 1 and result["graph_paths"] == [dict(document_id="visible", chunk_index=0)]


def test_acceptance_scores_the_final_visible_answer():
    from scripts.evaluate_professional import score, aggregate
    case = dict(id='test', category='fact', expect_sources=[], answer='125')
    events = [dict(type='chunk', content='125 [Source 1]'),
              dict(type='citation_check', sanitized_answer='无法确定'), dict(type='done')]
    result = score(case, events, 1, 1)
    assert result['answer'] == '无法确定'
    assert not result['answer_literal_match']
    assert result['visible_citations'] == 0
    assert aggregate([result])['citation_coverage'] == 0


def test_multihop_citation_source_recall_requires_both_documents():
    from scripts.evaluate_professional import score
    case = dict(id='chain', category='multihop', expect_sources=['a.md', 'b.md'], answer='终点')
    events = [dict(type='sources', sources=[dict(filename='a.md'), dict(filename='b.md')])]
    one = score(case, events + [dict(type='chunk', content='终点 [Source 2]'), dict(type='done')], 1, 1)
    both = score(case, events + [dict(type='chunk', content='终点 [Source 1-2]'), dict(type='done')], 1, 1)
    assert one['answer_literal_match'] and one['citation_source_recall'] == .5
    assert both['citation_source_recall'] == 1
    invalid = score(case, events + [dict(type='chunk', content='终点 [Source 1-999999999999]'), dict(type='done')], 1, 1)
    assert invalid['citation_source_recall'] == 0
    variant = score(case, events + [dict(type='chunk', content='终点 [source1 – 2]'), dict(type='done')], 1, 1)
    assert variant['citation_source_recall'] == 1 and variant['visible_citations'] == 1


def test_acceptance_reports_quote_fallback_separately_from_generation_quality():
    from scripts.evaluate_professional import score, aggregate
    case = dict(id='english', category='fact', expect_sources=['paper.pdf'], answer='1669')
    events = [dict(type='sources', sources=[dict(filename='paper.pdf')]),
              dict(type='citation_check', total=1, passed=1, fallback='source_quote', generation_check=dict(total=1, passed=0),
                   sanitized_answer='Original evidence: 1669 text chunks [Source 1]'), dict(type='done')]
    result = aggregate([score(case, events, 1, 1)])
    assert result['source_quote_fallback_cases'] == 1
    assert result['generation_citation_support_rate'] == 0
    assert result['citation_support_rate'] == 1
