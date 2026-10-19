import json
from types import SimpleNamespace

from scripts import evaluate_professional as runner


def test_correct_value_with_unresolved_refusal_is_not_a_correct_positive_answer():
    case = dict(id="positive", category="multihop", answer="玉兰园区", expect_sources=[])
    def events(answer):
        return [dict(type="chunk", content=answer), dict(type="done")]
    assert not runner.score(case, events("位于玉兰园区。无法确定，需要核实。"), 1, 1)["answer_literal_match"]
    assert runner.score(case, events("位于玉兰园区。"), 1, 1)["answer_literal_match"]


def test_paired_run_prewarms_each_query_before_measuring(monkeypatch, tmp_path):
    cases = [dict(id=f"case-{i}", category="fact") for i in range(2)]
    gold = dict(name="fixture", truth_origin="synthetic", cases=cases)
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(gold), encoding="utf-8")
    args = SimpleNamespace(gold=path, paired=True, condition="warm", config_version="test",
                           collection="kb", url="http://local", out=tmp_path / "result.json")
    calls = []
    class Client:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, path):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(id="owner"))
    def run_case(client, case, collection, strategy, conversations):
        checkpoint = json.loads((tmp_path / "result-hybrid.json").read_text())
        assert checkpoint["in_progress"]
        assert checkpoint["expected_cases"] == 2
        if case["id"] == "case-1":
            assert checkpoint["cases"][0]["id"] == "case-0"
        calls.append(case["id"])
        return dict(id=case["id"], category="fact", completed=True, error=None,
                    total_ms=1, stages_ms={}, **{"recall@5": 1})
    monkeypatch.setattr(runner.httpx, "Client", lambda **kwargs: Client())
    monkeypatch.setattr(runner, "run_case", run_case)
    monkeypatch.setattr(runner, "aggregate", lambda rows: dict(cases=len(rows), errors=0))
    assert not runner.run_report(args, gold, cases, "hybrid", "test-token", 6)
    assert calls == ["case-0", "case-0", "case-1", "case-1"]
    report = json.loads((tmp_path / "result-hybrid.json").read_text(encoding="utf-8"))
    assert report["summary"]["cases"] == 2
    assert not report["in_progress"]
    assert report["warmup_policy"] == "per_case"
    assert [row["id"] for row in report["warmups"]] == ["case-0", "case-1"]
