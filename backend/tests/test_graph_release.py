from copy import deepcopy

import pytest

from scripts.check_graph_release import check_release


def pair(counts, gold, kb, gain=False):
    rows = [dict(id=f"{kind}-{i}", category=kind, answer_literal_match=True,
                 citation_source_recall=1, **{"recall@5": 1})
            for kind, count in counts.items() for i in range(count)]
    baseline = dict(gold=gold, gold_sha256=gold * 64, collection_id=kb,
        config_version="same", scoring_version="professional_v3_no_mixed_refusal",
        account_scope={"id": "owner"}, condition="warm", strategy="hybrid",
        warmup_policy="per_case", cases=rows,
        warmups=[dict(id=row["id"], completed=True, error=None) for row in rows],
        summary={"retrieval_p95_ms": 100, "recall@5": 1, "citation_support_rate": 1,
                 "citation_coverage": 1, "errors": 0, "graph_used_cases": 0})
    graph = deepcopy(baseline)
    graph["strategy"] = "hybrid_graph"
    graph["summary"]["graph_used_cases"] = len(rows)
    if gain:
        for row in baseline["cases"]:
            if row["category"] == "multihop": row["answer_literal_match"] = False
    return baseline, graph


def reports():
    return (*pair({"fact": 40, "table": 25, "image": 15, "multihop": 20, "negative": 20}, "a", "full"),
            *pair({"multihop": 20, "fact": 4}, "b", "hard", gain=True))


def test_full_ceiling_is_not_a_fake_gain_and_ai_cannot_sign_release():
    result = check_release(*reports())
    assert result["automated_quality_pass"]
    assert not result["full_120"]["multihop_gate"]
    assert result["independent_24"]["multihop_gain_pp"] == 100
    assert not result["human_review_gate"] and not result["release_approved"]


def test_negative_regression_and_partial_formats_block_release():
    values = reports()
    values[1]["cases"][-1]["answer_literal_match"] = False
    assert not check_release(*values)["full_regression_gate"]
    values = reports()
    for report in values[:2]:
        report["cases"] = [row for row in report["cases"] if row["category"] != "image"]
    assert not check_release(*values)["full_regression_gate"]


@pytest.mark.parametrize("key,value", [("config_version", "other"), ("account_scope", {"id": "other"}),
                                      ("gold_sha256", "a" * 64), ("collection_id", "full")])
def test_mismatched_or_reused_experiment_is_rejected(key, value):
    values = reports()
    for report in values[2:]: report[key] = value
    with pytest.raises(ValueError): check_release(*values)


def test_late_warmup_failure_or_slow_graph_cannot_be_hidden_by_challenge_gain():
    values = reports()
    values[1]["warmups"][-1]["error"] = "timeout"
    assert not check_release(*values)["automated_quality_pass"]
    values = reports()
    values[1]["summary"]["retrieval_p95_ms"] = 121
    assert not check_release(*values)["automated_quality_pass"]


def test_unfinished_checkpoint_is_never_a_release_result():
    values = reports()
    values[1]["in_progress"] = True
    with pytest.raises(ValueError, match="运行中断"):
        check_release(*values)
