"""Check experimental graph gates on paired, warm runs; human review is still required."""
import argparse
import json
from collections import Counter
from pathlib import Path


def compare(baseline, graph):
    for key in ("gold", "gold_sha256", "config_version", "scoring_version", "collection_id", "account_scope", "condition"):
        if baseline.get(key) != graph.get(key): raise ValueError(f"不可比较：{key} 不一致")
    if baseline.get("condition") != "warm" or baseline.get("strategy") != "hybrid" or graph.get("strategy") != "hybrid_graph":
        raise ValueError("需要同范围下的预热 hybrid / hybrid_graph 成对报告")
    left = {row["id"]: row for row in baseline["cases"]}
    right = {row["id"]: row for row in graph["cases"]}
    if len(left) != len(baseline["cases"]) or len(right) != len(graph["cases"]):
        raise ValueError("重复用例 ID，不能用于门禁")
    if (not left or left.keys() != right.keys() or baseline.get("aborted") or graph.get("aborted")
            or baseline.get("in_progress") or graph.get("in_progress")):
        raise ValueError("用例不同或运行中断")
    multi = [key for key in left if left[key]["category"] == "multihop"]
    ordinary = [key for key in left if left[key]["category"] in {"fact", "table", "image"}]
    if not multi or not ordinary: raise ValueError("需要固定多跳和普通问题子集")
    gain = sum(int(right[key]["answer_literal_match"]) - int(left[key]["answer_literal_match"]) for key in multi) / len(multi)
    baseline_latency = baseline["summary"].get("retrieval_p95_ms")
    graph_latency = graph["summary"].get("retrieval_p95_ms")
    summary = graph["summary"]
    expected = {"fact": 40, "table": 25, "image": 15, "multihop": 20, "negative": 20}
    complete = dict(Counter(row["category"] for row in left.values())) == expected
    provenance = bool(baseline.get("gold_sha256") and len(baseline["gold_sha256"]) == 64
                      and baseline.get("config_version") and baseline.get("account_scope", {}).get("id"))
    warmed = all(report.get("warmup_policy") == "per_case"
                 and len(report.get("warmups", [])) == len(left)
                 and {row.get("id") for row in report.get("warmups", [])} == left.keys()
                 and all(row.get("completed") and not row.get("error") for row in report.get("warmups", []))
                 for report in (baseline, graph))
    return {"complete_dataset_gate": complete, "provenance_gate": provenance,
            "scoring_gate": graph.get("scoring_version") == "professional_v3_no_mixed_refusal",
            "warmup_gate": warmed,
            "multihop_gain_pp": round(gain * 100, 2), "multihop_gate": gain >= .1,
            "ordinary_no_regression": all(not left[key]["answer_literal_match"] or right[key]["answer_literal_match"] for key in ordinary),
            "latency_gate": bool(warmed and baseline_latency and graph_latency is not None and graph_latency <= 1.2 * baseline_latency),
            "recall_gate": (summary.get("recall@5") or 0) >= .95,
            "citation_gate": (summary.get("citation_support_rate") or 0) >= .98 and (summary.get("citation_coverage") or 0) == 1,
            "evidence_chain_gate": all(row.get("citation_source_recall") == 1 for row in right.values()
                if row["category"] != "negative" or row.get("recall@5") is not None),
            "error_gate": not summary.get("errors") and not baseline["summary"].get("errors"),
            "graph_usage_gate": (summary.get("graph_used_cases") or 0) > 0,
            "default_enable": False, "note": "自动字面检查不能替代答案人工复核与独立越权门禁；通过后仍需审核发布"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path); parser.add_argument("graph", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text(encoding="utf-8")), json.loads(args.graph.read_text(encoding="utf-8")))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(result[key] for key in ("complete_dataset_gate", "provenance_gate", "scoring_gate", "warmup_gate", "multihop_gate", "ordinary_no_regression", "latency_gate", "recall_gate", "citation_gate", "evidence_chain_gate", "error_gate", "graph_usage_gate")):
        raise SystemExit(1)


if __name__ == "__main__": main()
