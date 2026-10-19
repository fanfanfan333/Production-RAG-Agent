"""Keep the full format regression and independent hard multi-hop experiment separate."""
import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.compare_professional_reports import compare


QUALITY_GATES = (
    "provenance_gate", "scoring_gate", "warmup_gate", "ordinary_no_regression",
    "latency_gate", "recall_gate", "citation_gate", "evidence_chain_gate",
    "error_gate", "graph_usage_gate",
)


def check_release(full_baseline, full_graph, hard_baseline, hard_graph):
    full = compare(full_baseline, full_graph)
    hard = compare(hard_baseline, hard_graph)
    for key in ("config_version", "scoring_version", "account_scope", "condition"):
        if full_graph.get(key) != hard_graph.get(key):
            raise ValueError(f"完整回归与独立难题配置不一致：{key}")
    if (full_graph["gold_sha256"] == hard_graph["gold_sha256"]
            or full_graph["collection_id"] == hard_graph["collection_id"]):
        raise ValueError("难题必须使用独立金标及知识库，不能混入原完整集")
    counts = Counter(row["category"] for row in hard_graph["cases"])
    full_left = {row["id"]: row for row in full_baseline["cases"]}
    full_right = {row["id"]: row for row in full_graph["cases"]}
    regression = all(not row["answer_literal_match"] or full_right[key]["answer_literal_match"]
                     for key, row in full_left.items())
    full_pass = full["complete_dataset_gate"] and regression and all(full[key] for key in QUALITY_GATES)
    hard_pass = counts == {"multihop": 20, "fact": 4} and hard["multihop_gate"] and all(hard[key] for key in QUALITY_GATES)
    return {
        "full_120": full, "independent_24": hard,
        "full_regression_gate": full_pass, "all_categories_no_regression": regression,
        "independent_gain_gate": hard_pass,
        "automated_quality_pass": bool(full_pass and hard_pass),
        "human_review_gate": False, "release_approved": False,
        "note": "原120例检验完整格式与不退化；独立20道多跳验证至少10个百分点增益。"
                "不拼成144例金标。此工具不代签人工审核，也不代替独立权限与资源验收。",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full_baseline", "full_graph", "hard_baseline", "hard_graph"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = check_release(*(json.loads(getattr(args, name).read_text(encoding="utf-8"))
                             for name in ("full_baseline", "full_graph", "hard_baseline", "hard_graph")))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["automated_quality_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
