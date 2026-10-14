"""Serial API acceptance runner. Requires an existing account and explicitly uploaded fixture documents."""
import argparse
import hashlib
import ctypes
import json
import math
import os
import re
import time
import threading
from pathlib import Path

import httpx


def available_gb():
    if os.name != "nt":
        raise RuntimeError("真实推理验收请从 Windows 宿主运行，以检查宿主可用内存")
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in
            ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "extended")]
    state = Memory()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise RuntimeError("无法检查宿主内存")
    return state.avail_phys / 1024**3


def score(case, events, elapsed, first_token):
    sources, answer, stages, citation, done, error, paths = [], "", {}, {}, False, None, []
    for event in events:
        kind = event.get("type")
        if kind == "sources": sources = event.get("sources", [])
        elif kind == "chunk": answer += event.get("content", "")
        elif kind == "stage_timing": stages[event["stage"]] = stages.get(event["stage"], 0) + event["duration_ms"]
        elif kind == "citation_check": citation = event
        elif kind == "done": done = True
        elif kind == "error": error = event.get("message", "unknown error")
        elif kind == "graph_retrieval": paths = event.get("paths", [])
    expected = case["expect_sources"]
    locations = {row["filename"]: row for row in case.get("source_locations", [])}
    # Each expected document contributes once even if several chunks of it were returned.
    ranks = [next(i + 1 for i, source in enumerate(sources[:5]) if source.get("filename") == name
                  and source.get("page_number", 1) == locations.get(name, {}).get("page_number", 1))
             for name in expected if any(source.get("filename") == name
                 and source.get("page_number", 1) == locations.get(name, {}).get("page_number", 1) for source in sources[:5])]
    normalized = re.sub(r"\s+", "", answer)
    truth = case.get("answer")
    if case.get("expect_refusal"):
        correct = any(word in normalized for word in ("无法", "不足", "未找到", "未检索", "没有相关"))
    elif case.get("expect_conflict_notice"):
        correct = "冲突" in normalized or "不一致" in normalized
    else:
        literal = re.sub(r"\s+", "", truth or "")
        pattern = (r"(?<!\d)" if literal[:1].isdigit() else "") + re.escape(literal)
        pattern += r"(?!\d)" if literal[-1:].isdigit() else ""
        correct = bool(re.search(pattern, normalized))
    correct = bool(correct and done and not error)
    numeric_truth = re.findall(r"\d+(?:\.\d+)?(?:万元|瓦|天|年)", truth or "")
    numeric_match = all(re.search(r"(?<!\d)" + re.escape(value), normalized) for value in numeric_truth) if numeric_truth else None
    return {"id": case["id"], "category": case["category"], "recall@5": len(ranks) / len(expected) if expected else None,
            "mrr": 1 / min(ranks) if ranks else (0 if expected else None),
            "answer_literal_match": correct and numeric_match is not False, "numeric_unit_match": numeric_match,
            "citation_total": citation.get("total", 0), "citation_passed": citation.get("passed", 0),
            "ttft_ms": first_token, "total_ms": elapsed, "stages_ms": stages, "error": error,
            "completed": done, "answer": answer, "sources": sources, "graph_paths": paths}


def aggregate(rows):
    def mean(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None
    retrieval = sorted(row["stages_ms"]["retrieve"] for row in rows if "retrieve" in row["stages_ms"])
    total = sum(row["citation_total"] for row in rows)
    positives = [row for row in rows if row["category"] != "negative"]
    return {"cases": len(rows), "recall@5": mean("recall@5"), "mrr": mean("mrr"),
            "answer_literal_match": mean("answer_literal_match"), "numeric_unit_match": mean("numeric_unit_match"),
            "citation_support_rate": sum(row["citation_passed"] for row in rows) / total if total else None,
            "citation_coverage": sum(row["citation_total"] > 0 for row in positives) / len(positives) if positives else None,
            "graph_used_cases": sum(bool(row["graph_paths"]) for row in rows),
            "retrieval_p95_ms": retrieval[math.ceil(.95 * len(retrieval)) - 1] if retrieval else None,
            "ttft_mean_ms": mean("ttft_ms"), "total_mean_ms": mean("total_ms"),
            "errors": sum(bool(row["error"] or not row["completed"]) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path(__file__).resolve().parents[1] / "eval/professional_v1.json")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collection", required=True, help="Isolated, already-uploaded fixture knowledge base UUID")
    parser.add_argument("--strategy", choices=("hybrid", "hybrid_graph"), default="hybrid")
    parser.add_argument("--condition", choices=("cold", "warm", "cache_hit"), required=True, help="Operator-observed condition, not inferred from speed")
    parser.add_argument("--config-version", required=True, help="Same application/model/retrieval configuration version for paired runs")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--category", choices=("fact", "table", "image", "multihop", "negative"))
    args = parser.parse_args()
    if not 1 <= args.limit <= 120: parser.error("limit 必须在1至120之间")
    token = os.environ.get("RAG_EVAL_TOKEN")
    if not token: parser.error("请通过 RAG_EVAL_TOKEN 设置当前账号令牌；不要写入报告或命令行")
    if available_gb() < 5: parser.error("宿主可用内存不足5GB，本次不启动真实推理")
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    cases = [case for case in gold["cases"] if not args.category or case["category"] == args.category][:args.limit]
    rows, conversations = [], set()
    report = {"gold": gold["name"], "truth_origin": gold["truth_origin"], "condition": args.condition,
              "gold_sha256": hashlib.sha256(args.gold.read_bytes()).hexdigest(), "config_version": args.config_version,
              "strategy": args.strategy, "collection_id": args.collection,
              "method": "document/page recall, literal answer/numeric checks plus server citation verifier; requires human answer review", "cases": rows}
    with httpx.Client(base_url=args.url.rstrip("/"), headers={"Authorization": f"Bearer {token}"}, timeout=300, trust_env=False) as client:
        identity = client.get("/auth/me"); identity.raise_for_status()
        report["account_scope"] = {key: identity.json().get(key) for key in ("id", "username", "tenant_id", "department_id", "role")}
        try:
            for case in cases:
                if available_gb() < 5: raise RuntimeError("下一例启动前宿主内存不足5GB，已停止")
                events, first_token = [], None
                started = time.perf_counter()
                with client.stream("POST", "/query", json={"query": case["query"], "top_k": 5, "mode": "knowledge_qa",
                    "collection_id": args.collection, "retrieval_strategy": args.strategy}) as response:
                    response.raise_for_status()
                    stop, low_memory = threading.Event(), threading.Event()
                    def watch_memory():
                        while not stop.wait(1):
                            if available_gb() < 1:
                                low_memory.set(); response.close(); return
                    watcher = threading.Thread(target=watch_memory, daemon=True)
                    watcher.start()
                    try:
                        for line in response.iter_lines():
                            if low_memory.is_set(): raise RuntimeError("宿主可用内存低于1GB，已取消当前请求")
                            if not line.startswith("data: ") or line[6:] == "[DONE]": continue
                            event = json.loads(line[6:]); events.append(event)
                            if event.get("conversation_id"): conversations.add(event["conversation_id"])
                            if event.get("type") == "chunk" and first_token is None: first_token = (time.perf_counter() - started) * 1000
                    finally:
                        stop.set(); watcher.join(timeout=2)
                    if low_memory.is_set(): raise RuntimeError("宿主可用内存低于1GB，已取消当前请求")
                rows.append(score(case, events, (time.perf_counter() - started) * 1000, first_token))
                print(f"{case['id']}: completed={rows[-1]['completed']}, recall={rows[-1]['recall@5']}", flush=True)
        except (RuntimeError, httpx.HTTPError) as exc:
            report["aborted"] = str(exc)
        finally:
            report["summary"] = aggregate(rows)
            report["by_category"] = {kind: aggregate([row for row in rows if row["category"] == kind]) for kind in sorted({row["category"] for row in rows})}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            for identifier in conversations:
                response = client.delete(f"/conversations/{identifier}")
                if response.status_code not in (200, 204, 404): print("本轮测试会话清理失败，请按报告核查", flush=True)
    if report.get("aborted") or len(rows) != len(cases) or report["summary"]["errors"]: raise SystemExit(1)


if __name__ == "__main__": main()
