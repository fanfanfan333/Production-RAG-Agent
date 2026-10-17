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
from contextlib import closing, contextmanager
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
        elif kind == "citation_check":
            citation = event
            if isinstance(event.get("sanitized_answer"), str): answer = event["sanitized_answer"]
        elif kind == "output_guard" and isinstance(event.get("sanitized_answer"), str):
            answer = event["sanitized_answer"]
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
        if re.search(r"(?:\d+(?:\.\d+)?|[一二三四五六七八九十]+)(?:万元|元|瓦|天|年)", normalized):
            correct = False
    elif case.get("expect_conflict_notice"):
        amounts = {value for source in case.get("source_locations", []) for value in re.findall(r"\d+(?:\.\d+)?万元", source.get("evidence", ""))}
        uncertainty = any(word in normalized for word in ("无法确定", "无法唯一确定", "不能确定", "无法判断", "需核实", "需要核实", "有待核实"))
        correct = uncertainty and len(amounts) >= 2 and all(re.search(r"(?<!\d)" + re.escape(value), normalized) for value in amounts)
        if re.search(r"(?:没有|不存在|未发现|无)冲突|最终(?:预算|金额)(?:为|是)", normalized): correct = False
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
            "visible_citations": len(re.findall(r"\[Source\s+\d+(?:-\d+)?\]", answer)),
            "ttft_ms": first_token, "total_ms": elapsed, "stages_ms": stages, "error": error,
            "completed": done, "answer": answer, "sources": sources, "graph_paths": paths,
            "query": case.get("query"), "reference_answer": truth, "source_locations": case.get("source_locations", []),
            "human_review": "pending"}


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
            "citation_coverage": sum(row["visible_citations"] > 0 for row in positives) / len(positives) if positives else None,
            "graph_used_cases": sum(bool(row["graph_paths"]) for row in rows),
            "retrieval_p95_ms": retrieval[math.ceil(.95 * len(retrieval)) - 1] if retrieval else None,
            "ttft_mean_ms": mean("ttft_ms"), "total_mean_ms": mean("total_ms"),
            "errors": sum(bool(row["error"] or not row["completed"]) for row in rows)}


def wait_rate_limit(response):
    try:
        seconds = int(response.headers.get("Retry-After", "60"))
    except ValueError:
        raise RuntimeError("服务返回无效 Retry-After，已停止自动重试") from None
    if not 1 <= seconds <= 120:
        raise RuntimeError("服务限流等待超过本轮重试预算，保留报告后稍后继续")
    print(f"服务限流，按 Retry-After 等待 {seconds} 秒", flush=True)
    for _ in range(seconds):
        if available_gb() < 1: raise RuntimeError("限流等待期间可用内存低于1GB，已停止本轮")
        time.sleep(1)
    return seconds


@contextmanager
def memory_guard(client):
    stop, low_memory = threading.Event(), threading.Event()
    def watch():
        while not stop.wait(1):
            if available_gb() < 1:
                low_memory.set()
                client.close()
                return
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stop.set()
        watcher.join(timeout=2)
        if low_memory.is_set():
            raise RuntimeError("宿主可用内存低于1GB，已关闭本轮连接；服务端已开始的推理可能需要等待超时结束")


def run_case(client, case, collection, strategy, conversations):
    with memory_guard(client):
        return _run_case(client, case, collection, strategy, conversations)


def _run_case(client, case, collection, strategy, conversations):
    if available_gb() < 1:
        raise RuntimeError("运行中宿主可用内存低于1GB，已停止本轮")
    events, first_token = [], None
    started = time.perf_counter()
    response = None
    throttle_wait = 0
    for attempt in range(3):
        response = client.send(client.build_request("POST", "/query", json={"query": case["query"], "top_k": 5, "mode": "knowledge_qa",
            "collection_id": collection, "retrieval_strategy": strategy}), stream=True)
        if response.status_code != 429 or attempt == 2: break
        response.read(); response.close(); throttle_wait += wait_rate_limit(response)
    with closing(response):
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
        except httpx.HTTPError:
            if low_memory.is_set(): raise RuntimeError("宿主可用内存低于1GB，已取消当前请求") from None
            raise
        finally:
            stop.set(); watcher.join(timeout=2)
        if low_memory.is_set(): raise RuntimeError("宿主可用内存低于1GB，已取消当前请求")
    result = score(case, events, (time.perf_counter() - started) * 1000, first_token)
    result["throttle_wait_ms"] = throttle_wait * 1000
    return result


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
    parser.add_argument("--paired", action="store_true", help="Run hybrid then hybrid_graph in one serial, prewarmed session")
    args = parser.parse_args()
    if not 1 <= args.limit <= 120: parser.error("limit 必须在1至120之间")
    token = os.environ.get("RAG_EVAL_TOKEN")
    if not token: parser.error("请通过 RAG_EVAL_TOKEN 设置当前账号令牌；不要写入报告或命令行")
    initial_memory = available_gb()
    if initial_memory < 5: parser.error("宿主可用内存不足5GB，本次不启动真实推理")
    if args.paired and args.condition != "warm": parser.error("paired 只用于预热成对评测")
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    cases = [case for case in gold["cases"] if not args.category or case["category"] == args.category][:args.limit]
    if not cases: parser.error("当前筛选没有用例")
    strategies = ("hybrid", "hybrid_graph") if args.paired else (args.strategy,)
    failed = False
    for strategy in strategies:
        failed = run_report(args, gold, cases, strategy, token, initial_memory) or failed
        if failed: break
    if failed: raise SystemExit(1)


def run_report(args, gold, cases, strategy, token, initial_memory):
    rows, conversations = [], set()
    report = {"gold": gold["name"], "truth_origin": gold["truth_origin"], "condition": args.condition,
              "gold_sha256": hashlib.sha256(args.gold.read_bytes()).hexdigest(), "config_version": args.config_version,
              "strategy": strategy, "collection_id": args.collection, "session_start_available_gb": initial_memory,
              "method": "document/page recall, literal answer/numeric checks plus server citation verifier; requires human answer review", "cases": rows}
    with httpx.Client(base_url=args.url.rstrip("/"), headers={"Authorization": f"Bearer {token}"}, timeout=300, trust_env=False,
        limits=httpx.Limits(max_keepalive_connections=0)) as client:
        identity = client.get("/auth/me"); identity.raise_for_status()
        report["account_scope"] = {key: identity.json().get(key) for key in ("id", "username", "tenant_id", "department_id", "role")}
        try:
            if args.condition == "warm":
                warmup = run_case(client, cases[0], args.collection, strategy, conversations)
                report["warmup"] = {key: warmup[key] for key in ("id", "completed", "error", "total_ms", "stages_ms")}
                if not warmup["completed"] or warmup["error"]: raise RuntimeError("预热请求未完成，不能标记为预热验收")
            for case in cases:
                rows.append(run_case(client, case, args.collection, strategy, conversations))
                print(f"{case['id']}: completed={rows[-1]['completed']}, recall={rows[-1]['recall@5']}", flush=True)
        except (RuntimeError, httpx.HTTPError) as exc:
            report["aborted"] = str(exc)
        finally:
            report["summary"] = aggregate(rows)
            report["by_category"] = {kind: aggregate([row for row in rows if row["category"] == kind]) for kind in sorted({row["category"] for row in rows})}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            output = args.out.with_name(f"{args.out.stem}-{strategy}.json") if args.paired else args.out
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            for identifier in conversations:
                if getattr(client, "is_closed", False):
                    print("低内存导致连接关闭，本轮会话未清理，请按报告核查", flush=True)
                    break
                response = client.delete(f"/conversations/{identifier}")
                if response.status_code not in (200, 204, 404): print("本轮测试会话清理失败，请按报告核查", flush=True)
    return bool(report.get("aborted") or len(rows) != len(cases) or report["summary"]["errors"])


if __name__ == "__main__": main()
