"""Measure scoped, real-model retrieval latency and accuracy under bounded load."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


async def main(args) -> int:
    from sqlalchemy import select
    from app.db.postgres import get_db_session, get_engine
    from app.db.user_models import User
    from app.services.retrieval_service import retrieve_chunks_scoped
    from app.services.security_scope import request_security_scope
    from scripts.run_eval_baseline import resolve_labels

    async with get_db_session() as session:
        user = await session.scalar(select(User).where(User.username == args.username))
    assert user is not None, "benchmark user is missing"
    scope = await request_security_scope(user)
    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    cases, problems, excluded = await resolve_labels(golden, scope)
    assert not problems and cases, f"golden labels unavailable: {problems}"

    async def retrieve(case):
        started = time.perf_counter()
        chunks = await retrieve_chunks_scoped(case["query"], scope=scope, top_k=3)
        ids = {f"{c.document_id}::{c.chunk_index}" for c in chunks}
        return {"seconds": time.perf_counter() - started, "hit": bool(ids & set(case["relevant"]))}

    await retrieve(cases[0])  # Separate model loading from warm serving latency.
    results = []
    for concurrency in args.concurrency:
        semaphore = asyncio.Semaphore(concurrency)

        async def one(index):
            async with semaphore:
                try:
                    return await retrieve(cases[index % len(cases)])
                except Exception as exc:
                    return {"error": type(exc).__name__}

        started = time.perf_counter()
        samples = await asyncio.gather(*(one(i) for i in range(args.samples)))
        elapsed = time.perf_counter() - started
        times = [s["seconds"] for s in samples if "seconds" in s]
        results.append({
            "concurrency": concurrency,
            "requests": len(samples),
            "errors": sum("error" in s for s in samples),
            "hit_rate@3": sum(s.get("hit", False) for s in samples) / len(samples),
            "requests_per_second": round(len(samples) / elapsed, 3),
            "p50_seconds": round(percentile(times, .50), 3) if times else None,
            "p95_seconds": round(percentile(times, .95), 3) if times else None,
            "p99_seconds": round(percentile(times, .99), 3) if times else None,
            "samples": samples,
        })
        print(json.dumps({k: v for k, v in results[-1].items() if k != "samples"}), flush=True)
    Path(args.out).write_text(json.dumps({"scope": args.username, "golden": args.golden, "excluded_labels": excluded, "stage": "warm scoped retrieval including embedding, hybrid search and reranking; excludes generation", "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    await get_engine().dispose()
    return int(any(r["errors"] or r["hit_rate@3"] < 1.0 for r in results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default="admin")
    parser.add_argument("--golden", default="eval/golden_v1.json")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--out", default="/tmp/retrieval-load.json")
    args = parser.parse_args()
    if args.samples < 1 or not args.concurrency or any(c < 1 or c > 64 for c in args.concurrency):
        parser.error("samples must be positive and concurrency must be between 1 and 64")
    raise SystemExit(asyncio.run(main(args)))
