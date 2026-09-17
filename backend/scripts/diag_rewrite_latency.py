"""诊断：对比 qwen3 思考开关对改写耗时的真实影响（真实 _REWRITE_SYSTEM_PROMPT）.

背景（2026-09-17 实测，本地 qwen3:8b）：
    真实改写耗时 44.3s，而当时的生产超时是 12s → 改写**每轮都超时**，
    静默回退原查询。整条查询增强层（改写 / 多查询扩展 / 子问题 / HyDE）
    在生产里从未真正生效过。
    根因：query_router 与 graders/retrieval_grader 都写了 reasoning=False，
    唯独改写器漏了 —— 思考 token 既带来 4 倍延迟，又吃光 num_predict。

本脚本就是当时定性的那个对比（换模型后可重跑）：
    thinking=on  np=768 → 38.4s
    thinking=off np=768 → 10.7s   ← 修复后的配置
    thinking=off np=512 → 10.6s

用法（容器内）：
    docker cp backend/scripts/diag_rewrite_latency.py rag_backend:/tmp/diag_rw.py
    docker exec -d rag_backend sh -c "PYTHONIOENCODING=utf-8 python -u /tmp/diag_rw.py > /tmp/diag_rw.log 2>&1"
    docker exec rag_backend cat /tmp/diag_rw.log
"""

from __future__ import annotations

import asyncio
import json
import os
import time

os.environ["QUERY_REWRITE_TIMEOUT_SECONDS"] = "180"

import app.services.query_transform as qt  # noqa: E402
from app.config import get_settings  # noqa: E402

QUESTION = "反幻觉机制是怎么工作的？它和引用校验之间是什么关系？"


async def timed(label: str, **overrides) -> None:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    s = get_settings()
    base = dict(
        model=s.OLLAMA_MODEL,
        base_url=s.OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=768,
        num_ctx=min(4096, s.OLLAMA_NUM_CTX),
        format="json",
    )
    base.update(overrides)
    llm = ChatOllama(**base)
    started = time.perf_counter()
    try:
        res = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=qt._REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=f"当前问题：{QUESTION}"),
            ]),
            timeout=300,
        )
    except asyncio.TimeoutError:
        print(f"  {label}: TIMEOUT >300s", flush=True)
        return

    dt = time.perf_counter() - started
    content = str(res.content)
    parsed = qt._extract_json(content)
    fields = sorted(parsed.keys()) if parsed else None
    variants = len(parsed.get("variants") or []) if parsed else 0
    subqueries = len(parsed.get("subqueries") or []) if parsed else 0
    hyde = len(str(parsed.get("hyde") or "")) if parsed else 0
    print(f"  {label}: {dt:.1f}s  json_ok={parsed is not None} "
          f"variants={variants} subqueries={subqueries} hyde={hyde}chars",
          flush=True)
    print(f"      fields={fields}", flush=True)
    if parsed:
        print(f"      sample={json.dumps(parsed, ensure_ascii=False)[:180]}",
              flush=True)


async def main() -> None:
    print("真实 _REWRITE_SYSTEM_PROMPT，逐种配置测延迟与产出\n")
    await timed("thinking=on  np=768", reasoning=True)
    await timed("thinking=off np=768", reasoning=False)
    await timed("thinking=off np=512", reasoning=False, num_predict=512)


if __name__ == "__main__":
    asyncio.run(main())
