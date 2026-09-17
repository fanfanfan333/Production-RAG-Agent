"""端到端问答冒烟：走真实 POST /query（SSE），打印答案与来源概览.

    python backend/scripts/qa_probe.py "问题1" "问题2" ...

用于真实文档入库后快速验证「检索 → 依据 → 生成 → 引用校验」整条链路：
重点看 **sources 条数 / 类型分布 / 引用校验结果 / 是否拒答**，
而不是只看答案文字好不好看。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = os.getenv("RAG_API", "http://127.0.0.1:8000")
TIMEOUT = int(os.getenv("QA_TIMEOUT", "300"))


def ask(question: str) -> None:
    token = os.getenv("RAG_TOKEN", "")
    body = json.dumps({"query": question}).encode()
    req = urllib.request.Request(
        f"{BASE}/query",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    print("=" * 78)
    print("Q:", question)
    answer_parts: list[str] = []
    sources: list[dict] = []
    verdict = None
    citation = None
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            etype = event.get("type") or event.get("event")
            if etype == "chunk":
                answer_parts.append(event.get("content") or "")
            elif etype == "sources":
                sources = event.get("sources") or []
            elif etype == "intent":
                print("intent:", event.get("intent") or event.get("data"))
            elif etype == "grade":
                print("grade:", json.dumps(event, ensure_ascii=False)[:200])
            elif etype == "output_guard":
                print("output_guard:", json.dumps(event, ensure_ascii=False)[:260])
            elif etype == "error":
                print("SSE error:", event.get("message"))

    answer = "".join(answer_parts).strip()
    print("A:", answer[:900] or "(空)")
    print(f"sources: {len(sources)}")
    for s in sources[:6]:
        name = s.get("filename") or s.get("document_id", "?")
        print(
            f"  - {name} p{s.get('page_number')} "
            f"type={s.get('content_type') or s.get('type')} "
            f"score={s.get('score') or s.get('rerank_score')}"
        )
    if verdict is not None:
        print("verdict:", json.dumps(verdict, ensure_ascii=False)[:300])
    if citation is not None:
        print("citation_check:", json.dumps(citation, ensure_ascii=False)[:300])


if __name__ == "__main__":
    for q in sys.argv[1:]:
        try:
            ask(q)
        except Exception as exc:      # noqa: BLE001
            print("ERROR:", exc)
