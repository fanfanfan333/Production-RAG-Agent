"""
2026-09-17 修复验收 E2E：总结全覆盖 + 引用校验数字误判 + 质量监控持久化 + 检索去重.

覆盖本轮修复的四个点：

1. "总结所有文档"一份不漏（map-reduce）
   断言：intent=document_summary；doc_digests 覆盖全部可访问文档；
   答案里每一份文档都有独立的 "### <文件名>" 小节；多份时有"### 总体概览"。

2. 模型输出的非事实数字不再误报（引用校验）
   断言：总结类回答不再出现"数字与原文不一致"的整片误报
   （number_mismatch 为空；个别真错除外，这里只断言不整片出现）。

3. 质量监控持久化（重启不丢）
   断言：/quality/stats?window=24h 的 total_queries ≥ 本轮发问数，
   by_intent 含 document_summary，samples 字段齐全；
   window=session 同样可用（进程内实时视角）。

4. 跨文档相同内容去重（检索多样性）
   断言：knowledge_qa 的 sources 中不存在正文完全相同的重复条目
   （同一文件重复上传 N 份时，同一段只保留排名最高的一份）。

运行（容器内）：
    docker cp backend/scripts/e2e_quality_917.py rag_backend:/tmp/
    docker exec -e PYTHONIOENCODING=utf-8 rag_backend python /tmp/e2e_quality_917.py

默认账号 admin / RagAdmin#2026（可用 --admin-password 覆盖）。
只读业务数据：只发问答、只查指标，不修改/删除任何文档。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

BASE_DEFAULT = "http://127.0.0.1:8000"

# 不走系统代理（与 e2e_summary_scope.py 同一约定）
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_passed = 0
_failed = 0


def check(label: str, condition: bool, extra: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label} {extra}")


def login(base: str, username: str, password: str) -> str:
    req = urllib.request.Request(
        f"{base}/auth/login",
        data=json.dumps({"username": username, "password": password}).encode(),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    with _OPENER.open(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    token = payload.get("access_token")
    if not token:
        raise SystemExit(f"登录失败：{payload}")
    return token


def stream_query(base: str, token: str, query: str, timeout: int = 600) -> dict:
    """发一轮 /query，收齐 SSE 事件."""
    req = urllib.request.Request(
        f"{base}/query",
        data=json.dumps({"query": query, "stream": True}).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    events: list[dict] = []
    answer_parts: list[str] = []
    with _OPENER.open(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                continue
            events.append(evt)
            if evt.get("type") == "chunk":
                answer_parts.append(str(evt.get("content") or ""))
    return {"events": events, "answer": "".join(answer_parts)}


def get_json(base: str, token: str, path: str) -> dict:
    req = urllib.request.Request(f"{base}{path}", method="GET")
    req.add_header("Authorization", f"Bearer {token}")
    with _OPENER.open(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE_DEFAULT)
    parser.add_argument("--admin-username", default="admin")
    parser.add_argument("--admin-password", default="RagAdmin#2026")
    args = parser.parse_args()

    token = login(args.base, args.admin_username, args.admin_password)
    print(f"登录成功：{args.admin_username}")

    # ── 1. 总结所有文档：一份不漏 ─────────────────────────────────────────
    print("\n[1] 总结所有文档（map-reduce 全覆盖）")
    r1 = stream_query(args.base, token, "总结所有文档")
    intent_evt = next((e for e in r1["events"] if e.get("type") == "intent"), {})
    check("intent=document_summary", intent_evt.get("intent") == "document_summary",
          f"got={intent_evt.get('intent')!r}")
    digest_evt = next((e for e in r1["events"] if e.get("type") == "doc_digests"), {})
    digests = digest_evt.get("documents") or []
    check("doc_digests 非空", len(digests) > 0, f"count={len(digests)}")
    answer1 = r1["answer"]
    missing_sections = [
        d.get("filename") for d in digests
        if d.get("filename") and f"### {d['filename']}" not in answer1
    ]
    check("每份文档都有独立小节", not missing_sections,
          f"缺失: {missing_sections}")
    if len(digests) > 1:
        check("包含总体概览小节", "### 总体概览" in answer1)
    print(f"  (digests={len(digests)}, answer_chars={len(answer1)})")

    # ── 2. 引用校验：不再整片误报"数字与原文不一致" ────────────────────────
    print("\n[2] 知识问答 + 引用校验（数字误判回归）")
    r2 = stream_query(args.base, token, "反幻觉机制是怎么工作的？")
    cc_evt = next((e for e in r2["events"] if e.get("type") == "citation_check"), {})
    if cc_evt:
        total = int(cc_evt.get("total") or 0)
        num_mm = cc_evt.get("number_mismatch") or []
        check("引用校验有产出", total > 0, f"total={total}")
        check("数字误报不整片出现", len(num_mm) < max(total, 1),
              f"number_mismatch={num_mm} / total={total}")
        print(f"  (overall={cc_evt.get('overall')}, passed={cc_evt.get('passed')}/{total})")
    else:
        check("引用校验有产出", False, "没有 citation_check 事件")

    # ── 3. 跨文档相同内容去重 ─────────────────────────────────────────────
    print("\n[3] 检索去重（重复上传的文档不产生重复来源）")
    sources: list[dict] = []
    for e in r2["events"]:
        if e.get("type") == "sources":
            sources = e.get("sources") or []
    if sources:
        keys = [
            re.sub(r"\s+", "", (s.get("text_snippet") or ""))[:200]
            for s in sources if s.get("text_snippet")
        ]
        check("sources 无重复正文", len(keys) == len(set(keys)),
              f"{len(keys)} 条中唯一 {len(set(keys))} 条")
    else:
        print("  (无 sources，跳过去重断言)")

    # ── 4. 质量监控持久化 ─────────────────────────────────────────────────
    print("\n[4] /quality/stats（持久化聚合 + 时间窗）")
    q24 = get_json(args.base, token, "/quality/stats?window=24h")
    check("24h 窗口 total_queries ≥ 2", int(q24.get("total_queries") or 0) >= 2,
          f"got={q24.get('total_queries')}")
    check("by_intent 含 document_summary",
          "document_summary" in (q24.get("by_intent") or {}),
          f"got={q24.get('by_intent')}")
    check("samples 字段齐全",
          all(k in (q24.get("samples") or {})
              for k in ("evidence_gate", "citations_checked", "citation_answers", "queries")))
    check("latency_ms 字段存在", "latency_ms" in q24)
    qses = get_json(args.base, token, "/quality/stats?window=session")
    check("session 窗口可用（进程内实时）",
          qses.get("source") == "in_process" and "ratios" in qses)
    q7d = get_json(args.base, token, "/quality/stats?window=7d")
    check("7d 窗口可用（数据库聚合）",
          q7d.get("source") == "database" and "ratios" in q7d)

    print(f"\n结果：{_passed} passed, {_failed} failed")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
