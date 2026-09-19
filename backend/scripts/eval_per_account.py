"""按账号口径跑检索金标评测（T3）.

为什么需要它
────────────
此前评测用一份「全库金标」衡量所有账号：admin 的检索已收敛为
``content_scope = {所属租户} + 个人库``（测试公司被剔除、A/B 公司不可见），
用全库金标会让 admin 的指标**假性失真**（把"权限上本就看不到"误判成"检索失败"）。
本脚本按**账号口径**分别评测：每个账号用自己的 ``content_scope`` 建可见集，
金标逐账号定义（``backend/eval/gold/<account>.json``），并额外跑**跨公司探针**
（lisi 问 B公司/测试公司独有内容 → 必须零命中），把隔离断言计入选结果而非注释。

口径（与生产检索链路一致）
────────────────────────
    scope = tenancy.content_scope(user)   # 检索是内容消费路径
    retrieve = retrieve_chunks(query, top_k=10, **scope.acl_kwargs())

指标：recall@1/3/5/10、precision@1/3/5/10、MRR、MAP；逐例 rank/hit 明细 + 探针结果。

只读：不写任何数据库。报告写 ``/tmp/eval_per_account.md``（容器内），由宿主拷出。

用法（容器内）
──────────────
    docker exec -w /app rag_backend python -m scripts.eval_per_account
    docker exec -w /app rag_backend python -m scripts.eval_per_account --account lisi
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

GOLD_DIR = _BACKEND_ROOT / "eval" / "gold"
DEFAULT_ACCOUNTS = ("admin", "lisi", "wangwu")
TOP_K = 10


def _resolve_ids(by_name: dict[str, list[str]], filename: str) -> list[str]:
    return by_name.get(filename, [])


async def _load_doc_ids(names: set[str]) -> dict[str, list[str]]:
    from sqlalchemy import text

    from app.db.postgres import get_db_session

    if not names:
        return {}
    async with get_db_session() as session:
        rows = (
            await session.execute(
                text("select filename, id::text from documents where filename = any(:n)"),
                {"n": list(names)},
            )
        ).all()
    out: dict[str, list[str]] = {}
    for filename, doc_id in rows:
        out.setdefault(filename, []).append(doc_id)
    return out


async def run_account(account: str) -> dict:
    from app.services.evaluation import EvalCase, EvalSet, evaluate, item_from_chunk
    from app.services.retrieval_service import retrieve_chunks
    from app.services.tenancy import content_scope, request_scope

    gold = json.loads((GOLD_DIR / f"{account}.json").read_text(encoding="utf-8"))
    username = gold.get("username", gold["account"])

    # ── 解析逻辑坐标 + 探针禁用文档 ──────────────────────────────────────────
    wanted: set[str] = set()
    for case in gold["cases"]:
        wanted.update(it["filename"] for it in case["expect"])
    forbidden_names: set[str] = set()
    for probe in gold["probes"]:
        forbidden_names.update(probe["forbidden_docs"])
    by_name = await _load_doc_ids(wanted | forbidden_names)

    problems: list[str] = []
    cases: list[EvalCase] = []
    for case in gold["cases"]:
        rel: list[str] = []
        for it in case["expect"]:
            ids = _resolve_ids(by_name, it["filename"])
            if not ids:
                problems.append(f"金标文档缺失：{it['filename']!r}（query={case['query']!r}）")
                continue
            rel.extend(f"{doc_id}::{it['chunk_index']}" for doc_id in ids)
        cases.append(EvalCase(query=case["query"], relevant=frozenset(rel),
                              modality="text", note=case.get("note", "")))

    from sqlalchemy import select

    from app.db.postgres import get_db_session
    from app.db.user_models import User

    async with get_db_session() as session:
        user = (
            await session.execute(select(User).where(User.username == username))
        ).scalars().first()
    if user is None:
        return {"account": account, "error": f"账号不存在：{username}"}

    req = await request_scope(user)
    scope = await content_scope(user)

    # ── scope 口径断言 ───────────────────────────────────────────────────────
    exp = gold.get("expected_scope", {})
    scope_checks: list[dict] = []

    def _chk(label: str, got, want) -> None:
        g = sorted(got)
        w = sorted(want)
        scope_checks.append({"label": label, "got": g, "want": w, "ok": g == w})

    if exp.get("request_tenant_ids") is not None:
        _chk("request_scope.tenant_ids", req.tenant_ids or frozenset(), exp["request_tenant_ids"])
    if exp.get("content_tenant_ids") is not None:
        _chk("content_scope.tenant_ids", scope.tenant_ids or frozenset(), exp["content_tenant_ids"])
    if exp.get("content_owns_tenant_ids") is not None:
        _chk("content_scope.owns_tenant_ids", scope.owns_tenant_ids, exp["content_owns_tenant_ids"])

    # ── 评测 ─────────────────────────────────────────────────────────────────
    async def _retrieve(query: str):
        chunks = await retrieve_chunks(query, top_k=TOP_K, **scope.acl_kwargs())
        return [item_from_chunk(c) for c in chunks]

    eval_set = EvalSet(name=f"gold-{account}", cases=tuple(cases),
                       description=gold.get("note", ""))
    report = await evaluate(_retrieve, eval_set, k_values=(1, 3, 5, 10))

    # ── 跨公司 / 越权探针（必须零命中）───────────────────────────────────────
    probe_results: list[dict] = []
    for probe in gold["probes"]:
        forbidden_ids: set[str] = set()
        for name in probe["forbidden_docs"]:
            forbidden_ids.update(_resolve_ids(by_name, name))
        chunks = await retrieve_chunks(probe["query"], top_k=TOP_K, **scope.acl_kwargs())
        got_ids = {str(c.document_id) for c in chunks}
        leaked_ids = got_ids & forbidden_ids
        leaked_names = sorted(
            n for n in probe["forbidden_docs"]
            if set(_resolve_ids(by_name, n)) & leaked_ids
        )
        probe_results.append({
            "query": probe["query"],
            "why": probe.get("why", ""),
            "returned": len(chunks),
            "leaked_docs": leaked_names,
            "ok": not leaked_ids,
        })

    return {
        "account": account,
        "username": username,
        "problems": problems,
        "scope": {
            "request_tenant_ids": sorted(req.tenant_ids or []),
            "content_tenant_ids": sorted(scope.tenant_ids or []),
            "content_owns_tenant_ids": sorted(scope.owns_tenant_ids),
            "department_id": scope.department_id,
            "tenant_wide": scope.tenant_wide,
        },
        "scope_checks": scope_checks,
        "report": report,
        "probes": probe_results,
    }


def _cells(metrics: dict, ks) -> str:
    """把一个 k→值 的字典渲染成 4 个表格单元格（`a | b | c | d`）。"""
    return " | ".join(
        f"{metrics.get(k)}" if metrics.get(k) is not None else "—" for k in ks
    )


def render_markdown(results: list[dict]) -> str:
    from datetime import datetime, timezone

    ks = (1, 3, 5, 10)
    lines: list[str] = []
    lines.append("# 逐账号检索金标评测（T3）")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now(tz=timezone.utc).isoformat()}")
    lines.append("- 口径：`scope = tenancy.content_scope(user)`；`retrieve_chunks(query, top_k=10, **scope.acl_kwargs())`")
    lines.append("- 金标定义：`backend/eval/gold/<account>.json`（(filename, chunk_index) 逻辑坐标）")
    lines.append("- 探针：跨公司 / 测试公司 / 他人个人库内容**必须零命中**，计入断言")
    lines.append("")

    for r in results:
        lines.append(f"## 账号 `{r['account']}`（{r.get('username')}）")
        lines.append("")
        if r.get("error"):
            lines.append(f"**错误**：{r['error']}")
            lines.append("")
            continue

        sc = r["scope"]
        lines.append(
            f"- content_scope.tenant_ids = `{sc['content_tenant_ids']}`；"
            f"owns = `{sc['content_owns_tenant_ids']}`；"
            f"department_id = `{sc['department_id']}`；tenant_wide = `{sc['tenant_wide']}`"
        )
        lines.append(f"- request_scope.tenant_ids = `{sc['request_tenant_ids']}`")
        bad_scope = [c for c in r["scope_checks"] if not c["ok"]]
        lines.append(
            f"- 口径断言：{'全部通过' if not bad_scope else '**失败** ' + json.dumps(bad_scope, ensure_ascii=False)}"
        )
        lines.append("")

        rep = r["report"]
        lines.append("### 指标")
        lines.append("")
        lines.append("| 指标 | @1 | @3 | @5 | @10 |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| recall | {_cells(rep.recall, ks)} |")
        lines.append(f"| precision | {_cells(rep.precision, ks)} |")
        lines.append(f"| ndcg | {_cells(rep.ndcg, ks)} |")
        lines.append(f"| hit | {_cells(rep.hit_rate, ks)} |")
        lines.append("")
        lines.append(f"- **MRR = {rep.mrr}**，MAP = {rep.map_score}，scored = {rep.scored_cases}/{rep.total_cases}")
        lines.append("")

        lines.append("### 逐例明细")
        lines.append("")
        lines.append("| # | query | recall@10 | precision@10 | rr | 命中(rank) | 漏召回 |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, c in enumerate(rep.cases, 1):
            missed = "、".join(x.split("::")[-1] for x in c.missed) or "—"
            lines.append(
                f"| {i} | {c.query} | {c.recall.get(10)} | {c.precision.get(10)} | "
                f"{c.rr} | {c.hits.get(10)} | {missed} |"
            )
        lines.append("")

        lines.append("### 跨公司 / 越权探针（期望零命中）")
        lines.append("")
        lines.append("| # | query | 返回条数 | 泄漏文档 | 结果 |")
        lines.append("|---|---|---|---|---|")
        for i, p in enumerate(r["probes"], 1):
            lines.append(
                f"| {i} | {p['query']} | {p['returned']} | "
                f"{'、'.join(p['leaked_docs']) or '无'} | {'PASS' if p['ok'] else 'FAIL'} |"
            )
        lines.append("")
        if r["problems"]:
            lines.append("### 金标解析问题")
            lines.append("")
            for p in r["problems"]:
                lines.append(f"- {p}")
            lines.append("")

    # 汇总
    lines.append("## 汇总（recall@10 / precision@3 / MRR / 探针）")
    lines.append("")
    lines.append("| 账号 | recall@1 | recall@3 | recall@5 | recall@10 | precision@3 | MRR | 探针 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['account']} | — | — | — | — | — | — | 错误 |")
            continue
        rep = r["report"]
        probes_ok = all(p["ok"] for p in r["probes"])
        lines.append(
            f"| {r['account']} | {rep.recall.get(1)} | {rep.recall.get(3)} | "
            f"{rep.recall.get(5)} | {rep.recall.get(10)} | {rep.precision.get(3)} | "
            f"{rep.mrr} | {'全部零命中' if probes_ok else '**有泄漏**'} |"
        )
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", default="", help="只跑指定账号（默认全部）")
    ap.add_argument("--out", default="/tmp/eval_per_account.md")
    args = ap.parse_args()

    accounts = (
        [args.account] if args.account else list(DEFAULT_ACCOUNTS)
    )
    results: list[dict] = []
    for acc in accounts:
        results.append(await run_account(acc))

    md = render_markdown(results)
    Path(args.out).write_text(md, encoding="utf-8")

    # stdout：给宿主机捕获的机读摘要
    summary = {
        "accounts": [
            {
                "account": r["account"],
                "error": r.get("error"),
                "recall": r["report"].recall if not r.get("error") else None,
                "precision": r["report"].precision if not r.get("error") else None,
                "mrr": r["report"].mrr if not r.get("error") else None,
                "probes_ok": all(p["ok"] for p in r.get("probes", [])) if not r.get("error") else None,
                "scope_checks_ok": all(c["ok"] for c in r.get("scope_checks", [])) if not r.get("error") else None,
            }
            for r in results
        ],
        "report_path": args.out,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failed = any(
        (r.get("error") is not None)
        or any(not c["ok"] for c in r.get("scope_checks", []))
        or any(not p["ok"] for p in r.get("probes", []))
        for r in results
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
