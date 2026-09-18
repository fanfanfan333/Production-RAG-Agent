"""跑一轮金标评测并判定是否回归（企业级回归门禁）.

它解决的问题
────────────
在它之前，评测这件事只有"跑一次看数字"：``/eval/history`` 是进程内 deque，
后端一重启就清空，于是"上周 0.91、这周 0.86"这种判断根本做不出来 —— 每次
改检索参数都只能看单次绝对值拍脑袋。

本脚本：
  1. 从 ``backend/eval/golden_v1.json`` 读金标（标注用 filename+chunk_index
     逻辑坐标，运行期解析成 ``document_id::chunk_index``，语料重导不用改标注）；
  2. 以管理员身份 ``POST /eval/run``，真实走生产同一条检索链路；
  3. 复核 ``GET /eval/history`` 里确实出现了这一轮（验证**落库**生效，
     而不是"跑完就没了"）；
  4. 单独做负例拒答检查（库里没有的问题必须拒答）—— 召回率单指标会被
     "把阈值调低换取召回"刷分，必须与拒答一起看；
  5. 与 ``thresholds`` 逐项比对，回归则退出码 2（可直接当 CI 门禁用）。

用法（容器内）
    RAG_EVAL_PASSWORD=... python -u /tmp/run_eval_baseline.py
凭证刻意不写进代码：从环境变量读，避免把口令提交进仓库。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone


def _http(base: str, path: str, *, method: str = "GET", body: dict | None = None,
          token: str | None = None, timeout: int = 300) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{base}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"ok": True, "status": resp.status, "body": json.loads(resp.read())}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        return {"ok": False, "status": exc.code, "error": raw[:800]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": 0, "error": f"{type(exc).__name__}: {exc}"}


async def resolve_labels(golden: dict) -> tuple[list[dict], list[str]]:
    """把 (filename, chunk_index) 逻辑坐标解析成 ``document_id::chunk_index``."""
    from sqlalchemy import text

    from app.db.postgres import get_db_session

    wanted = {
        item["filename"]
        for case in golden["cases"]
        for item in case["expect"]
    }
    async with get_db_session() as session:
        rows = (
            await session.execute(
                text(
                    "select filename, id::text from documents "
                    "where filename = any(:names)"
                ),
                {"names": list(wanted)},
            )
        ).all()
    by_name: dict[str, list[str]] = {}
    for filename, doc_id in rows:
        by_name.setdefault(filename, []).append(doc_id)

    problems: list[str] = []
    cases: list[dict] = []
    for case in golden["cases"]:
        relevant: list[str] = []
        for item in case["expect"]:
            name = item["filename"]
            ids = by_name.get(name)
            if not ids:
                problems.append(f"金标引用的文档不存在：{name!r}（query={case['query']!r}）")
                continue
            for doc_id in ids:
                relevant.append(f"{doc_id}::{item['chunk_index']}")
        cases.append({"query": case["query"], "relevant": relevant,
                      "modality": "text", "note": case.get("note", "")})
    return cases, problems


async def check_negatives(golden: dict, username: str) -> list[dict]:
    """
    负例拒答检查：库里没有的问题，精排最高分必须低于证据闸阈值.

    scope 刻意与 ``POST /eval/run`` **完全同源**（同一个 ``tenancy.scope_for``），
    而不是"随便给一个大范围"：
      * 用窄口径 → 会把"权限上看不到"误判成"库里不存在"，负例检查退化成
        永远通过的假测试；
      * 用 ``owner_id=None`` 的伪全量口径 → 私有库整片看不见，而本语料里
        绝大多数内容是 private，等于把检查范围悄悄缩小了 90%。
    所以这里把调用者自己按 scope_for 展开，与评测链路一字不差。
    """
    from sqlalchemy import select

    from app.config import get_settings
    from app.db.postgres import get_db_session
    from app.db.user_models import User
    from app.services.retrieval_service import retrieve_chunks
    from app.services.tenancy import scope_for

    settings = get_settings()
    async with get_db_session() as session:
        user = (
            await session.execute(select(User).where(User.username == username))
        ).scalars().first()
    if user is None:
        return [{"query": nc["query"], "error": f"user {username!r} not found",
                 "refused": False} for nc in golden["negative_cases"]]
    scope = scope_for(user)

    out: list[dict] = []
    for case in golden["negative_cases"]:
        try:
            chunks = await retrieve_chunks(
                case["query"], top_k=10,
                owner_id=str(scope.owner_id) if scope.owner_id else None,
                tenant_id=scope.tenant_id,
                user_department_id=scope.department_id,
                tenant_wide=scope.tenant_wide,
                platform_wide=scope.platform_wide,
            )
            top = max((c.score for c in chunks), default=0.0)
            top_doc = chunks[0].document_id if chunks else None
            passed = top < float(settings.EVIDENCE_GATE_MIN_TOP_SCORE)
            out.append({
                "query": case["query"], "returned": len(chunks),
                "top_score": round(top, 6), "top_document_id": top_doc,
                "threshold": settings.EVIDENCE_GATE_MIN_TOP_SCORE,
                "refused": passed, "note": case.get("note", ""),
            })
        except Exception as exc:  # noqa: BLE001
            out.append({"query": case["query"], "error": f"{type(exc).__name__}: {exc}",
                        "refused": False})
    return out


async def measure_band_headroom(
    golden: dict, username: str, resolved_cases: list[dict], top_k: int = 10
) -> dict:
    """
    在**关闭阈值过滤**的那一态下测量相对分带的真实安全上界.

    为什么必须单独测这一态
    ──────────────────────
    ``POST /eval/run`` 走的是生产配置（过滤开）。在那一态下，**被分带砍掉的
    次要证据根本不会出现在返回结果里** —— 于是评分时它被算作"本来就不存在"，
    ``min_gold_ratio`` 被算得偏高。这正是旧金标集测不出分带风险的同一盲区：
    观测手段本身把要观测的现象抹掉了（幸存者偏差）。

    所以这里刻意跑第二态（``RERANK_MIN_SCORE_FILTER=False``），让被砍掉的
    证据显形，再算 ``min(次证分 / 头名分)`` —— 它就是"ratio 还能调到多高"的
    真实上界。两态对照还有第二个用处：态 B 里"找得到、态 A 里找不到"的那些，
    正是被分带误杀的**直接证据**。

    注：该值取自**过完整条流水线之后**的分数，已含 ``PARENT_SCORE_DECAY``
    折损，因此偏保守（真实可容忍上界略高）。对护栏而言保守方向正确。
    """
    from sqlalchemy import select

    from app.config import get_settings
    from app.db.postgres import get_db_session
    from app.db.user_models import User
    from app.services.evaluation import gold_score_profile, item_from_chunk
    from app.services.retrieval_service import retrieve_chunks
    from app.services.tenancy import scope_for

    settings = get_settings()
    async with get_db_session() as session:
        user = (
            await session.execute(select(User).where(User.username == username))
        ).scalars().first()
    if user is None:
        return {"error": f"user {username!r} not found"}
    scope = scope_for(user)

    # 只测「金标 ≥ 2 条」的用例：单金标用例 gold/head ≡ 1.0，无论如何都有上界
    # 1.0 —— 放进来看起来"余量充足"，其实是恒真命题，会掩盖真实风险。
    targets = [c for c in resolved_cases if len(c["relevant"]) >= 2]
    if not targets:
        return {"error": "金标集里没有一条需要 ≥2 条证据的用例 —— 分带风险仍然测不出"}

    async def _retrieve(query: str):
        return await retrieve_chunks(
            query, top_k=top_k,
            owner_id=str(scope.owner_id) if scope.owner_id else None,
            tenant_id=scope.tenant_id,
            user_department_id=scope.department_id,
            tenant_wide=scope.tenant_wide,
            platform_wide=scope.platform_wide,
        )

    per_case: list[dict] = []
    original = settings.RERANK_MIN_SCORE_FILTER
    try:
        # 态 B：关掉阈值过滤 —— 被分带砍掉的证据才会显形
        settings.RERANK_MIN_SCORE_FILTER = False
        for case in targets:
            chunks = await _retrieve(case["query"])
            ranked = [item_from_chunk(c) for c in chunks]
            profile = gold_score_profile(ranked, case["relevant"])
            per_case.append({
                "query": case["query"],
                "gold_count": len(case["relevant"]),
                "found_without_filter": sum(
                    1 for r in case["relevant"] if any(i.matches({r}) for i in ranked)
                ),
                "best_score": profile["best_score"],
                "min_gold_ratio": profile["min_gold_ratio"],
                "gold_scores": profile["gold_scores"],
            })
    finally:
        settings.RERANK_MIN_SCORE_FILTER = original   # 必须还原：进程内单例共享

    ratios = [c["min_gold_ratio"] for c in per_case if c["min_gold_ratio"] is not None]
    return {
        "method": "容器内复跑 retrieve_chunks（admin scope, top_k=%d），两态对照："
                  "态A=生产配置（过滤开）；态B=关闭阈值过滤" % top_k,
        "measured_state": "B（过滤关）。态A会把被砍掉的证据算成『本来就不存在』，"
                          "使 min_gold_ratio 偏高 → 不能用来定上界。",
        "cases": per_case,
        "safe_ceiling": min(ratios) if ratios else None,
        "cases_total": len(per_case),
        "cases_fully_found_without_filter": sum(
            1 for c in per_case if c["found_without_filter"] == c["gold_count"]
        ),
        "conservatism": "比值取的是过完整条流水线之后的分数（含 PARENT_SCORE_DECAY "
                        "折损），因此是偏保守的一侧；对护栏而言方向正确。",
    }


def judge(report: dict, negatives: list[dict], thresholds: dict,
          band: dict | None = None) -> dict:
    """把「指标 / 负例拒答 / 多证据 / 分带余量」四类判据合并成一个结论.

    ``band`` 为 ``measure_band_headroom`` 的返回值（可为 None —— 拿不到测量时
    只跳过"实测上界"那一条，其余判据照常生效，不会静默放行）。
    """
    recall10 = (report.get("recall") or {}).get("10")
    if recall10 is None:
        recall10 = (report.get("recall") or {}).get("10.0")
    prec3 = (report.get("precision") or {}).get("3")
    if prec3 is None:
        prec3 = (report.get("precision") or {}).get("3.0")
    mrr = report.get("mrr")
    slices = report.get("evidence_slices") or {}
    multi = slices.get("multi_evidence") or {}
    all_found = multi.get("all_gold_found_rate")
    checks = [
        ("recall@10", recall10, thresholds.get("min_recall_at_10"), ">="),
        ("mrr", mrr, thresholds.get("min_mrr"), ">="),
        ("precision@3", prec3, thresholds.get("min_precision_at_3"), ">="),
        # 多证据全召回率：比整体 recall 更直白 —— 漏一条就是"答案只答了一半"。
        # 它同时是分带护栏的**行为侧**判据（分带砍掉次要证据 → 该用例漏召回
        # → 这里的速率掉下来），而下面的 ratio 对账是**数值侧**判据。两者互补：
        # 数值侧能看到"贴着边界跑"的余量侵蚀，行为侧能抓住数值侧漏掉的失效。
        ("multi_evidence_all_found_rate", all_found,
         thresholds.get("min_multi_evidence_all_found_rate"), ">="),
    ]
    failures: list[str] = []
    detail = []
    for name, value, floor, op in checks:
        if value is None or floor is None:
            ok = False
            detail.append({"metric": name, "value": value, "floor": floor, "ok": ok,
                           "why": "指标无样本或未设下限"})
        else:
            # JSON 往返后 dict 的键可能变成字符串，比较前统一取 float
            v = float(value)
            ok = v >= float(floor)
            detail.append({"metric": name, "value": round(v, 4), "floor": floor, "ok": ok})
        if not ok:
            failures.append(f"{name}: {value} < {floor}")

    # ── 分带（relative band）护栏 ────────────────────────────────────────────
    # 这一节守的是本次修复最核心的失效模式：ratio 调过头 → 次要证据被带砍掉
    # → 答一半，且**不报错**。三道判据从松到紧：
    #   1) 金标集声明的上界（静态、可评审、随金标集版本走）
    #   2) 代码里的护栏常量必须与它一致（防两处声明各自漂移）
    #   3) 本次实测上界（过滤关那一态；随模型/语料实时变化）
    from app.config import RERANK_MIN_SCORE_RATIO_CEILING, get_settings

    settings = get_settings()
    current = float(settings.RERANK_MIN_SCORE_RATIO)
    declared = thresholds.get("max_rerank_min_score_ratio")
    band_detail: dict = {"current_ratio": current, "declared_ceiling": declared,
                         "guard_constant": RERANK_MIN_SCORE_RATIO_CEILING}
    if declared is not None and current >= float(declared):
        failures.append(
            f"RERANK_MIN_SCORE_RATIO={current} ≥ 金标集声明上界 {declared}"
            " —— 至少一条合法次要证据会被相对分带砍掉"
        )
    if declared is not None and abs(RERANK_MIN_SCORE_RATIO_CEILING - float(declared)) > 1e-9:
        failures.append(
            f"护栏常量 {RERANK_MIN_SCORE_RATIO_CEILING} 与金标集声明上界 {declared} 不一致"
            " —— 两处声明必须同步修改，否则其中一处形同虚设"
        )
    if band and band.get("safe_ceiling") is not None:
        measured = float(band["safe_ceiling"])
        band_detail["measured_ceiling"] = measured
        band_detail["margin"] = (round(measured / current, 3) if current > 0 else None)
        if current >= measured:
            failures.append(
                f"RERANK_MIN_SCORE_RATIO={current} ≥ 本次实测上界 {measured}"
                f"（态B/过滤关实测，见 band.cases）—— 已在误杀次要证据"
            )
    elif band and band.get("error"):
        band_detail["measure_error"] = band["error"]
        failures.append(f"分带余量未测出：{band['error']}")

    refused = sum(1 for n in negatives if n.get("refused"))
    neg_ok = refused == len(negatives)
    if not neg_ok:
        for n in negatives:
            if not n.get("refused"):
                failures.append(f"负例未拒答：{n['query']!r} top_score={n.get('top_score')}")
    return {
        "passed": not failures,
        "failures": failures,
        "metrics": detail,
        "band": band_detail,
        "negative_cases_total": len(negatives),
        "negative_cases_refused": refused,
        "negative_gate_ok": neg_ok,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    # 默认路径写死 /app 而不是相对 __file__：本脚本在容器里通常被拷到 /tmp
    # 运行（/app 不是 bind mount），相对 __file__ 会解析到 /eval 这个不存在的
    # 位置。可用 RAG_EVAL_GOLDEN / --golden 覆盖。
    ap.add_argument(
        "--golden",
        default=os.getenv("RAG_EVAL_GOLDEN", "/app/eval/golden_v1.json"),
    )
    ap.add_argument("--base-url", default=os.getenv("RAG_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--out", default="/tmp/eval_baseline_report.json")
    ap.add_argument("--top-k", type=int, default=10)
    # 分带余量测量要额外跑一轮检索（过滤关那一态）；排障时可以跳过换速度，
    # 但**跳过会少一道护栏**，CI 门禁不应加这个开关。
    ap.add_argument("--skip-band", action="store_true",
                    default=os.getenv("RAG_EVAL_SKIP_BAND", "") not in ("", "0", "false"),
                    help="跳过相对分带余量实测（仅排障用，会少一道判据）")
    args = ap.parse_args()

    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)

    user = os.getenv("RAG_EVAL_USERNAME", "admin")
    password = os.getenv("RAG_EVAL_PASSWORD", "")
    if not password:
        print("缺少环境变量 RAG_EVAL_PASSWORD（评测需要 audit.read 权限的账号）")
        return 3

    cases, problems = await resolve_labels(golden)
    if problems:
        print("金标解析失败：")
        for p in problems:
            print("  -", p)
        return 4
    print(f"golden={golden['name']} cases={len(cases)} negatives={len(golden['negative_cases'])}")

    login = _http(args.base_url, "/auth/login", method="POST",
                  body={"username": user, "password": password}, timeout=30)
    if not login.get("ok"):
        print("登录失败:", login)
        return 3
    token = login["body"].get("access_token")
    print(f"login ok as {login['body'].get('user', {}).get('username')}")

    started = datetime.now(timezone.utc)
    run = _http(args.base_url, "/eval/run", method="POST", token=token, timeout=1200, body={
        "name": golden["name"],
        "description": golden.get("description", ""),
        "cases": cases,
        "k_values": [1, 3, 5, 10],
        "top_k": args.top_k,
    })
    if not run.get("ok"):
        print("评测请求失败:", run)
        return 5
    report = run["body"]
    print("eval/run ->", json.dumps({
        "total_cases": report.get("total_cases"),
        "scored_cases": report.get("scored_cases"),
        "mrr": report.get("mrr"),
        "map": report.get("map"),
        "recall": report.get("recall"),
        "precision": report.get("precision"),
        "ndcg": report.get("ndcg"),
    }, ensure_ascii=False))

    hist = _http(args.base_url, "/eval/history?limit=3", token=token, timeout=60)
    runs = (hist.get("body") or {}).get("runs") if hist.get("ok") else []
    persisted = bool(runs) and runs[0].get("name", runs[0].get("eval_set")) == golden["name"]
    print(f"eval/history -> runs={len(runs or [])} persisted={persisted} "
          f"source={(runs or [{}])[0].get('source')}")

    negatives = await check_negatives(golden, user)
    for n in negatives:
        print(f"  negative [{'refuse' if n.get('refused') else 'LEAK'}] "
              f"top={n.get('top_score')} {n['query']!r}")

    # ── 分带余量（第二态实测）──────────────────────────────────────────────
    # 只在需要时才跑：它能发现"贴着边界跑"的余量侵蚀，但要额外一轮检索。
    band: dict = {}
    if not args.skip_band:
        print("measuring relative-band headroom (filter OFF pass) …")
        band = await measure_band_headroom(golden, user, cases, top_k=args.top_k)
        if band.get("error"):
            print(f"  band: ERROR {band['error']}")
        else:
            print(f"  band: safe_ceiling={band.get('safe_ceiling')} "
                  f"(state B, filter off) over {band.get('cases_total')} multi-evidence cases")
            for c in band.get("cases", []):
                flag = "" if c["found_without_filter"] == c["gold_count"] else "  <-- MISSING"
                print(f"    ratio={c['min_gold_ratio']} head={c['best_score']} "
                      f"found={c['found_without_filter']}/{c['gold_count']} "
                      f"{c['query'][:42]!r}{flag}")

    verdict = judge(report, negatives, golden.get("thresholds", {}), band=band)
    verdict["eval_persisted_to_db"] = persisted
    if not persisted:
        verdict["failures"].append("评测结果未落库：/eval/history 里看不到本轮（跨重启基线不成立）")
        verdict["passed"] = False

    lag = (datetime.now(timezone.utc) - started).total_seconds()
    out = {
        "golden_set": golden["name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(lag, 1),
        "report": report,
        "negatives": negatives,
        "band": band,
        "verdict": verdict,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(f"report -> {args.out}")
    return 0 if verdict["passed"] else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
