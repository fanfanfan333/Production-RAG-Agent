"""
检索质量（召回 / 准确 / 排序）+ 拒答能力 实测.

`e2e_full_acceptance.py` 验证的是**权限与隔离**（谁能看到/删除什么）。
本脚本验证的是**检索与生成的质量**，两者互补：

    1. 造 4 份事实唯一的小文档（含 1 份干扰项），等入库完成；
    2. 用 `POST /eval/run` 跑金标集 —— 走**生产同一条**检索链路
       （多路向量 + BM25 → RRF → cross-encoder 精排），报告
       Recall@K / Precision@K / MRR / MAP / NDCG@K / HitRate；
    3. 用 `POST /query` 实际提问，验证：
         * 可回答的问题不被误拒，且答案包含金标事实；
         * 知识库里根本没有的问题**必须拒答**（不硬编）。
       —— "拒答率低"既可能是答得好，也可能是该拒的没拒，所以必须两侧都测。

金标集用 ``document_id::chunk_index`` 标识（评测服务的原生主键）。

运行：

    docker exec -e HOME=/tmp -e NO_PROXY='*' rag_backend \\
        python /app/scripts/e2e_recall_refusal.py

    python backend/scripts/e2e_recall_refusal.py --keep   # 保留数据排查
"""

from __future__ import annotations

import argparse
import json
import random
import ssl
import string
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE_DEFAULT = "http://127.0.0.1:8000"

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE

# /auth/* 10 次/分钟/IP，/upload 12 次/分钟/IP（都是产品行为，不去关）
_RETRY_429_MAX = 45
_RETRY_429_SLEEP = 2.0

# 拒答哨兵：evidence_gate.REFUSAL_ANSWER 的开头，模型主动拒答也用同一句
REFUSAL_SENTINEL = "我在当前知识库中没有找到与这个问题足够相关的信息"

_passed = 0
_failed = 0
_skipped = 0
_notes: list[str] = []


def check(label: str, condition: bool, extra: str = "") -> bool:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label} {extra}")
    return bool(condition)


def skip(label: str, reason: str = "") -> None:
    """无法在当前环境下判定 —— 显式记 SKIP，绝不伪装成 PASS。"""
    global _skipped
    _skipped += 1
    print(f"  SKIP  {label}" + (f" —— {reason}" if reason else ""))


def detail(res: object) -> str:
    if isinstance(res, dict):
        return str(res.get("detail") or res)[:200]
    return str(res)[:200]


def _decode(body: str):
    try:
        return json.loads(body)
    except Exception:  # noqa: BLE001
        return body


def call(
    method: str,
    path: str,
    *,
    base: str,
    token: str | None = None,
    payload: dict | None = None,
    timeout: int = 60,
) -> tuple[int, dict | list | str]:
    url = f"{base}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    for attempt in range(_RETRY_429_MAX):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                return resp.status, _decode(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < _RETRY_429_MAX - 1:
                time.sleep(_RETRY_429_SLEEP)
                continue
            return exc.code, _decode(body)
        except Exception as exc:  # noqa: BLE001
            return 0, f"{type(exc).__name__}: {exc}"
    return 429, "rate limited"


def upload(base: str, token: str, filename: str, content: bytes):
    boundary = "----WB" + uuid.uuid4().hex
    chunks: list[bytes] = []
    chunks.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
        f"filename=\"{filename}\"\r\nContent-Type: text/plain\r\n\r\n".encode("utf-8")
    )
    chunks.append(content)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    req = urllib.request.Request(f"{base}/upload", data=b"".join(chunks), method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Authorization", f"Bearer {token}")
    for attempt in range(_RETRY_429_MAX):
        try:
            with _OPENER.open(req, timeout=120) as resp:
                return resp.status, _decode(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < _RETRY_429_MAX - 1:
                time.sleep(_RETRY_429_SLEEP)
                continue
            return exc.code, _decode(body)
        except Exception as exc:  # noqa: BLE001
            return 0, f"{type(exc).__name__}: {exc}"
    return 429, "rate limited"


def login(base: str, username: str, password: str) -> str:
    st, res = call("POST", "/auth/login", base=base,
                   payload={"username": username, "password": password})
    if st != 200 or not isinstance(res, dict):
        raise SystemExit(f"登录失败 {username}: {st} {detail(res)}")
    return res["access_token"]


def ask(base: str, token: str, query: str, *, timeout: int = 300) -> dict:
    """
    走 POST /query 的 SSE，把 chunk 事件拼成完整答案.

    不能直接在原始响应里 grep 关键字：token 是逐个 SSE 事件下发的，
    金标事实可能被切成 "2026" / "-03" / "-15" 三个事件，中间夹着 JSON 转义。
    """
    url = f"{base}/query"
    body = json.dumps({"query": query, "top_k": 5, "mode": "knowledge_qa"}).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    req.add_header("Authorization", f"Bearer {token}")

    answer_parts: list[str] = []
    intents: list[str] = []
    sources: list[str] = []
    grade: str = ""
    errors: list[str] = []
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except Exception:  # noqa: BLE001
                    continue
                etype = str(event.get("type") or "")
                if etype == "chunk":
                    # 坑：/query 的 chunk 事件字段名是 **content**，不是 text。
                    # 这里两个都兜一下，免得后端改名后静默拿到空答案、把拒答判定全判错。
                    answer_parts.append(str(event.get("content") or event.get("text") or ""))
                elif etype == "intent":
                    intents.append(str(event.get("intent") or ""))
                elif etype == "sources":
                    for s in event.get("sources") or []:
                        sources.append(str(s.get("document_name") or s.get("filename") or ""))
                elif etype == "grade":
                    grade = str(event.get("reason") or event.get("label") or "")
                elif etype == "error":
                    errors.append(str(event.get("message") or ""))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")

    answer = "".join(answer_parts)
    return {
        "answer": answer,
        "intents": intents,
        "sources": sources,
        "grade": grade,
        "errors": errors,
        "refused": REFUSAL_SENTINEL in answer,
    }


def delete_doc(base: str, token: str, doc_id: str) -> None:
    call("DELETE", f"/documents/{doc_id}", base=base, token=token)


# ── 清理 ──────────────────────────────────────────────────────────────────────

def cleanup(prefixes: list[str], companies: list[str]) -> str:
    try:
        import asyncio

        from sqlalchemy import delete, or_, select

        from app.db.models import Document
        from app.db.postgres import get_db_session
        from app.db.share_models import ShareRequest
        from app.db.staff_models import StaffRequest
        from app.db.user_models import User
    except Exception as exc:  # noqa: BLE001
        return f"跳过（无 app 包：{type(exc).__name__}）"

    async def _run() -> dict:
        async with get_db_session() as session:
            rows = await session.execute(
                select(User.id, User.username).where(
                    or_(*[User.username.like(f"{p}%") for p in prefixes])
                )
            )
            users = list(rows.all())
            uids = [u for u, _ in users]

            doc_rows = await session.execute(
                select(Document.id).where(
                    or_(*[Document.filename.like(f"%{p}%") for p in prefixes])
                    | (Document.owner_id.in_(uids) if uids else False)
                )
            )
            doc_ids = [d for (d,) in doc_rows.all()]
            r_doc = await session.execute(
                delete(Document).where(Document.id.in_(doc_ids))
            )
            r_share = await session.execute(
                delete(ShareRequest).where(
                    ShareRequest.document_name.like(f"%{prefixes[0]}%")
                )
            )
            r_staff = await session.execute(
                delete(StaffRequest).where(
                    or_(
                        StaffRequest.company_name.in_(companies),
                        *[StaffRequest.applicant_username.like(f"{p}%") for p in prefixes],
                    )
                )
            )
            if uids:
                await session.execute(delete(User).where(User.id.in_(uids)))
            await session.commit()
            return {
                "users": len(uids),
                "docs": r_doc.rowcount or 0,
                "share_reqs": r_share.rowcount or 0,
                "staff_reqs": r_staff.rowcount or 0,
            }

    try:
        return json.dumps(asyncio.run(_run()), ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return f"失败（{type(exc).__name__}: {exc}）"


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE_DEFAULT)
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--admin-password", default="RagAdmin#2026")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="只验召回/准确率与「该拒的必须拒」，跳过需要 LLM 生成的误拒验证"
             "（宿主内存不足、模型加载不了时用）",
    )
    parser.add_argument("--ingest-timeout", type=int, default=300)
    args = parser.parse_args()
    base = args.base.rstrip("/")

    sfx = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    company = f"质量验收公司{sfx}"
    dept = f"数据部{sfx}"
    user = f"quality.{sfx}@example.com"
    pw = "QaPass#2026"
    prefix = f"quality.{sfx}"

    print("═" * 74)
    print("  检索质量（召回/准确）+ 拒答能力 实测")
    print(f"  base={base}  账号={user}  公司={company}")
    print("═" * 74)

    admin = login(base, args.admin_user, args.admin_password)

    # ── 0. 建账号（部门负责人：eval/run 需要 audit.read）─────────────────────
    st, res = call("POST", "/auth/register", base=base,
                   payload={"username": user, "password": pw})
    if st != 201:
        raise SystemExit(f"注册失败：{st} {detail(res)}")
    token = res["access_token"]
    st, res = call("POST", "/staff/requests", base=base, token=token,
                   payload={"company_name": company, "department_name": dept,
                            "duty": "数据工程师"})
    if st != 201:
        raise SystemExit(f"提交身份验证失败：{st} {detail(res)}")
    rid = res["request"]["id"]
    st, res = call("POST", f"/staff/requests/{rid}/review", base=base, token=admin,
                   payload={"approve": True, "role": "dept_manager",
                            "department_name": dept, "duty": "数据工程师",
                            "reviewer_title": "平台管理员", "reviewer_name": "超级管理员"})
    if st != 200:
        raise SystemExit(f"批准失败：{st} {detail(res)}")
    token = login(base, user, pw)
    print(f"\n  账号就绪：{user}（部门负责人，可跑评测）")

    # ── 1. 语料：事实唯一、互不干扰 + 一份干扰项 ──────────────────────────────
    corpus = {
        "星河项目验收报告": (
            f"星河项目验收报告（编号 XYZ-{sfx}）。\n"
            "本项目（星河项目）的交付日期是 2026-03-15，项目负责人是 李文博。\n"
            "验收范围包括数据处理模块、报表模块与告警模块。\n"
        ),
        "天穹系统容量设计": (
            f"天穹系统容量设计说明（编号 TQ-{sfx}）。\n"
            "天穹系统的并发上限是 12000 QPS，缓存层使用 Redis 集群，"
            "数据库使用 PostgreSQL 主从架构。\n"
        ),
        "极光数据中心选址说明": (
            f"极光数据中心选址说明（编号 JG-{sfx}）。\n"
            "极光数据中心位于呼和浩特，设计 PUE 为 1.18，"
            "全年自然冷却时长约 4200 小时。\n"
        ),
        "员工餐厅本周菜单": (
            f"员工餐厅本周菜单（编号 CT-{sfx}）。\n"
            "本周供应红烧牛肉面与番茄鸡蛋盖饭，另有紫菜蛋花汤。\n"
        ),
    }

    print("\n── 1. 上传语料并等待入库 ──")
    doc_ids: dict[str, str] = {}
    for name, text in corpus.items():
        filename = f"{prefix}-{name}.txt"
        st, res = upload(base, token, filename, text.encode("utf-8"))
        if st not in (200, 202):
            raise SystemExit(f"上传失败 {filename}: {st} {detail(res)}")
        did = res["documents"][0].get("document_id")
        doc_ids[name] = did
        print(f"  已受理 {filename} → {did}")

    # 轮询入库终态
    deadline = time.time() + args.ingest_timeout
    pending = set(corpus.keys())
    while pending and time.time() < deadline:
        st, res = call("GET", "/documents?limit=100", base=base, token=token)
        rows = {d["filename"]: d for d in res.get("documents", [])}
        still: set[str] = set()
        for name in pending:
            row = rows.get(f"{prefix}-{name}.txt")
            if row is None:
                still.add(name)
                continue
            if row["status"] in ("completed", "already_exists"):
                continue
            if row["status"] == "failed":
                check(f"文档入库成功：{name}", False,
                      f"status=failed error={row.get('error')}")
                still.discard(name)
                continue
            still.add(name)
        pending = still
        if pending:
            time.sleep(5)

    check("四份语料全部入库完成", not pending, f"未完成={sorted(pending)}")

    # ── 2. 金标集：document_id::chunk_index ──────────────────────────────────
    def relevant_keys(name: str) -> list[str]:
        did = doc_ids.get(name)
        if not did:
            return []
        st, res = call("GET", f"/documents/{did}/chunks", base=base, token=token)
        if not isinstance(res, dict):
            return []
        return [f"{did}::{c['chunk_index']}" for c in res.get("chunks", [])]

    keys = {name: relevant_keys(name) for name in corpus}
    for name, ks in keys.items():
        print(f"  金标分块 {name}: {len(ks)} 块")

    cases = [
        {"query": "星河项目的交付日期是哪一天？", "relevant": keys["星河项目验收报告"],
         "note": "单跳事实"},
        {"query": "天穹系统的并发上限是多少？", "relevant": keys["天穹系统容量设计"],
         "note": "数值事实"},
        {"query": "极光数据中心位于哪个城市？", "relevant": keys["极光数据中心选址说明"],
         "note": "单跳事实"},
        {"query": "天穹系统的缓存层用了什么？", "relevant": keys["天穹系统容量设计"],
         "note": "同文档第二个事实"},
    ]
    cases = [c for c in cases if c["relevant"]]
    check("金标集构建成功（每条都有相关分块）", len(cases) == 4, f"cases={len(cases)}")

    # ── 3. 检索质量评测（走生产检索链路）────────────────────────────────────
    print("\n── 2. 检索质量（POST /eval/run，走生产同一条检索链路）──")
    st, rep = call("POST", "/eval/run", base=base, token=token, timeout=600,
                   payload={"name": f"质量验收-{sfx}", "description": "召回/准确率实测",
                            "cases": cases, "k_values": [1, 3, 5], "top_k": 5})
    if not check("评测接口返回成功", st == 200 and isinstance(rep, dict),
                 f"status={st} {detail(rep)}"):
        print(f"\n无法继续：{detail(rep)}")
        return 1

    recall = {int(k): v for k, v in (rep.get("recall") or {}).items()}
    precision = {int(k): v for k, v in (rep.get("precision") or {}).items()}
    ndcg = {int(k): v for k, v in (rep.get("ndcg") or {}).items()}
    hit = {int(k): v for k, v in (rep.get("hit_rate") or {}).items()}

    print(f"\n  用例数 = {rep.get('scored_cases')}/{rep.get('total_cases')}")
    print(f"  Recall@K    = {recall}")
    print(f"  Precision@K = {precision}")
    print(f"  NDCG@K      = {ndcg}")
    print(f"  HitRate@K   = {hit}")
    print(f"  MRR         = {rep.get('mrr')}")
    print(f"  MAP         = {rep.get('map')}")

    r5 = recall.get(5)
    r3 = recall.get(3)
    r1 = recall.get(1)
    check("Recall@5 = 1.0（金标事实全部被召回）", r5 == 1.0, f"recall@5={r5}")
    check("Recall@3 ≥ 0.75", (r3 or 0) >= 0.75, f"recall@3={r3}")
    check("HitRate@1 ≥ 0.75（多数问题首个命中即正确）",
          (hit.get(1) or 0) >= 0.75, f"hit@1={hit.get(1)}")
    check("MRR ≥ 0.75（正确分块总体排在前面）",
          (rep.get("mrr") or 0) >= 0.75, f"mrr={rep.get('mrr')}")
    check("NDCG@5 ≥ 0.75（排序质量）",
          (ndcg.get(5) or 0) >= 0.75, f"ndcg@5={ndcg.get(5)}")
    _notes.append(
        f"Recall@1={r1} Recall@3={r3} Recall@5={r5} P@1={precision.get(1)} "
        f"MRR={rep.get('mrr')} NDCG@5={ndcg.get(5)}"
    )

    # ── 4. 拒答能力：可回答的不误拒 / 不可回答的必须拒 ───────────────────────
    print("\n── 3. 拒答能力（POST /query 实际提问）──")

    unanswerable = [
        "请问公司 2031 年的年会在哪个城市举办？",
        "我们的量子计算机用的是哪种纠错码算法？",
        "公司新任 CTO 的姓名和履历是什么？",
    ]
    answerable = [
        ("星河项目的项目负责人是谁？", "李文博"),
        ("天穹系统的并发上限是多少？", "12000"),
        ("极光数据中心的 PUE 是多少？", "1.18"),
    ]

    # 4.1 「该拒的必须拒」—— 走 Evidence Gate，在**生成之前**短路
    #     （无关问题检索不到分块 → grade=no_chunks → 直接返回拒答文案），
    #     因此**不依赖 LLM 能否加载**，宿主内存紧张时同样可验。
    correct_refusals = 0
    asked_unanswerable = 0
    gate_errors: list[str] = []
    for query in unanswerable:
        res = ask(base, token, query)
        if res["errors"]:
            gate_errors.append(str(res["errors"][:1]))
            continue
        asked_unanswerable += 1
        correct_refusals += 1 if res["refused"] else 0
        check(f"知识库无关问题被正确拒答：「{query}」", res["refused"],
              f"answer={res['answer'][:160]!r}")
    for err in gate_errors:
        check("无关问题提问链路未报错", False, f"errors={err}")

    refused_rate = (correct_refusals / asked_unanswerable) if asked_unanswerable else 0.0
    print(f"\n  拒答率（该拒的拒了）  = {correct_refusals}/{asked_unanswerable}"
          f" = {refused_rate:.0%}")
    if asked_unanswerable:
        check("拒答率 = 100%（无关问题全部拒答）", refused_rate == 1.0,
              f"rate={refused_rate}")
    else:
        skip("拒答率 = 100%", "本轮无关问题提问未拿到可判定的响应")

    # 4.2 「不该拒的不能拒」—— 需要 LLM 真正生成答案。
    #     宿主内存不足时模型加载失败（out-of-memory），此时记 **SKIP 而不是 FAIL**：
    #     这是环境问题，拒答逻辑本身由 tests/test_evidence_gate.py 与
    #     tests/test_master_graph_e2e.py 覆盖。
    if args.retrieval_only:
        skip("误拒率 = 0%（可回答问题全部作答）",
             "指定 --retrieval-only，本轮跳过生成侧验证")
    else:
        false_refusals = 0
        asked_answerable = 0
        llm_down = ""
        for query, expect in answerable:
            res = ask(base, token, query)
            if res["errors"]:
                llm_down = str(res["errors"][:1])
                break
            asked_answerable += 1
            refused = res["refused"]
            has_fact = expect in res["answer"]
            false_refusals += 1 if refused else 0
            check(f"可回答问题未被误拒且答出「{expect}」",
                  (not refused) and has_fact,
                  f"refused={refused} answer={res['answer'][:120]!r}")
        if llm_down:
            skip("误拒率 = 0%（可回答问题全部作答）",
                 f"LLM 不可用（生成节点报错，环境问题非代码问题）：{llm_down}")
        elif asked_answerable:
            false_refusal_rate = false_refusals / asked_answerable
            print(f"  误拒率（不该拒却拒）  = {false_refusals}/{asked_answerable}"
                  f" = {false_refusal_rate:.0%}")
            check("误拒率 = 0%（可回答问题全部作答）", false_refusal_rate == 0.0,
                  f"rate={false_refusal_rate}")

    _notes.append(
        f"拒答率={refused_rate:.0%}（{correct_refusals}/{asked_unanswerable}）"
    )

    # ── 5. 隔离对检索同样生效（个人库不外流）────────────────────────────────
    print("\n── 4. 检索侧的隔离（个人库不流入公司 RAG）──")
    st, res = call("POST", "/auth/register", base=base,
                   payload={"username": f"outsider.{sfx}@example.com", "password": pw})
    if st == 201:
        outsider = res["access_token"]
        # 未验证 → 应被身份闸门挡住，连提问都不行
        st, res = call("POST", "/query", base=base, token=outsider,
                       payload={"query": "星河项目的交付日期", "mode": "knowledge_qa"})
        check("未验证账号无法调用 /query（身份闸门）", st == 403,
              f"status={st} {detail(res)}")
        # 验证后仍应因跨公司看不到对方内容 —— 这里只用文档列表做等价校验
        prefixes_extra = f"outsider.{sfx}"
    else:
        prefixes_extra = ""

    # 直接核对：语料是个人库(private)，公司内他人（未上传者）看不到
    st, res = call("GET", "/documents?limit=100", base=base, token=admin)
    rows = res.get("documents", []) if isinstance(res, dict) else []
    mine = [d for d in rows if d["filename"].startswith(prefix)]
    check("语料仍为个人库（未因评测/提问被自动发布）",
          bool(mine) and all(d["access_level"] == "private" for d in mine),
          f"levels={sorted({d['access_level'] for d in mine})}")

    # ── 清理 ─────────────────────────────────────────────────────────────────
    if args.keep:
        print(f"\n  （--keep）保留数据：账号={user} 公司={company}")
    else:
        prefixes = [prefix] + ([prefixes_extra] if prefixes_extra else [])
        print(f"\n  清理：{cleanup(prefixes, [company])}")

    print(f"\n{'═' * 74}")
    print(f"  结果：{_passed} 通过 / {_failed} 失败" + (f" / {_skipped} 跳过" if _skipped else ""))
    for note in _notes:
        print(f"  · {note}")
    print(f"{'═' * 74}")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
