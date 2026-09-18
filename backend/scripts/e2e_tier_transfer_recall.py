"""
层级转换是否真正落到**检索链路** —— 真实提问验证.

`e2e_full_acceptance.py` 的 C3/C4 验证的是 PG 侧的可见性（文档列表接口）。
但提问走的不是 PG，而是 Qdrant 里那份 **ACL 载荷副本**：

    set_document_access_level()
        ├─ PG      access_level / department_id      ← 列表接口读这份
        └─ Qdrant  payload(access_level, department_id)  ← 检索前置过滤读这份

两者分叉时的症状极具欺骗性：列表页一切正常（"转换成功了"），但目标部门的
同事一提问就检索不到。而代码注释里点名的**最危险时序**恰恰是最常见的操作
顺序 —— 上传后立刻转换：此时向量点往往还没写进去，
``update_document_access_payload`` 匹配 0 个点直接返回，全靠入库收尾的
``resync_document_acl_payload`` 追平。

本脚本用**现场新建**的账号与文档，把两个时序各测一遍，并以"真实提问能否
答出唯一事实"作为最终判据（用户视角的功能是否成立），而不是读代码推断。

    场景 A：上传公司库文档 → **立刻**转市场部 → 等入库 → 提问
    场景 B：上传公司库文档 → 等入库 → 转市场部 → 提问

判据两侧都测：目标部门（市场部）必须答得出，原部门（研发部）必须答不出
—— 只看一侧无法区分"隔离生效"与"谁都看不到"。

用法（容器内或宿主机均可，只用标准库 + app 包）：

    docker exec -e HOME=/tmp -e NO_PROXY='*' rag_backend \
        python /app/scripts/e2e_tier_transfer_recall.py

    python backend/scripts/e2e_tier_transfer_recall.py --keep   # 保留数据排查
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

# /auth/* 10 次/分钟/IP、/upload 12 次/分钟/IP（均为产品行为，不去关它）
_RETRY_429_MAX = 45
_RETRY_429_SLEEP = 2.0

# 拒答哨兵：evidence_gate 与模型主动拒答共用同一句开头
REFUSAL_SENTINEL = "我在当前知识库中没有找到与这个问题足够相关的信息"

_passed = 0
_failed = 0


def section(title: str) -> None:
    print(f"\n── {title} ──")


def check(label: str, condition: bool, extra: str = "") -> bool:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label} {extra}")
    return bool(condition)


def detail(res: object) -> str:
    if isinstance(res, dict):
        return str(res.get("detail") or res)[:200]
    return str(res)[:200]


def _decode(body: str):
    try:
        return json.loads(body)
    except Exception:  # noqa: BLE001
        return body


# ── HTTP ──────────────────────────────────────────────────────────────────────

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


def upload(
    base: str, token: str, filename: str, content: bytes, *, access_level: str | None = None
) -> tuple[int, dict | list | str]:
    boundary = "----WB" + uuid.uuid4().hex
    chunks: list[bytes] = []

    def field(name: str, value: str) -> None:
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n".encode("utf-8")
        )

    if access_level:
        field("access_level", access_level)
    chunks.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
        f"filename=\"{filename}\"\r\nContent-Type: text/plain\r\n\r\n".encode("utf-8")
    )
    chunks.append(content)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)

    req = urllib.request.Request(f"{base}/upload", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    for attempt in range(_RETRY_429_MAX):
        try:
            with _OPENER.open(req, timeout=120) as resp:
                return resp.status, _decode(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < _RETRY_429_MAX - 1:
                time.sleep(_RETRY_429_SLEEP)
                continue
            return exc.code, _decode(text)
        except Exception as exc:  # noqa: BLE001
            return 0, f"{type(exc).__name__}: {exc}"
    return 429, "rate limited"


def login(base: str, username: str, password: str) -> str:
    st, res = call("POST", "/auth/login", base=base,
                   payload={"username": username, "password": password})
    if st != 200 or not isinstance(res, dict):
        raise SystemExit(f"登录失败 {username}: {st} {detail(res)}")
    return res["access_token"]


def make_user(
    base: str, admin_token: str, *, username: str, password: str, company: str,
    dept: str, duty: str, role: str,
) -> str:
    token = register(base, username, password)
    st, res = call("POST", "/staff/requests", base=base, token=token,
                   payload={"company_name": company, "department_name": dept, "duty": duty})
    if st != 201:
        raise SystemExit(f"提交身份验证失败 {username}: {st} {detail(res)}")
    rid = res["request"]["id"]
    st, res = call("POST", f"/staff/requests/{rid}/review", base=base, token=admin_token,
                   payload={"approve": True, "role": role, "department_name": dept,
                            "duty": duty, "reviewer_title": "平台管理员",
                            "reviewer_name": "超级管理员",
                            "comment": f"同意 {username} 加入 {dept}"})
    if st != 200:
        raise SystemExit(f"批准失败 {username}: {st} {detail(res)}")
    return login(base, username, password)


def register(base: str, username: str, password: str) -> str:
    st, res = call("POST", "/auth/register", base=base,
                   payload={"username": username, "password": password})
    if st != 201 or not isinstance(res, dict):
        raise SystemExit(f"注册失败 {username}: {st} {detail(res)}")
    return res["access_token"]


def list_docs(base: str, token: str) -> list[dict]:
    st, res = call("GET", "/documents?limit=100", base=base, token=token)
    if not isinstance(res, dict):
        return []
    rows = res.get("documents") or res.get("items") or []
    return rows if isinstance(rows, list) else []


def find_doc(rows: list[dict], name: str) -> dict | None:
    return next((d for d in rows if d.get("filename") == name), None)


# ── 提问（走生产同一条 /query SSE 链路）──────────────────────────────────────

def ask(base: str, token: str, query: str, *, timeout: int = 300) -> dict:
    """
    走 POST /query 的 SSE，把 chunk 事件拼成完整答案.

    不能直接在原始响应里 grep 关键字：token 是逐个 SSE 事件下发的，
    金标事实可能被切成多个事件，中间夹着 JSON 转义。
    """
    url = f"{base}/query"
    body = json.dumps({"query": query, "top_k": 5, "mode": "knowledge_qa"}).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    req.add_header("Authorization", f"Bearer {token}")

    answer_parts: list[str] = []
    sources: list[str] = []
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
                    answer_parts.append(str(event.get("content") or event.get("text") or ""))
                elif etype == "sources":
                    for s in event.get("sources") or []:
                        sources.append(
                            str(s.get("document_name") or s.get("filename") or "")
                        )
                elif etype == "error":
                    errors.append(str(event.get("message") or ""))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")

    return {
        "answer": "".join(answer_parts).strip(),
        "sources": sources,
        "errors": errors,
    }


# ── 诊断辅助：读向量层的 ACL 副本（**仅在断言失败时**用来定位）────────────────

def vector_acl(doc_id: str) -> list[dict] | str:
    """
    读该文档所有向量点的 ACL 载荷；失败返回一段错误字符串.

    只在断言失败时打印 —— 断言本身走的是用户视角（提问能否答出），
    读 payload 是为了把"检索不到"定位到"PG 改了但 Qdrant 没改"这一步。
    """
    try:
        import asyncio

        from qdrant_client import models as qm

        from app.config import get_settings
        from app.services.vector_service import get_qdrant_client
    except Exception as exc:  # noqa: BLE001
        return f"（无法加载向量层：{type(exc).__name__}: {exc}）"

    async def _run() -> list[dict]:
        settings = get_settings()
        client = get_qdrant_client()
        points, _ = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(
                    key="document_id", match=qm.MatchValue(value=doc_id))]
            ),
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "access_level": p.payload.get("access_level"),
                "department_id": p.payload.get("department_id"),
            }
            for p in points
        ]

    try:
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        return f"（读取失败：{type(exc).__name__}: {exc}）"


def describe_acl(doc_id: str) -> str:
    rows = vector_acl(doc_id)
    if isinstance(rows, str):
        return rows
    if not rows:
        return "向量点 0 个（入库未产出向量）"
    levels = sorted({str(r["access_level"]) for r in rows})
    depts = sorted({str(r["department_id"]) for r in rows})
    return f"点数={len(rows)} access_level={levels} department_id={depts}"


# ── 入库等待 ──────────────────────────────────────────────────────────────────

def wait_ingest(base: str, token: str, names: list[str], *, timeout: int) -> set[str]:
    """轮询直到这些文档进入终态；返回仍未完成的文件名集合。"""
    deadline = time.time() + timeout
    pending = set(names)
    while pending and time.time() < deadline:
        rows = {d["filename"]: d for d in list_docs(base, token)}
        still: set[str] = set()
        for name in pending:
            row = rows.get(name)
            if row is None:
                still.add(name)
                continue
            status = str(row.get("status") or "")
            if status in ("completed", "already_exists", "failed"):
                continue
            still.add(name)
        pending = still
        if pending:
            time.sleep(5)
    return pending


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
        from app.services.tenancy import company_id_from_name
    except Exception as exc:  # noqa: BLE001
        return f"跳过（无 app 包：{type(exc).__name__}）"

    async def _run() -> dict:
        async with get_db_session() as session:
            user_rows = await session.execute(
                select(User.id, User.username).where(
                    or_(*[User.username.like(f"{p}%") for p in prefixes])
                )
            )
            users = list(user_rows.all())
            uids = [u for u, _ in users]
            unames = [n for _, n in users]
            tenant_ids = [company_id_from_name(c) for c in companies if c]

            doc_rows = await session.execute(
                select(Document.id).where(
                    or_(
                        Document.owner_id.in_(uids) if uids else False,
                        or_(*[Document.filename.like(f"{p}%") for p in prefixes]),
                    )
                )
            )
            doc_ids = [d for (d,) in doc_rows.all()]

            r_share = await session.execute(
                delete(ShareRequest).where(
                    or_(
                        ShareRequest.tenant_id.in_(tenant_ids) if tenant_ids else False,
                        ShareRequest.requester_username.in_(unames) if unames else False,
                        or_(*[ShareRequest.requester_username.like(f"{p}%") for p in prefixes]),
                        ShareRequest.document_id.in_(doc_ids) if doc_ids else False,
                    )
                )
            )
            # 删 PG 行前先清 Qdrant 向量（否则留下孤儿向量，挤占检索候选池；
            # 详见 scripts/_e2e_purge.py）
            if doc_ids:
                from _e2e_purge import delete_vectors_for_documents

                await delete_vectors_for_documents(doc_ids)

            r_doc = await session.execute(
                delete(Document).where(Document.id.in_(doc_ids)) if doc_ids else None
            )
            r_staff = await session.execute(
                delete(StaffRequest).where(
                    or_(
                        or_(*[StaffRequest.applicant_username.like(f"{p}%") for p in prefixes]),
                        StaffRequest.company_name.in_(companies),
                    )
                )
            )
            if uids:
                await session.execute(delete(User).where(User.id.in_(uids)))
            await session.commit()
            return {
                "users": len(uids),
                "docs": (r_doc.rowcount or 0) if r_doc is not None else 0,
                "share_reqs": r_share.rowcount or 0,
                "staff_reqs": r_staff.rowcount or 0,
            }

    try:
        return json.dumps(asyncio.run(_run()), ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return f"失败（{type(exc).__name__}: {exc}）"


# ── LLM 预检 ──────────────────────────────────────────────────────────────────

def llm_ready() -> str:
    """
    真实发一次最小生成请求，返回空串表示可用，否则返回原因.

    只看 ``/api/tags`` 是不够的：Ollama 主进程在跑时它永远 200，而真正干活的
    ``llama-server`` 子进程可能刚崩掉（宿主内存紧张时常见）—— 那时
    ``/api/chat`` 会 500 或连接被拒。「Ollama 在跑」与「模型能推理」是两件事，
    环境问题不能被算成"层级转换没落到检索层"。
    """
    try:
        import asyncio

        import httpx

        from app.config import get_settings
    except Exception as exc:  # noqa: BLE001
        return f"无法加载配置：{type(exc).__name__}: {exc}"

    base = get_settings().OLLAMA_BASE_URL.rstrip("/")
    model = get_settings().OLLAMA_MODEL

    async def _run() -> str:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(
                f"{base}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "只回答数字：1+1=?"}],
                    "stream": False,
                    "think": False,
                    "options": {"num_predict": 8},
                },
            )
            if r.status_code != 200:
                return f"{base}/api/chat → HTTP {r.status_code}: {r.text[:120]}"
            return ""

    try:
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        return f"{base} 不可达：{type(exc).__name__}: {exc}"


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE_DEFAULT)
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--admin-password", default="RagAdmin#2026")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--ingest-timeout", type=int, default=300)
    args = parser.parse_args()
    base = args.base.rstrip("/")

    sfx = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    company = f"转入检索公司{sfx}"
    dept_dev = f"研发部{sfx}"
    dept_mkt = f"市场部{sfx}"
    pw = "TrPass#2026"
    users = {
        "kb": f"tr.kb.{sfx}@example.com",
        "mkt": f"tr.mkt.{sfx}@example.com",
        "dev": f"tr.dev.{sfx}@example.com",
    }
    prefixes = [f"tr.kb.{sfx}", f"tr.mkt.{sfx}", f"tr.dev.{sfx}"]

    print("═" * 74)
    print("  层级转换 → 检索链路 真实提问验证")
    print(f"  base={base}  公司={company}")
    print(f"  部门={dept_dev}（原部门，转换后不该看到） / {dept_mkt}（目标部门，应看到）")
    print("═" * 74)

    admin = login(base, args.admin_user, args.admin_password)

    # 本地模型不可用时直接停下：继续跑只会把环境问题记成"检索链路没生效"。
    reason = llm_ready()
    if reason:
        print(f"\n  预检失败：{reason}")
        print("  → 这不是层级转换的问题，而是本地 LLM 不可用（常见于宿主内存不足时")
        print("    Ollama 卸载/重启 llama-server）。请待模型可正常回答后重跑。")
        return 2
    print("  预检：本地模型可正常推理")

    # 三个新账号：上传并用转换的管理员 + 目标部门员工 + 原部门员工
    t_kb = make_user(base, admin, username=users["kb"], password=pw, company=company,
                     dept=dept_dev, duty="知识库管理员", role="kb_admin")
    t_mkt = make_user(base, admin, username=users["mkt"], password=pw, company=company,
                      dept=dept_mkt, duty="市场专员", role="employee")
    t_dev = make_user(base, admin, username=users["dev"], password=pw, company=company,
                      dept=dept_dev, duty="嵌入式软件工程师", role="employee")
    check("三个临时账号已建好（知识库管理员 / 市场部员工 / 研发部员工）", True)

    # 部门清单里拿到市场部的 department_id
    probe = f"探测载体-{sfx}.txt"
    upload(base, t_kb, probe, f"用于读取部门清单 {sfx}".encode("utf-8"),
           access_level="tenant")
    probe_row = find_doc(list_docs(base, t_kb), probe)
    probe_id = probe_row["document_id"] if probe_row else ""
    st, res = call("GET", f"/documents/{probe_id}/transfer-targets", base=base, token=t_kb)
    options = res.get("options", []) if isinstance(res, dict) else []
    by_name = {o["department_name"]: o for o in options}
    mkt_opt = by_name.get(dept_mkt)
    if not check("部门清单里找到市场部", bool(mkt_opt), f"names={list(by_name)}"):
        print("\n无法继续：拿不到目标部门 ID")
        return 1

    # ══ 场景 A：上传 → **立刻**转换（向量点尚不存在，最危险时序）══════════════
    section("场景 A. 上传公司库文档后**立刻**转为市场部（转换早于入库）")

    doc_a = f"紫罗兰计划-{sfx}.txt"
    fact_a = "8842"
    upload(base, t_kb, doc_a,
           f"紫罗兰计划的年度预算是 {fact_a} 万元，由财务部张明远负责审批。"
           f"本文件仅用于检索链路验证 {sfx}。".encode("utf-8"),
           access_level="tenant")
    row_a = find_doc(list_docs(base, t_kb), doc_a)
    a_id = row_a["document_id"] if row_a else ""
    check("场景 A 文档已上传（公司库）", bool(a_id))

    if a_id:
        st, res = call("POST", f"/documents/{a_id}/transfer-department", base=base,
                       token=t_kb,
                       payload={"department_id": mkt_opt["department_id"],
                                "note": f"上传后立即转换 {sfx}"})
        check("上传后立刻转为市场部文档（转换请求发生在入库之前）",
              st == 200 and res.get("access_level") == "department",
              f"status={st} {detail(res)}")

        pending = wait_ingest(base, t_kb, [doc_a], timeout=args.ingest_timeout)
        check("场景 A 文档入库完成", not pending, f"未完成={sorted(pending)}")

        q = ask(base, t_mkt, "紫罗兰计划的年度预算是多少？")
        ok_mkt = fact_a in q["answer"]
        check("【核心】市场部员工提问能答出转换后的文档内容（检索层 ACL 已跟随）",
              ok_mkt,
              f"answer={q['answer'][:160]!r} errors={q['errors'][:1]} "
              f"drilldown={describe_acl(a_id) if not ok_mkt else ''}")

        q2 = ask(base, t_dev, "紫罗兰计划的年度预算是多少？")
        ok_dev = (fact_a not in q2["answer"]) or (REFUSAL_SENTINEL in q2["answer"])
        check("研发部员工提问拿不到该内容（原部门已失去检索可见性）",
              ok_dev, f"answer={q2['answer'][:160]!r}")

    # ══ 场景 B：入库完成后再转换 ═══════════════════════════════════════════════
    section("场景 B. 上传公司库文档 → 等入库完成 → 再转为市场部")

    doc_b = f"翡翠计划-{sfx}.txt"
    fact_b = "2027 年 4 月 18 日"
    upload(base, t_kb, doc_b,
           f"翡翠计划的交付日期是 {fact_b}，验收由质量管理部负责。"
           f"本文件仅用于检索链路验证 {sfx}。".encode("utf-8"),
           access_level="tenant")
    row_b = find_doc(list_docs(base, t_kb), doc_b)
    b_id = row_b["document_id"] if row_b else ""
    check("场景 B 文档已上传（公司库）", bool(b_id))

    if b_id:
        pending = wait_ingest(base, t_kb, [doc_b], timeout=args.ingest_timeout)
        check("场景 B 文档入库完成", not pending, f"未完成={sorted(pending)}")

        st, res = call("POST", f"/documents/{b_id}/transfer-department", base=base,
                       token=t_kb,
                       payload={"department_id": mkt_opt["department_id"],
                                "note": f"入库后转换 {sfx}"})
        check("入库完成后转为市场部文档",
              st == 200 and res.get("access_level") == "department",
              f"status={st} {detail(res)}")

        q = ask(base, t_mkt, "翡翠计划的交付日期是什么时候？")
        ok_mkt = "2027" in q["answer"] and "4" in q["answer"]
        check("【核心】市场部员工提问能答出（入库后转换同样落到检索层）",
              ok_mkt,
              f"answer={q['answer'][:160]!r} "
              f"drilldown={describe_acl(b_id) if not ok_mkt else ''}")

        q2 = ask(base, t_dev, "翡翠计划的交付日期是什么时候？")
        ok_dev = ("2027" not in q2["answer"]) or (REFUSAL_SENTINEL in q2["answer"])
        check("研发部员工提问拿不到该内容", ok_dev,
              f"answer={q2['answer'][:160]!r}")

    # ── 清理 ─────────────────────────────────────────────────────────────────
    if args.keep:
        print("\n  （--keep：保留临时数据）")
    else:
        print(f"\n  清理：{cleanup(prefixes, [company])}")

    print("\n" + "═" * 74)
    print(f"  结果：{_passed} 通过 / {_failed} 失败")
    print("═" * 74)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
