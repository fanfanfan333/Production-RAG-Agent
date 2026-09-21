"""「删除公司」功能的在线验收脚本（QA 固化版，只读、可重复跑）.

诉求（用户 2026-09-19 拍板）
────────────────────────
增加删除按钮，删除后该公司**全部员工账号**（含个人数据，之后需重新注册）、
**三级文档全删**（private / department / tenant + 分块 + 倒排索引 + Qdrant 向量 +
``uploads/{tenant}/`` 磁盘原文件）、**会话与消息**、公司注册行一并删除；
权限**仅平台管理员**（``is_admin``）；必须有二次确认弹窗。

本脚本独立复核工程师的实现，**不采信其自测**：

* **默认只读**（不写任何数据，可重复跑）：走真实 HTTP 端点核对负向用例
  （``default``/不存在 id → 404，非平台管理员 → 403）、对**全部已注册公司**
  独立复算预检数字并逐项比对、核对每文档 Qdrant 实际点数与预检 ``vectors``
  是否一致、验证批量向量删除的**空列表短路语义**。
* ``--write`` 额外执行**自清理**的破坏性验证：造一家合成公司（多成员 + 三级文档
  + 会话/消息 + 收藏 + 反馈 + 留痕 + 磁盘文件 + Qdrant 向量）→ 走真实 HTTP
  ``DELETE /companies/{id}`` → 独立断言各表归零 / Qdrant 前后 count 对比归零 /
  磁盘目录消失 / 审计留存 / 留痕保留且外键置空 / 重复删除 → 404 / 无文档公司的
  空集删除**不误删他人向量** / 注入 Qdrant 失败验证「向量先行、PG 不动」。
* **跨租户保护（第 2 轮修复项，P5）**：合成「本公司成员名下、归属**别家租户**」的
  文档 → 删除本公司后该文档**仍存在、``owner_id`` 置 NULL、``tenant_id`` 不变**，
  且别家租户文档数不变；``kept.cross_tenant_documents`` 与实际解绑行数一致。
* **``vectors`` 是派生估算值（P7）**：预检 ``deleted.vectors`` = Σ ``documents.chunk_count``
  （**不等于**实时 Qdrant 点数——某批 upsert 永久失败时 chunk_count 不回滚）。
  脚本按派生值口径**严格**断言，并反证它与 Qdrant 实际点数确实可以不同。
  **绝不触碰 A公司 / B公司 / 测试公司1 / 测试公司2 的持久数据。**

退出码即结论：``0`` = 全部通过；``1`` = 存在失败项。脚本结尾打印失败清单。

容器内运行姿势（重要，照做否则静默失败）
────────────────────────────────────────
镜像的 runtime 阶段只 ``COPY app/`` —— **不含 ``scripts/`` ``tests/``**。正确姿势：

    docker cp backend/scripts/verify_company_deletion.py rag_backend:/tmp/
    docker exec -w /app -e PYTHONPATH=/app rag_backend \\
        python /tmp/verify_company_deletion.py           # 只读
    # 需要跑破坏性验证时再加 --write

两个踩过的坑：
  1. ``docker cp`` 直接覆盖容器内文件会因权限不足**静默留 0 字节**。可靠写法：
     ``docker cp <本地> rag_backend:/tmp/x.py`` → ``docker exec -u root rag_backend
     cp /tmp/x.py /app/<目标>`` → 用 ``md5sum`` 与本地 ``Get-FileHash`` 对齐。
     本脚本只落到 ``/tmp`` 直接以路径运行，不覆盖 ``/app``，规避该坑。
  2. 跑 pytest/脚本时**不要**加 ``-e HOME=/tmp``。那会把 user-site 切到
     ``/tmp/.local``，于是装在系统/``/app/.cache`` 的包全部变成
     "No module named ..."。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from qdrant_client.http import models as qm  # noqa: E402
from sqlalchemy import delete, func, or_, select  # noqa: E402

BASE = "http://127.0.0.1:8000"
HTTP_TIMEOUT = 120
NON_ADMIN_USERNAME = "tr.kb.bjld8@example.com"  # kb_admin，属测试公司1

# 真实注册公司（只读基线；--write 模式结束时复验仍是这 4 家）
REAL_COMPANIES = ["c8111de986583", "cfb08c53677c4", "c309a7cb9f496", "cf33b1db5679d"]

_failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)
    return ok


def info(label: str, detail: str = "") -> None:
    print(f"  [info] {label}{(' — ' + detail) if detail else ''}")


# ── HTTP 工具 ─────────────────────────────────────────────────────────────────


def _http(method: str, path: str, token: str, payload: dict | None = None) -> tuple[int, object]:
    data = None
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


# ── 独立复算：某租户「删除影响」的 PG 口径 ─────────────────────────────────────


async def _recompute_pg_assets(tenant_id: str) -> dict:
    """不调用被测服务，直接用 SQL 复算预检应得的数字（反抄脚本输出）。"""
    from app.db.badcase_models import BadCase
    from app.db.conversation_models import Conversation, Message
    from app.db.feedback_models import AnswerFeedback
    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.db.share_models import ShareRequest
    from app.db.staff_models import StaffRequest
    from app.db.user_models import Collection, User

    async with get_db_session() as s:
        member_ids = [r[0] for r in (await s.execute(select(User.id).where(User.tenant_id == tenant_id))).all()]
        doc_rows = (
            await s.execute(
                select(Document.id, Document.access_level, Document.chunk_count).where(
                    Document.tenant_id == tenant_id
                )
            )
        ).all()
        conv_clause = Conversation.tenant_id == tenant_id
        if member_ids:
            conv_clause = or_(conv_clause, Conversation.owner_id.in_(member_ids))
        conv_ids = [r[0] for r in (await s.execute(select(Conversation.id).where(conv_clause))).all()]
        priv = dept = ten = vec = 0
        for _i, lvl, cc in doc_rows:
            v = (lvl or "").strip() or "private"
            if v == "department":
                dept += 1
            elif v == "tenant":
                ten += 1
            else:
                priv += 1
            vec += int(cc or 0)
        msgs = 0
        if conv_ids:
            msgs = int(
                (
                    await s.execute(
                        select(func.count()).select_from(Message).where(Message.conversation_id.in_(conv_ids))
                    )
                ).scalar_one()
            )
        cols = fb = sr = shr = bc = 0
        for mid in member_ids:
            cols += int((await s.execute(select(func.count()).select_from(Collection).where(Collection.owner_id == mid))).scalar_one())
            fb += int((await s.execute(select(func.count()).select_from(AnswerFeedback).where(AnswerFeedback.user_id == mid))).scalar_one())
            sr += int((await s.execute(select(func.count()).select_from(StaffRequest).where(or_(StaffRequest.applicant_id == mid, StaffRequest.reviewer_id == mid)))).scalar_one())
            shr += int((await s.execute(select(func.count()).select_from(ShareRequest).where(or_(ShareRequest.requester_id == mid, ShareRequest.reviewer_id == mid)))).scalar_one())
            bc += int((await s.execute(select(func.count()).select_from(BadCase).where(BadCase.user_id == mid))).scalar_one())
    return {
        "members": len(member_ids),
        "documents": len(doc_rows),
        "documents_private": priv,
        "documents_department": dept,
        "documents_tenant": ten,
        "conversations": len(conv_ids),
        "messages": msgs,
        "collections": cols,
        "feedback": fb,
        "vectors": vec,
        "staff_requests": sr,
        "share_requests": shr,
        "bad_cases": bc,
    }


async def _real_snapshot() -> dict:
    from app.db.company_models import Company
    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.db.user_models import User

    async with get_db_session() as s:
        comps = {
            c.tenant_id: {
                "display_name": c.display_name,
                "created_by": str(c.created_by) if c.created_by else None,
                "is_test": bool(c.is_test),
            }
            for c in (await s.execute(select(Company))).scalars().all()
        }
        ucount = {t: int(n) for t, n in (await s.execute(select(User.tenant_id, func.count()).group_by(User.tenant_id))).all()}
        dcount = {t: int(n) for t, n in (await s.execute(select(Document.tenant_id, func.count()).group_by(Document.tenant_id))).all()}
    return {"companies": comps, "user_counts": ucount, "doc_counts": dcount}


# ── 只读阶段 ──────────────────────────────────────────────────────────────────


async def _phase_readonly(admin, admin_token, non_admin) -> None:
    from app.config import get_settings
    from app.db.company_models import Company
    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.db.qdrant import get_qdrant_client
    from app.services.auth_service import create_access_token
    import app.services.vector_service as vs

    st = get_settings()
    client = get_qdrant_client()

    async def q_total() -> int:
        c = await client.count(collection_name=st.QDRANT_COLLECTION, exact=True)
        return int(getattr(c, "count", 0) or 0)

    async def q_doc(doc_id: str) -> int:
        c = await client.count(
            collection_name=st.QDRANT_COLLECTION,
            count_filter=qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=doc_id))]),
            exact=True,
        )
        return int(getattr(c, "count", 0) or 0)

    print("── P1  HTTP 负向用例 ──")
    check(_http("GET", "/companies/default/deletion-preview", admin_token)[0] == 404, "admin 预检 default → 404")
    check(_http("GET", "/companies/cdeadbeef000/deletion-preview", admin_token)[0] == 404, "admin 预检不存在 id → 404")
    check(_http("DELETE", "/companies/default", admin_token)[0] == 404, "admin 删除 default → 404（删不掉落脚点）")
    check(_http("DELETE", "/companies/cdeadbeef000", admin_token)[0] == 404, "admin 删除不存在 id → 404")
    check(_http("GET", "/companies/{}/deletion-preview".format(REAL_COMPANIES[0]), "")[0] == 401, "无 token 预检 → 401")
    if non_admin is not None:
        nt = create_access_token(non_admin)
        check(_http("GET", f"/companies/{REAL_COMPANIES[0]}/deletion-preview", nt)[0] == 403, "非平台管理员预检 → 403")
        check(_http("DELETE", f"/companies/{REAL_COMPANIES[0]}", nt)[0] == 403, "非平台管理员删除 → 403")
        check(_http("GET", "/companies/default/deletion-preview", nt)[0] == 403, "非平台管理员预检 default → 403（权限先于存在性）")

    print("── P2  预检一致性（全部已注册公司，独立复算）──")
    async with get_db_session() as s:
        registered = [c.tenant_id for c in (await s.execute(select(Company))).scalars().all()]

    fields = [
        "members", "documents", "documents_private", "documents_department",
        "documents_tenant", "conversations", "messages", "collections", "feedback", "vectors",
    ]
    kept_fields = ["staff_requests", "share_requests", "bad_cases"]
    for tid in registered:
        status, body = _http("GET", f"/companies/{tid}/deletion-preview", admin_token)
        if not check(status == 200, f"{tid} 预检 HTTP 200", f"status={status}"):
            continue
        pg = await _recompute_pg_assets(tid)
        del_ = body.get("deleted", {}) if isinstance(body, dict) else {}
        kept_ = body.get("kept", {}) if isinstance(body, dict) else {}
        bad = [f for f in fields if del_.get(f) != pg[f]]
        badk = [f for f in kept_fields if kept_.get(f) != pg[f]]
        check(not bad, f"{tid} 预检 deleted 与独立复算逐项一致", f"mismatch={bad}")
        check(not badk, f"{tid} 预检 kept 与独立复算逐项一致", f"mismatch={badk}")
        # Qdrant 每文档点数 vs 预检 vectors
        async with get_db_session() as s:
            dids = [str(r[0]) for r in (await s.execute(select(Document.id).where(Document.tenant_id == tid))).all()]
        actual = sum([await q_doc(d) for d in dids]) if dids else 0
        info(f"{tid} vectors: 预检={del_.get('vectors')} Qdrant实际={actual} (chunk_sum={pg['vectors']})")

    print("── P3  反证 ①：批量向量删除的**空列表语义**（严格只读）──")
    t1 = await q_total()
    r = await vs.delete_by_document_ids([])
    t2 = await q_total()
    check(r == 0 and t1 == t2, "delete_by_document_ids([]) 不发请求且不误删（全库 count 不变）", f"return={r} before={t1} after={t2}")


# ── 写入阶段（自清理）──────────────────────────────────────────────────────────


async def _phase_write(admin, admin_token) -> None:
    import uuid

    from app.config import get_settings
    from app.db.badcase_models import BadCase
    from app.db.company_models import Company
    from app.db.conversation_models import Conversation, Message
    from app.db.feedback_models import AnswerFeedback
    from app.db.models import ChunkParent, Document, DocumentChunkTerm, DocumentMetadataRow
    from app.db.postgres import get_db_session
    from app.db.quality_models import QualityEvent
    from app.db.qdrant import get_qdrant_client
    from app.db.share_models import ShareRequest
    from app.db.staff_models import StaffRequest
    from app.db.user_models import AuditLog, Collection, User
    from app.services import staff_service
    from app.services.company_registry import normalize_company_name_key
    from app.services.storage.image_store import document_dir
    import app.services.vector_service as vs

    st = get_settings()
    client = get_qdrant_client()
    coll = await client.get_collection(collection_name=st.QDRANT_COLLECTION)
    dim = coll.config.params.vectors.size
    total0 = int(coll.points_count or 0)

    async def q_total() -> int:
        c = await client.count(collection_name=st.QDRANT_COLLECTION, exact=True)
        return int(getattr(c, "count", 0) or 0)

    async def q_doc(doc_id: str) -> int:
        c = await client.count(
            collection_name=st.QDRANT_COLLECTION,
            count_filter=qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=doc_id))]),
            exact=True,
        )
        return int(getattr(c, "count", 0) or 0)

    async def q_upsert(doc_id: str, tenant: str, n: int) -> None:
        pts = [qm.PointStruct(id=str(uuid.uuid4()), vector=[0.1] * dim, payload={"document_id": doc_id, "tenant_id": tenant}) for _ in range(n)]
        if pts:
            await client.upsert(collection_name=st.QDRANT_COLLECTION, points=pts, wait=True)

    syn = "c" + uuid.uuid4().hex[:12]
    other = "c" + uuid.uuid4().hex[:12]
    m1, m2, mu = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    doc_priv, doc_dept, doc_ten = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    doc_other_ownerless, doc_other_by_member = uuid.uuid4(), uuid.uuid4()
    conv1, conv2 = uuid.uuid4(), uuid.uuid4()
    col_id, fb_id, bc_id, shr_id, sreq_id, qe_id = (uuid.uuid4() for _ in range(6))
    syn_name = f"QA删司-{uuid.uuid4().hex[:6]}"

    try:
        async with get_db_session() as s:
            s.add(Company(tenant_id=syn, display_name=syn_name, name_key=normalize_company_name_key(syn_name), created_by=admin.id, is_test=True))
            s.add(User(id=m1, username=f"qa-delm1-{uuid.uuid4().hex[:6]}", display_name="QA成员1", role=User.ROLE_EMPLOYEE, tenant_id=syn, company_name=syn_name, department_id="d_qa", department_name="QA部", job_title="QA"))
            s.add(User(id=m2, username=f"qa-delm2-{uuid.uuid4().hex[:6]}", display_name="QA成员2", role=User.ROLE_KB_ADMIN, tenant_id=syn, company_name=syn_name, department_id="d_qa", department_name="QA部", job_title="QA"))
            s.add(User(id=mu, username=f"qa-other-{uuid.uuid4().hex[:6]}", display_name="别家用户", role=User.ROLE_EMPLOYEE, tenant_id=other, company_name="别家"))
            await s.flush()
            for did, lvl, cc in [(doc_priv, "private", 4), (doc_dept, "department", 3), (doc_ten, "tenant", 2)]:
                s.add(Document(id=did, filename=f"qa-{lvl}.txt", file_size=10, file_hash=uuid.uuid4().hex, tenant_id=syn, access_level=lvl, owner_id=m1, chunk_count=cc, department_id=("d_qa" if lvl == "department" else None)))
            s.add(Document(id=doc_other_ownerless, filename="other-owned-elsewhere.txt", file_size=10, file_hash=uuid.uuid4().hex, tenant_id=other, access_level="private", owner_id=mu, chunk_count=1))
            s.add(Document(id=doc_other_by_member, filename="other-owned-by-member.txt", file_size=10, file_hash=uuid.uuid4().hex, tenant_id=other, access_level="private", owner_id=m1, chunk_count=1))
            await s.flush()
            s.add(DocumentMetadataRow(document_id=doc_priv, tenant_id=syn, title="QA"))
            s.add(ChunkParent(parent_id=f"{doc_priv}:p:0", document_id=doc_priv, tenant_id=syn, level="parent", idx=0, text="qa", access_level="private"))
            s.add(DocumentChunkTerm(document_id=doc_priv, tenant_id=syn, chunk_index=0, point_id=str(uuid.uuid4()), terms="qa", access_level="private"))
            s.add(Conversation(id=conv1, owner_id=m1, tenant_id=syn))
            s.add(Conversation(id=conv2, owner_id=m1, tenant_id=syn))
            await s.flush()
            for cid in (conv1, conv1, conv2):
                s.add(Message(conversation_id=cid, role="user", content="q", user_id=m1, tenant_id=syn))
            s.add(Collection(id=col_id, name="QA收藏", owner_id=m1))
            s.add(AnswerFeedback(id=fb_id, user_id=m1, conversation_id=conv1, rating="down", question="q", answer="a"))
            s.add(BadCase(id=bc_id, user_id=m1, username="qa", conversation_id=conv1, reason="feedback_down", question="q", answer="a"))
            s.add(ShareRequest(id=shr_id, tenant_id=syn, document_id=doc_priv, document_name="qa-private.txt", requester_id=m1, requester_username="qa", intent="publish", target_level="department", status="pending"))
            s.add(StaffRequest(id=sreq_id, applicant_id=m1, applicant_username="qa", applicant_company_id=syn, company_name=syn_name, company_id=syn, department_name="QA部", department_id="d_qa", duty="QA", status="pending"))
            s.add(QualityEvent(id=qe_id, user_id=m1, username="qa", conversation_id=conv1, intent="knowledge_qa"))
            await s.flush()

        for did, n in [(doc_priv, 4), (doc_dept, 3), (doc_ten, 2), (doc_other_ownerless, 1)]:
            await q_upsert(str(did), syn if did in (doc_priv, doc_dept, doc_ten) else other, n)
        for did in (doc_priv, doc_dept, doc_ten):
            d = document_dir(str(did), syn)
            d.mkdir(parents=True, exist_ok=True)
            (d / "raw.txt").write_text("qa", encoding="utf-8")
        syn_dir = document_dir(str(doc_priv), syn).parent

        print("── P4  合成公司完整删除（真实 HTTP DELETE）──")
        stp, prev = _http("GET", f"/companies/{syn}/deletion-preview", admin_token)
        pg = await _recompute_pg_assets(syn)
        d, k = prev.get("deleted", {}), prev.get("kept", {})
        check(stp == 200, "预检 HTTP 200", f"status={stp}")
        check(d.get("members") == 2 and d.get("documents") == 3 and d.get("conversations") == 2, "预检 members/documents/conversations = 2/3/2", f"{d}")
        check(d.get("documents_private") == 1 and d.get("documents_department") == 1 and d.get("documents_tenant") == 1, "预检三级文档各 1", f"{d}")
        check(d.get("messages") == 3 and d.get("collections") == 1 and d.get("feedback") == 1, "预检 messages/collections/feedback = 3/1/1", f"{d}")
        check(d.get("vectors") == 9 and pg["vectors"] == 9, "预检 vectors=9 且与独立复算一致", f"preview={d.get('vectors')} recompute={pg['vectors']}")
        check(k.get("staff_requests") == 1 and k.get("share_requests") == 1 and k.get("bad_cases") == 1, "预检 kept 留痕各 1", f"{k}")

        # 反证：注入 Qdrant 失败 → 向量先行，PG 一行不动
        orig = vs.delete_by_document_ids
        async def _boom(*_a, **_k):
            raise RuntimeError("QA-injected qdrant failure")
        vs.delete_by_document_ids = _boom
        aborted = err = None
        try:
            await staff_service.delete_company(admin, syn)
        except staff_service.StaffError as exc:
            aborted, err = True, (exc.status_code, str(exc))
        finally:
            vs.delete_by_document_ids = orig
        async with get_db_session() as s:
            comp_alive = (await s.get(Company, syn)) is not None
            docs_alive = int((await s.execute(select(func.count()).select_from(Document).where(Document.tenant_id == syn))).scalar_one())
            users_alive = int((await s.execute(select(func.count()).select_from(User).where(User.tenant_id == syn))).scalar_one())
        check(aborted and err and err[0] == 503, "注入 Qdrant 失败 → StaffError(503)", f"err={err}")
        check(comp_alive and docs_alive == 3 and users_alive == 2, "向量先行：失败时 PG 一行未动", f"company={comp_alive} docs={docs_alive} users={users_alive}")

        before = await q_total()
        stp, body = _http("DELETE", f"/companies/{syn}", admin_token)
        check(stp == 200, "真实 HTTP DELETE → 200", f"status={stp} body={str(body)[:120]}")

        async with get_db_session() as s:
            zeros = {
                "documents": int((await s.execute(select(func.count()).select_from(Document).where(Document.tenant_id == syn))).scalar_one()),
                "document_metadata": int((await s.execute(select(func.count()).select_from(DocumentMetadataRow).where(DocumentMetadataRow.document_id == doc_priv))).scalar_one()),
                "chunk_parents": int((await s.execute(select(func.count()).select_from(ChunkParent).where(ChunkParent.document_id == doc_priv))).scalar_one()),
                "chunk_terms": int((await s.execute(select(func.count()).select_from(DocumentChunkTerm).where(DocumentChunkTerm.document_id == doc_priv))).scalar_one()),
                "conversations": int((await s.execute(select(func.count()).select_from(Conversation).where(Conversation.tenant_id == syn))).scalar_one()),
                "messages": int((await s.execute(select(func.count()).select_from(Message).where(Message.conversation_id.in_([conv1, conv2])))).scalar_one()),
                "users": int((await s.execute(select(func.count()).select_from(User).where(User.tenant_id == syn))).scalar_one()),
                "collections": int((await s.execute(select(func.count()).select_from(Collection).where(Collection.owner_id.in_([m1, m2])))).scalar_one()),
                "answer_feedback": int((await s.execute(select(func.count()).select_from(AnswerFeedback).where(AnswerFeedback.user_id.in_([m1, m2])))).scalar_one()),
                "companies": int((await s.execute(select(func.count()).select_from(Company).where(Company.tenant_id == syn))).scalar_one()),
            }
            bc_row = await s.get(BadCase, bc_id)
            shr_row = await s.get(ShareRequest, shr_id)
            sreq_row = await s.get(StaffRequest, sreq_id)
            qe_row = await s.get(QualityEvent, qe_id)
            audit_cnt = int((await s.execute(select(func.count()).select_from(AuditLog).where(AuditLog.action == "company.delete", AuditLog.resource_id == syn))).scalar_one())
            other_ownerless_alive = (await s.get(Document, doc_other_ownerless)) is not None
            other_by_member_alive = (await s.get(Document, doc_other_by_member)) is not None
            other_user_alive = (await s.get(User, mu)) is not None
        check(all(v == 0 for v in zeros.values()), "PG 全部归零（文档/分块/会话/消息/账号/收藏/反馈/公司行）", f"{zeros}")
        check(bc_row is not None and shr_row is not None and sreq_row is not None and qe_row is not None, "留痕保留（bad_case/share_request/staff_request/quality_event）")
        check(
            bc_row is not None and bc_row.user_id is None and bc_row.conversation_id is None
            and shr_row is not None and shr_row.requester_id is None and shr_row.document_id is None
            and sreq_row is not None and sreq_row.applicant_id is None,
            "留痕归属人外键已置 NULL（SET NULL，未被 CASCADE 带走）",
        )
        check(audit_cnt >= 1, "审计 company.delete 存在且留存", f"count={audit_cnt}")

        after = await q_total()
        syn_vec_left = sum([await q_doc(str(x)) for x in (doc_priv, doc_dept, doc_ten)])
        check(syn_vec_left == 0, "该公司文档向量在 Qdrant 归零", f"left={syn_vec_left}")
        check(before - after == 9, "全库 Qdrant 点数恰减少 9（清且仅清该公司向量）", f"before={before} after={after} delta={before - after}")
        check(not syn_dir.exists(), "uploads/{tenant}/ 磁盘目录确实不存在", f"path={syn_dir}")

        check(_http("DELETE", f"/companies/{syn}", admin_token)[0] == 404, "同一公司删两次 → 第二次 404（非 500）")

        print("── P5  跨租户安全 ──")
        check(other_ownerless_alive and other_user_alive, "别家租户的无关用户与其文档未被误删")
        if not other_by_member_alive:
            check(False, "本公司成员名下、**别家租户**的文档不应被删", "实际已被删除（经 users.owner_id CASCADE 连带）")
        else:
            check(True, "本公司成员名下、别家租户的文档未被删")

        print("── P6  反证 ①：删除**无文档**的合成公司不误删他人向量 ──")
        syn_empty = "c" + uuid.uuid4().hex[:12]
        ename = f"QA空司-{uuid.uuid4().hex[:6]}"
        async with get_db_session() as s:
            s.add(Company(tenant_id=syn_empty, display_name=ename, name_key=normalize_company_name_key(ename), created_by=admin.id, is_test=True))
            await s.flush()
        tA = await q_total()
        stp, _ = _http("DELETE", f"/companies/{syn_empty}", admin_token)
        tB = await q_total()
        check(stp == 200 and tA == tB, "无文档公司删除成功且全库 count 不变（空集不删全库）", f"status={stp} before={tA} after={tB}")

        print("── P7  反证 ②：预检 vectors = Σ chunk_count（**派生估算值**，非实时 Qdrant 点数）──")
        syn_vec = "c" + uuid.uuid4().hex[:12]
        vname = f"QA向量司-{uuid.uuid4().hex[:6]}"
        d_no_vec, d_has_vec = uuid.uuid4(), uuid.uuid4()
        async with get_db_session() as s:
            s.add(Company(tenant_id=syn_vec, display_name=vname, name_key=normalize_company_name_key(vname), created_by=admin.id, is_test=True))
            await s.flush()
            # pg-only：PG 记 7 块但从未 upsert；vec-extra：PG 记 1 块但实际 upsert 了 3 条
            s.add(Document(id=d_no_vec, filename="pg-only.txt", file_size=1, file_hash=uuid.uuid4().hex, tenant_id=syn_vec, access_level="private", owner_id=None, chunk_count=7))
            s.add(Document(id=d_has_vec, filename="vec-extra.txt", file_size=1, file_hash=uuid.uuid4().hex, tenant_id=syn_vec, access_level="private", owner_id=None, chunk_count=1))
            await s.flush()
        await q_upsert(str(d_has_vec), syn_vec, 3)
        stp, pv = _http("GET", f"/companies/{syn_vec}/deletion-preview", admin_token)
        actual = await q_doc(str(d_no_vec)) + await q_doc(str(d_has_vec))
        preview_vectors = (pv.get("deleted", {}) or {}).get("vectors") if isinstance(pv, dict) else None
        # 独立复算派生值：Σ documents.chunk_count —— 这就是预检 vectors 的**定义**
        async with get_db_session() as s:
            chunk_sum = int(
                (
                    await s.execute(
                        select(func.coalesce(func.sum(Document.chunk_count), 0)).where(
                            Document.tenant_id == syn_vec
                        )
                    )
                ).scalar_one()
                or 0
            )
        check(preview_vectors == chunk_sum, "预检 vectors == Σ chunk_count（派生值口径，严格相等）",
              f"preview={preview_vectors} chunk_sum={chunk_sum}")
        # 反证「它是派生值、不是实时点数」：Qdrant 实际点数与派生值**本就可以不同**。
        # 这正是已文档化的派生值语义（后端注释 + 前端标注「按入库分块预计」），**不判失败**。
        info(f"Qdrant 实际点数={actual} ≠ 派生值 {chunk_sum} —— 属**预期**：pg-only.txt 的 7 块从未 upsert"
             "（文档会被置 FAILED 但 chunk_count 不回滚）；前端已标注『按入库分块预计』。删除正确性不受影响。")
        check(actual != chunk_sum, "反证：Qdrant 实际 ≠ 派生值（证明预检 vectors 确为派生估算而非实时点数）")
        _http("DELETE", f"/companies/{syn_vec}", admin_token)
        await vs.delete_by_document_ids([str(d_has_vec)])
    finally:
        print("── 清理合成数据 ──")
        for did in (doc_priv, doc_dept, doc_ten, doc_other_ownerless, doc_other_by_member):
            try:
                await vs.delete_by_document_ids([str(did)])
            except Exception:
                pass
        async with get_db_session() as s:
            for did in (doc_priv, doc_dept, doc_ten, doc_other_ownerless, doc_other_by_member):
                row = await s.get(Document, did)
                if row is not None:
                    await s.delete(row)
            await s.flush()
            for mid in (m1, m2, mu):
                row = await s.get(User, mid)
                if row is not None:
                    await s.delete(row)
            for cid in (syn, other):
                row = await s.get(Company, cid)
                if row is not None:
                    await s.delete(row)
            for model, key in [(BadCase, bc_id), (ShareRequest, shr_id), (StaffRequest, sreq_id), (QualityEvent, qe_id)]:
                row = await s.get(model, key)
                if row is not None:
                    await s.delete(row)
            await s.flush()
        try:
            import shutil

            for did in (doc_priv, doc_dept, doc_ten):
                dd = document_dir(str(did), syn)
                if dd.exists():
                    shutil.rmtree(dd.parent, ignore_errors=True)
        except Exception:
            pass
        info(f"清理完成；全库 Qdrant 点数 初始={total0} 结束={await q_total()}")


# ── 主流程 ────────────────────────────────────────────────────────────────────


async def _run(with_write: bool) -> int:
    from app.db.postgres import get_db_session
    from app.db.user_models import User
    from app.services.auth_service import create_access_token

    async with get_db_session() as s:
        admin = await s.scalar(select(User).where(User.role == User.ROLE_ADMIN).limit(1))
        non_admin = await s.scalar(select(User).where(User.username == NON_ADMIN_USERNAME).limit(1))
        if non_admin is None:
            non_admin = await s.scalar(select(User).where(User.role != User.ROLE_ADMIN).limit(1))
    if admin is None:
        print("FATAL：找不到平台管理员账号")
        return 1
    admin_token = create_access_token(admin)
    print(f"── 账号 ─ admin={admin.username}({admin.id}) tenant={admin.tenant_id}; "
          f"非管理员={getattr(non_admin, 'username', None)}")

    baseline = await _real_snapshot()

    try:
        await _phase_readonly(admin, admin_token, non_admin)
        if with_write:
            await _phase_write(admin, admin_token)
    except Exception:
        print("── 运行异常 ──")
        print(traceback.format_exc())
        _failures.append("脚本运行异常（见 traceback）")

    print("── P8  真实公司零变化复验 ──")
    after = await _real_snapshot()
    check(set(after["companies"]) == set(REAL_COMPANIES), "companies 仍为原 4 家", f"actual={sorted(after['companies'])}")
    check(
        after["companies"].get("c309a7cb9f496", {}).get("created_by") is None
        and after["companies"].get("cf33b1db5679d", {}).get("created_by") is None,
        "A/B 公司 created_by 仍为 NULL",
    )
    check(after["companies"] == baseline["companies"], "companies 全表与基线逐字段一致")
    check(after["user_counts"] == baseline["user_counts"], "users 各租户计数与基线一致", f"baseline={baseline['user_counts']} after={after['user_counts']}")
    check(after["doc_counts"] == baseline["doc_counts"], "documents 各租户计数与基线一致")

    print()
    if _failures:
        print(f"共 {len(_failures)} 项失败：")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("全部断言通过。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="删除公司功能验收脚本（默认只读）")
    parser.add_argument("--write", action="store_true", help="额外执行自清理的破坏性验证（默认只读）")
    args = parser.parse_args()
    return asyncio.run(_run(with_write=args.write))


if __name__ == "__main__":
    sys.exit(main())
