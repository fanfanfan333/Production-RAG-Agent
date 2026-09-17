"""
三级知识库全链路验收（企业 RAG）.

一次跑完产品要求的**全部**功能面，并把「谁该看到什么、谁该能删什么」用
真实 HTTP 请求钉死，而不是靠读代码推断：

    A. 身份与申请      注册 → 身份验证申请 → 上级同意 / 拒绝 → 更换岗位
    B. 上传与三层标志   个人 / 部门 / 公司 三层的上传权限与 access_level 落库
    C. 申请共享闭环     普通员工申请 → 部门负责人 / 知识库管理员 同意或拒绝
    D. 删除权限矩阵     个人删个人；部门库由部门负责人及以上；公司库由知识库管理员及以上
    E. 隔离性           公司之间 / 部门之间 / 个人之间（可见 + 可删 + 检索）
    F. 文档显示         个人看到的 = 本人文档 + 本部门文档 + 公司文档

数据是**两公司 × 两部门 × 六账号**的真实拓扑，全部临时账号，跑完自动清理。

运行（容器内或宿主机均可，只用标准库）：

    docker exec -e HOME=/tmp -e NO_PROXY='*' rag_backend \
        python /app/scripts/e2e_full_acceptance.py

    python backend/scripts/e2e_full_acceptance.py --keep   # 保留数据排查
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

# 企业网里 http_proxy 常把 127.0.0.1 也劫持成假 502 —— 显式绕过代理。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE

# /auth/* 有 10 次/分钟/IP 的限流（产品行为，不去关它）。验收脚本要建 8 个账号、
# 登录 8 次，必然撞到 429 —— 所以脚本侧做退避重试，而不是把限流调松。
_RETRY_429_MAX = 45
_RETRY_429_SLEEP = 2.0

_passed = 0
_failed = 0
_section = ""


# ── 断言 ──────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    global _section
    _section = title
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


# ── HTTP ──────────────────────────────────────────────────────────────────────

def call(
    method: str,
    path: str,
    *,
    base: str,
    token: str | None = None,
    payload: dict | None = None,
) -> tuple[int, dict | list | str]:
    url = f"{base}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None

    # /auth/* 限流命中（429）时退避重试；其它状态码一律原样返回。
    for attempt in range(_RETRY_429_MAX):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with _OPENER.open(req, timeout=60) as resp:
                return resp.status, _decode(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < _RETRY_429_MAX - 1:
                if attempt == 0:
                    print(f"    · 触发限流，退避重试中（{path}）…")
                time.sleep(_RETRY_429_SLEEP)
                continue
            return exc.code, _decode(body)
        except Exception as exc:  # noqa: BLE001
            return 0, f"{type(exc).__name__}: {exc}"
    return 429, "rate limited"


def _decode(body: str):
    try:
        return json.loads(body)
    except Exception:  # noqa: BLE001
        return body


def upload(
    base: str,
    token: str,
    filename: str,
    content: bytes,
    *,
    access_level: str | None = None,
) -> tuple[int, dict | list | str]:
    """multipart/form-data 上传（urllib 不内置，手写 boundary）。"""
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
    # 上传同样有限流（12 次/分钟/IP），验收脚本要连传十几份 → 退避重试。
    for attempt in range(_RETRY_429_MAX):
        try:
            with _OPENER.open(req, timeout=120) as resp:
                return resp.status, _decode(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < _RETRY_429_MAX - 1:
                if attempt == 0:
                    print("    · 上传触发限流，退避重试中…")
                time.sleep(_RETRY_429_SLEEP)
                continue
            return exc.code, _decode(body_text)
        except Exception as exc:  # noqa: BLE001
            return 0, f"{type(exc).__name__}: {exc}"
    return 429, "rate limited"


def login(base: str, username: str, password: str) -> str:
    status, res = call(
        "POST", "/auth/login", base=base,
        payload={"username": username, "password": password},
    )
    if status != 200 or not isinstance(res, dict):
        raise SystemExit(f"登录失败 {username}: {status} {detail(res)}")
    return res["access_token"]


# ── 业务动作 ──────────────────────────────────────────────────────────────────

def register(base: str, username: str, password: str) -> str:
    status, res = call(
        "POST", "/auth/register", base=base,
        payload={"username": username, "password": password},
    )
    if status != 201 or not isinstance(res, dict):
        raise SystemExit(f"注册失败 {username}: {status} {detail(res)}")
    return res["access_token"]


def submit_identity(base: str, token: str, company: str, dept: str, duty: str):
    return call(
        "POST", "/staff/requests", base=base, token=token,
        payload={"company_name": company, "department_name": dept, "duty": duty},
    )


def approve_identity(
    base: str, reviewer_token: str, request_id: str, role: str, dept: str, duty: str,
    *, approve: bool = True, title: str = "部门负责人", name: str = "审核人",
    comment: str = "",
):
    return call(
        "POST", f"/staff/requests/{request_id}/review", base=base, token=reviewer_token,
        payload={
            "approve": approve, "role": role, "department_name": dept, "duty": duty,
            "reviewer_title": title, "reviewer_name": name, "comment": comment,
        },
    )


def list_docs(base: str, token: str, **params) -> tuple[int, dict | list | str]:
    query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    path = f"/documents?limit=100{'&' + query if query else ''}"
    status, res = call("GET", path, base=base, token=token)
    # /documents 的列表字段叫 ``documents``（不是 items）。这里补一个 items 别名，
    # 让下面所有取列表的代码只有一个口径，不必在 20 处分别记住字段名。
    if isinstance(res, dict) and "documents" in res and "items" not in res:
        res = {**res, "items": res["documents"]}
    return status, res


def docs_items(res: object) -> list[dict]:
    """/documents 的列表字段叫 ``documents``（不是 items）——统一在这里取。"""
    if isinstance(res, dict):
        rows = res.get("documents") or res.get("items") or []
        return rows if isinstance(rows, list) else []
    return []


def find_doc(items: list[dict], name: str) -> dict | None:
    return next((d for d in items if d.get("filename") == name), None)


def make_user(
    base: str, admin_token: str, *, username: str, password: str, company: str,
    dept: str, duty: str, role: str,
    reviewer_token: str | None = None, reviewer_title: str = "平台管理员",
    reviewer_name: str = "超级管理员",
) -> str:
    """
    注册 → 提身份验证 → 由 reviewer 批准并授予 role → 返回该用户 token.

    默认由平台管理员批准；传入 reviewer_token 可测「部门负责人批准普通员工」
    这条层级链路。
    """
    token = register(base, username, password)
    status, res = submit_identity(base, token, company, dept, duty)
    if status != 201:
        raise SystemExit(f"提交身份验证失败 {username}: {status} {detail(res)}")
    rid = res["request"]["id"]
    status, res = approve_identity(
        base, reviewer_token or admin_token, rid, role, dept, duty,
        title=reviewer_title, name=reviewer_name,
        comment=f"同意 {username} 加入 {dept}",
    )
    if status != 200:
        raise SystemExit(f"批准失败 {username}: {status} {detail(res)}")
    return login(base, username, password)


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

            # 公司名 → 租户 id，用来兜住「文档已删、document_id 已置空」的申请记录
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
                        # 坑：删除申请通过后 document_id 会被置空（ON DELETE SET NULL，
                        # 有意保留审计记录），于是「按 document_id 清」永远漏掉这一批。
                        # 必须同时按 tenant_id / 申请人来清，否则每跑一轮就留 6 条残渣。
                        ShareRequest.tenant_id.in_(tenant_ids) if tenant_ids else False,
                        ShareRequest.requester_username.in_(unames) if unames else False,
                        or_(*[ShareRequest.requester_username.like(f"{p}%") for p in prefixes]),
                        ShareRequest.document_id.in_(doc_ids) if doc_ids else False,
                        or_(*[ShareRequest.document_name.like(f"{p}%") for p in prefixes]),
                    )
                )
            )
            r_doc = await session.execute(
                delete(Document).where(Document.id.in_(doc_ids)) if doc_ids
                else None
            )
            r_staff = await session.execute(
                delete(StaffRequest).where(
                    or_(
                        or_(*[StaffRequest.applicant_username.like(f"{p}%") for p in prefixes]),
                        StaffRequest.company_name.in_(companies),
                        *([StaffRequest.company_name.like(f"{c}%") for c in companies] if companies else []),
                    )
                )
            )
            if uids:
                await session.execute(
                    delete(User).where(User.id.in_(uids))
                )
            await session.commit()
            return {
                "users": len(uids),
                "docs": (r_doc.rowcount or 0) if r_doc is not None else 0,
                "share_reqs": r_share.rowcount or 0,
                "staff_reqs": r_staff.rowcount or 0,
                "usernames": unames,
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
    parser.add_argument("--keep", action="store_true", help="保留临时数据（默认清理）")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    sfx = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    comp_a = f"甲公司{sfx}"
    comp_b = f"乙公司{sfx}"
    dept_dev = f"研发部{sfx}"
    dept_mkt = f"市场部{sfx}"
    duty_dev = "嵌入式软件工程师"
    duty_mkt = "市场专员"
    pw = "E2ePass#2026"

    users = {
        "a_kb": f"a.kb.{sfx}@example.com",
        "a_mgr": f"a.mgr.{sfx}@example.com",
        "a_emp1": f"a.emp1.{sfx}@example.com",
        "a_emp2": f"a.emp2.{sfx}@example.com",
        "b_kb": f"b.kb.{sfx}@example.com",
        "b_emp": f"b.emp.{sfx}@example.com",
    }
    prefixes = [f"a.kb.{sfx}", f"a.mgr.{sfx}", f"a.emp1.{sfx}", f"a.emp2.{sfx}",
                f"b.kb.{sfx}", f"b.emp.{sfx}"]

    print("═" * 74)
    print("  三级知识库全链路验收")
    print(f"  base={base}")
    print(f"  甲公司={comp_a}  乙公司={comp_b}")
    print(f"  部门={dept_dev} / {dept_mkt}")
    print("═" * 74)

    admin = login(base, args.admin_user, args.admin_password)
    check("平台管理员登录", bool(admin))

    # ══ A. 身份与申请 ══════════════════════════════════════════════════════════
    section("A. 身份与申请：注册 → 申请 → 同意 / 拒绝 → 更换岗位")

    # 新注册 = 未验证，业务接口被拦
    tmp_token = register(base, f"probe.{sfx}@example.com", pw)
    prefixes.append(f"probe.{sfx}")
    st, res = call("GET", "/documents", base=base, token=tmp_token)
    check("未验证账号访问 /documents 被身份闸门拦下（403）", st == 403,
          f"status={st} {detail(res)}")
    st, me = call("GET", "/staff/me", base=base, token=tmp_token)
    check("未验证身份状态 = none（去验证）",
          st == 200 and isinstance(me, dict) and me.get("identity_status") == "none",
          f"identity={me.get('identity_status') if isinstance(me, dict) else me}")

    # 甲公司知识库管理员（平台管理员批准）
    t_a_kb = make_user(
        base, admin, username=users["a_kb"], password=pw, company=comp_a,
        dept=dept_dev, duty="知识库管理员", role="kb_admin",
    )
    # 甲公司研发部负责人
    t_a_mgr = make_user(
        base, admin, username=users["a_mgr"], password=pw, company=comp_a,
        dept=dept_dev, duty="研发部负责人", role="dept_manager",
    )
    # 甲公司普通员工（由**本部门负责人**批准 —— 验证层级链路）
    t_a_emp1 = make_user(
        base, admin, username=users["a_emp1"], password=pw, company=comp_a,
        dept=dept_dev, duty=duty_dev, role="employee",
        reviewer_token=t_a_mgr, reviewer_title="部门负责人", reviewer_name="研发部负责人",
    )
    # 甲公司市场部普通员工
    t_a_emp2 = make_user(
        base, admin, username=users["a_emp2"], password=pw, company=comp_a,
        dept=dept_mkt, duty=duty_mkt, role="employee",
    )
    # 乙公司知识库管理员 + 普通员工
    t_b_kb = make_user(
        base, admin, username=users["b_kb"], password=pw, company=comp_b,
        dept=dept_dev, duty="知识库管理员", role="kb_admin",
    )
    t_b_emp = make_user(
        base, admin, username=users["b_emp"], password=pw, company=comp_b,
        dept=dept_dev, duty=duty_dev, role="employee",
    )
    check("六个账号全部通过身份验证", True)

    st, me = call("GET", "/staff/me", base=base, token=t_a_emp1)
    ok = isinstance(me, dict)
    check("批准后身份状态 = approved", ok and me.get("identity_status") == "approved",
          f"identity={me.get('identity_status') if ok else me}")
    check("公司名称为中文原文（非内部 ID）",
          ok and me.get("company_name") == comp_a,
          f"company_name={me.get('company_name') if ok else me}")
    check("部门 / 职责已落地",
          ok and me.get("department_name") == dept_dev and me.get("job_title") == duty_dev,
          f"dept={me.get('department_name') if ok else me} duty={me.get('job_title') if ok else me}")

    # 拒绝路径：乙公司员工提交，甲公司知识库管理员无权审（跨公司）
    reject_token = register(base, f"reject.{sfx}@example.com", pw)
    prefixes.append(f"reject.{sfx}")
    st, res = submit_identity(base, reject_token, comp_b, dept_dev, duty_dev)
    reject_rid = res["request"]["id"] if st == 201 else ""
    check("跨公司申请人提交身份验证申请", st == 201, f"status={st} {detail(res)}")
    if reject_rid:
        st, res = approve_identity(
            base, t_a_kb, reject_rid, "employee", dept_dev, duty_dev,
        )
        check("甲公司管理员无法审核乙公司的申请（403，公司隔离）", st == 403,
              f"status={st} {detail(res)}")
        st, res = approve_identity(
            base, admin, reject_rid, "employee", dept_dev, duty_dev, approve=False,
            title="平台管理员", name="超级管理员", comment="材料不全，暂不通过",
        )
        ok = st == 200 and isinstance(res, dict)
        check("平台管理员拒绝该申请", ok, f"status={st} {detail(res)}")
        check("拒绝后负责人记录为「职务 + 名称」（后台可见）",
              ok and res["request"].get("reviewer_label") == "平台管理员 超级管理员",
              f"reviewer_label={res['request'].get('reviewer_label') if ok else ''}")

    # 更换岗位
    st, members = call("GET", f"/staff/members?keyword={users['a_emp1']}",
                       base=base, token=admin)
    rows = members.get("items", []) if isinstance(members, dict) else []
    if check("管理后台可查到成员", len(rows) == 1, f"found={len(rows)}"):
        uid = rows[0]["id"]
        st, res = call("PATCH", f"/staff/members/{uid}", base=base, token=admin,
                       payload={"role": "dept_manager", "job_title": "技术总监"})
        ok = st == 200 and isinstance(res, dict)
        check("上级可更换下级岗位（employee → dept_manager）", ok,
              f"status={st} {detail(res)}")
        check("新岗位落库",
              ok and res["member"]["role"] == "dept_manager"
              and res["member"]["job_title"] == "技术总监")
        # 还原，避免影响后续文档权限用例
        call("PATCH", f"/staff/members/{uid}", base=base, token=admin,
             payload={"role": "employee", "job_title": duty_dev})
        st, res = call("PATCH", f"/staff/members/{uid}", base=base, token=admin,
                       payload={"is_active": False})
        deactivated = st == 200 and res.get("member", {}).get("is_active") is False
        check("上级可停用（删除）成员岗位", deactivated, f"status={st} {detail(res)}")
        call("PATCH", f"/staff/members/{uid}", base=base, token=admin,
             payload={"is_active": True})

    # ══ B. 上传与三层标志 ══════════════════════════════════════════════════════
    section("B. 上传与三层知识库标志")

    doc_priv = f"个人私有-{sfx}.txt"
    st, res = upload(base, t_a_emp1, doc_priv,
                     f"甲公司研发部内部研发记录 {sfx}，代号 Alpha-{sfx}，负责人张三。"
                     .encode("utf-8"))
    check("普通员工上传文档成功（202 受理）", st in (200, 202), f"status={st} {detail(res)}")

    doc_dept = f"部门共享-{sfx}.txt"
    st, res = upload(base, t_a_emp1, doc_dept,
                     f"部门级内容 {sfx}".encode("utf-8"), access_level="department")
    check("普通员工直接上传到部门库被拒（403 + 引导申请共享）",
          st == 403 and "申请共享" in detail(res), f"status={st} {detail(res)}")

    doc_company = f"公司共享-{sfx}.txt"
    st, res = upload(base, t_a_emp1, doc_company,
                     f"公司级内容 {sfx}".encode("utf-8"), access_level="tenant")
    check("普通员工直接上传到公司库被拒（403）", st == 403, f"status={st} {detail(res)}")

    st, res = upload(base, t_a_mgr, doc_dept,
                     f"部门级知识 {sfx}：研发部编码规范，接口命名统一 snake_case。"
                     .encode("utf-8"), access_level="department")
    check("部门负责人可直接上传到部门库", st in (200, 202), f"status={st} {detail(res)}")

    st, res = upload(base, t_a_kb, doc_company,
                     f"公司级知识 {sfx}：全公司通用的信息安全管理规范，密码长度不少于 12 位。"
                     .encode("utf-8"), access_level="tenant")
    check("知识库管理员可直接上传到公司库", st in (200, 202), f"status={st} {detail(res)}")

    # 落库校验：三层标志
    # 个人库文档只有归属人（t_a_emp1）能看到，所以用**归属人**的清单校验它；
    # 知识库管理员的清单里应当只有部门库 + 公司库。
    st, res = list_docs(base, t_a_emp1)
    own_items = res.get("items", []) if isinstance(res, dict) else []
    st, res = list_docs(base, t_a_kb)
    items = res.get("items", []) if isinstance(res, dict) else []
    p = find_doc(own_items, doc_priv)
    d = find_doc(items, doc_dept)
    c = find_doc(items, doc_company)
    if check("部门库 / 公司库文档在知识库管理员清单里能查到", all([d, c]),
             f"dept={bool(d)} company={bool(c)}"):
        check("部门文档 access_level=department / 标注「部门」",
              d["access_level"] == "department" and d["access_label"] == "部门",
              f"level={d['access_level']} label={d['access_label']}")
        check("公司文档 access_level=tenant / 标注「公司」",
              c["access_level"] == "tenant" and c["access_label"] == "公司",
              f"level={c['access_level']} label={c['access_label']}")
        check("部门文档带部门归属（department_id 非空）",
              bool(d.get("department_id")), f"department_id={d.get('department_id')}")
        check("公司文档不带部门归属", not c.get("department_id"),
              f"company={c.get('department_id')}")
        check("知识库管理员**看不到**同事的个人库文档",
              find_doc(items, doc_priv) is None,
              "他人个人库泄漏到管理员清单")
    if check("个人文档在归属人自己的清单里能查到", bool(p),
             f"priv={bool(p)}"):
        check("个人文档 access_level=private / 标注「个人」",
              p["access_level"] == "private" and p["access_label"] == "个人",
              f"level={p['access_level']} label={p['access_label']}")
        check("个人文档不带部门归属", not p.get("department_id"),
              f"priv={p.get('department_id')}")

    # access_level 筛选（用归属人身份：个人库只对本人可见）
    st, res = call("GET", "/documents?limit=100&access_level=private", base=base, token=t_a_emp1)
    priv_items = docs_items(res)
    check("归属人按 access_level=private 筛选只返回个人库",
          all(i["access_level"] == "private" for i in priv_items) and len(priv_items) > 0,
          f"n={len(priv_items)}")
    st, res = call("GET", "/documents?limit=100&access_level=private", base=base, token=t_a_kb)
    kb_priv_items = docs_items(res)
    check("知识库管理员按 private 筛选看不到同事的个人库",
          find_doc(kb_priv_items, doc_priv) is None,
          f"n={len(kb_priv_items)}")
    st, res = call("GET", "/documents?limit=100&access_level=tenant", base=base, token=t_a_kb)
    tenant_items = docs_items(res)
    check("按 access_level=tenant 筛选只返回公司库",
          all(i["access_level"] == "tenant" for i in tenant_items) and len(tenant_items) > 0,
          f"n={len(tenant_items)}")

    # ══ C. 申请共享闭环 ════════════════════════════════════════════════════════
    section("C. 申请共享：普通员工申请 → 上级同意 / 拒绝")

    doc_req = f"申请共享-{sfx}.txt"
    upload(base, t_a_emp1, doc_req, f"待共享内容 {sfx}".encode("utf-8"))
    st, res = list_docs(base, t_a_emp1)
    items = res.get("items", []) if isinstance(res, dict) else []
    req_doc = find_doc(items, doc_req)
    req_doc_id = req_doc["document_id"] if req_doc else ""

    if check("申请前拿到目标文档 ID", bool(req_doc_id)):
        st, res = call("POST", "/share-requests", base=base, token=t_a_emp1,
                       payload={"document_id": req_doc_id, "target_level": "department",
                                "reason": f"研发部同事需要参考 {sfx}"})
        ok = st == 201 and isinstance(res, dict)
        check("普通员工提交「发布到部门库」申请", ok, f"status={st} {detail(res)}")
        share_rid = res["request"]["id"] if ok else ""

        st, res = call("POST", "/share-requests", base=base, token=t_a_emp1,
                       payload={"document_id": req_doc_id, "target_level": "department"})
        check("同文档重复申请被拒（409 防刷队列）", st == 409, f"status={st} {detail(res)}")

        st, res = call("POST", "/share-requests", base=base, token=t_a_mgr,
                       payload={"document_id": req_doc_id, "target_level": "tenant"})
        check("非归属人不能替他人申请共享（403，只能由本人提交）", st == 403,
              f"status={st} {detail(res)}")

        if share_rid:
            # 部门负责人：本部门申请应可审
            st, inbox = call("GET", "/share-requests/inbox", base=base, token=t_a_mgr)
            rows = inbox.get("items", []) if isinstance(inbox, dict) else []
            mine = next((r for r in rows if r["id"] == share_rid), None)
            check("本部门负责人在审核队列看到该申请", mine is not None, f"inbox={len(rows)}")
            check("该申请对部门负责人标记为可审核",
                  bool(mine and mine.get("can_review")),
                  f"can_review={mine.get('can_review') if mine else None}")

            # 跨公司审核人不可见
            st, inbox_b = call("GET", "/share-requests/inbox", base=base, token=t_b_kb)
            rows_b = inbox_b.get("items", []) if isinstance(inbox_b, dict) else []
            check("乙公司知识库管理员看不到甲公司的申请（公司隔离）",
                  not any(r["id"] == share_rid for r in rows_b), f"inbox_b={len(rows_b)}")

            # 不能自审
            st, res = call("POST", f"/share-requests/{share_rid}/review",
                           base=base, token=t_a_emp1, payload={"approve": True})
            check("申请人不能审核自己的申请", st in (403, 422), f"status={st} {detail(res)}")

            # 同意 → 自动发布
            st, res = call("POST", f"/share-requests/{share_rid}/review",
                           base=base, token=t_a_mgr,
                           payload={"approve": True, "comment": "同意共享到研发部"})
            ok = st == 200 and isinstance(res, dict)
            check("部门负责人同意申请", ok, f"status={st} {detail(res)}")
            check("批准后自动发布到部门库",
                  ok and res.get("published", {}).get("access_level") == "department",
                  f"published={res.get('published') if ok else ''}")

            st, res = list_docs(base, t_a_emp1)
            items = res.get("items", []) if isinstance(res, dict) else []
            after = find_doc(items, doc_req)
            check("文档层级已升级为部门库",
                  bool(after) and after["access_level"] == "department",
                  f"level={after['access_level'] if after else None}")

            st, res = call("GET", "/share-requests/mine", base=base, token=t_a_emp1)
            rows = res.get("items", []) if isinstance(res, dict) else []
            check("申请人可看到「已通过」结论",
                  any(r["id"] == share_rid and r["status"] == "approved" for r in rows))

    # 拒绝路径
    doc_rej = f"申请被拒-{sfx}.txt"
    upload(base, t_a_emp1, doc_rej, f"会被拒绝的内容 {sfx}".encode("utf-8"))
    st, res = list_docs(base, t_a_emp1)
    items = res.get("items", []) if isinstance(res, dict) else []
    rej_doc = find_doc(items, doc_rej)
    if rej_doc:
        st, res = call("POST", "/share-requests", base=base, token=t_a_emp1,
                       payload={"document_id": rej_doc["document_id"],
                                "target_level": "tenant", "reason": "希望全公司可见"})
        if st == 201:
            rid2 = res["request"]["id"]
            st, res = call("POST", f"/share-requests/{rid2}/review", base=base,
                           token=t_b_kb, payload={"approve": True})
            check("乙公司管理员审核甲公司申请被拒（403，公司隔离）", st == 403,
                  f"status={st} {detail(res)}")
            st, res = call("POST", f"/share-requests/{rid2}/review", base=base,
                           token=t_a_kb,
                           payload={"approve": False, "comment": "内容尚未定稿，暂不发布"})
            check("知识库管理员拒绝该申请", st == 200, f"status={st} {detail(res)}")
            st, res = list_docs(base, t_a_emp1)
            items = res.get("items", []) if isinstance(res, dict) else []
            after = find_doc(items, doc_rej)
            check("被拒后文档仍留在个人库（未被发布）",
                  bool(after) and after["access_level"] == "private",
                  f"level={after['access_level'] if after else None}")

    # 管理员无需申请
    st, res = call("POST", "/share-requests", base=base, token=t_a_mgr,
                   payload={"document_id": req_doc_id, "target_level": "department"})
    check("有直接发布权限的角色申请部门库被提示「无需申请」（409）",
          st in (409, 404), f"status={st} {detail(res)}")

    # ══ D. 删除权限矩阵 ════════════════════════════════════════════════════════
    section("D. 删除权限矩阵（按层级裁定 + 「申请删除」闭环）")

    def list_one(token: str, name: str):
        """某账号视角下的一份文档（含 can_delete / can_request_delete 能力字段）。"""
        _, res = list_docs(base, token)
        return find_doc(docs_items(res), name)

    def new_doc(owner_token: str, name: str, level: str | None = None):
        return upload(base, owner_token, name,
                      f"{name} 内容 {sfx}".encode("utf-8"), access_level=level)[0]

    # ── D1. 个人知识库：只有归属人能删 ───────────────────────────────────────
    n1 = f"删-个人-{sfx}.txt"
    new_doc(t_a_emp1, n1)
    d1 = list_one(t_a_emp1, n1)
    if check("个人库文档已就位", bool(d1)):
        check("归属人对个人库文档 can_delete=True",
              d1.get("can_delete") is True, f"can_delete={d1.get('can_delete')}")
        st, res = call("DELETE", f"/documents/{d1['document_id']}", base=base,
                       token=t_a_emp2)
        check("同公司他人删我的个人文档被拒（403 + 中文原因）", st == 403,
              f"status={st} {detail(res)}")
        st, res = call("DELETE", f"/documents/{d1['document_id']}", base=base,
                       token=t_b_emp)
        check("跨公司删除被拒且不泄漏存在性（404）", st == 404,
              f"status={st} {detail(res)}")
        st, res = call("DELETE", f"/documents/{d1['document_id']}", base=base,
                       token=t_a_emp1)
        check("本人可删自己的个人文档", st in (200, 204), f"status={st} {detail(res)}")

    # ── D2. 部门知识库：他人可申请删除，部门负责人同意后生效 ─────────────────
    n2 = f"删-部门他传-{sfx}.txt"
    new_doc(t_a_mgr, n2, "department")      # 部门负责人直接上传 → owner = a_mgr
    d2 = list_one(t_a_emp1, n2)
    if check("同部门员工看得见部门库文档", bool(d2)):
        check("员工对部门库文档 can_delete=False",
              d2.get("can_delete") is False, f"can_delete={d2.get('can_delete')}")
        check("员工被拒原因指向「申请删除」",
              "申请删除" in str(d2.get("delete_denied_reason", "")),
              f"reason={d2.get('delete_denied_reason')}")
        check("员工对部门库文档 can_request_delete=True",
              d2.get("can_request_delete") is True,
              f"can_request_delete={d2.get('can_request_delete')}")

        st, res = call("DELETE", f"/documents/{d2['document_id']}", base=base,
                       token=t_a_emp1)
        check("员工实际删除部门库他人文档被拒（403）", st == 403,
              f"status={st} {detail(res)}")

        st, res = call("POST", "/share-requests", base=base, token=t_a_emp1,
                       payload={"document_id": d2["document_id"], "intent": "delete",
                                "reason": f"内容已过时，申请删除 {sfx}"})
        ok = st == 201 and isinstance(res, dict)
        check("员工提交「申请删除」成功", ok, f"status={st} {detail(res)}")
        del_rid = res["request"]["id"] if ok else ""
        if ok:
            check("申请意图标记为 delete",
                  res["request"].get("intent") == "delete"
                  and res["request"].get("intent_label") == "申请删除",
                  f"intent={res['request'].get('intent')}")
            check("删除申请的目标层级 = 文档当前层级（部门）",
                  res["request"].get("target_level") == "department")

            st, res = call("POST", f"/share-requests/{del_rid}/review", base=base,
                           token=t_b_kb, payload={"approve": True})
            check("跨公司审核删除申请被拒（403，公司隔离）", st == 403,
                  f"status={st} {detail(res)}")

            st, inbox = call("GET", "/share-requests/inbox", base=base, token=t_a_mgr)
            rows = inbox.get("items", []) if isinstance(inbox, dict) else []
            mine = next((r for r in rows if r["id"] == del_rid), None)
            check("本部门负责人在队列看到删除申请", mine is not None, f"inbox={len(rows)}")

            st, res = call("POST", f"/share-requests/{del_rid}/review", base=base,
                           token=t_a_mgr,
                           payload={"approve": True, "comment": "确认内容已过时，同意删除"})
            ok = st == 200 and isinstance(res, dict)
            check("部门负责人同意删除申请", ok, f"status={st} {detail(res)}")
            check("响应回传删除结果（deleted 非空）",
                  ok and res.get("deleted"), f"deleted={res.get('deleted') if ok else ''}")

            st, res = list_docs(base, t_a_mgr)
            check("批准后文档确实从知识库消失",
                  find_doc(docs_items(res), n2) is None)

            st, res = call("GET", "/share-requests/mine", base=base, token=t_a_emp1)
            rows = res.get("items", []) if isinstance(res, dict) else []
            done = next((r for r in rows if r["id"] == del_rid), None)
            check("申请人看到「已通过」且文档引用已置空（审计记录保留）",
                  bool(done) and done["status"] == "approved"
                  and done["document_id"] is None,
                  f"status={done['status'] if done else None} "
                  f"doc={done.get('document_id') if done else None}")

    # ── D3. 归属人也不能直接删已发布到部门库的文档 ──────────────────────────
    n3 = f"删-部门自传-{sfx}.txt"
    upload(base, t_a_emp1, n3, f"{n3} 内容 {sfx}".encode("utf-8"))
    d3 = list_one(t_a_emp1, n3)
    if d3:
        st, res = call("POST", "/share-requests", base=base, token=t_a_emp1,
                       payload={"document_id": d3["document_id"],
                                "target_level": "department", "reason": "部门需要"})
        if st == 201:
            call("POST", f"/share-requests/{res['request']['id']}/review", base=base,
                 token=t_a_mgr, payload={"approve": True, "comment": "ok"})
        d3 = list_one(t_a_emp1, n3)
        check("文档已升为部门库",
              bool(d3) and d3["access_level"] == "department")
        if d3:
            check("归属人对已发布的部门库文档 can_delete=False（收归上级）",
                  d3.get("can_delete") is False, f"can_delete={d3.get('can_delete')}")
            check("归属人可提交「申请删除」",
                  d3.get("can_request_delete") is True,
                  f"can_request_delete={d3.get('can_request_delete')}")
            st, res = call("DELETE", f"/documents/{d3['document_id']}", base=base,
                           token=t_a_emp1)
            check("归属人实际删除被拒（403）", st == 403,
                  f"status={st} {detail(res)}")
            d3_mgr = list_one(t_a_mgr, n3)
            check("部门负责人对同部门文档 can_delete=True",
                  bool(d3_mgr) and d3_mgr.get("can_delete") is True)
            st, res = call("DELETE", f"/documents/{d3['document_id']}", base=base,
                           token=t_a_mgr)
            check("部门负责人可删本部门范围内的文档", st in (200, 204),
                  f"status={st} {detail(res)}")

    # ── D4. 公司知识库：知识库管理员可删；其他人走申请 ──────────────────────
    n4 = f"删-公司-{sfx}.txt"
    new_doc(t_a_kb, n4, "tenant")
    d4 = list_one(t_a_kb, n4)
    if check("公司库文档已就位", bool(d4)):
        check("知识库管理员对公司库文档 can_delete=True",
              d4.get("can_delete") is True, f"can_delete={d4.get('can_delete')}")
        d4_mgr = list_one(t_a_mgr, n4)
        check("部门负责人对公司库文档 can_delete=False",
              bool(d4_mgr) and d4_mgr.get("can_delete") is False,
              f"can_delete={d4_mgr.get('can_delete') if d4_mgr else None}")
        check("部门负责人对公司库文档 can_request_delete=True",
              bool(d4_mgr) and d4_mgr.get("can_request_delete") is True)
        st, res = call("DELETE", f"/documents/{d4['document_id']}", base=base,
                       token=t_a_mgr)
        check("部门负责人删公司库文档被拒（403）", st == 403,
              f"status={st} {detail(res)}")

        # 拒绝路径：申请了但被驳回 → 文档必须还在
        st, res = call("POST", "/share-requests", base=base, token=t_a_mgr,
                       payload={"document_id": d4["document_id"], "intent": "delete",
                                "reason": "希望删除"})
        if check("部门负责人可提交针对公司库文档的删除申请", st == 201,
                 f"status={st} {detail(res)}"):
            rid = res["request"]["id"]
            st, res = call("POST", f"/share-requests/{rid}/review", base=base,
                           token=t_a_kb,
                           payload={"approve": False, "comment": "该文档仍在使用，暂不删除"})
            check("知识库管理员拒绝删除申请", st == 200, f"status={st} {detail(res)}")
            check("被拒后文档仍然存在（未被删除）",
                  list_one(t_a_kb, n4) is not None)

        st, res = call("DELETE", f"/documents/{d4['document_id']}", base=base,
                       token=t_a_kb)
        check("知识库管理员可删公司库文档", st in (200, 204),
              f"status={st} {detail(res)}")

    # ── D5. 能直接删的人不该被引导去申请 ────────────────────────────────────
    n5 = f"删-无需申请-{sfx}.txt"
    new_doc(t_a_kb, n5)
    d5 = list_one(t_a_kb, n5)
    if d5:
        check("有删除权时 can_request_delete=False（按钮不会重复出现）",
              d5.get("can_request_delete") is False,
              f"can_request_delete={d5.get('can_request_delete')}")
        st, res = call("POST", "/share-requests", base=base, token=t_a_kb,
                       payload={"document_id": d5["document_id"], "intent": "delete"})
        check("有删除权的人提交删除申请被提示「可直接删除」（409）", st == 409,
              f"status={st} {detail(res)}")
        call("DELETE", f"/documents/{d5['document_id']}", base=base, token=t_a_kb)

    # ══ E. 隔离性 ══════════════════════════════════════════════════════════════
    section("E. 隔离性：公司 / 部门 / 个人")

    iso_priv = f"隔离-甲研个人-{sfx}.txt"
    iso_dept = f"隔离-甲研部门-{sfx}.txt"
    iso_comp = f"隔离-甲公司-{sfx}.txt"
    iso_mkt = f"隔离-甲市场-{sfx}.txt"
    iso_b = f"隔离-乙公司-{sfx}.txt"

    upload(base, t_a_emp1, iso_priv, f"甲公司研发部员工个人 {sfx}".encode("utf-8"))
    upload(base, t_a_mgr, iso_dept, f"甲公司研发部部门级 {sfx}".encode("utf-8"),
           access_level="department")
    upload(base, t_a_kb, iso_comp, f"甲公司公司级 {sfx}".encode("utf-8"),
           access_level="tenant")
    upload(base, t_a_emp2, iso_mkt, f"甲公司市场部员工个人 {sfx}".encode("utf-8"))
    upload(base, t_b_kb, iso_b, f"乙公司公司级 {sfx}".encode("utf-8"),
           access_level="tenant")

    # 说明：向量层的检索隔离（跨公司 / 跨部门 / 个人库不外流）由
    # scripts/e2e_recall_refusal.py 用真实提问验证 —— 那里需要等入库完成。

    # 可见性：甲研员工应看到 自己 + 本部门 + 公司；看不到市场部 / 乙公司
    st, res = list_docs(base, t_a_emp1)
    items = res.get("items", []) if isinstance(res, dict) else []
    names = {i["filename"] for i in items}
    check("甲研员工可见自己的个人文档", iso_priv in names)
    check("甲研员工可见本部门文档", iso_dept in names)
    check("甲研员工可见本公司文档", iso_comp in names)
    check("甲研员工**看不到**同公司其他部门他人的个人文档", iso_mkt not in names)
    check("甲研员工**看不到**乙公司文档", iso_b not in names)

    # 市场部员工：看不到研发部的部门文档
    st, res = list_docs(base, t_a_emp2)
    items = res.get("items", []) if isinstance(res, dict) else []
    names_mkt = {i["filename"] for i in items}
    check("市场部员工看不到研发部的部门文档（部门隔离）", iso_dept not in names_mkt,
          f"visible={sorted(n for n in names_mkt if n.startswith('隔离'))}")
    check("市场部员工可见本公司文档", iso_comp in names_mkt)

    # 乙公司：只看得到自己的
    st, res = list_docs(base, t_b_emp)
    items = res.get("items", []) if isinstance(res, dict) else []
    names_b = {i["filename"] for i in items}
    check("乙公司员工只看得到本公司文档", iso_b in names_b)
    check("乙公司员工看不到甲公司的任何隔离文档",
          not ({iso_priv, iso_dept, iso_comp, iso_mkt} & names_b),
          f"leaked={sorted(names_b & {iso_priv, iso_dept, iso_comp, iso_mkt})}")

    # 知识库管理员：同公司所有**部门库 + 公司库**可见（跨部门宽口径），
    # 但**他人个人库不可见**（个人库对任何人私密）。
    st, res = list_docs(base, t_a_kb)
    items = res.get("items", []) if isinstance(res, dict) else []
    names_kb = {i["filename"] for i in items}
    check("知识库管理员可见本公司全部部门库与公司库文档",
          {iso_dept, iso_comp} <= names_kb,
          f"missing={sorted({iso_dept, iso_comp} - names_kb)}")
    check("知识库管理员**看不到**他人的个人库文档",
          iso_priv not in names_kb and iso_mkt not in names_kb,
          f"leaked={sorted(names_kb & {iso_priv, iso_mkt})}")
    check("知识库管理员仍看不到其他公司文档", iso_b not in names_kb)

    # 平台管理员：跨公司可见所有公司的部门库 / 公司库，但同样看不到他人个人库
    st, res = list_docs(base, admin)
    items = res.get("items", []) if isinstance(res, dict) else []
    names_admin = {i["filename"] for i in items}
    check("平台管理员跨公司可见各公司的公司库文档",
          iso_b in names_admin and iso_comp in names_admin)
    check("平台管理员跨公司可见部门库文档", iso_dept in names_admin)
    check("平台管理员**看不到**他人的个人库文档",
          iso_priv not in names_admin and iso_mkt not in names_admin,
          f"leaked={sorted(names_admin & {iso_priv, iso_mkt})}")

    # 直接访问他人文档（越权取详情）
    st, res = list_docs(base, t_a_kb)
    items = res.get("items", []) if isinstance(res, dict) else []
    priv_doc = find_doc(items, iso_priv)
    if priv_doc is None:
        # 个人库对知识库管理员已不可见 → 改用「公司内他人个人库」路径验证：
        # 直接拿归属人自己的列表找出该文档 id
        st, res = list_docs(base, t_a_emp1)
        items = res.get("items", []) if isinstance(res, dict) else []
        priv_doc = find_doc(items, iso_priv)
    if priv_doc:
        st, res = call("GET", f"/documents/{priv_doc['document_id']}/chunks",
                       base=base, token=t_b_emp)
        check("乙公司用户读取甲公司文档详情被拒（404/403，不泄漏存在性）",
              st in (403, 404), f"status={st} {detail(res)}")

        # 平台管理员：跨公司能读公司库，但读不了别人的个人库
        st, res = call("GET", f"/documents/{priv_doc['document_id']}/chunks",
                       base=base, token=admin)
        check("平台管理员读取他人个人库文档被拒（404）", st in (403, 404),
              f"status={st} {detail(res)}")

    # ══ F. 文档显示统计 ════════════════════════════════════════════════════════
    section("F. 文档显示：个人 = 本人 + 本部门 + 公司")
    st, res = list_docs(base, t_a_emp1)
    items = res.get("items", []) if isinstance(res, dict) else []
    levels = {i["access_level"] for i in items}
    check("甲研员工列表同时包含 个人 / 部门 / 公司 三类文档",
          {"private", "department", "tenant"} <= levels, f"levels={sorted(levels)}")
    check("列表中每条都带中文标注 access_label",
          all(i.get("access_label") in ("个人", "部门", "公司") for i in items))
    check("列表中每条都带能力字段（can_delete / needs_share_request）",
          all("can_delete" in i and "needs_share_request" in i for i in items))

    # ══ 结果 ═══════════════════════════════════════════════════════════════════
    if args.keep:
        print(f"\n  （--keep）保留临时数据：公司={comp_a}/{comp_b}")
    else:
        print(f"\n  清理：{cleanup(prefixes, [comp_a, comp_b])}")

    print(f"\n{'═' * 74}")
    print(f"  结果：{_passed} 通过 / {_failed} 失败")
    print(f"{'═' * 74}")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
