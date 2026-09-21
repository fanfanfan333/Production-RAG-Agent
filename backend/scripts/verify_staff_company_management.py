"""「公司管理」面板升级的在线验收脚本（QA 固化版）.

覆盖诉求：「在管理公司界面，怎么看不到之前的普通公司」→ 修复后
``GET /staff/companies`` 对平台管理员返回**全部已注册公司**，并新增
``can_rename``（平台管理员可给「自建」或「无主（created_by IS NULL）」公司改名）。

设计原则
────────
* **默认只读**（不写任何数据，可重复跑）：走**真实 HTTP 端点**核对公司清单、
  改名准入（403/404/409/400/422 边界）、``/companies/accessible`` 候选、
  以及 P0-6 文档隔离未击穿。
* ``--write`` 额外执行一次**自清理**的「无主合成公司」改名往返：临时插入
  合成公司 + 合成成员 + 合成文档 → 改名 → 断言 ``created_by`` 仍为 ``None``
  且成员/文档归属零变化 → 复原并删除。**绝不触碰 A公司 / B公司 / 测试公司
  的持久展示名**。
* 退出码即结论：``0`` = 全部通过；``1`` = 存在失败项。

容器内运行姿势（重要，照做否则静默失败）
────────────────────────────────────────
镜像的 runtime 阶段只 ``COPY app/`` ``alembic/`` —— **不含 ``scripts/``**，
所以文档里 ``python -m scripts.xxx`` 那套在容器内根本跑不通。正确姿势：

    docker cp backend/scripts/verify_staff_company_management.py rag_backend:/tmp/
    docker exec -w /app -e PYTHONPATH=/app rag_backend \\
        python /tmp/verify_staff_company_management.py           # 只读
    # 需要跑改名往返时再加 --write

两个踩过的坑：
  1. ``docker cp`` 直接覆盖容器内文件会因权限不足**静默留 0 字节**。可靠写法：
     ``docker cp <本地> rag_backend:/tmp/x.py`` → ``docker exec -u root rag_backend
     cp /tmp/x.py /app/<目标>`` → 用 ``md5sum`` 与本地 ``Get-FileHash`` 对齐。
     本脚本只落到 ``/tmp`` 直接以路径运行，不覆盖 ``/app``，规避该坑。
  2. 跑测试时**不要**加 ``-e HOME=/tmp``。那会把 user-site 切到 ``/tmp/.local``，
     于是装在系统/``/app/.cache`` 的包（含 pytest）全部变成 "No module named ..."。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

BASE = "http://127.0.0.1:8000"
HTTP_TIMEOUT = 30

# 数据库真实现状（本次验收的基准）。
TEST_COMPANY_1 = "c8111de986583"
TEST_COMPANY_2 = "cfb08c53677c4"
COMPANY_A = "c309a7cb9f496"  # A公司（created_by = NULL，无主历史公司）
COMPANY_B = "cf33b1db5679d"  # B公司（created_by = NULL）
OWNERLESS_COMPANIES = {COMPANY_A, COMPANY_B}
TEST_COMPANIES = {TEST_COMPANY_1, TEST_COMPANY_2}
KNOWN_COMPANIES = TEST_COMPANIES | OWNERLESS_COMPANIES
NON_ADMIN_USERNAME = "tr.kb.bjld8@example.com"  # kb_admin，属测试公司1

_failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)
    return ok


# ── HTTP 工具 ─────────────────────────────────────────────────────────────────


def _http(method: str, path: str, token: str, payload: dict | None = None) -> tuple[int, object]:
    """发一次请求，返回 (status, parsed_json_or_text)。"""
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


def _items(body: object) -> list[dict]:
    if isinstance(body, dict) and isinstance(body.get("items"), list):
        return [r for r in body["items"] if isinstance(r, dict)]
    return []


# ── 主流程 ────────────────────────────────────────────────────────────────────


async def _run(with_write: bool) -> int:
    from sqlalchemy import func, select

    from app.db.company_models import Company
    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.db.user_models import User
    from app.services.auth_service import create_access_token
    from app.services.company_registry import (
        CompanyError,
        assert_admin_owns,
        can_platform_admin_rename,
        list_registered_companies,
        tenant_ids_created_by,
    )
    from app.services.tenancy import (
        DEFAULT_TENANT_ID,
        effective_tenant_id,
        request_scope,
    )

    # ── 取真实账号 & 签发真 JWT ────────────────────────────────────────────
    async with get_db_session() as session:
        admin = await session.scalar(
            select(User).where(User.role == User.ROLE_ADMIN).limit(1)
        )
        non_admin = await session.scalar(
            select(User).where(User.username == NON_ADMIN_USERNAME).limit(1)
        )
        if non_admin is None:
            non_admin = await session.scalar(
                select(User).where(User.role != User.ROLE_ADMIN).limit(1)
            )
    if admin is None:
        print("FATAL：找不到平台管理员账号")
        return 1
    admin_token = create_access_token(admin)
    print(f"── 账号 ─ admin={admin.username}({admin.id}) "
          f"非管理员={getattr(non_admin, 'username', None)} "
          f"tenant={getattr(non_admin, 'tenant_id', None)}")

    # ── 基线快照（用于收尾复验）────────────────────────────────────────────
    async def snapshot() -> dict:
        async with get_db_session() as session:
            comps = {
                c.tenant_id: {
                    "display_name": c.display_name,
                    "created_by": (str(c.created_by) if c.created_by else None),
                    "is_test": bool(c.is_test),
                }
                for c in (await session.execute(select(Company))).scalars().all()
            }
            ucount = {
                t: int(n)
                for t, n in (
                    await session.execute(
                        select(User.tenant_id, func.count()).group_by(User.tenant_id)
                    )
                ).all()
            }
            dcount = {
                t: int(n)
                for t, n in (
                    await session.execute(
                        select(Document.tenant_id, func.count()).group_by(Document.tenant_id)
                    )
                ).all()
            }
        return {"companies": comps, "user_counts": ucount, "doc_counts": dcount}

    baseline = await snapshot()

    # ── P1：admin 走真实 HTTP 读 /staff/companies ──────────────────────────
    print("── P1  GET /staff/companies（admin，真 JWT）──")
    status, body = _http("GET", "/staff/companies", admin_token)
    rows = _items(body)
    by_id = {str(r.get("company_id")): r for r in rows}
    check(status == 200, "HTTP 200", f"status={status}")
    check(len(rows) == 4, "返回 4 家公司", f"actual={len(rows)} ids={sorted(by_id)}")
    check(
        set(by_id) == KNOWN_COMPANIES,
        "4 家 = 测试公司1/2 + A公司/B公司",
        f"ids={sorted(by_id)}",
    )
    expected_rows = {
        TEST_COMPANY_1: ("测试公司1", True, 3, True),
        TEST_COMPANY_2: ("测试公司2", True, 3, True),
        COMPANY_A: ("A公司", False, 2, True),
        COMPANY_B: ("B公司", False, 2, True),
    }
    for tid, (name, is_test, members, can_rename) in expected_rows.items():
        row = by_id.get(tid, {})
        check(
            row.get("company_name") == name,
            f"{tid} company_name == {name!r}",
            f"actual={row.get('company_name')!r}",
        )
        check(row.get("is_test") is is_test, f"{tid} is_test == {is_test}", f"actual={row.get('is_test')!r}")
        check(
            row.get("member_count") == members,
            f"{tid} member_count == {members}",
            f"actual={row.get('member_count')!r}",
        )
        check(
            "can_rename" in row,
            f"{tid} HTTP 响应含 can_rename 字段（未被 response_model 吞掉）",
            f"keys={sorted(row)}",
        )
        check(
            row.get("can_rename") is can_rename,
            f"{tid} can_rename == {can_rename}",
            f"actual={row.get('can_rename')!r}",
        )

    # ── P2：非管理员走真实 HTTP 读 /staff/companies ────────────────────────
    print("── P2  GET /staff/companies（非管理员 kb_admin）──")
    if non_admin is None:
        check(False, "存在可用于验证的非管理员账号")
    else:
        non_token = create_access_token(non_admin)
        status, body = _http("GET", "/staff/companies", non_token)
        rows = _items(body)
        own = effective_tenant_id(non_admin)
        check(status == 200, "HTTP 200", f"status={status}")
        check(len(rows) == 1, "只返回本公司 1 家", f"actual={len(rows)} rows={rows}")
        check(
            rows and str(rows[0].get("company_id")) == own,
            f"该公司 == 其所属租户 {own}",
            f"actual={[r.get('company_id') for r in rows]}",
        )
        check(
            rows and rows[0].get("can_rename") is False,
            "非管理员 can_rename == False",
            f"actual={rows[0].get('can_rename') if rows else None!r}",
        )

    # ── P3：/companies/accessible（admin）候选不泄漏 A/B ────────────────────
    print("── P3  GET /companies/accessible（admin）──")
    from app.services.document_query_service import count_by_tenant

    scope_p3 = await request_scope(admin)
    status, body = _http("GET", "/companies/accessible", admin_token)
    cand = body if isinstance(body, list) else []
    cand_ids = {str(c.get("company_id")) for c in cand if isinstance(c, dict)}
    cand_names = {str(c.get("display_name")) for c in cand if isinstance(c, dict)}
    print(
        "  [info] 候选明细 "
        + str([(c.get("company_id"), c.get("display_name"), c.get("doc_count")) for c in cand if isinstance(c, dict)])
    )
    check(status == 200, "HTTP 200", f"status={status}")
    check(
        DEFAULT_TENANT_ID in cand_ids,
        "候选含 default 租户（显示为「管理员」）",
        f"ids={sorted(cand_ids)}",
    )
    check("管理员" in cand_names, "default 显示名 == 「管理员」", f"names={sorted(cand_names)}")
    check(
        "default" not in cand_names,
        "不出现裸 'default' 展示名",
        f"names={sorted(cand_names)}",
    )
    check(
        not (cand_ids & OWNERLESS_COMPANIES),
        "A/B 公司**不得**出现在可见候选里（P0-6）",
        f"ids={sorted(cand_ids)}",
    )
    check(
        cand_ids == ({DEFAULT_TENANT_ID} | TEST_COMPANIES),
        "候选恰为 default + 测试公司1/2",
        f"ids={sorted(cand_ids)}",
    )
    # doc_count 与 count_by_tenant 同源（同一 document_scope_clause 入口）——
    # 对 admin 而言本次改动只换展示名，不改变 tenant_ids/计数口径。
    counts = await count_by_tenant(
        owner_id=scope_p3.owner_id,
        tenant_ids=scope_p3.tenant_ids,
        owns_tenant_ids=scope_p3.owns_tenant_ids,
        department_id=scope_p3.department_id,
        tenant_wide=scope_p3.tenant_wide,
    )
    for c in cand:
        if not isinstance(c, dict):
            continue
        tid = str(c.get("company_id"))
        check(
            c.get("doc_count") == counts.get(tid, 0),
            f"{tid} doc_count 与 count_by_tenant 同源",
            f"http={c.get('doc_count')} recomp={counts.get(tid, 0)}",
        )

    # ── P4：P0-6 文档隔离专项（服务层口径，独立复算）────────────────────────
    print("── P4  P0-6 文档隔离（request_scope / tenant_ids_created_by）──")
    scope = await request_scope(admin)
    owned = set(scope.owns_tenant_ids)
    scoped = set(scope.tenant_ids)
    created = set(await tenant_ids_created_by(admin.id))
    check(
        owned == TEST_COMPANIES,
        "owns_tenant_ids 恰为 2 家测试公司",
        f"actual={sorted(owned)}",
    )
    check(
        created == TEST_COMPANIES,
        "tenant_ids_created_by(admin.id) 恰为 2 家（反证：未被改动放大）",
        f"actual={sorted(created)}",
    )
    check(
        not (scoped & OWNERLESS_COMPANIES),
        "scope.tenant_ids 未混入 A/B 公司",
        f"actual={sorted(scoped)}",
    )
    check(
        owned == created,
        "scope.owns_tenant_ids == tenant_ids_created_by（同一来源）",
        f"owned={sorted(owned)} created={sorted(created)}",
    )

    # ── P5：边界 / 反向用例 ────────────────────────────────────────────────
    print("── P5  边界与反向用例 ──")
    # 5a. can_platform_admin_rename 对不存在的 tenant 返回 False（不抛异常）
    try:
        ghost = await can_platform_admin_rename(admin, "cdeadbeef000")
        check(ghost is False, "can_platform_admin_rename(不存在 tenant) == False（不抛异常）", f"actual={ghost!r}")
    except Exception as exc:  # noqa: BLE001
        check(False, "can_platform_admin_rename(不存在 tenant) 应返回 False", f"{type(exc).__name__}: {exc}")

    # 5b. assert_admin_owns 仍保持「严格自建」语义（不应被顺手放宽）
    try:
        await assert_admin_owns(admin, COMPANY_A)
        check(False, "assert_admin_owns(A公司) 应拒绝", "未抛异常（被放宽了！）")
    except CompanyError as exc:
        check(exc.status_code == 403, "assert_admin_owns(A公司) → 403（严格自建语义未放宽）", f"status={exc.status_code}")
    except Exception as exc:  # noqa: BLE001
        check(False, "assert_admin_owns(A公司) 应抛 CompanyError", f"{type(exc).__name__}: {exc}")

    # 5c. HTTP PATCH 不存在的公司 → 403（无主闸先于存在性判定）
    st, bd = _http("PATCH", "/companies/cdeadbeef000", admin_token, {"display_name": "QA幽灵公司"})
    check(st in (403, 404), "PATCH 不存在公司 → 403/404（拒绝）", f"status={st} body={str(bd)[:80]}")

    # 5d. HTTP PATCH 重名 → 409
    st, bd = _http("PATCH", f"/companies/{COMPANY_A}", admin_token, {"display_name": "测试公司1"})
    check(st == 409, "PATCH A公司 → 已存在名「测试公司1」→ 409", f"status={st} body={str(bd)[:80]}")
    # 5e. 空格变体重名仍 409
    st, bd = _http("PATCH", f"/companies/{COMPANY_A}", admin_token, {"display_name": "  测试公司1  "})
    check(st == 409, "PATCH A公司 → 空格变体「  测试公司1  」→ 409", f"status={st} body={str(bd)[:80]}")

    # 5f. 纯空白 → 400（服务层 _clean_name）；空串 / 超长 → 422（Pydantic schema 闸）
    st, bd = _http("PATCH", f"/companies/{COMPANY_A}", admin_token, {"display_name": "   "})
    check(st == 400, "PATCH A公司 → 纯空白 → 400", f"status={st} body={str(bd)[:80]}")
    st, bd = _http("PATCH", f"/companies/{COMPANY_A}", admin_token, {"display_name": ""})
    check(st == 422, "PATCH A公司 → 空串 → 422（schema min_length）", f"status={st}")
    st, bd = _http("PATCH", f"/companies/{COMPANY_A}", admin_token, {"display_name": "x" * 129})
    check(st == 422, "PATCH A公司 → 超长(129) → 422（schema max_length）", f"status={st}")

    # 5g. 非管理员 HTTP PATCH → 403（require_admin 前置）
    if non_admin is not None:
        st, bd = _http("PATCH", f"/companies/{COMPANY_A}", create_access_token(non_admin), {"display_name": "越权改"})
        check(st == 403, "非管理员 PATCH → 403", f"status={st} body={str(bd)[:80]}")

    # 关键：以上 5c-5f 均对 A公司 发出，但都应在**改名前**被拦下 —— 复核 A公司 名未变
    async with get_db_session() as session:
        a_row = await session.get(Company, COMPANY_A)
    check(a_row is not None and a_row.display_name == "A公司", "A公司 展示名未被边界用例污染", f"actual={a_row.display_name if a_row else None!r}")

    # ── P6（可选 --write）：无主合成公司改名往返 ───────────────────────────
    synthetic_summary = "（未执行，默认只读；加 --write 启用）"
    if with_write:
        import uuid as _uuid

        synthetic_summary = await _ownerless_rename_roundtrip(
            admin, admin_token, get_db_session, Company, Document, User
        )

    # ── P7：收尾复验 ──────────────────────────────────────────────────────
    print("── P7  收尾复验（基线未被污染）──")
    after = await snapshot()
    check(
        set(after["companies"]) == KNOWN_COMPANIES,
        "companies 表仍为原 4 行",
        f"actual={sorted(after['companies'])}",
    )
    check(
        after["companies"].get(COMPANY_A, {}).get("created_by") is None
        and after["companies"].get(COMPANY_B, {}).get("created_by") is None,
        "A/B 的 created_by 仍为 NULL",
        f"A={after['companies'].get(COMPANY_A, {}).get('created_by')} "
        f"B={after['companies'].get(COMPANY_B, {}).get('created_by')}",
    )
    check(
        after["companies"].get(COMPANY_A, {}).get("display_name") == "A公司"
        and after["companies"].get(COMPANY_B, {}).get("display_name") == "B公司",
        "A/B 展示名未被改动",
        f"A={after['companies'].get(COMPANY_A, {}).get('display_name')} "
        f"B={after['companies'].get(COMPANY_B, {}).get('display_name')}",
    )
    check(
        after["user_counts"] == baseline["user_counts"],
        "users 各租户计数与基线一致",
        f"baseline={baseline['user_counts']} after={after['user_counts']}",
    )
    check(
        after["doc_counts"] == baseline["doc_counts"],
        "documents 各租户计数与基线一致",
        f"baseline={baseline['doc_counts']} after={after['doc_counts']}",
    )
    check(
        after["companies"] == baseline["companies"],
        "companies 全表与基线逐字段一致",
    )
    print(f"  [info] 改名往返：{synthetic_summary}")

    print()
    if _failures:
        print(f"共 {len(_failures)} 项失败：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print("全部断言通过。")
    return 0


async def _ownerless_rename_roundtrip(admin, admin_token, get_db_session, Company, Document, User) -> str:
    """临时插入无主合成公司 + 成员 + 文档 → 改名 → 断言零副作用 → 清理。"""
    import uuid as _uuid

    from sqlalchemy import func, select

    from app.services.company_registry import normalize_company_name_key

    tid = "c" + _uuid.uuid4().hex[:12]
    orig_name = "QA无主合成公司"
    new_name = "QA无主合成公司-已改名"
    uname = f"qa-synthetic-{_uuid.uuid4().hex[:8]}"
    uid = _uuid.uuid4()

    print("── P6  无主合成公司改名往返（--write，自清理）──")
    try:
        async with get_db_session() as session:
            session.add(
                User(
                    id=uid,
                    username=uname,
                    password_hash=None,
                    display_name="QA 合成成员",
                    role=User.ROLE_EMPLOYEE,
                    tenant_id=tid,
                    company_name=orig_name,
                )
            )
            # 先落 User 再落 Document：Document.owner_id 有 FK，显式 flush 保证顺序
            # （不依赖 SQLAlchemy 的隐式排序，避免 FK 违例）。
            await session.flush()
            session.add(
                Company(
                    tenant_id=tid,
                    display_name=orig_name,
                    name_key=normalize_company_name_key(orig_name),
                    created_by=None,
                    is_test=False,
                )
            )
            session.add(
                Document(
                    filename="qa-synthetic.pdf",
                    file_size=1,
                    file_hash=_uuid.uuid4().hex,
                    tenant_id=tid,
                    access_level="tenant",
                    owner_id=uid,
                )
            )
            await session.flush()

        # 改名走真实 HTTP 端点
        st, bd = _http("PATCH", f"/companies/{tid}", admin_token, {"display_name": new_name})
        check(st == 200, "无主合成公司改名 → HTTP 200", f"status={st} body={str(bd)[:120]}")
        check(
            isinstance(bd, dict) and bd.get("company_name") == new_name,
            "响应 company_name == 新名",
            f"actual={bd.get('company_name') if isinstance(bd, dict) else bd!r}",
        )

        async with get_db_session() as session:
            comp = await session.get(Company, tid)
            user = await session.get(User, uid)
            doc = await session.scalar(select(Document).where(Document.tenant_id == tid).limit(1))

        check(comp is not None and comp.display_name == new_name, "companies.display_name == 新名", f"actual={comp.display_name if comp else None!r}")
        check(comp is not None and comp.created_by is None, "改名后 created_by **仍为 None**（未回填 → 不击穿 P0-6）", f"actual={comp.created_by if comp else None!r}")
        check(comp is not None and comp.tenant_id == tid, "tenant_id 不变")
        check(user is not None and user.tenant_id == tid, "成员 tenant_id 不变", f"actual={user.tenant_id if user else None!r}")
        check(user is not None and user.company_name == new_name, "成员 company_name 已同步为新名（展示副本）", f"actual={user.company_name if user else None!r}")
        check(doc is not None and doc.tenant_id == tid, "文档 tenant_id 归属不变", f"actual={doc.tenant_id if doc else None!r}")
        check(doc is not None and doc.owner_id == uid, "文档 owner_id 归属不变")

        # 反证：改这家无主公司**不会**让它进入 admin 的 owns（created_by 未回填）
        from app.services.company_registry import tenant_ids_created_by

        owned_now = set(await tenant_ids_created_by(admin.id))
        check(tid not in owned_now, "改名后该公司未进入 admin.owns_tenant_ids（P0-6 反证）", f"owns={sorted(owned_now)}")
    finally:
        # 清理：document → user → company（顺序满足 FK）
        async with get_db_session() as session:
            for row in (await session.execute(select(Document).where(Document.tenant_id == tid))).scalars().all():
                await session.delete(row)
            u = await session.get(User, uid)
            if u is not None:
                await session.delete(u)
            c = await session.get(Company, tid)
            if c is not None:
                await session.delete(c)

    return f"已插入并清理合成租户 {tid}（原名→新名→删除）"


def main() -> int:
    parser = argparse.ArgumentParser(description="公司管理面板升级验收脚本")
    parser.add_argument(
        "--write",
        action="store_true",
        help="额外执行（自清理的）无主合成公司改名往返；默认只读",
    )
    args = parser.parse_args()
    return asyncio.run(_run(with_write=args.write))


if __name__ == "__main__":
    sys.exit(main())
