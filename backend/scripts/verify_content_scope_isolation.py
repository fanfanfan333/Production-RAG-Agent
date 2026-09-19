"""内容消费隔离验收（#13）—— 摘要 / 文档关联 / 对话内文档列表 = content_scope.

只读，可重复运行。退出码：0 = 全部断言通过；1 = 存在失败项。

验收目标（用户裁决）
────────────────────
把「admin 排除测试公司」从**检索**扩展到**内容消费**全链路，并收敛为**唯一实现点**
``tenancy.content_scope(user)``（= ``request_scope`` 后再按
``tenancy.exclude_test_tenants`` 剔除测试公司，仅平台管理员生效）。

规则（与检索口径一致）
  1. 测试公司成员（测试账号）：一切照旧 —— 本公司内容可见、可总结、可关联。
  2. 平台管理员 admin：**列表/管理**仍可见测试公司文档（走 request_scope），
     但**内容消费**（摘要 collect_document_digests / 关联 / 对话内文档列表
     _list_completed_documents）**不得命中**测试公司文档（走 content_scope）。
  3. A公司 / B公司：行为完全不变。
  4. admin 本人文档（含 default 私库）：content_scope 下**保留**。

本脚本验证**语义与派生点**（函数级，直接消费真实 DB）。端到端 SSE 见
``probe_summary_scope_isolation.py``。

用法（容器内）
──────────────
    docker cp scripts/verify_content_scope_isolation.py rag_backend:/app/scripts/
    docker exec -w /app rag_backend python -m scripts.verify_content_scope_isolation
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

TEST_COMPANY_1 = "c8111de986583"      # 测试公司1
TEST_COMPANY_2 = "cfb08c53677c4"      # 测试公司2
A_COMPANY = "c309a7cb9f496"
B_COMPANY = "cf33b1db5679d"
TEST_TENANTS = {TEST_COMPANY_1, TEST_COMPANY_2}

ADMIN_USERNAME = "admin"
A_USER_USERNAME = "lisi@a-company.com"
TEST_MEMBER_USERNAME = "tr.kb.bjld8@example.com"   # 测试公司1 的 kb_admin

_failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)


def _scopes_equal(a, b) -> bool:
    """DocumentScope 逐字段相等（非 admin 的 content_scope 必须与 request_scope 一致）."""
    return (
        a.owner_id == b.owner_id
        and a.tenant_ids == b.tenant_ids
        and a.owns_tenant_ids == b.owns_tenant_ids
        and a.department_id == b.department_id
        and a.tenant_wide == b.tenant_wide
    )


def _tenants_of(doc_ids, id2tenant) -> set[str]:
    return {id2tenant[d] for d in doc_ids if d in id2tenant}


async def _run() -> int:
    from sqlalchemy import select

    from app.db.company_models import Company
    from app.db.models import Document, DocumentStatus
    from app.db.postgres import get_db_session
    from app.db.user_models import User
    from app.services.company_registry import tenant_ids_created_by, test_tenant_ids
    from app.services.relation_service import (
        collect_document_digests,
        list_accessible_documents,
    )
    from app.services.tenancy import content_scope, effective_tenant_id, request_scope

    # ── 载入数据快照 ───────────────────────────────────────────────────────────
    async with get_db_session() as session:
        users = {
            u.username: u
            for u in (await session.execute(select(User))).scalars().all()
        }
        docs = (
            await session.execute(
                select(Document).where(Document.status == DocumentStatus.COMPLETED)
            )
        ).scalars().all()
        comps = (await session.execute(select(Company))).scalars().all()

    id2tenant = {str(d.id): d.tenant_id for d in docs}
    admin = users.get(ADMIN_USERNAME)
    a_user = users.get(A_USER_USERNAME)
    test_member = users.get(TEST_MEMBER_USERNAME)
    for label, u in (
        (ADMIN_USERNAME, admin), (A_USER_USERNAME, a_user),
        (TEST_MEMBER_USERNAME, test_member),
    ):
        if u is None:
            print(f"FATAL：找不到账号 {label}")
            return 1

    admin_own_doc_ids = {
        str(d.id) for d in docs
        if d.tenant_id == "default" and d.access_level == "private"
    }
    test_doc_ids = {str(d.id) for d in docs if d.tenant_id in TEST_TENANTS}
    a_doc_ids = {str(d.id) for d in docs if d.tenant_id == A_COMPANY}

    print(f"快照：completed docs={len(docs)}；测试公司文档={len(test_doc_ids)}；"
          f"admin 私库文档={len(admin_own_doc_ids)}；A公司文档={len(a_doc_ids)}")
    print(f"公司表：{[ (getattr(c,'tenant_id',None), getattr(c,'is_test',None), str(getattr(c,'created_by',None))[:8]) for c in comps ]}")

    # ── 0. 唯一来源：test_tenant_ids ───────────────────────────────────────────
    print("── 0. test_tenant_ids（唯一来源：companies.is_test = true）──")
    tset = await test_tenant_ids()
    check(frozenset(TEST_TENANTS) <= tset, "test_tenant_ids 含两家测试公司",
          f"{sorted(tset)}")
    check(A_COMPANY not in tset and B_COMPANY not in tset,
          "test_tenant_ids 不含 A/B 公司", f"{sorted(tset)}")

    # ── 1. 派生点：content_scope = request_scope − 测试公司（仅 admin）──────────
    print("── 1. content_scope 派生（唯一实现点）──")
    req_admin = await request_scope(admin)
    cnt_admin = await content_scope(admin)
    req_a = await request_scope(a_user)
    cnt_a = await content_scope(a_user)
    req_tm = await request_scope(test_member)
    cnt_tm = await content_scope(test_member)

    home = frozenset({effective_tenant_id(admin)})
    owned = await tenant_ids_created_by(admin.id)
    check(
        req_admin.tenant_ids == (home | owned) and req_admin.owns_tenant_ids == owned,
        "request_scope(admin) = 所属租户 ∪ 自建集合 —— 管理/列表仍见测试公司",
        f"tenants={sorted(req_admin.tenant_ids or ())}",
    )
    check(
        cnt_admin.tenant_ids == ((home | owned) - tset)
        and cnt_admin.owns_tenant_ids == (owned - tset),
        "content_scope(admin) 剔除测试公司 → tenant_ids 剩 {所属租户}、owns 空集",
        f"tenants={sorted(cnt_admin.tenant_ids or ())} owns={sorted(cnt_admin.owns_tenant_ids or ())}",
    )
    check(
        cnt_admin.owner_id == admin.id,
        "content_scope(admin).owner_id 仍为本人 → 私库保留",
        f"owner={cnt_admin.owner_id}",
    )
    check(
        _scopes_equal(cnt_tm, req_tm)
        and cnt_tm.tenant_ids == frozenset({TEST_COMPANY_1}),
        "content_scope(测试公司成员) == request_scope —— 测试账号照旧",
        f"tenants={sorted(cnt_tm.tenant_ids or ())}",
    )
    check(
        _scopes_equal(cnt_a, req_a) and cnt_a.tenant_ids == frozenset({A_COMPANY}),
        "content_scope(A公司) == request_scope —— 行为不变",
        f"tenants={sorted(cnt_a.tenant_ids or ())}",
    )

    # ── 2. 摘要 / 关联（collect_document_digests 的底层：list_accessible_documents）
    print("── 2. 摘要/关联候选集（list_accessible_documents，@document_scope_clause）──")
    req_admin_ids = {
        str(r[0]) for r in await list_accessible_documents(
            limit=200, owner_id=str(req_admin.owner_id), tenant_ids=req_admin.tenant_ids,
            owns_tenant_ids=req_admin.owns_tenant_ids,
            user_department_id=req_admin.department_id,
            tenant_wide=req_admin.tenant_wide,
        )
    }
    cnt_admin_ids = {
        str(r[0]) for r in await list_accessible_documents(
            limit=200, owner_id=str(cnt_admin.owner_id), tenant_ids=cnt_admin.tenant_ids,
            owns_tenant_ids=cnt_admin.owns_tenant_ids,
            user_department_id=cnt_admin.department_id,
            tenant_wide=cnt_admin.tenant_wide,
        )
    }
    check(
        test_doc_ids <= req_admin_ids,
        "【管理视角】admin 的 request_scope 候选集**含**全部测试公司文档",
        f"命中 {len(test_doc_ids & req_admin_ids)}/{len(test_doc_ids)}",
    )
    check(
        not (cnt_admin_ids & test_doc_ids),
        "【核心】admin 的 content_scope 候选集**不含任何**测试公司文档",
        f"泄漏={sorted(_tenants_of(cnt_admin_ids & test_doc_ids, id2tenant))}",
    )
    check(
        admin_own_doc_ids <= cnt_admin_ids,
        "【阳性对照】admin 私库（default/private）仍在 content_scope 候选集内",
        f"命中 {len(admin_own_doc_ids & cnt_admin_ids)}/{len(admin_own_doc_ids)}",
    )

    tm_ids = {
        str(r[0]) for r in await list_accessible_documents(
            limit=200, owner_id=str(cnt_tm.owner_id), tenant_ids=cnt_tm.tenant_ids,
            owns_tenant_ids=cnt_tm.owns_tenant_ids,
            user_department_id=cnt_tm.department_id, tenant_wide=cnt_tm.tenant_wide,
        )
    }
    tm_own = {d for d in tm_ids if id2tenant.get(d) == TEST_COMPANY_1}
    check(
        bool(tm_own),
        "【阳性对照】测试公司成员 content_scope 候选集**含**本公司文档（可总结/关联）",
        f"本公司文档 {len(tm_own)} 份",
    )

    a_ids = {
        str(r[0]) for r in await list_accessible_documents(
            limit=200, owner_id=str(cnt_a.owner_id), tenant_ids=cnt_a.tenant_ids,
            owns_tenant_ids=cnt_a.owns_tenant_ids,
            user_department_id=cnt_a.department_id, tenant_wide=cnt_a.tenant_wide,
        )
    }
    check(
        bool(a_ids & a_doc_ids) and not (a_ids & test_doc_ids),
        "A公司 content_scope：见本公司、不见测试公司（行为不变）",
        f"A命中={len(a_ids & a_doc_ids)} 测试泄漏={len(a_ids & test_doc_ids)}",
    )

    # ── 3. 真实摘要入口：collect_document_digests（含 Qdrant 采样）──────────────
    print("── 3. 真实摘要/关联入口 collect_document_digests ──")
    dig_admin = await collect_document_digests(
        owner_id=str(cnt_admin.owner_id),
        tenant_ids=cnt_admin.tenant_ids,
        owns_tenant_ids=cnt_admin.owns_tenant_ids,
        user_department_id=cnt_admin.department_id,
        tenant_wide=cnt_admin.tenant_wide,
    )
    dig_admin_ids = {d.document_id for d in dig_admin}
    check(
        not (dig_admin_ids & test_doc_ids),
        "【核心】admin 的 collect_document_digests **不含**测试公司文档",
        f"文档数={len(dig_admin_ids)} 泄漏租户={sorted(_tenants_of(dig_admin_ids & test_doc_ids, id2tenant))}",
    )
    check(
        bool(dig_admin_ids & admin_own_doc_ids),
        "【阳性对照】admin 的 collect_document_digests **含**本人私库文档",
        f"私库命中={len(dig_admin_ids & admin_own_doc_ids)}",
    )

    dig_tm = await collect_document_digests(
        owner_id=str(cnt_tm.owner_id),
        tenant_ids=cnt_tm.tenant_ids,
        owns_tenant_ids=cnt_tm.owns_tenant_ids,
        user_department_id=cnt_tm.department_id,
        tenant_wide=cnt_tm.tenant_wide,
    )
    dig_tm_ids = {d.document_id for d in dig_tm}
    check(
        bool(dig_tm_ids & {d for d, t in id2tenant.items() if t == TEST_COMPANY_1}),
        "【阳性对照】测试公司成员的 collect_document_digests **含**本公司文档",
        f"文档数={len(dig_tm_ids)}",
    )

    # ── 4. 对话内文档列表：rag_graph._list_completed_documents（读 state）────────
    print("── 4. 对话内文档列表 _list_completed_documents ──")
    from app.services.rag_graph import _list_completed_documents

    list_admin_content = await _list_completed_documents(
        owner_id=str(cnt_admin.owner_id), collection_id=None,
        tenant_ids=cnt_admin.tenant_ids, user_department_id=cnt_admin.department_id,
        tenant_wide=cnt_admin.tenant_wide, owns_tenant_ids=cnt_admin.owns_tenant_ids,
    )
    list_admin_req = await _list_completed_documents(
        owner_id=str(req_admin.owner_id), collection_id=None,
        tenant_ids=req_admin.tenant_ids, user_department_id=req_admin.department_id,
        tenant_wide=req_admin.tenant_wide, owns_tenant_ids=req_admin.owns_tenant_ids,
    )
    # 该函数只回 filename，用 filename→tenant 映射反查
    name2tenant = {d.filename: d.tenant_id for d in docs}
    content_tenants = {name2tenant.get(r["filename"]) for r in list_admin_content}
    req_names = {r["filename"] for r in list_admin_req}
    test_names = {d.filename for d in docs if d.tenant_id in TEST_TENANTS}
    check(
        test_names <= req_names,
        "【管理视角】request_scope 的对话内列表**含**测试公司文档",
        f"命中 {len(test_names & req_names)}/{len(test_names)}",
    )
    check(
        not (content_tenants & set(TEST_TENANTS)),
        "【核心】content_scope 的对话内列表**不含**测试公司文档",
        f"泄漏租户={sorted(t for t in content_tenants if t in TEST_TENANTS)}",
    )

    # ── 5. 派生点唯一性（静态）：query.py 消费 content_scope ────────────────────
    print("── 5. 派生点 wiring（query.py 走 content_scope）──")
    query_src = (_BACKEND_ROOT / "app" / "api" / "query.py").read_text(encoding="utf-8")
    check(
        "content_scope(user)" in query_src and "await request_scope(" not in query_src,
        "app/api/query.py 内容消费入口改用 content_scope（不再调用 request_scope）",
        "命中 content_scope(user)" if "content_scope(user)" in query_src else "未命中",
    )
    rs_src = (_BACKEND_ROOT / "app" / "services" / "retrieval_service.py").read_text(
        encoding="utf-8"
    )
    check(
        "exclude_test_tenants(" in rs_src and "test_tenant_ids()" not in rs_src,
        "retrieval_service 复用 exclude_test_tenants（不再内联重复排除）",
        "命中 exclude_test_tenants" if "exclude_test_tenants(" in rs_src else "未命中",
    )

    print()
    if _failures:
        print(f"共 {len(_failures)} 项失败：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print("全部断言通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
