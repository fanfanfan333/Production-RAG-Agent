"""测试公司文档隔离验收（用户新规则 1–4）—— 只读，可重复运行.

规则（用户修正后，以此为准）
────────────────────────────
  1. **测试公司成员（测试账号）**：在自己测试公司范围内一切照旧 —— 列表可见、
     检索**能命中**、可上传、可预览。（用知识库测 bug 的前提。）
  2. **平台管理员 admin**：文档列表**仍能看到**测试公司文档（管理用），但
     **检索结果里不得命中**测试公司文档。
  3. **A公司 / B公司**：行为完全不变。
  4. **admin 本人的文档**（含 default 租户私库）：保留 —— 正常可见、正常可检索。

退出码：0 = 全部断言通过；1 = 存在失败项。

用法（容器内）
──────────────
    docker cp scripts/verify_test_company_isolation.py rag_backend:/app/scripts/
    docker exec -w /app rag_backend python -m scripts.verify_test_company_isolation
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

# 测试公司文档里独有的正文（见各 .txt）：命中即说明检索到了测试公司内容。
TEST_CONTENT_QUERY = "紫罗兰计划的年度预算是多少 由财务部张明远负责审批 8842万元"
# admin 自己 default 私库文档（研发部-2024年度技术方案）里独有的正文。
ADMIN_OWN_QUERY = "研发部技术方案 里程碑对照表 交付物 元数据体系与结构分块"
# A公司文档里独有的正文。
A_DOC_QUERY = "用印申请流程 分级审批 印章保管"

_failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)


def _hit_summary(chunks) -> str:
    return str([(c.filename, c.tenant_id) for c in chunks])


async def _list_for(user, scope):
    from app.services.document_query_service import list_documents

    return await list_documents(
        page=1, limit=100,
        owner_id=scope.owner_id,
        tenant_ids=scope.tenant_ids,
        user_department_id=scope.department_id,
        tenant_wide=scope.tenant_wide,
        owns_tenant_ids=scope.owns_tenant_ids,
        viewer=user,
    )


def _hits(chunks, tid: str) -> list:
    return [c for c in chunks if c.tenant_id == tid]


async def _run() -> int:
    from sqlalchemy import select

    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.db.user_models import User
    from app.services.company_registry import tenant_ids_created_by, test_tenant_ids
    from app.services.tenancy import (
        ACCESS_TENANT,
        can_access_document,
        delete_permission_for,
        effective_tenant_id,
        request_scope,
    )
    from app.services.retrieval_service import retrieve_chunks

    async with get_db_session() as session:
        users = {
            u.username: u
            for u in (await session.execute(select(User))).scalars().all()
        }

    admin = users.get(ADMIN_USERNAME)
    a_user = users.get(A_USER_USERNAME)
    test_member = users.get(TEST_MEMBER_USERNAME)
    for label, u in (
        ("admin", admin), (A_USER_USERNAME, a_user), (TEST_MEMBER_USERNAME, test_member),
    ):
        if u is None:
            print(f"FATAL：找不到账号 {label}")
            return 1

    # ── 0. 测试公司集合（唯一来源）──────────────────────────────────────────────
    print("── 0. test_tenant_ids（唯一来源：companies.is_test = true）──")
    tset = await test_tenant_ids()
    check(TEST_TENANTS <= tset, "test_tenant_ids 含两家测试公司", f"{sorted(tset)}")
    check(
        A_COMPANY not in tset and B_COMPANY not in tset,
        "test_tenant_ids 不含 A/B 公司（is_test=false）",
        f"{sorted(tset)}",
    )

    # ── scope 分流（非 admin 路径一行未动）──────────────────────────────────────
    print("── scope 分流（admin = 所属租户 ∪ 自建集合；其余保持原样）──")
    sc_admin = await request_scope(admin)
    sc_a = await request_scope(a_user)
    sc_tm = await request_scope(test_member)
    owned = await tenant_ids_created_by(admin.id)
    home = frozenset({effective_tenant_id(admin)})
    check(
        sc_admin.tenant_ids == (home | owned) and sc_admin.owns_tenant_ids == owned,
        "admin scope = 所属租户(default) ∪ 自建集合（列表/管理可见测试公司）",
        f"tenants={sorted(sc_admin.tenant_ids or ())}",
    )
    check(
        sc_tm.tenant_ids == frozenset({TEST_COMPANY_1}) and sc_tm.owner_id == test_member.id,
        "测试公司成员 scope **照旧**（本公司 + 自己的个人库）",
        f"tenants={sorted(sc_tm.tenant_ids or ())}",
    )
    check(
        sc_a.tenant_ids == frozenset({A_COMPANY}),
        "A公司成员 scope 不变（本公司）",
        f"tenants={sorted(sc_a.tenant_ids or ())}",
    )

    # ── 规则 1：测试公司成员在自己公司范围内一切照旧（列表 + 检索）─────────────
    print("── 规则1 测试公司成员照旧可用（列表 + 检索**能命中**）──")
    page_tm = await _list_for(test_member, sc_tm)
    tm_tenants = {d.tenant_id for d in page_tm.documents}
    check(
        page_tm.total > 0 and tm_tenants <= {TEST_COMPANY_1},
        "测试公司成员列表**能看到**自己公司的文档",
        f"total={page_tm.total} tenants={sorted(tm_tenants)}",
    )

    hits_tm = await retrieve_chunks(
        query=TEST_CONTENT_QUERY, top_k=5,
        owner_id=str(sc_tm.owner_id),
        tenant_ids=sc_tm.tenant_ids,
        owns_tenant_ids=sc_tm.owns_tenant_ids,
        tenant_wide=sc_tm.tenant_wide,
    )
    check(
        bool(_hits(hits_tm, TEST_COMPANY_1)),
        "【阳性对照】测试公司成员检索自己的测试公司文档**能命中**",
        f"hits={_hit_summary(hits_tm)}",
    )

    # ── 规则 2：admin 列表可见测试公司文档，但检索零命中 ────────────────────────
    print("── 规则2 admin 列表可见 / 检索零命中 ──")
    page_admin = await _list_for(admin, sc_admin)
    admin_tenants = {d.tenant_id for d in page_admin.documents}
    check(
        TEST_TENANTS <= admin_tenants,
        "admin 列表**能看到**测试公司文档（用于管理）",
        f"tenants={sorted(admin_tenants)} total={page_admin.total}",
    )

    hits_admin_test = await retrieve_chunks(
        query=TEST_CONTENT_QUERY, top_k=5,
        owner_id=str(sc_admin.owner_id),
        tenant_ids=sc_admin.tenant_ids,
        owns_tenant_ids=sc_admin.owns_tenant_ids,
        tenant_wide=sc_admin.tenant_wide,
    )
    check(
        not _hits(hits_admin_test, TEST_COMPANY_1)
        and not _hits(hits_admin_test, TEST_COMPANY_2),
        "【核心】admin 检索测试公司内容**零命中**",
        f"hits={_hit_summary(hits_admin_test)}",
    )

    # ── 规则 4：admin 自己的 default 私库仍可检索（阳性对照）───────────────────
    print("── 规则4 admin 自己的私库仍可检索（阳性对照）──")
    hits_admin_own = await retrieve_chunks(
        query=ADMIN_OWN_QUERY, top_k=5,
        owner_id=str(sc_admin.owner_id),
        tenant_ids=sc_admin.tenant_ids,
        owns_tenant_ids=sc_admin.owns_tenant_ids,
        tenant_wide=sc_admin.tenant_wide,
    )
    check(
        bool(_hits(hits_admin_own, "default")),
        "【阳性对照】admin 检索**命中自己的 default 私库**",
        f"hits={_hit_summary(hits_admin_own)}",
    )

    # ── 规则 3：A公司行为完全不变 ───────────────────────────────────────────────
    print("── 规则3 A公司行为不变 ──")
    page_a = await _list_for(a_user, sc_a)
    a_tenants = {d.tenant_id for d in page_a.documents}
    check(
        A_COMPANY in a_tenants and not (a_tenants & TEST_TENANTS),
        "A公司成员列表：见本公司、不见测试公司",
        f"tenants={sorted(a_tenants)}",
    )
    hits_a = await retrieve_chunks(
        query=A_DOC_QUERY, top_k=5,
        owner_id=str(sc_a.owner_id),
        tenant_ids=sc_a.tenant_ids,
        owns_tenant_ids=sc_a.owns_tenant_ids,
        tenant_wide=sc_a.tenant_wide,
    )
    check(
        bool(_hits(hits_a, A_COMPANY)),
        "A公司成员检索本公司文档**能命中**",
        f"hits={_hit_summary(hits_a)}",
    )
    hits_a_test = await retrieve_chunks(
        query=TEST_CONTENT_QUERY, top_k=5,
        owner_id=str(sc_a.owner_id),
        tenant_ids=sc_a.tenant_ids,
        owns_tenant_ids=sc_a.owns_tenant_ids,
        tenant_wide=sc_a.tenant_wide,
    )
    check(
        not (_hits(hits_a_test, TEST_COMPANY_1) or _hits(hits_a_test, TEST_COMPANY_2)),
        "A公司成员检索测试公司内容零命中（跨公司隔离）",
        f"hits={_hit_summary(hits_a_test)}",
    )

    # ── 边界：admin 对测试公司文档的管理能力保留（预览/删除权判定，不删除）──────
    print("── admin 管理能力保留（/chunks、图片、下载、删除判定，不实际删除）──")
    async with get_db_session() as session:
        test_doc = (
            await session.execute(
                select(Document).where(
                    Document.tenant_id == TEST_COMPANY_1,
                    Document.access_level == ACCESS_TENANT,
                )
            )
        ).scalars().first()
    if test_doc is None:
        check(False, "测试公司1 存在公司库文档（用于管理能力断言）")
    else:
        check(
            can_access_document(
                test_doc, admin,
                tenant_ids=sc_admin.tenant_ids,
                owns_tenant_ids=sc_admin.owns_tenant_ids,
            ),
            "admin **仍可访问/预览**测试公司文档",
            f"doc={test_doc.filename}",
        )
        allowed, reason = delete_permission_for(
            test_doc, admin,
            tenant_ids=sc_admin.tenant_ids,
            owns_tenant_ids=sc_admin.owns_tenant_ids,
        )
        check(allowed, "admin **仍可删除**测试公司公司库文档", reason or "allowed")
        check(
            can_access_document(
                test_doc, test_member,
                tenant_ids=sc_tm.tenant_ids,
                owns_tenant_ids=sc_tm.owns_tenant_ids,
            ),
            "测试公司成员**仍可访问**本公司文档（照旧）",
            f"doc={test_doc.filename}",
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
