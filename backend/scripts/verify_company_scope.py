"""公司改名（P0-4 / P1-2）与 admin 作用域收敛（P0-6/7/8）的在线验收脚本.

只读断言（**不写任何数据**），可重复运行。覆盖 PRD §6 的可执行验收点：

  P0-4 改名   「转入检索公司bjld8」→「测试公司1」(c8111de986583)、
              「转入检索公司1z5pp」→「测试公司2」(cfb08c53677c4)；
              tenant_id 不变、成员数/文档数按租户核对、users.company_name 已同步。
  P1-2 旧名   旧名按注册表解析不到（find_by_name → None），新名可解析。
  P0-6 列表   admin 可见文档（走 document_scope_clause —— 列表/检索/关键词腿的
              唯一 SQL 入口）只落在其自建测试公司内（含该公司 private），
              不含 A/B 公司任何文档。
  P0-7 检索   检索与列表共用同一 document_scope_clause（结构性同源，见
              pg_keyword_search / master_graph 的调用点），本脚本另断言
              「count_by_tenant == 按公司过滤的 list_documents」逐家一致。
  P0-8 私库   A/B 公司成员的 private 文档对 admin 不可见；
              测试公司内他人 private 文档对 admin 可见（可读、不可删由
              tests/test_scope_isolation_tenant_set.py 单测兜底）。

退出码：0 = 全部断言通过；1 = 存在失败项。

用法（容器内）
──────────────
    docker exec -w /app rag_backend python -m scripts.verify_company_scope
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

TEST_COMPANY_1 = "c8111de986583"
TEST_COMPANY_2 = "cfb08c53677c4"
OLD_NAME_1 = "转入检索公司bjld8"
OLD_NAME_2 = "转入检索公司1z5pp"
OTHER_COMPANIES = {"c309a7cb9f496", "cf33b1db5679d"}  # A公司 / B公司

_failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)


async def _run() -> int:
    from sqlalchemy import func, select

    from app.db.company_models import Company
    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.db.user_models import User
    from app.services.company_registry import (
        company_display_names,
        find_by_name,
        tenant_ids_created_by,
    )
    from app.services.document_query_service import count_by_tenant, list_documents
    from app.services.tenancy import (
        ACCESS_PRIVATE,
        document_scope_clause,
        effective_tenant_id,
        scope_for,
    )

    async with get_db_session() as session:
        admin = await session.scalar(
            select(User).where(User.username == "admin").limit(1)
        )
    if admin is None:
        print("FATAL：找不到 admin 账号")
        return 1

    # ── P0-4：改名结果（tenant_id 不变、展示名已更新、成员/文档按租户核对）────
    print("── P0-4 公司改名（bjld8→测试公司1 / 1z5pp→测试公司2）──")
    names = await company_display_names(None)
    check(names.get(TEST_COMPANY_1) == "测试公司1", "测试公司1 展示名", names.get(TEST_COMPANY_1, "<缺失>"))
    check(names.get(TEST_COMPANY_2) == "测试公司2", "测试公司2 展示名", names.get(TEST_COMPANY_2, "<缺失>"))

    async with get_db_session() as session:
        member_rows = await session.execute(
            select(User.tenant_id, User.company_name, func.count())
            .group_by(User.tenant_id, User.company_name)
        )
        doc_rows = await session.execute(
            select(Document.tenant_id, func.count()).group_by(Document.tenant_id)
        )
    members = {t: (n, c) for t, n, c in member_rows.all()}
    docs = {t: c for t, c in doc_rows.all()}

    for tid, expect_name in ((TEST_COMPANY_1, "测试公司1"), (TEST_COMPANY_2, "测试公司2")):
        member_name, member_count = members.get(tid, (None, 0))
        check(member_count > 0, f"{expect_name} 成员存在", f"{member_count} 人")
        check(
            member_name == expect_name,
            f"{expect_name} 成员 company_name 已同步",
            f"users.company_name={member_name!r}",
        )
        check(tid in docs, f"{expect_name} 文档存在", f"{docs.get(tid, 0)} 份")
        print(f"  [info] {expect_name}({tid})：成员 {member_count} 人 / 文档 {docs.get(tid, 0)} 份")

    # ── P1-2：旧名失效、新名可解析 ─────────────────────────────────────────
    print("── P1-2 改名后旧名不可再用于身份验证 ──")
    old1 = await find_by_name(OLD_NAME_1)
    old2 = await find_by_name(OLD_NAME_2)
    check(old1 is None, f"旧名 {OLD_NAME_1!r} 解析不到（按未注册拒绝）")
    check(old2 is None, f"旧名 {OLD_NAME_2!r} 解析不到（按未注册拒绝）")
    new1 = await find_by_name("测试公司1")
    new2 = await find_by_name("测试公司2")
    check(new1 is not None and new1.tenant_id == TEST_COMPANY_1, "新名「测试公司1」解析到原 tenant_id")
    check(new2 is not None and new2.tenant_id == TEST_COMPANY_2, "新名「测试公司2」解析到原 tenant_id")

    # ── P0-6/7/8：admin 作用域（同一 document_scope_clause 入口）─────────────
    print("── P0-6/7/8 admin 作用域收敛（列表/检索同一 SQL 入口）──")
    owned = await tenant_ids_created_by(admin.id)
    # 只断言「包含两家目标测试公司」且「不含非 admin 创建的 A/B 公司」——
    # 不锁定精确集合：admin 随时可能再建测试公司（如 QA 留下的临时公司），
    # 集合规模变化是合法产品行为，不应让验收脚本变红。
    check(
        {TEST_COMPANY_1, TEST_COMPANY_2} <= owned,
        "admin 自建集合包含 测试公司1/2（tenant_id 未变）",
        f"{sorted(owned)}",
    )
    check(
        owned.isdisjoint(OTHER_COMPANIES),
        "admin 自建集合不含 A/B 公司（非其创建）",
        f"{sorted(owned)}",
    )
    home = frozenset({effective_tenant_id(admin)})
    scope = scope_for(admin, owned_tenant_ids=owned)
    check(
        scope.tenant_ids == (home | owned) and scope.owns_tenant_ids == owned,
        "scope tenant_ids == 所属租户 ∪ 自建集合；owns == 自建集合",
        f"tenants={sorted(scope.tenant_ids or ())}",
    )

    async with get_db_session() as session:
        visible_rows = (
            await session.execute(
                select(Document.id, Document.tenant_id, Document.access_level, Document.owner_id)
                .where(
                    document_scope_clause(
                        owner_id=scope.owner_id,
                        department_id=scope.department_id,
                        tenant_ids=scope.tenant_ids,
                        owns_tenant_ids=scope.owns_tenant_ids,
                        tenant_wide=scope.tenant_wide,
                    )
                )
            )
        ).all()

    visible_ids = {str(r.id) for r in visible_rows}
    foreign = [r for r in visible_rows if r.tenant_id in OTHER_COMPANIES and r.owner_id != admin.id]
    check(not foreign, "可见集不含 A/B 公司任何文档", f"可见 {len(visible_ids)} 份")

    # 强化断言（QA 补齐）：原 #4 只断言「包含两家 + 不含 A/B」，且 foreign 检查只覆盖
    # OTHER_COMPANIES —— 一家非 A/B、又非 admin 自建的第三家公司若泄漏可见仍会通过。
    # 这里把口径钉在**可从 DB 计算**的集合：admin 可见公司 = {所属租户} ∪
    # {companies.created_by == admin.id}，并断言任何可见文档的 tenant_id 都落在
    # 该集合内（admin 本人私库除外）。
    async with get_db_session() as session:
        db_created = set(
            (
                await session.execute(
                    select(Company.tenant_id).where(Company.created_by == admin.id)
                )
            )
            .scalars()
            .all()
        )
    check(
        owned == db_created,
        "owned == companies(created_by==admin.id)（DB 计算，防注册表漂移）",
        f"owned={sorted(owned)} db={sorted(db_created)}",
    )
    allowed_tenants = db_created | {effective_tenant_id(admin)}
    stray = [
        r for r in visible_rows if r.owner_id != admin.id and r.tenant_id not in allowed_tenants
    ]
    check(
        not stray,
        "可见文档 tenant_id 全部 ∈ 所属租户 ∪ admin 自建公司集合（DB 计算）",
        f"越权 {len(stray)} 份: {[(r.tenant_id, str(r.id)) for r in stray]}",
    )

    # admin 可见测试公司内的 private 文档（若存在）；A/B 公司 private 恒不可见
    owned_private = [r for r in visible_rows if r.tenant_id in owned and r.access_level == ACCESS_PRIVATE and r.owner_id != admin.id]
    print(f"  [info] admin 可见测试公司内他人私库 {len(owned_private)} 份（可读、不可删）")

    async with get_db_session() as session:
        other_private = (
            await session.execute(
                select(Document.id).where(
                    Document.tenant_id.in_(sorted(OTHER_COMPANIES)),
                    Document.access_level == ACCESS_PRIVATE,
                )
            )
        ).scalars().all()
    leaked = [str(d) for d in other_private if str(d) in visible_ids]
    check(not leaked, "A/B 公司成员的 private 文档对 admin 不可见")

    # P0-7 计数同源：count_by_tenant == list_documents(company_id=...) 逐家一致
    # （含所属租户 default —— admin 的公司筛选也应对它计数同源）
    counts = await count_by_tenant(
        owner_id=scope.owner_id,
        tenant_ids=scope.tenant_ids,
        owns_tenant_ids=scope.owns_tenant_ids,
        department_id=scope.department_id,
        tenant_wide=scope.tenant_wide,
    )
    for tid in sorted(scope.tenant_ids):
        page = await list_documents(
            page=1, limit=1,
            owner_id=scope.owner_id,
            tenant_ids=scope.tenant_ids,
            user_department_id=scope.department_id,
            tenant_wide=scope.tenant_wide,
            owns_tenant_ids=scope.owns_tenant_ids,
            company_id=tid,
        )
        check(
            counts.get(tid, 0) == page.total,
            f"doc_count 与列表一致（{names.get(tid, tid)}）",
            f"count={counts.get(tid, 0)} list={page.total}",
        )

    # P0-6 边界：无自建公司时收敛为「仅所属租户」（编译期断言，不动数据）——
    # 不回退全平台，也不再漏掉 admin 自己租户的公司库/部门库。
    empty_scope = scope_for(admin, owned_tenant_ids=frozenset())
    check(
        empty_scope.tenant_ids == home and empty_scope.owns_tenant_ids == frozenset(),
        "无自建公司 → tenant_ids 仅所属租户（不回退全平台）；owns 为空",
        f"tenants={sorted(empty_scope.tenant_ids or ())}",
    )

    # ── P0-1 三层标注数据（服务端：能取到即干净字符串，取不到即 None）──────────
    print("── P0-1 三层标注数据（GET /documents 的 tenant_name / department_name）──")
    page_all = await list_documents(
        page=1, limit=100,
        owner_id=scope.owner_id,
        tenant_ids=scope.tenant_ids,
        user_department_id=scope.department_id,
        tenant_wide=scope.tenant_wide,
        owns_tenant_ids=scope.owns_tenant_ids,
        viewer=admin,
    )
    by_level: dict[str, int] = {}
    dirty: list[str] = []
    for summ in page_all.documents:
        by_level[summ.access_level] = by_level.get(summ.access_level, 0) + 1
        print(
            f"    - [{summ.access_level}] {summ.filename} | "
            f"tenant_name={summ.tenant_name!r} department_name={summ.department_name!r}"
        )
        if summ.access_level == "department" and not (summ.tenant_name and summ.department_name):
            dirty.append(f"部门文档 {summ.filename} 缺公司名/部门名")
        if summ.access_level == "tenant" and not summ.tenant_name:
            dirty.append(f"公司文档 {summ.filename} 缺公司名")
        for value in (summ.tenant_name, summ.department_name):
            if value is not None and (
                not value.strip() or value.strip().lower() in {"none", "null", "undefined"}
            ):
                dirty.append(f"{summ.filename} 展示名脏值 {value!r}")
    print(f"  [info] 可见文档层级分布 {by_level}")
    check(
        not dirty,
        "三层标注展示名：部门=公司名+部门名 / 公司=公司名 / 无 None-undefined 脏值",
        "; ".join(dirty) if dirty else "",
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
