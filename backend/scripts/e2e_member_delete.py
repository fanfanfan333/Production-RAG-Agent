"""删除成员（账号注销）的回归验收脚本.

覆盖产品约定：「删除后该成员所有信息消失、个人文档消失，但部门文档和公司文档
不会消失」，以及五条权限护栏。

    docker cp backend/scripts/e2e_member_delete.py rag_backend:/tmp/e2e_member_delete.py
    docker exec -e PYTHONIOENCODING=utf-8 rag_backend \\
        sh -c "cd /app && python /tmp/e2e_member_delete.py"

为什么必须跑在**容器内**：脚本要在真实 PostgreSQL 上核对"删除后到底还剩什么"
（``documents`` 的 CASCADE 与解绑行为只能在库上看），而库只对容器网络开放。

安全约束：只创建并清理自己的 ``purge_mdel_*`` 临时账号、临时公司与临时文档，
**不触碰任何真实成员**；跑完自动清理（可重复运行）。退出码 0 = 全部通过。

覆盖点：

    护栏   不能删自己 / 不能删平台管理员 / 不能删同级 / 不能跨公司 / 需管理权限
    预检   个人库文档数、保留文档数、会话与消息数、文档集合数、身份快照
    执行   个人库文档整行删除（含 document_metadata 级联）
           部门库 / 公司库文档**保留**且仅解除归属（owner_id = NULL，层级不变）
           会话与消息级联清除、账号行消失
"""

from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import delete, func, or_, select

from app.db.conversation_models import Conversation, Message
from app.db.models import Document, DocumentMetadataRow, DocumentStatus
from app.db.postgres import get_db_session
from app.db.user_models import AuditLog, Collection, User
from app.services.staff_service import (
    StaffError,
    delete_member,
    preview_member_deletion,
)

TAG = "purge_mdel"
COMPANY_A = "c_purgemdel_a"
COMPANY_B = "c_purgemdel_b"

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    RESULTS.append((bool(ok), name, "" if ok else str(detail)))
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"   << {detail}"))


async def fetch_user(user_id: uuid.UUID) -> User:
    async with get_db_session() as s:
        return await s.get(User, user_id)


async def expect_staff_error(coro, keyword: str) -> tuple[bool, str]:
    """断言某个操作用 StaffError 拒绝，且理由里含指定关键字。"""
    try:
        await coro
    except StaffError as exc:
        return (keyword in str(exc), f"{exc.status_code} {exc}")
    except Exception as exc:                      # noqa: BLE001
        return (False, f"非 StaffError 异常：{type(exc).__name__}: {exc}")
    return (False, "居然成功了（本应被拒绝）")


async def purge() -> None:
    """清掉本脚本可能留下的全部临时数据（开头与结尾各跑一次）。

    ⚠️ 删 ``Document`` 行之前必须先清 Qdrant 向量。历史上这里只删 PG 行，
    每跑一轮就往向量库里留一批孤儿点；本仓库实测积累到 177 个孤儿 document_id
    （占集合 95%），并实测把平台管理员的检索候选池挤掉 80%。详见
    ``scripts/_e2e_purge.py`` 的模块说明。
    """
    from _e2e_purge import delete_vectors_for_documents

    async with get_db_session() as s:
        uids = list(
            (
                await s.execute(
                    select(User.id).where(User.username.like(f"{TAG}\\_%", escape="\\"))
                )
            ).scalars().all()
        )
        # 先把待删文档的 id 收集齐（按 owner 与按文件名前缀两路），统一清向量
        doc_rows = await s.execute(
            select(Document.id).where(
                or_(
                    Document.owner_id.in_(uids) if uids else False,
                    Document.filename.like(f"{TAG}%"),
                )
            )
        )
        doc_ids = [d for (d,) in doc_rows.all()]
        if doc_ids:
            await delete_vectors_for_documents(doc_ids)

        if uids:
            await s.execute(delete(Conversation).where(Conversation.owner_id.in_(uids)))
            await s.execute(delete(Collection).where(Collection.owner_id.in_(uids)))
            await s.execute(delete(Document).where(Document.owner_id.in_(uids)))
        await s.execute(delete(Document).where(Document.filename.like(f"{TAG}%")))
        await s.execute(
            delete(Conversation).where(Conversation.tenant_id.in_([COMPANY_A, COMPANY_B]))
        )
        if uids:
            await s.execute(delete(User).where(User.id.in_(uids)))
        # 测试动作写下的审计条目也是残留（正式审计里不该出现临时账号）
        await s.execute(
            delete(AuditLog).where(AuditLog.username.like(f"{TAG}\\_%", escape="\\"))
        )


async def count(model, *conditions) -> int:
    async with get_db_session() as s:
        stmt = select(func.count()).select_from(model)
        if conditions:
            stmt = stmt.where(*conditions)
        return int((await s.execute(stmt)).scalar_one() or 0)


async def main() -> int:
    await purge()

    actor_id = uuid.uuid4()          # 临时平台管理员（执行者）
    other_admin_id = uuid.uuid4()    # 临时平台管理员（被删对象）
    target_id = uuid.uuid4()         # 被删成员：普通员工 + 个人库/部门库/公司库文档
    peer_id = uuid.uuid4()           # 同级成员（kb_admin 删 kb_admin 的场景）
    boss_id = uuid.uuid4()           # 临时 kb_admin（同级 / 跨公司场景的执行者）
    outsider_id = uuid.uuid4()       # 别的公司的员工

    private_doc = uuid.uuid4()
    dept_doc = uuid.uuid4()
    company_doc = uuid.uuid4()
    conv_id = uuid.uuid4()

    # ── 造数据 ────────────────────────────────────────────────────────────────
    async with get_db_session() as s:
        s.add(User(id=actor_id, username=f"{TAG}_actor", role=User.ROLE_ADMIN,
                   tenant_id="default", is_active=True))
        s.add(User(id=other_admin_id, username=f"{TAG}_admin2", role=User.ROLE_ADMIN,
                   tenant_id="default", is_active=True))
        s.add(User(id=boss_id, username=f"{TAG}_boss", role=User.ROLE_KB_ADMIN,
                   tenant_id=COMPANY_A, company_name="测试A公司", is_active=True))
        s.add(User(id=peer_id, username=f"{TAG}_peer", role=User.ROLE_KB_ADMIN,
                   tenant_id=COMPANY_A, company_name="测试A公司", is_active=True))
        s.add(User(id=outsider_id, username=f"{TAG}_outsider", role=User.ROLE_EMPLOYEE,
                   tenant_id=COMPANY_B, company_name="测试B公司", is_active=True))
        s.add(User(
            id=target_id, username=f"{TAG}_target", display_name="临时成员",
            role=User.ROLE_EMPLOYEE, tenant_id=COMPANY_A, company_name="测试A公司",
            department_id="d_purgemdel", department_name="测试部门",
            job_title="测试工程师", is_active=True,
        ))
    async with get_db_session() as s:
        for did, level in ((private_doc, "private"), (dept_doc, "department"),
                           (company_doc, "tenant")):
            s.add(Document(
                id=did, filename=f"{TAG}_{level}.pdf", file_size=10,
                file_hash=uuid.uuid4().hex, status=DocumentStatus.COMPLETED,
                owner_id=target_id, tenant_id=COMPANY_A, access_level=level,
                department_id="d_purgemdel" if level == "department" else None,
            ))
        # 个人库文档的元数据行：验证删文档时 document_metadata 的 CASCADE 是否生效
        s.add(DocumentMetadataRow(document_id=private_doc, tenant_id=COMPANY_A,
                                  title="临时"))
        s.add(Conversation(id=conv_id, owner_id=target_id, tenant_id=COMPANY_A))
        s.add(Message(id=uuid.uuid4(), conversation_id=conv_id, role="user",
                      content="临时提问", user_id=target_id, tenant_id=COMPANY_A))
        s.add(Collection(id=uuid.uuid4(), name=f"{TAG}_集合", owner_id=target_id))

    # ── A. 权限护栏 ───────────────────────────────────────────────────────────
    ok, why = await expect_staff_error(
        delete_member(await fetch_user(actor_id), actor_id), "不能删除自己")
    check("护栏①：不能删除自己", ok, why)

    ok, why = await expect_staff_error(
        delete_member(await fetch_user(actor_id), other_admin_id), "平台管理员")
    check("护栏②：平台管理员账号不可被删除", ok, why)

    ok, why = await expect_staff_error(
        delete_member(await fetch_user(boss_id), peer_id), "同级或更高等级")
    check("护栏③：不能删除同级成员（kb_admin → kb_admin）", ok, why)

    ok, why = await expect_staff_error(
        delete_member(await fetch_user(boss_id), outsider_id), "不属于你所在的公司")
    check("护栏④：非平台管理员不能删别的公司的成员", ok, why)

    ok, why = await expect_staff_error(
        delete_member(await fetch_user(target_id), peer_id), "没有管理成员身份的权限")
    check("护栏⑤：普通员工没有成员管理权限", ok, why)

    # ── B. 预检（确认弹窗的数据来源）──────────────────────────────────────────
    impact = await preview_member_deletion(await fetch_user(actor_id), target_id)
    check("预检：个人库文档 1 份待删",
          impact["deleted"]["personal_documents"] == 1, impact["deleted"])
    check("预检：部门/公司库文档 2 份保留",
          impact["kept"]["shared_documents"] == 2, impact["kept"])
    check("预检：会话 1 / 消息 1",
          impact["deleted"]["conversations"] == 1 and impact["deleted"]["messages"] == 1,
          impact["deleted"])
    check("预检：文档集合 1", impact["deleted"]["collections"] == 1, impact["deleted"])
    check("预检：身份快照带公司/部门/职责",
          impact["member"]["company_name"] == "测试A公司"
          and impact["member"]["department_name"] == "测试部门"
          and impact["member"]["job_title"] == "测试工程师", impact["member"])

    # ── C. 真正删除 ───────────────────────────────────────────────────────────
    result = await delete_member(await fetch_user(actor_id), target_id)

    check("删除结果：个人库文档 1 份被删",
          result["deleted"]["personal_documents"] == 1, result["deleted"])
    check("删除结果：会话 1 个被删",
          result["deleted"]["conversations"] == 1, result["deleted"])
    check("删除结果：保留文档 2 份",
          result["kept"]["shared_documents"] == 2, result["kept"])

    async with get_db_session() as s:
        gone_user = await s.get(User, target_id)
        kept = list(
            (
                await s.execute(
                    select(Document.id, Document.owner_id, Document.access_level)
                    .where(Document.id.in_([private_doc, dept_doc, company_doc]))
                )
            ).all()
        )
        meta = await count(DocumentMetadataRow, DocumentMetadataRow.document_id == private_doc)
        conv = await s.get(Conversation, conv_id)
        msgs = await count(Message, Message.conversation_id == conv_id)

    check("终态：账号行已消失", gone_user is None, gone_user)
    kept_ids = {row[0] for row in kept}
    check("终态：个人库文档整行已删除", private_doc not in kept_ids,
          sorted(str(i) for i in kept_ids))
    check("终态：个人库文档的元数据行被级联清除", meta == 0, meta)
    check("终态：部门库文档仍在", dept_doc in kept_ids, sorted(str(i) for i in kept_ids))
    check("终态：公司库文档仍在", company_doc in kept_ids, sorted(str(i) for i in kept_ids))
    owners = {row[0]: (row[1], row[2]) for row in kept}
    check("终态：保留文档仅解除归属（owner_id = NULL，层级不变）",
          owners.get(dept_doc) == (None, "department")
          and owners.get(company_doc) == (None, "tenant"), owners)
    check("终态：会话已删除", conv is None, conv)
    check("终态：消息已级联清除", msgs == 0, msgs)
    check("终态：删完不能再删（成员不存在）",
          (await expect_staff_error(
              delete_member(await fetch_user(actor_id), target_id), "成员不存在"))[0])

    # ── D. 清理 ───────────────────────────────────────────────────────────────
    await purge()
    check("清理：临时账号无残留",
          await count(User, User.username.like(f"{TAG}\\_%", escape="\\")) == 0)
    check("清理：临时文档无残留",
          await count(Document, Document.filename.like(f"{TAG}%")) == 0)

    failed = [name for ok, name, _ in RESULTS if not ok]
    print()
    print(f"===== {len(RESULTS) - len(failed)}/{len(RESULTS)} passed =====")
    if failed:
        print("failed: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
