"""
回归测试：`delete_company` 不得删掉**其它租户**的文档（跨租户级联误删）.

缺陷：``documents.owner_id`` 是 ``ON DELETE CASCADE``。``delete_company`` 先按
``tenant_id == 本公司`` 删文档，再删本公司 ``User`` 行 —— 删用户会把它名下**任意
tenant** 的文档一起级联删除。可达路径：身份验证审核通过把成员的 ``tenant_id`` 改到
新公司，而他此前在原公司租户下上传的文档仍留在原租户。

修复后语义：**本租户文档照删，他租户文档只解绑归属（``owner_id=NULL``）不删**。

运行方式（容器内，需可达数据库）：
    docker exec -w /app -e PYTHONPATH=/app rag_backend \
        python -m pytest tests/test_delete_company_cross_tenant.py -q -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

try:
    import pytest

    from sqlalchemy import func, select, text

    from app.db.company_models import Company
    from app.db.models import Document, DocumentStatus
    from app.db.postgres import get_db_session
    from app.db.user_models import User
    from app.services.staff_service import (
        delete_company,
        preview_company_deletion,
    )

    _IMPORT_OK = True
except ImportError as exc:  # pragma: no cover — 宿主机缺依赖 → 跳过
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")
    _IMPORT_OK = False


async def _count_docs(tenant_id: str) -> int:
    async with get_db_session() as s:
        return int(
            (await s.execute(
                select(func.count()).select_from(Document).where(Document.tenant_id == tenant_id)
            )).scalar_one() or 0
        )


async def _cleanup(tid: str, other: str) -> None:
    """删掉本测试在同一次运行里创建的行（tenant 为随机值，绝不误伤真实数据）。"""
    async with get_db_session() as s:
        await s.execute(text("DELETE FROM documents WHERE tenant_id = :t"), {"t": tid})
        await s.execute(text("DELETE FROM documents WHERE tenant_id = :t"), {"t": other})
        await s.execute(text("DELETE FROM conversations WHERE tenant_id = :t"), {"t": tid})
        await s.execute(text("DELETE FROM users WHERE tenant_id = :t"), {"t": tid})
        await s.execute(text("DELETE FROM companies WHERE tenant_id = :t"), {"t": tid})
        await s.execute(text("DELETE FROM companies WHERE tenant_id = :t"), {"t": other})


async def _scenario() -> None:
    tid = "c" + uuid.uuid4().hex[:12]   # 被删的公司
    other = "c" + uuid.uuid4().hex[:12]  # 另一家公司（成员跨租户文档的归属租户）

    async with get_db_session() as s:
        admin = await s.scalar(select(User).where(User.username == "admin").limit(1))
    if admin is None or not admin.is_admin:
        pytest.skip("找不到平台管理员 admin — 跳过（需容器内数据库）")

    try:
        cname = "跨租户回归公司_" + uuid.uuid4().hex[:6]
        async with get_db_session() as s:
            member = User(
                username="crsynth_" + uuid.uuid4().hex[:6], password_hash="x",
                display_name="跨租户成员", role=User.ROLE_EMPLOYEE, tenant_id=tid,
            )
            s.add(member)
            await s.flush()
            member_id = member.id

            own_doc = Document(
                filename="own.pdf", file_size=1, file_hash=uuid.uuid4().hex,
                status=DocumentStatus.COMPLETED, owner_id=member_id, tenant_id=tid,
                access_level="private",
            )
            cross_doc = Document(
                filename="cross.pdf", file_size=1, file_hash=uuid.uuid4().hex,
                status=DocumentStatus.COMPLETED, owner_id=member_id, tenant_id=other,
                access_level="private",
            )
            s.add_all([own_doc, cross_doc])
            await s.flush()
            cross_doc_id = cross_doc.id

            s.add(Company(
                tenant_id=tid, display_name=cname,
                name_key=cname.casefold().replace(" ", ""),
                created_by=admin.id, is_test=True,
            ))

        # 预检：跨租户文档应在 kept 组里计数为 1
        preview = await preview_company_deletion(admin, tid)
        assert preview["kept"]["cross_tenant_documents"] == 1, preview["kept"]

        other_docs_before = await _count_docs(other)

        payload = await delete_company(admin, tid)

        # ① 跨租户文档仍然存在、owner 置空、tenant 不变
        async with get_db_session() as s:
            cross = await s.get(Document, cross_doc_id)
        assert cross is not None, "跨租户文档被误删（级联缺陷）"
        assert cross.owner_id is None, "跨租户文档的归属人未被解绑"
        assert cross.tenant_id == other, "跨租户文档的 tenant_id 被改动"

        # ② 其它租户的文档数不变
        assert await _count_docs(other) == other_docs_before

        # ③ 本公司租户内的文档仍然全部删除
        assert await _count_docs(tid) == 0

        # ④ 预检与执行共用口径
        assert payload["kept"]["cross_tenant_documents"] == 1, payload["kept"]

        # ⑤ 公司注册行已删
        async with get_db_session() as s:
            assert await s.get(Company, tid) is None
    finally:
        await _cleanup(tid, other)


def test_delete_company_unbinds_cross_tenant_document_without_deleting_it():
    if not _IMPORT_OK:
        return
    asyncio.run(_scenario())
