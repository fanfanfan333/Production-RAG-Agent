"""
存量"部门库但无部门"退化文档的修复脚本（不变量：department ⇒ department_id 非空）.

背景
────
``documents.access_level = 'department'`` 但 ``department_id IS NULL`` 是一个**退化态**：
``tenancy.document_scope_clause`` 的 department 分支要求 ``department_id`` 相等，而
``can_access_document`` 的 department 分支也没有 owner 快放 —— 于是这份文档**连 owner
都检索不到**（列表里看得见、提问搜不到），且对象级 ``allows()`` 与文档级判定同时失效，
表现为一个"查不出来的静默缺陷"。

成因：历史写点 ``knowledge_tier_service.set_document_access_level`` 会静默写 NULL，
上传路径 ``api/documents.py`` 也会在操作者无部门时静默落库成 "department + NULL"
（典型是平台管理员）。两个写点现已加闸（拒绝再产出退化态）；本脚本修**存量**。

修复口径
────────
目标部门 = **owner 的 ``users.department_id``**（文档归属人所在的部门）。
owner 不存在 / 无部门 → 列入「无法自动修复」并**跳过**（绝不瞎猜一个值）。

本脚本**只走唯一写点** ``knowledge_tier_service.set_document_access_level`` ——
它一次性覆盖三处事实源（file:line 证据）：

    documents 行          knowledge_tier_service.py:318–324
    document_objects 行   → security_cascade.sync_access_level  (security_cascade.py:787–794)
    Qdrant 载荷           → vector_service.update_document_access_payload (vector_service.py:480–482)

默认 **dry-run**（只出报告，一个字节都不写）。``--apply`` 才写；幂等（修完再跑 = 0 篇）。

用法（容器内）：

    # 1) 先看报告，不写任何东西（默认 dry-run）
    docker exec -w /app -e HOME=/tmp rag_backend python scripts/repair_department_acl.py

    # 2) 运维窗口确认后再写入
    docker exec -w /app -e HOME=/tmp rag_backend python scripts/repair_department_acl.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from sqlalchemy import select


def _norm(value) -> str | None:
    """None / 空串 → None，便于比较。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


async def _collect_targets():
    """找出全部退化文档，并解析各自的目标部门（owner 的部门）。"""
    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.db.user_models import User

    async with get_db_session() as session:
        docs = (
            await session.execute(
                select(Document).where(
                    Document.access_level == "department",
                    Document.department_id.is_(None),
                )
            )
        ).scalars().all()

        owner_ids = {d.owner_id for d in docs if d.owner_id is not None}
        owners: dict[str, str | None] = {}
        if owner_ids:
            for uid, dept in (
                await session.execute(
                    select(User.id, User.department_id).where(User.id.in_(owner_ids))
                )
            ).all():
                owners[str(uid)] = _norm(dept)

    fixable: list[dict] = []
    unfixable: list[dict] = []
    for d in docs:
        rec = {
            "id": str(d.id),
            "filename": d.filename or "-",
            "owner_id": str(d.owner_id) if d.owner_id else None,
        }
        if d.owner_id is None:
            rec["reason"] = "owner_id 为空（无法定位部门）"
            unfixable.append(rec)
            continue
        dept = owners.get(str(d.owner_id))
        if dept is None:
            rec["reason"] = "owner 无部门归属"
            unfixable.append(rec)
            continue
        rec["target_department_id"] = dept
        fixable.append(rec)
    return fixable, unfixable


async def _document_row_status(doc_id: str):
    from app.db.models import Document
    from app.db.postgres import get_db_session

    async with get_db_session() as session:
        doc = (
            await session.execute(
                select(Document).where(Document.id == uuid.UUID(doc_id))
            )
        ).scalar_one_or_none()
    if doc is None:
        return None
    return _norm(doc.access_level), _norm(doc.department_id)


async def _object_rows_status(doc_id: str):
    """``document_objects`` 行里 level/dept 的组合分布（判定三处是否一致）。"""
    from app.db.postgres import get_db_session
    from app.db.security_models import DocumentObject

    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(DocumentObject.access_level, DocumentObject.department_id).where(
                    DocumentObject.document_id == uuid.UUID(doc_id)
                )
            )
        ).all()
    variants: dict[tuple[str | None, str | None], int] = {}
    for level, dept in rows:
        key = (_norm(level) or "private", _norm(dept))
        variants[key] = variants.get(key, 0) + 1
    return variants


async def _payload_status(doc_id: str):
    """Qdrant 载荷里 level/dept 的组合分布（只读 scroll）。"""
    from app.config import get_settings
    from app.db.qdrant import get_qdrant_client
    from qdrant_client import models as qmodels

    settings = get_settings()
    client = get_qdrant_client()
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="document_id", match=qmodels.MatchValue(value=doc_id)
            )
        ]
    )
    variants: dict[tuple[str | None, str | None], int] = {}
    offset = None
    while True:
        pts, offset = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break
        for p in pts:
            pl = p.payload or {}
            key = (_norm(pl.get("access_level")) or "private", _norm(pl.get("department_id")))
            variants[key] = variants.get(key, 0) + 1
        if offset is None:
            break
    return variants


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正写入修复（默认只出报告，不修改任何数据）",
    )
    args = parser.parse_args()

    fixable, unfixable = await _collect_targets()

    print(f"[repair-dept] 退化文档 (department + department_id IS NULL) = {len(fixable) + len(unfixable)}")
    print(f"[repair-dept] 待修 = {len(fixable)}   无法自动修复 = {len(unfixable)}")

    for rec in fixable:
        print(
            f"    - {rec['filename']}  {rec['id']}\n"
            f"        owner={rec['owner_id']}  →  目标部门={rec['target_department_id']}"
        )
    for rec in unfixable:
        print(f"    ! {rec['filename']}  {rec['id']}  ({rec['reason']})")

    if not fixable:
        print("\n[repair-dept] 无待修文档（幂等：不变量已满足）。")
        return

    if not args.apply:
        print(
            "\n[repair-dept] DRY-RUN — 未修改任何数据。"
            "确认上面列出的文档之后再追加 --apply。"
        )
        return

    from app.services.knowledge_tier_service import (
        ACCESS_DEPARTMENT,
        set_document_access_level,
    )

    fixed = failed = 0
    for rec in fixable:
        doc_id = rec["id"]
        try:
            await set_document_access_level(
                uuid.UUID(doc_id),
                level=ACCESS_DEPARTMENT,
                department_id=rec["target_department_id"],
                actor_username="repair_department_acl",
                action="repair.department_acl",
                detail_extra="backfill department_id from owner",
            )
        except Exception as exc:      # noqa: BLE001 — 单份失败不影响其余
            failed += 1
            print(f"    !! {rec['filename']} ({doc_id}) 修复失败: {exc}")
            continue

        fixed += 1
        doc_row = await _document_row_status(doc_id)
        obj_rows = await _object_rows_status(doc_id)
        payload = await _payload_status(doc_id)
        target = rec["target_department_id"]
        print(
            f"    ok {rec['filename']} ({doc_id})\n"
            f"        documents        = {doc_row}   (want department/{target}: "
            f"{doc_row == ('department', target)})\n"
            f"        document_objects = {obj_rows}   (want single department/{target}: "
            f"{obj_rows == {('department', target): sum(obj_rows.values())}})\n"
            f"        qdrant payload   = {payload}   (want single department/{target}: "
            f"{payload == {('department', target): sum(payload.values())}})"
        )

    print(f"\n[repair-dept] done: fixed={fixed} failed={failed} / total={len(fixable)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
