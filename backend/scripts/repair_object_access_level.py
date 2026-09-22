"""
存量「对象级权限行」与文档层级追平脚本（PG 内部两处事实的分叉）.

背景：三层知识库的可见性在 PostgreSQL 里写在**两处**，而"发布 / 收回 / 转为部门
文档"历史上只改了其中一处：

    documents          access_level / department_id   ← 列表、详情、文档级判定
    document_objects   access_level / department_id   ← 第 11 / 12 环逐块 allows()、
                                                         GET /documents/{id}/chunks
                                                         的逐块复核

对象行是**入库那一刻的快照**（文档入库时必然是 private）。于是在
``security_cascade.sync_access_level`` 上线之前被发布 / 转为部门文档 / 收回过的
文档，对象行仍然停在 private：

    documents         access_level=department  d_tech
    document_objects  access_level=private     NULL      ← 陈旧

症状与 ``repair_acl_payload.py`` 那类"PG ↔ Qdrant 分叉"几乎一样，但**修的位置
不同**（那个脚本改向量载荷，本脚本改 PG 对象行；两者的症状完全重合，所以经常是
同一份文档两个都要修）：

    * 列表里看得见、点得开、文档级权限判断全对；
    * "原文预览"对**除 owner 外**的所有人返回 0 条分块；
    * 检索第 11 / 12 环把该文档的全部 chunk 丢弃 → 表现为拒答。

**owner 自测永远正常**（``allows()`` 的 ``_source_gate`` 对 owner 恒放行），
所以这个缺陷必须换账号才复现 —— 这是它长期没被发现的原因。

代码侧已在层级变更的**唯一实现点** ``set_document_access_level`` 里接入
``sync_access_level``（此后新操作不会再分叉）；**在此之前**变更过的文档仍然是
错的，本脚本负责把存量追平。

以 PostgreSQL 的 ``documents`` 行为权威来源，只改**确实分叉**的对象行。

用法（容器内）：

    # 1) 先看报告，不写任何东西（默认 dry-run）
    docker exec -e HOME=/tmp rag_backend sh -c "cd /app && python scripts/repair_object_access_level.py"

    # 2) 确认报告无误后再真正写入
    docker exec -e HOME=/tmp rag_backend sh -c "cd /app && python scripts/repair_object_access_level.py --apply"

    # 3) 只修某一份（排查用）
    docker exec -e HOME=/tmp rag_backend sh -c "cd /app && python scripts/repair_object_access_level.py --doc <document_id>"

建议在**停写窗口**执行（或至少确认此刻没有并发的层级变更）：本脚本与
``set_document_access_level`` 写的是同一批行，并发下后写者胜，可能出现"刚修完
又被一次旧请求覆盖"的竞态。脚本本身幂等，重跑无害。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

# 分叉判定：与 sync_all_access_levels 的 WHERE 子句**逐条同构** —— 报告里数出来的
# 就是修复会改的那些行。写成同构是有意的：如果两边各写一份条件，报告会与实际
# 修复量不符，而这正是运维脚本最容易失去信任的地方（"报告 3 条、改了 0 行"）。
_STALE_PREDICATE = """
   (o.access_level IS DISTINCT FROM LOWER(COALESCE(d.access_level, 'private'))
 OR o.department_id IS DISTINCT FROM (
        CASE WHEN LOWER(COALESCE(d.access_level, 'private')) = 'department'
             THEN d.department_id ELSE NULL END))
"""

_REPORT_SELECT = """
SELECT o.document_id,
       d.filename,
       o.access_level   AS obj_level,
       o.department_id  AS obj_dept,
       LOWER(COALESCE(d.access_level, 'private')) AS doc_level,
       CASE WHEN LOWER(COALESCE(d.access_level, 'private')) = 'department'
            THEN d.department_id ELSE NULL END    AS doc_dept
  FROM document_objects AS o
  JOIN documents AS d ON d.id = o.document_id
"""


async def _report(session, doc_id: uuid.UUID | None) -> list:
    """只读报告：列出**确实分叉**的对象行（不写任何东西）。"""
    from sqlalchemy import text

    where = _STALE_PREDICATE
    params: dict[str, object] = {}
    if doc_id is not None:
        where += " AND o.document_id = :doc_id"
        params["doc_id"] = doc_id
    stmt = text(_REPORT_SELECT + " WHERE " + where + " ORDER BY o.document_id")
    return (await session.execute(stmt, params)).all()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正写入修复（默认只出报告，不修改任何数据）",
    )
    parser.add_argument(
        "--doc",
        default=None,
        help="只处理指定的 document_id（UUID），默认处理全部",
    )
    args = parser.parse_args()

    from app.db.postgres import get_db_session

    doc_id: uuid.UUID | None = None
    if args.doc:
        try:
            doc_id = uuid.UUID(str(args.doc))
        except ValueError:
            print(f"[objfix] --doc 不是合法 UUID: {args.doc!r}")
            sys.exit(2)

    # ── 1. 只读报告 ─────────────────────────────────────────────────────────
    async with get_db_session() as session:
        rows = await _report(session, doc_id)

    docs = {str(r.document_id) for r in rows}
    stale_rows = len(rows)          # 一行 = 一个陈旧对象行

    print(f"[objfix] 分叉文档 = {len(docs)}，陈旧对象行 = {stale_rows}")
    if not rows:
        print(
            "\n[objfix] 没有分叉 —— 对象行与 documents 行一致，无需修复。\n"
            "          （若用户仍在报『原文预览为空 / 检索不到』，那更可能是\n"
            "           Qdrant 载荷分叉，请改跑 scripts/repair_acl_payload.py）"
        )
        return

    # 同一文档可能有多行（多个对象），报告里按文档聚合展示
    per_doc: dict[str, list] = {}
    for r in rows:
        per_doc.setdefault(str(r.document_id), []).append(r)

    for doc_key, group in list(per_doc.items())[:40]:
        head = group[0]
        print(
            f"    - {head.filename or '-'}  {doc_key}"
            f"  [陈旧行 {len(group)}]\n"
            f"        documents        = access_level={head.doc_level} dept={head.doc_dept}\n"
            f"        document_objects = access_level={head.obj_level} dept={head.obj_dept}"
            + (
                f"  (+{len(group) - 1} 行同值)"
                if len(group) > 1
                and all(r.obj_level == head.obj_level for r in group)
                else ""
            )
        )
    if len(per_doc) > 40:
        print(f"    ... 其余 {len(per_doc) - 40} 份文档省略")

    if not args.apply:
        print(
            "\n[objfix] DRY-RUN — 未修改任何数据。"
            "确认上面列出的文档之后再追加 --apply。"
        )
        return

    # ── 2. 写入（以 documents 为权威）────────────────────────────────────────
    if doc_id is not None:
        from app.db.models import Document
        from app.services.security_cascade import sync_access_level
        from sqlalchemy import select

        async with get_db_session() as session:
            doc = (
                await session.execute(select(Document).where(Document.id == doc_id))
            ).scalar_one_or_none()
            if doc is None:
                print(f"\n[objfix] documents 里没有 {doc_id}，未做任何修改。")
                sys.exit(1)
            written = await sync_access_level(
                doc_id,
                access_level=doc.access_level,
                department_id=doc.department_id,
                session=session,
            )
        print(f"\n[objfix] done: document={doc_id} 已对齐对象行 = {written}")
        return

    from app.services.security_cascade import sync_all_access_levels

    written = await sync_all_access_levels()
    print(f"\n[objfix] done: 已对齐对象行 = {written}")
    if written == 0:
        # 报告说有条目却一行没改 —— 竞态或权限问题，必须让人看见
        print("    !! 报告有分叉但 UPDATE 改写 0 行 —— 请检查并发写入或表权限")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
