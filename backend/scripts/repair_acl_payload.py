"""
存量向量的 ACL 载荷修复脚本（PG ↔ Qdrant 分叉追平）.

背景：``POST /upload`` 是异步的（受理即返回 202，管线丢后台），上传那一刻
``access_level`` 就固定成 private。而"传完立刻点发布"是正常操作路径 —— 发布走
Qdrant ``set_payload`` 按 ``document_id`` 更新，**此时向量点往往还没写入**，
于是更新匹配 0 个点并静默成功。随后入库又用上传时刻的快照 private 把点写进去，
两份事实永久分叉：

    PostgreSQL  access_level=department
    Qdrant      access_level=private

症状极具误导性：**列表里看得见、点得开，提问时却谁都检索不到**（owner 除外，
因为 private 对 owner 本来就放行）。

代码侧已在**入库收尾**加了 ``resync_document_acl_payload`` 追平（此后新入库的
文档不会再分叉），但**在此之前入库的存量文档仍然是错的** —— 本脚本负责修它们。

以 PostgreSQL 为权威来源，只改**确实分叉**的文档（一致的文档一个字节都不动）。

用法（容器内）：

    # 1) 先看报告，不写任何东西（默认 dry-run）
    docker exec -e HOME=/tmp rag_backend sh -c "cd /app && python scripts/repair_acl_payload.py"

    # 2) 确认报告无误后再真正写入
    docker exec -e HOME=/tmp rag_backend sh -c "cd /app && python scripts/repair_acl_payload.py --apply"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict

from sqlalchemy import select


def _norm(value) -> str | None:
    """把 payload / PG 里可能出现的 None、空串统一成 None，便于比较。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正写入修复（默认只出报告，不修改任何数据）",
    )
    args = parser.parse_args()

    from app.config import get_settings
    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.db.qdrant import get_qdrant_client

    settings = get_settings()
    client = get_qdrant_client()

    # ── 1. PG 权威映射：document_id → (access_level, department_id) ─────────
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(
                    Document.id,
                    Document.access_level,
                    Document.department_id,
                    Document.filename,
                )
            )
        ).all()

    truth: dict[str, tuple[str | None, str | None]] = {}
    names: dict[str, str] = {}
    for doc_id, level, dept, filename in rows:
        key = str(doc_id)
        level = _norm(level) or "private"
        # 与 set_document_access_level 同口径：只有部门库保留部门归属
        truth[key] = (level, _norm(dept) if level == "department" else None)
        names[key] = filename or "-"

    print(f"[repair] PG documents = {len(truth)}")

    # ── 2. 滚动向量库，聚合每个 document_id 上实际生效的 ACL ────────────────
    observed: dict[str, set[tuple[str | None, str | None]]] = defaultdict(set)
    point_counts: dict[str, int] = defaultdict(int)

    offset = None
    scanned = 0
    while True:
        points, offset = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        scanned += len(points)
        for p in points:
            payload = p.payload or {}
            doc_id = _norm(payload.get("document_id"))
            if doc_id is None:
                continue
            point_counts[doc_id] += 1
            level = _norm(payload.get("access_level")) or "private"
            dept = _norm(payload.get("department_id"))
            observed[doc_id].add((level, dept if level == "department" else None))
        if offset is None:
            break

    print(f"[repair] Qdrant points scanned = {scanned}, documents = {len(observed)}")

    # ── 3. 分叉判定 ─────────────────────────────────────────────────────────
    divergent: list[str] = []
    for doc_id, variants in observed.items():
        want = truth.get(doc_id)
        if want is None:
            continue          # 孤儿向量（PG 无此文档）—— 不在本脚本职责内
        if variants != {want}:
            divergent.append(doc_id)

    orphans = sorted(set(observed) - set(truth))
    no_vectors = sorted(set(truth) - set(observed))

    print(f"[repair] divergent (PG != Qdrant) = {len(divergent)}")
    print(f"[repair] orphan vectors (no PG row) = {len(orphans)}")
    print(f"[repair] PG docs with no vectors     = {len(no_vectors)}")

    for doc_id in divergent[:20]:
        pg = truth[doc_id]
        qd = sorted(observed[doc_id], key=str)
        print(
            f"    - {names.get(doc_id)}  {doc_id}\n"
            f"        PG     = access_level={pg[0]} dept={pg[1]}\n"
            f"        Qdrant = {qd}  (points={point_counts[doc_id]})"
        )
    if len(divergent) > 20:
        print(f"    ... 其余 {len(divergent) - 20} 份省略")

    if not args.apply:
        print(
            "\n[repair] DRY-RUN — 未修改任何数据。"
            "确认上面列出的文档之后再追加 --apply。"
        )
        return

    if not divergent:
        print("\n[repair] 无分叉，无需修复。")
        return

    # ── 4. 逐份追平（以 PG 为权威）──────────────────────────────────────────
    from app.services.knowledge_tier_service import resync_document_acl_payload

    fixed = failed = 0
    for doc_id in divergent:
        try:
            applied = await resync_document_acl_payload(
                doc_id, reason="repair_script", expected=None
            )
        except Exception as exc:      # noqa: BLE001 — 单份失败不影响其余
            failed += 1
            print(f"    !! {names.get(doc_id)} ({doc_id}) 修复失败: {exc}")
            continue
        if applied:
            fixed += 1
        else:
            failed += 1
            print(f"    !! {names.get(doc_id)} ({doc_id}) 没匹配到向量点")

    print(f"\n[repair] done: fixed={fixed} failed={failed} / divergent={len(divergent)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
