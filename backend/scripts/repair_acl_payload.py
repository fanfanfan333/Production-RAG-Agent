"""
存量向量的 ACL 载荷修复脚本（PG ↔ Qdrant 分叉追平）.

覆盖**两类**分叉：

  ① 文档级 ACL（``access_level`` / ``department_id``）
     ``POST /upload`` 是异步的（受理即返回 202，管线丢后台），上传那一刻
     ``access_level`` 就固定成 private。而"传完立刻点发布"是正常操作路径 ——
     发布走 Qdrant ``set_payload`` 按 ``document_id`` 更新，**此时向量点往往还没
     写入**，于是更新匹配 0 个点并静默成功。随后入库又用上传时刻的快照 private
     把点写进去，两份事实永久分叉：

         PostgreSQL  access_level=department
         Qdrant      access_level=private

     症状极具误导性：**列表里看得见、点得开，提问时却谁都检索不到**（owner 除外，
     因为 private 对 owner 本来就放行）。修复走
     ``resync_document_acl_payload``（入库收尾同一路径）。

  ② 对象级 ACL 物化（``object_id`` / ``visibility_mode`` / ``security_level`` /
     ``acl_sync_state`` / ``excluded`` / ``acl_allow`` / ``effective_security_level``）
     ``build_object_rows`` 为五种对象（doc / text_chunk / table / code / image）各建
     ``document_objects`` 行，并把权限字段冗余进载荷供第 11 环 ``allows()`` 读。但
     **image 行**曾经漏写 ``_payload`` ⇒ ``push_payload_async`` 逐行跳过 ⇒ image 点的
     Qdrant 载荷**完全缺**这 7 个字段 ⇒ ``ObjectACLView.from_payload`` 拿不到
     ``object_id`` ⇒ ``allows()`` 直接 ``missing_object_view`` ⇒ **所有账号都检索不到
     图片**（能力丧失，不是越权）。

     这类分叉**①的判定看不见**：image 点的 ``access_level`` / ``department_id`` 恰恰
     来自 ``upsert_vectors``、与 PG 一致 ⇒ 旧的逐项比对判它"没分叉"、直接跳过。
     本脚本②按 ``document_id`` 调 ``materialize_document_objects`` 重新物化并推载荷。

以 PostgreSQL 为权威来源；默认 **dry-run**（只出报告，一个字节都不写）。
``--apply`` 才写：① 逐份 ``resync_document_acl_payload``；② 逐份
``materialize_document_objects``；随后重新扫描打印 before/after 覆盖表。

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
import uuid
from collections import defaultdict

from sqlalchemy import select

#: 对象级 ACL 物化字段（第 11 环 ``allows()`` 读的全部关键字段；缺任一个都可能误判）。
OBJECT_MATERIALIZED_KEYS = (
    "object_id",
    "visibility_mode",
    "security_level",
    "acl_sync_state",
    "excluded",
    "acl_allow",
    "effective_security_level",
)


def _norm(value) -> str | None:
    """把 payload / PG 里可能出现的 None、空串统一成 None，便于比较。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bucket(content_type) -> str:
    """载荷 ``content_type`` → 报告分桶（image / table / code / text / other）。"""
    ctype = _norm(content_type) or "text"
    if ctype in ("image", "table", "code", "text"):
        return ctype
    return "other"


async def _scan(client, collection: str):
    """滚动整库一次：返回 (每文档点列表, 覆盖计数, 总点数).

    每文档点列表 = ``[(point_id, payload), ...]``；覆盖计数 =
    ``{bucket: {"total": n, "has_object_id": m}}``。
    """
    per_doc_points: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    coverage: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "has_object_id": 0}
    )
    scanned = 0

    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
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
            pid = str(p.id)
            doc_id = _norm(payload.get("document_id"))
            if doc_id is not None:
                per_doc_points[doc_id].append((pid, payload))
            bucket = _bucket(payload.get("content_type"))
            coverage[bucket]["total"] += 1
            if _norm(payload.get("object_id")):
                coverage[bucket]["has_object_id"] += 1
        if offset is None:
            break

    return per_doc_points, coverage, scanned


def _print_coverage(title: str, coverage: dict[str, dict[str, int]]) -> None:
    print(f"\n[repair] ── {title} ──")
    print(f"    {'content_type':<14}{'points':>8}{'has_object_id':>16}{'missing':>10}")
    for bucket in ("image", "table", "code", "text", "other"):
        stat = coverage.get(bucket)
        if not stat or not stat["total"]:
            continue
        missing = stat["total"] - stat["has_object_id"]
        print(
            f"    {bucket:<14}{stat['total']:>8}{stat['has_object_id']:>16}"
            f"{missing:>10}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正写入修复（默认只出报告，不修改任何数据）",
    )
    parser.add_argument(
        "--level-only",
        action="store_true",
        help="只修①文档级 ACL，跳过②对象级物化（默认两者都修）",
    )
    args = parser.parse_args()

    from app.config import get_settings
    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.db.qdrant import get_qdrant_client

    settings = get_settings()
    client = get_qdrant_client()

    # ── 1. PG 权威映射：document_id → (access_level, department_id, filename) ──
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

    # ── 2. 滚动向量库：每文档点 + 对象级覆盖 ──────────────────────────────────
    per_doc_points, coverage, scanned = await _scan(
        client, settings.QDRANT_COLLECTION
    )
    print(f"[repair] Qdrant points scanned = {scanned}, documents = {len(per_doc_points)}")

    # ── 3. ① 文档级分叉判定（access_level / department_id）────────────────────
    observed: dict[str, set[tuple[str | None, str | None]]] = defaultdict(set)
    point_counts: dict[str, int] = defaultdict(int)
    for doc_id, pts in per_doc_points.items():
        for _pid, payload in pts:
            point_counts[doc_id] += 1
            level = _norm(payload.get("access_level")) or "private"
            dept = _norm(payload.get("department_id"))
            observed[doc_id].add((level, dept if level == "department" else None))

    divergent: list[str] = []
    for doc_id, variants in observed.items():
        want = truth.get(doc_id)
        if want is None:
            continue          # 孤儿向量（PG 无此文档）—— 不在本脚本职责内
        if variants != {want}:
            divergent.append(doc_id)

    orphans = sorted(set(observed) - set(truth))
    no_vectors = sorted(set(truth) - set(observed))

    print(f"[repair] ① divergent (level: PG != Qdrant) = {len(divergent)}")
    print(f"[repair]    orphan vectors (no PG row) = {len(orphans)}")
    print(f"[repair]    PG docs with no vectors     = {len(no_vectors)}")

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

    # ── 4. ② 对象级覆盖（object_id 等物化字段）────────────────────────────────
    _print_coverage("对象级载荷覆盖（before）", coverage)

    backfill_docs: list[str] = []      # 有任一点缺 object_id、且 PG 有权威行的文档
    image_missing_total = 0
    skipped_orphans = 0
    for doc_id, pts in per_doc_points.items():
        missing_buckets: dict[str, int] = defaultdict(int)
        for _pid, payload in pts:
            if not _norm(payload.get("object_id")):
                missing_buckets[_bucket(payload.get("content_type"))] += 1
        if not missing_buckets:
            continue
        if doc_id not in truth:
            # 孤儿向量（Qdrant 有点、PG 无 documents 行）：无权威源、不可物化 ——
            # **不是失败**。单列计数，绝不并入 obj_failed 触发假红退出码。
            skipped_orphans += 1
            continue
        backfill_docs.append(doc_id)
        image_missing_total += missing_buckets.get("image", 0)

    backfill_docs.sort()
    print(
        f"[repair] ② 待补文档数 = {len(backfill_docs)}   待补 image 点数 = "
        f"{image_missing_total}"
    )
    print(f"[repair] ② 跳过孤儿向量（PG 无行） = {skipped_orphans}")
    for doc_id in backfill_docs[:50]:
        missing_buckets: dict[str, int] = defaultdict(int)
        for _pid, payload in per_doc_points[doc_id]:
            if not _norm(payload.get("object_id")):
                missing_buckets[_bucket(payload.get("content_type"))] += 1
        detail = ", ".join(
            f"{b}={n}" for b, n in sorted(missing_buckets.items())
        )
        print(f"    - {names.get(doc_id)}  {doc_id}  ({detail})")
    if len(backfill_docs) > 50:
        print(f"    ... 其余 {len(backfill_docs) - 50} 份省略")

    if not args.apply:
        print(
            "\n[repair] DRY-RUN — 未修改任何数据。"
            "确认上面列出的文档之后再追加 --apply。"
        )
        return

    # ── 5. 写入 ───────────────────────────────────────────────────────────────
    # ① 文档级：逐份 resync（以 PG 为权威）
    if divergent:
        from app.services.knowledge_tier_service import resync_document_acl_payload

        fixed = failed = 0
        for doc_id in divergent:
            try:
                applied = await resync_document_acl_payload(
                    doc_id, reason="repair_script", expected=None
                )
            except Exception as exc:      # noqa: BLE001 — 单份失败不影响其余
                failed += 1
                print(f"    !! {names.get(doc_id)} ({doc_id}) ①修复失败: {exc}")
                continue
            if applied:
                fixed += 1
            else:
                failed += 1
                print(f"    !! {names.get(doc_id)} ({doc_id}) 没匹配到向量点")
        print(f"\n[repair] ① done: fixed={fixed} failed={failed} / divergent={len(divergent)}")

    # ② 对象级：逐份重新物化并推载荷（本脚本旧版**做不到** —— 只比 level/dept）
    if backfill_docs and not args.level_only:
        from app.services.security_cascade import materialize_document_objects

        obj_fixed = obj_failed = 0
        for doc_id in backfill_docs:
            try:
                doc_uuid = uuid.UUID(doc_id)
            except ValueError:
                obj_failed += 1
                continue
            async with get_db_session() as session:
                doc = (
                    await session.execute(
                        select(Document).where(Document.id == doc_uuid)
                    )
                ).scalar_one_or_none()
            if doc is None:
                obj_failed += 1
                print(f"    !! {names.get(doc_id)} ({doc_id}) ②PG 已无此行，跳过")
                continue
            points = [
                {"id": pid, "payload": payload}
                for pid, payload in per_doc_points[doc_id]
            ]
            try:
                stats = await materialize_document_objects(doc, points)
                obj_fixed += 1
                print(
                    f"    ok {names.get(doc_id)} ({doc_id}) ②物化 {stats}"
                )
            except Exception as exc:      # noqa: BLE001 — 单份失败不影响其余
                obj_failed += 1
                print(f"    !! {names.get(doc_id)} ({doc_id}) ②物化失败: {exc}")
        print(
            f"\n[repair] ② done: fixed={obj_fixed} failed={obj_failed} / "
            f"backfill={len(backfill_docs)}"
        )

    # ── 6. 复核：重新扫描打印覆盖表（对齐 before/after）────────────────────────
    _per_doc2, coverage_after, scanned_after = await _scan(
        client, settings.QDRANT_COLLECTION
    )
    _print_coverage("对象级载荷覆盖（after）", coverage_after)

    if divergent and failed:
        sys.exit(1)
    if backfill_docs and not args.level_only and obj_failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
