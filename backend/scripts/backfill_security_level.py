"""存量密级 / 对象级权限行的**幂等**回填脚本（T1，§8.4）.

本脚本只做三件事，且**每一步都保证存量行为零变化**：

    1. 确认 ``documents`` 的新列默认值已经生效（security_level=1 /
       visibility_mode='tier' / project_ids=acl_allow=acl_deny='[]'）
    2. 为每份文档物化 ``document_objects`` 行：
           1 行 doc  +  N 行 text_chunk/table/code  +  M 行 image
       OCR 派生块（``image_id`` 非空且 ``content_type != 'image'``）的
       ``parent_object_id`` = **源图片的 object_id**（不是文档），
       ``effective_security_level`` = ``max(父文档, 源图片)``（决策 12）
    3. 把新字段推到 Qdrant payload（副本；缺字段的老向量在检索侧 fail-open，
       推了也只会更准，不会更宽松）

为什么不在 alembic 迁移里做：迁移**只加列、不写业务规则**（沿用上游决策 5 的
纪律）。物化需要按 Qdrant 的实际分块结果逐块生成，那是业务规则。

幂等性
──────
``object_id`` 是主键 + ``ON CONFLICT (object_id) DO UPDATE``（只更新权限字段，
不动定位字段）。连跑两次 ⇒ 第二次 ``inserted=0 / updated=N``，不产生重复行。

用法（容器内）
──────────────
    # 1) 先看计划（默认 dry-run，不写任何东西）
    docker exec -u root -e HOME=/tmp rag_backend \
        sh -c "cd /app && python scripts/backfill_security_level.py"

    # 2) 确认后再真正写入
    docker exec -u root -e HOME=/tmp rag_backend \
        sh -c "cd /app && python scripts/backfill_security_level.py --apply"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# 允许「容器内直接 `python scripts/backfill_security_level.py`」也能找到 app 包
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


# ── 权限字段（ON CONFLICT 时只更新这些；定位字段不动）──────────────────────────
_PERMISSION_FIELDS = (
    "object_type",
    "parent_object_id",
    "inherited_from",
    "inherited_at",
    "tenant_id",
    "owner_id",
    "department_id",
    "access_level",
    "visibility_mode",
    "project_ids",
    "visible_scope",
    "security_level",
    "parent_security_level",
    "effective_security_level",
    "acl_allow",
    "acl_deny",
    "acl_expires_at",
    "acl_sync_state",
    "excluded",
    "share_status",
    "share_grant_scope",
    "updated_at",
)


def _norm(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def build_object_rows(
    doc: dict,
    points: list[dict],
    *,
    default_level: int,
    now: datetime,
) -> tuple[list[dict], dict]:
    """
    一份文档 → ``document_objects`` 行（纯函数，便于单测与重复运行）.

    ``points`` 是 Qdrant 里属于该文档的向量块（``{"id":..., "payload":{...}}``）。

    返回 ``(rows, stats)``。stats 里的 ``chunk_index_collisions`` 是
    ``(document_id, chunk_index)`` 上有重复的分块数 —— ``uq_dobj_doc_chunk``
    是唯一索引，重复会被数据库拒；这里**保留第一条**并记数，让问题可见而不是
    让整批插入失败。
    """
    from app.db.security_models import (
        OBJECT_TYPE_DOC,
        OBJECT_TYPE_IMAGE,
        make_object_id,
        object_type_from_content_type,
    )
    from app.services.security_policy import P_ACL_EXPIRES_AT_TS

    doc_id = str(doc["id"])
    doc_level = doc.get("security_level")
    doc_level = int(doc_level) if doc_level is not None else int(default_level)
    doc_tenant = doc.get("tenant_id") or "default"
    doc_acl_allow = _as_list(doc.get("acl_allow"))
    doc_acl_deny = _as_list(doc.get("acl_deny"))
    doc_projects = _as_list(doc.get("project_ids"))
    doc_visibility = doc.get("visibility_mode") or "tier"

    def _base_row(object_id: str, object_type: str) -> dict:
        return {
            "object_id": object_id,
            "document_id": doc["id"],
            "object_type": object_type,
            "parent_object_id": None,
            "inherited_from": None,
            "inherited_at": now,
            "tenant_id": doc_tenant,
            "owner_id": doc.get("owner_id"),
            "department_id": doc.get("department_id"),
            "access_level": doc.get("access_level") or "private",
            "visibility_mode": doc_visibility,
            "project_ids": doc_projects,
            "visible_scope": None,
            "security_level": doc_level,
            "parent_security_level": None,
            "effective_security_level": doc_level,
            "acl_allow": doc_acl_allow,
            "acl_deny": doc_acl_deny,
            "acl_expires_at": doc.get("acl_expires_at"),
            "acl_sync_state": "synced",
            "excluded": False,
            "share_status": doc.get("share_status") or "none",
            "share_grant_scope": doc.get("share_grant_scope"),
            "chunk_index": None,
            "page_number": None,
            "image_id": None,
            "image_path": None,
            "content_type": None,
            "created_at": now,
            "updated_at": now,
        }

    rows: list[dict] = []
    stats: dict = {
        "doc": 0, "chunk": 0, "image": 0, "derived": 0,
        "chunk_index_collisions": 0,
    }

    # ── 1. doc 镜像行 ─────────────────────────────────────────────────────────
    doc_row = _base_row(doc_id, OBJECT_TYPE_DOC)
    doc_row["parent_security_level"] = None
    rows.append(doc_row)
    stats["doc"] += 1

    # ── 2. 图片对象（同一 image_id 只建一行；一图多块时后者合并）─────────────────
    image_rows: dict[str, dict] = {}
    for p in points:
        payload = p.get("payload") or {}
        image_id = _norm(payload.get("image_id"))
        if not image_id:
            continue
        content_type = _norm(payload.get("content_type")) or "text"
        if content_type == "image":
            obj_id = make_object_id(doc_id, image_id, object_type=OBJECT_TYPE_IMAGE)
            if obj_id in image_rows:
                continue
            row = _base_row(obj_id, OBJECT_TYPE_IMAGE)
            row.update(
                parent_object_id=doc_id,
                inherited_from=doc_id,
                parent_security_level=doc_level,
                image_id=image_id,
                image_path=_norm(payload.get("image_path")),
                content_type=content_type,
                page_number=payload.get("page_number"),
            )
            image_rows[obj_id] = row
            rows.append(row)
            stats["image"] += 1
        else:
            # OCR 派生块也意味着存在一张源图片 —— 为它补一行（幂等：已存在则跳过）
            obj_id = make_object_id(doc_id, image_id, object_type=OBJECT_TYPE_IMAGE)
            if obj_id not in image_rows:
                row = _base_row(obj_id, OBJECT_TYPE_IMAGE)
                row.update(
                    parent_object_id=doc_id,
                    inherited_from=doc_id,
                    parent_security_level=doc_level,
                    image_id=image_id,
                    image_path=_norm(payload.get("image_path")),
                    content_type="image",
                    page_number=payload.get("page_number"),
                )
                image_rows[obj_id] = row
                rows.append(row)
                stats["image"] += 1

    # ── 3. 分块对象（含 OCR 派生）──────────────────────────────────────────────
    seen_chunk_index: dict[tuple[str, int], str] = {}
    for p in points:
        payload = p.get("payload") or {}
        point_id = str(p.get("id"))
        content_type = _norm(payload.get("content_type")) or "text"
        image_id = _norm(payload.get("image_id"))

        if content_type == "image" and image_id:
            # 图片对象本体已在第 2 步建过（它的向量块就是这张图，不另建 chunk 行）
            continue

        object_type = object_type_from_content_type(content_type)
        obj_id = make_object_id(doc_id, point_id, object_type=object_type)
        row = _base_row(obj_id, object_type)
        row.update(
            chunk_index=payload.get("chunk_index"),
            page_number=payload.get("page_number"),
            image_id=image_id,
            image_path=_norm(payload.get("image_path")),
            content_type=content_type,
        )

        if image_id:
            # ── OCR 派生：父 = **源图片**（不是文档），有效密级取 max ──────────
            src_obj_id = make_object_id(doc_id, image_id, object_type=OBJECT_TYPE_IMAGE)
            src = image_rows.get(src_obj_id)
            src_eff = int(src["effective_security_level"]) if src else doc_level
            row.update(
                parent_object_id=src_obj_id,
                inherited_from=src_obj_id,
                parent_security_level=src_eff,
                effective_security_level=max(doc_level, src_eff, int(row["security_level"])),
                # 派生对象**不得**通过 acl_allow 获得父之外的可见性（PRD 3.2）
                acl_allow=[],
                acl_deny=doc_acl_deny,
                visible_scope="project" if doc_visibility == "project" else None,
            )
            stats["derived"] += 1
        else:
            row.update(
                parent_object_id=doc_id,
                inherited_from=doc_id,
                parent_security_level=doc_level,
                effective_security_level=max(doc_level, int(row["security_level"])),
            )
            stats["chunk"] += 1

        # (document_id, chunk_index) 唯一索引的碰撞保护
        ci = row.get("chunk_index")
        if ci is not None:
            key = (doc_id, int(ci))
            if key in seen_chunk_index:
                stats["chunk_index_collisions"] += 1
                row["chunk_index"] = None      # 让唯一索引放行（NULL 不进索引）
            else:
                seen_chunk_index[key] = obj_id

        # payload 推送用（仅内部传递，不落库）
        row["_point_id"] = point_id
        row["_payload"] = {
            "object_id": obj_id,
            "object_type": object_type,
            "parent_object_id": row["parent_object_id"],
            "visibility_mode": row["visibility_mode"],
            "project_ids": sorted(row["project_ids"]),
            "security_level": row["security_level"],
            "parent_security_level": row["parent_security_level"],
            "effective_security_level": row["effective_security_level"],
            "acl_allow": sorted(row["acl_allow"]),
            "acl_deny": sorted(row["acl_deny"]),
            "acl_expires_at": row["acl_expires_at"].isoformat() if row["acl_expires_at"] else None,
            P_ACL_EXPIRES_AT_TS: (
                row["acl_expires_at"].timestamp() if row["acl_expires_at"] else None
            ),
            "excluded": row["excluded"],
            "acl_sync_state": row["acl_sync_state"],
            "share_status": row["share_status"],
        }
        rows.append(row)

    return rows, stats


async def _load_documents(*, only_completed: bool) -> list[dict]:
    from sqlalchemy import select

    from app.db.models import Document, DocumentStatus
    from app.db.postgres import get_db_session

    stmt = select(
        Document.id,
        Document.tenant_id,
        Document.owner_id,
        Document.department_id,
        Document.access_level,
        Document.security_level,
        Document.visibility_mode,
        Document.project_ids,
        Document.acl_allow,
        Document.acl_deny,
        Document.acl_expires_at,
        Document.share_status,
        Document.share_grant_scope,
        Document.status,
    )
    if only_completed:
        stmt = stmt.where(Document.status == DocumentStatus.COMPLETED)

    async with get_db_session() as session:
        result = await session.execute(stmt)
        cols = [
            "id", "tenant_id", "owner_id", "department_id", "access_level",
            "security_level", "visibility_mode", "project_ids", "acl_allow",
            "acl_deny", "acl_expires_at", "share_status", "share_grant_scope", "status",
        ]
        return [dict(zip(cols, row)) for row in result.all()]


async def _scroll_points_by_document(collection: str) -> dict[str, list[dict]]:
    from app.db.qdrant import get_qdrant_client

    client = get_qdrant_client()
    by_doc: dict[str, list[dict]] = defaultdict(list)
    offset = None
    scanned = 0
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
            doc_id = _norm(payload.get("document_id"))
            if doc_id is None:
                continue
            by_doc[doc_id].append({"id": p.id, "payload": payload})
        if offset is None:
            break
    print(f"[backfill] Qdrant 扫描点数 = {scanned}，覆盖文档 = {len(by_doc)}")
    return by_doc


async def _apply_rows(rows: list[dict]) -> tuple[int, int]:
    """幂等 upsert（ON CONFLICT (object_id) DO UPDATE 只更新权限字段）。"""
    from sqlalchemy.dialects.postgresql import insert

    from app.db.postgres import get_db_session
    from app.db.security_models import DocumentObject

    inserted = updated = 0
    async with get_db_session() as session:
        for row in rows:
            payload_row = {k: v for k, v in row.items() if not k.startswith("_")}
            stmt = insert(DocumentObject).values(**payload_row)
            stmt = stmt.on_conflict_do_update(
                index_elements=["object_id"],
                set_={k: getattr(stmt.excluded, k) for k in _PERMISSION_FIELDS},
                where=None,
            )
            result = await session.execute(stmt)
            # SQLAlchemy 对 PG 的 ON CONFLICT 不提供稳定的 inserted/updated 区分，
            # 用 rowcount 近似：1 = 写入或更新成功
            if result.rowcount:
                inserted += 1
    # 幂等语义下统一按"已确保存在"计数；真正的 inserted/updated 拆分由第二次
    # 运行的对象总数不变来证明（见主流程的 before/after 对比）。
    return inserted, updated


async def _push_payloads(rows: list[dict], collection: str, *, batch: int) -> int:
    """把权限字段推到 Qdrant payload（副本）。按 payload 分组批量写。"""
    from app.db.qdrant import get_qdrant_client

    client = get_qdrant_client()
    groups: dict[str, tuple[dict, list]] = {}
    for row in rows:
        payload = row.get("_payload")
        point_id = row.get("_point_id")
        if not payload or not point_id:
            continue
        import json

        key = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        entry = groups.setdefault(key, (payload, []))
        entry[1].append(point_id)

    pushed = 0
    for _key, (payload, ids) in groups.items():
        for start in range(0, len(ids), batch):
            chunk_ids = ids[start:start + batch]
            try:
                await client.set_payload(
                    collection_name=collection,
                    payload=payload,
                    points=chunk_ids,
                )
                pushed += len(chunk_ids)
            except Exception as exc:      # noqa: BLE001 — 单批失败不影响其余
                print(f"    !! set_payload 失败（{len(chunk_ids)} 点）: {exc}")
    return pushed


async def _count_objects() -> int:
    from sqlalchemy import func, select

    from app.db.postgres import get_db_session
    from app.db.security_models import DocumentObject

    async with get_db_session() as session:
        return int(
            (await session.execute(
                select(func.count()).select_from(DocumentObject)
            )).scalar_one()
        )


async def _count_null_effective() -> int:
    from sqlalchemy import func, select

    from app.db.postgres import get_db_session
    from app.db.security_models import DocumentObject

    async with get_db_session() as session:
        return int(
            (await session.execute(
                select(func.count()).select_from(DocumentObject).where(
                    DocumentObject.effective_security_level.is_(None)
                )
            )).scalar_one()
        )


async def _check_document_defaults() -> dict:
    """确认迁移的 server_default 已生效（存量行为零变化的前提）。"""
    from sqlalchemy import func, select

    from app.db.models import Document
    from app.db.postgres import get_db_session

    async with get_db_session() as session:
        total = int((await session.execute(
            select(func.count()).select_from(Document))).scalar_one())
        null_level = int((await session.execute(
            select(func.count()).select_from(Document).where(
                Document.security_level.is_(None)))).scalar_one())
        null_vis = int((await session.execute(
            select(func.count()).select_from(Document).where(
                Document.visibility_mode.is_(None)))).scalar_one())
        not_tier = int((await session.execute(
            select(func.count()).select_from(Document).where(
                Document.visibility_mode != "tier"))).scalar_one())
    return {
        "documents": total,
        "security_level_is_null": null_level,
        "visibility_mode_is_null": null_vis,
        "visibility_mode_not_tier": not_tier,
    }


async def _run(args: argparse.Namespace) -> int:
    from app.config import get_settings
    from app.db.security_models import DEFAULT_SECURITY_LEVEL

    settings = get_settings()
    collection = settings.QDRANT_COLLECTION
    now = datetime.now(timezone.utc)

    if settings.DEFAULT_SECURITY_LEVEL != DEFAULT_SECURITY_LEVEL:
        print(
            f"[backfill] ✗ settings.DEFAULT_SECURITY_LEVEL="
            f"{settings.DEFAULT_SECURITY_LEVEL} 与 security_models."
            f"DEFAULT_SECURITY_LEVEL={DEFAULT_SECURITY_LEVEL} 不一致 —— 拒绝运行"
        )
        return 2

    print("=" * 72)
    print(f"[backfill] 模式 = {'APPLY（写库）' if args.apply else 'DRY-RUN（不写任何东西）'}")
    print(f"[backfill] 集合 = {collection}　默认密级 = {DEFAULT_SECURITY_LEVEL}")

    # ── 1. 列默认值确认 ───────────────────────────────────────────────────────
    defaults = await _check_document_defaults()
    print(f"[backfill] documents 总数 = {defaults['documents']}")
    print(f"[backfill] security_level IS NULL       = {defaults['security_level_is_null']}（应为 0）")
    print(f"[backfill] visibility_mode IS NULL      = {defaults['visibility_mode_is_null']}（应为 0）")
    print(f"[backfill] visibility_mode != 'tier'    = {defaults['visibility_mode_not_tier']}（应为 0）")
    if defaults["security_level_is_null"] or defaults["visibility_mode_is_null"]:
        print("[backfill] ✗ 迁移未生效（存在 NULL）—— 请先跑 alembic upgrade head")
        return 2

    # ── 2. 载入文档与向量 ─────────────────────────────────────────────────────
    docs = await _load_documents(only_completed=not args.all_status)
    by_doc = await _scroll_points_by_document(collection)
    print(f"[backfill] 参与回填的文档 = {len(docs)}（only_completed={not args.all_status}）")

    # ── 3. 物化行 ─────────────────────────────────────────────────────────────
    all_rows: list[dict] = []
    totals: dict = defaultdict(int)
    collision_docs: list[str] = []
    for doc in docs:
        points = by_doc.get(str(doc["id"]), [])
        rows, stats = build_object_rows(
            doc, points, default_level=DEFAULT_SECURITY_LEVEL, now=now,
        )
        all_rows.extend(rows)
        for k, v in stats.items():
            totals[k] += v
        if stats["chunk_index_collisions"]:
            collision_docs.append(f"{doc['id']}（{stats['chunk_index_collisions']} 处）")

    print(
        f"[backfill] 将生成 document_objects 行 = {len(all_rows)}　"
        f"doc={totals['doc']} chunk={totals['chunk']} image={totals['image']} "
        f"derived={totals['derived']}"
    )
    if totals["chunk_index_collisions"]:
        print(
            f"[backfill] ⚠ (document_id, chunk_index) 重复 {totals['chunk_index_collisions']} 处，"
            f"已把这些行的 chunk_index 置 NULL 以避开唯一索引："
        )
        for item in collision_docs[:10]:
            print(f"    - {item}")
        if len(collision_docs) > 10:
            print(f"    ... 其余 {len(collision_docs) - 10} 份省略")

    points_with_payload = sum(1 for r in all_rows if r.get("_point_id"))
    print(f"[backfill] 将推送 payload 的向量点 = {points_with_payload}")

    if not args.apply:
        print("\n[backfill] DRY-RUN —— 未写入任何数据。确认后再追加 --apply。")
        return 0

    # ── 4. 写库（幂等）────────────────────────────────────────────────────────
    before = await _count_objects()
    written, _ = await _apply_rows(all_rows)
    after = await _count_objects()
    print(f"[backfill] document_objects 行数 before={before} after={after}（写入/更新 {written} 行）")
    print(f"[backfill] 新增行数 = {after - before}（第二次运行应 ≈ 0 ⇒ 幂等）")

    # ── 5. 推 payload ─────────────────────────────────────────────────────────
    if args.push_payload:
        pushed = await _push_payloads(all_rows, collection, batch=args.batch)
        print(f"[backfill] 已推送 payload 的向量点 = {pushed}")
    else:
        print("[backfill] 跳过 payload 推送（--no-push-payload）")

    # ── 6. 严格模式提示（§8.4 要求的结束语）───────────────────────────────────
    nulls = await _count_null_effective()
    print("\n" + "=" * 72)
    print("[backfill] 回填完成。")
    print(
        "若要将密级前置过滤切换为严格白名单，请先确认 "
        f"`effective_security_level IS NULL` 的对象数为 0（当前 = {nulls}），"
        "再设置 ACL_SECURITY_PREFILTER_STRICT=true。"
    )
    print("=" * 72)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="存量密级 / 对象级权限行回填（幂等，默认 dry-run）",
    )
    ap.add_argument("--apply", action="store_true", help="真正写库（默认只出计划）")
    ap.add_argument("--dry-run", action="store_true", help="显式 dry-run（默认行为）")
    ap.add_argument(
        "--all-status", action="store_true",
        help="包含非 COMPLETED 的文档（默认只处理 COMPLETED）",
    )
    ap.add_argument(
        "--no-push-payload", dest="push_payload", action="store_false",
        help="只写 PG，不推 Qdrant payload",
    )
    ap.add_argument("--batch", type=int, default=256, help="payload 推送批大小")
    args = ap.parse_args()
    apply_flag = args.apply and not args.dry_run
    args.apply = apply_flag
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
