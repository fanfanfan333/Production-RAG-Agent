"""PG ↔ Qdrant payload 一致性校验（QA 用，**非阻断**）.

PG 是权限权威源、payload 是副本。两者允许**短暂不一致**（收紧同步 / 放宽异步），
但不允许**长期静默分叉** —— 那正是上一轮 ``repair_acl_payload.py`` 修的那类
"列表看得见、检索搜不到"的病。

本脚本只**报告**，不修复（修复走 ``scripts/backfill_security_level.py`` 或
T4 的 ``security_cascade.push_payload_async``）。

比对字段（权限相关，定位字段不比）：
    effective_security_level / security_level / visibility_mode /
    project_ids / acl_allow / acl_deny / excluded / acl_sync_state

用法（容器内）：

    docker exec -u root -e HOME=/tmp rag_backend \
        sh -c "cd /app && python scripts/verify_acl_payload_sync.py"
    docker exec -u root -e HOME=/tmp rag_backend \
        sh -c "cd /app && python scripts/verify_acl_payload_sync.py --sample 200 --verbose"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_FIELDS = (
    "security_level",
    "effective_security_level",
    "visibility_mode",
    "project_ids",
    "acl_allow",
    "acl_deny",
    "excluded",
    "acl_sync_state",
)


def _norm_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return sorted(str(v) for v in value)
    return [str(value)]


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=0,
                    help="只抽查前 N 个对象（0 = 全部）")
    ap.add_argument("--verbose", action="store_true", help="逐条打印不一致")
    args = ap.parse_args()

    from sqlalchemy import select

    from app.config import get_settings
    from app.db.postgres import get_db_session
    from app.db.qdrant import get_qdrant_client
    from app.db.security_models import DocumentObject

    settings = get_settings()
    client = get_qdrant_client()

    async with get_db_session() as session:
        rows = (await session.execute(select(DocumentObject))).scalars().all()

    truth: dict[str, dict] = {}
    for r in rows:
        truth[r.object_id] = {
            "security_level": r.security_level,
            "effective_security_level": r.effective_security_level,
            "visibility_mode": r.visibility_mode,
            "project_ids": _norm_list(r.project_ids),
            "acl_allow": _norm_list(r.acl_allow),
            "acl_deny": _norm_list(r.acl_deny),
            "excluded": bool(r.excluded),
            "acl_sync_state": r.acl_sync_state,
        }

    print(f"[verify] PG document_objects = {len(truth)}")

    by_object: dict[str, dict] = {}
    offset = None
    scanned = 0
    while True:
        points, offset = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=512, offset=offset, with_payload=True, with_vectors=False,
        )
        if not points:
            break
        scanned += len(points)
        for p in points:
            payload = p.payload or {}
            oid = payload.get("object_id")
            if not oid:
                continue
            by_object.setdefault(str(oid), payload)
        if offset is None:
            break

    print(f"[verify] Qdrant 扫描点数 = {scanned}，带 object_id 的对象 = {len(by_object)}")

    items = sorted(truth.items())
    if args.sample:
        items = items[: args.sample]

    mismatched: list[tuple[str, str, object, object]] = []
    missing_payload: list[str] = []
    for oid, want in items:
        payload = by_object.get(oid)
        if payload is None:
            missing_payload.append(oid)
            continue
        for field in _FIELDS:
            got = payload.get(field)
            if field in ("project_ids", "acl_allow", "acl_deny"):
                got = _norm_list(got)
            elif field == "excluded":
                got = bool(got) if got is not None else None
            if got != want[field]:
                mismatched.append((oid, field, want[field], got))

    print(f"[verify] 比对对象 = {len(items)}")
    print(f"[verify] 缺 payload 副本的对象 = {len(missing_payload)}"
          f"（回填前或已被删除的向量，属正常）")
    print(f"[verify] 字段不一致 = {len(mismatched)}")

    if args.verbose:
        for oid, field, want, got in mismatched[:50]:
            print(f"    - {oid}: {field} PG={want!r} Qdrant={got!r}")
        if len(mismatched) > 50:
            print(f"    ... 其余 {len(mismatched) - 50} 条省略")
        for oid in missing_payload[:20]:
            print(f"    - {oid}: Qdrant 无副本")
        if len(missing_payload) > 20:
            print(f"    ... 其余 {len(missing_payload) - 20} 条省略")

    per_field: dict[str, int] = defaultdict(int)
    for _oid, field, _w, _g in mismatched:
        per_field[field] += 1
    for field, count in sorted(per_field.items()):
        print(f"    按字段：{field} = {count}")

    if not mismatched and not missing_payload:
        print("\n[verify] ✓ 完全一致")
        return 0
    print("\n[verify] 存在不一致 —— 非阻断，请按需重跑回填或让 cascade 追平")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
