"""
继承与级联（设计文档 §7 / 决策 12）—— 五维权限在"派生对象"上的取严落地.

本模块是 T4 的落点之一，回答三个问题：

    1. **写入时机**（§7.1）：入库收尾（标记 COMPLETED 之前）为一份文档的**五种对象**
       （``doc`` / ``text_chunk`` / ``table`` / ``code`` / ``image``）各写一行
       ``document_objects``，并把权限字段冗余推给 Qdrant payload。PG 始终是权威源。
    2. **只收紧不放宽**（PRD 3.2）：派生对象的 ``effective_security_level`` 恒
       ``>= 父文档``；派生对象 ``acl_allow`` 恒写空集（不得通过 need-to-know 获得
       父之外的可见性）。
    3. **OCR 派生取严**（决策 12 / 红线 2）：OCR 出的 ``text`` / ``table`` / ``code``
       块的 ``parent_object_id`` 指向**源图片对象**而不是文档，有效密级取
       ``max(父文档, 源图片)``。堵住"图看不了、但字还能搜到"的泄密面。

为什么单独一个模块
──────────────────
``materialize`` 会在**入库收尾**与**权限变更**两条路径被调用；``cascade_image_derived``
会在**图片提级 / 剔除**时被调用。把它们收在一处，才能保证"父文档权限"与"派生对象
权限"的唯一计算入口 —— 这正是 §7 要的结构保证。

纯函数与副作用分离
──────────────────
``build_object_rows`` 与 ``derive_child_fields`` 是**纯函数**（不碰 DB / 网络），
可以直接单测；``materialize_document_objects`` / ``cascade_image_derived`` 只是
把它们的结果写进 PG 与 Qdrant。这样"取严"这条红线就有可执行、可回归的断言，
而不是埋在几十行 DB 代码里。
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from app.db.security_models import (
    ACL_SYNC_PENDING,
    ACL_SYNC_STALE,
    ACL_SYNC_SYNCED,
    DEFAULT_SECURITY_LEVEL,
    OBJECT_TYPE_DOC,
    OBJECT_TYPE_IMAGE,
    make_object_id,
    object_type_from_content_type,
)
from app.services.security_policy import (
    P_ACL_EXPIRES_AT_TS,
    ObjectACLView,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


#: 父块对象类型（设计 §19.2 方案 A）。
#:
#: small-to-big 回填会把 ``chunk_parents`` 的父块正文**替换**子块正文送进 LLM
#: （``context_builder.build_context`` / ``multimodal_context_node``），而父块原
#: 不在 ``document_objects`` ⇒ 第 12 环 ``allows()`` 覆盖不到这条通道（设计缺口 1）。
#: 方案 A 把父块也物化成一行，与其余五种对象走同一套判定。
#:
#: ⚠️ 规范归属应为 ``app.db.security_models``（§19.5 记 T1 补齐）。此处先本地定义并
#: 在 ``__all__`` 暴露，避免与 T1 的落地时点强耦合；两者取值恒为 ``"parent_chunk"``。
OBJECT_TYPE_PARENT_CHUNK = "parent_chunk"


# ═══════════════════════════════════════════════════════════════════════════════
# 小工具（纯）
# ═══════════════════════════════════════════════════════════════════════════════


def _norm(value: Any) -> str | None:
    """字符串归一化：``None`` / 空串 → ``None``（避免空串混入 object_id 与 ACL）。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_list(value: Any) -> list[str]:
    """JSONB 数组 → 去空字符串的 ``list[str]``（``None`` / 非序列 → 空表）。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v) for v in value if _norm(v)]
    return []


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _doc_get(doc: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    """从 Mapping 或 ORM 对象读字段（两处来源共用一套读取）。"""
    if isinstance(doc, Mapping):
        return doc.get(key, default)
    return getattr(doc, key, default)


def _raw_from_object_id(object_id: str, document_id: str) -> str | None:
    """``{document_id}::{raw}`` → ``raw``；doc 行（``object_id == document_id``）→ ``None``.

    ``document_objects`` 行**没有** ``parent_id`` 列，父块的原始键只能从
    ``object_id`` 反解（父块 ``object_id = make_object_id(doc, parent_id, "parent_chunk")``）。
    """
    from app.db.security_models import OBJECT_ID_SEPARATOR

    if object_id == document_id:
        return None
    marker = f"{document_id}{OBJECT_ID_SEPARATOR}"
    if object_id.startswith(marker):
        return object_id[len(marker):]
    return object_id     # OBJECT_ID_MODE='raw' 时 object_id 即 raw_object_id


# ═══════════════════════════════════════════════════════════════════════════════
# ① 行构造（纯函数）—— "五种对象 + OCR 取严" 的唯一算法
# ═══════════════════════════════════════════════════════════════════════════════


def build_object_rows(
    doc: Mapping[str, Any] | Any,
    points: Sequence[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    default_level: int = DEFAULT_SECURITY_LEVEL,
    parents: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict], dict]:
    """
    为一份文档构造全部 ``document_objects`` 行（**纯函数，不碰 DB**）.

    Args:
        doc:    文档权限快照（Mapping 或 ORM）。至少读 ``id`` / ``tenant_id`` /
                ``security_level`` / ``access_level`` / ``visibility_mode`` /
                ``project_ids`` / ``acl_allow`` / ``acl_deny`` / ``owner_id`` /
                ``department_id``。
        points: Qdrant 向量块 ``[{"id": point_id, "payload": {...}}, ...]``。
                payload 里只关心 ``image_id`` / ``image_path`` / ``content_type`` /
                ``chunk_index`` / ``page_number``。
        now:    时间戳注入（测试可控）。

    Returns:
        ``(rows, stats)``。``rows`` 里每行是可直接 INSERT 的 dict（额外带
        ``_point_id`` / ``_payload`` 两个内部键，供 payload 推送；落库前需剥离）。

    OCR 取严的关键（决策 12）
    ------------------------
    ``image_id`` 非空且 ``content_type != 'image'`` 的块是 **OCR 派生块**：
    它的 ``parent_object_id`` / ``inherited_from`` 指向**源图片对象**，
    ``parent_security_level = 源图.effective_security_level``，
    ``effective_security_level = max(文档, 源图, 自身)``，``acl_allow`` 恒空 ——
    源图被提级 / 剔除时，这块跟着一起严。

    父块取严（设计 §19.2 / 决策 16）
    ------------------------------
    ``parents`` 给出该文档的 ``chunk_parents`` 行时，为每个父块再建一行
    ``object_type='parent_chunk'``（``object_id = make_object_id(doc, parent_id,
    "parent_chunk")``）。small-to-big 回填的父块正文因此也落入第 12 环的
    ``allows()`` 覆盖范围 —— 堵住"子块通过、父块正文却绕过判定进上下文"的缺口。
    父块 ``owner_id`` / ``access_level`` / ``department_id`` 从父块行取（缺省回落
    文档行）；``acl_allow`` 恒空（派生对象不得单独放行）。
    """
    now = now or datetime.now(timezone.utc)
    doc_id = str(_doc_get(doc, "id"))
    doc_level = _as_int(_doc_get(doc, "security_level"), default_level)
    if doc_level is None:
        doc_level = default_level
    doc_tenant = _norm(_doc_get(doc, "tenant_id")) or "default"
    doc_acl_allow = _as_list(_doc_get(doc, "acl_allow"))
    doc_acl_deny = _as_list(_doc_get(doc, "acl_deny"))
    doc_projects = _as_list(_doc_get(doc, "project_ids"))
    doc_visibility = _norm(_doc_get(doc, "visibility_mode")) or "tier"

    def _base_row(object_id: str, object_type: str) -> dict:
        return {
            "object_id": object_id,
            "document_id": _doc_get(doc, "id"),
            "object_type": object_type,
            "parent_object_id": None,
            "inherited_from": None,
            "inherited_at": now,
            "tenant_id": doc_tenant,
            "owner_id": _doc_get(doc, "owner_id"),
            "department_id": _doc_get(doc, "department_id"),
            "access_level": _norm(_doc_get(doc, "access_level")) or "private",
            "visibility_mode": doc_visibility,
            "project_ids": doc_projects,
            "visible_scope": None,
            "security_level": doc_level,
            "parent_security_level": None,
            "effective_security_level": doc_level,
            "acl_allow": doc_acl_allow,
            "acl_deny": doc_acl_deny,
            "acl_expires_at": _doc_get(doc, "acl_expires_at"),
            "acl_sync_state": ACL_SYNC_SYNCED,
            "excluded": False,
            "share_status": _norm(_doc_get(doc, "share_status")) or "none",
            "share_grant_scope": _norm(_doc_get(doc, "share_grant_scope")),
            "chunk_index": None,
            "page_number": None,
            "image_id": None,
            "image_path": None,
            "content_type": None,
            "created_at": now,
            "updated_at": now,
        }

    rows: list[dict] = []
    stats: dict = {"doc": 0, "chunk": 0, "image": 0, "derived": 0, "parent": 0, "collisions": 0}

    # ── 1. doc 镜像行（供第 11 / 12 环的对象级统一视图消费）─────────────────────
    doc_row = _base_row(doc_id, OBJECT_TYPE_DOC)
    rows.append(doc_row)
    stats["doc"] += 1

    # ── 2. 图片对象（同一 image_id 只建一行；一图多块时后者合并）────────────────
    image_rows: dict[str, dict] = {}

    def _ensure_image_row(image_id: str, payload: Mapping[str, Any]) -> dict:
        obj_id = make_object_id(doc_id, image_id, object_type=OBJECT_TYPE_IMAGE)
        row = image_rows.get(obj_id)
        if row is not None:
            return row
        row = _base_row(obj_id, OBJECT_TYPE_IMAGE)
        row.update(
            parent_object_id=doc_id,
            inherited_from=doc_id,
            parent_security_level=doc_level,
            image_id=image_id,
            image_path=_norm(payload.get("image_path")),
            content_type=_norm(payload.get("content_type")) or "image",
            page_number=_as_int(payload.get("page_number")),
        )
        image_rows[obj_id] = row
        rows.append(row)
        stats["image"] += 1
        return row

    for p in points:
        payload = (p.get("payload") or {}) if isinstance(p, Mapping) else {}
        image_id = _norm(payload.get("image_id"))
        if not image_id:
            continue
        # 只要看到 image_id 就为该图建行：图片本体块会走到这里；
        # OCR 派生块（content_type=table/code/text）也意味着一张源图存在。
        _ensure_image_row(image_id, payload)

    # ── 3. 分块对象（含 OCR 派生取严）───────────────────────────────────────────
    seen_chunk_index: dict[tuple[str, int], str] = {}
    for p in points:
        payload = (p.get("payload") or {}) if isinstance(p, Mapping) else {}
        point_id = _norm(p.get("id")) or ""
        content_type = _norm(payload.get("content_type")) or "text"
        image_id = _norm(payload.get("image_id"))

        if content_type == "image" and image_id:
            # 图片本体：其权限行已在第 2 步建过（图片对象不另建 chunk 行）
            continue

        object_type = object_type_from_content_type(content_type)
        obj_id = make_object_id(doc_id, point_id, object_type=object_type)
        row = _base_row(obj_id, object_type)
        row.update(
            chunk_index=_as_int(payload.get("chunk_index")),
            page_number=_as_int(payload.get("page_number")),
            image_id=image_id,
            image_path=_norm(payload.get("image_path")),
            content_type=content_type,
        )

        if image_id:
            # ── OCR 派生：父 = 源图片（不是文档），有效密级取 max ─────────────
            src_row = image_rows.get(
                make_object_id(doc_id, image_id, object_type=OBJECT_TYPE_IMAGE)
            )
            src_eff = (
                _as_int(src_row.get("effective_security_level"), doc_level)
                if src_row
                else doc_level
            )
            row.update(
                parent_object_id=src_row["object_id"] if src_row else None,
                inherited_from=src_row["object_id"] if src_row else None,
                parent_security_level=src_eff,
                effective_security_level=max(doc_level, src_eff, int(row["security_level"])),
                # 派生对象不得通过 acl_allow 获得父之外的可见性（PRD 3.2 / 共享知识 9）
                acl_allow=[],
                acl_deny=sorted(set(doc_acl_deny) | set(row["acl_deny"])),
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

        # uq_dobj_doc_chunk（(document_id, chunk_index) 部分唯一索引）的碰撞保护：
        # 保留第一条，其余置 NULL（NULL 不进部分索引），让问题可见而非整批插入失败。
        ci = row.get("chunk_index")
        if ci is not None:
            key = (doc_id, int(ci))
            if key in seen_chunk_index:
                stats["collisions"] += 1
                row["chunk_index"] = None
            else:
                seen_chunk_index[key] = obj_id

        row["_point_id"] = point_id
        row["_payload"] = {
            "object_id": obj_id,
            "object_type": object_type,
            "parent_object_id": row["parent_object_id"],
            "inherited_from": row["inherited_from"],
            "visibility_mode": row["visibility_mode"],
            "project_ids": sorted(row["project_ids"]),
            "security_level": row["security_level"],
            "parent_security_level": row["parent_security_level"],
            "effective_security_level": row["effective_security_level"],
            "acl_allow": sorted(row["acl_allow"]),
            "acl_deny": sorted(row["acl_deny"]),
            "acl_expires_at": (
                row["acl_expires_at"].isoformat()
                if isinstance(row["acl_expires_at"], datetime)
                else None
            ),
            P_ACL_EXPIRES_AT_TS: (
                row["acl_expires_at"].timestamp()
                if isinstance(row["acl_expires_at"], datetime)
                else None
            ),
            "excluded": row["excluded"],
            "acl_sync_state": row["acl_sync_state"],
            "share_status": row["share_status"],
        }
        rows.append(row)

    # ── 4. 父块对象（设计 §19.2-A）───────────────────────────────────────────────
    # small-to-big 回填的父块正文也进 LLM，必须与其它对象同一套判定。父块自身没有
    # security_level 列 ⇒ 以文档密级为基线（父块 ≡ 文档），owner/department 取父块行
    # （历史行可能为空 → 回落文档行）；acl_allow 恒空（派生不得单独放行）。
    seen_parent_ids: set[str] = set()
    for parent in parents or []:
        parent_id = _norm(_doc_get(parent, "parent_id"))
        if not parent_id or parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        obj_id = make_object_id(doc_id, parent_id, object_type=OBJECT_TYPE_PARENT_CHUNK)
        row = _base_row(obj_id, OBJECT_TYPE_PARENT_CHUNK)
        row.update(
            parent_object_id=doc_id,
            inherited_from=doc_id,
            parent_security_level=doc_level,
            effective_security_level=max(doc_level, int(row["security_level"])),
            acl_allow=[],
            owner_id=_doc_get(parent, "owner_id") or _doc_get(doc, "owner_id"),
            access_level=(
                _norm(_doc_get(parent, "access_level")) or row["access_level"]
            ),
            department_id=(
                _doc_get(parent, "department_id") or _doc_get(doc, "department_id")
            ),
            tenant_id=(_norm(_doc_get(parent, "tenant_id")) or doc_tenant),
        )
        stats["parent"] += 1
        rows.append(row)

    return rows, stats


def derive_child_fields(child: Mapping[str, Any], src: Mapping[str, Any]) -> dict:
    """
    由**源图片权限**重算一个派生对象的权威字段（纯函数）—— 收紧同步的核心.

    规则（设计 §7.4）：
        effective_security_level = max(子自身, 源图 effective)   ← 只收紧不放宽
        excluded                 = 子 excluded OR 源图 excluded   ← 剔除向下传染
        acl_deny                 = 子 acl_deny ∪ 源图 acl_deny     ← deny 向下传染
        acl_allow                = 空集                            ← 派生不得单独放行
        acl_sync_state           = pending（PG 已写、payload 待推）

    绝不"放宽"：即使源图被降级，派生对象的有效密级也只会是 ``max``，不会低于自身。
    """
    child_self = _as_int(child.get("security_level"), 0) or 0
    child_eff = _as_int(child.get("effective_security_level"), child_self)
    src_eff = _as_int(src.get("security_level"), 0) or 0
    src_eff_final = _as_int(src.get("effective_security_level"), src_eff)
    if src_eff_final is None:
        src_eff_final = src_eff

    effective = max(int(child_self), int(child_eff or 0), int(src_eff_final or 0))
    return {
        "parent_security_level": int(src_eff_final or 0),
        "effective_security_level": effective,
        "excluded": bool(child.get("excluded")) or bool(src.get("excluded")),
        "acl_deny": sorted(set(_as_list(child.get("acl_deny"))) | set(_as_list(src.get("acl_deny")))),
        "acl_allow": [],
        "acl_sync_state": ACL_SYNC_PENDING,
        "updated_at": datetime.now(timezone.utc),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ② 落库：materialize / sync_doc_row / cascade_image_derived
# ═══════════════════════════════════════════════════════════════════════════════

#: 落库时剔除的内部键（只用于 payload 推送）
_INTERNAL_KEYS = ("_point_id", "_payload")


def _row_for_db(row: Mapping[str, Any]) -> dict:
    return {k: v for k, v in row.items() if k not in _INTERNAL_KEYS}


async def _upsert_rows(session: Any, rows: Sequence[Mapping[str, Any]]) -> int:
    """幂等 upsert ``document_objects``（``ON CONFLICT(object_id) DO UPDATE``）."""
    if not rows:
        return 0
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from app.db.security_models import DocumentObject

    written = 0
    for row in rows:
        payload = _row_for_db(row)
        stmt = pg_insert(DocumentObject).values(**payload)
        update_cols = {
            k: stmt.excluded[k]
            for k in payload
            if k not in ("object_id", "document_id", "created_at")
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["object_id"], set_=update_cols
        )
        await session.execute(stmt)
        written += 1
    return written


async def materialize_document_objects(
    doc: Mapping[str, Any] | Any,
    points: Sequence[Mapping[str, Any]],
    *,
    session: Any = None,
    push_payload: bool = True,
    parents: Sequence[Mapping[str, Any]] | None = None,
) -> dict:
    """
    入库收尾 / 回填：为一份文档幂等生成全部 ``document_objects`` 行并推 payload.

    ``session`` 为 ``None`` 时自建会话（独立事务）。入库路径里传入外层 session
    可与"标记 COMPLETED"同一事务提交（收紧同步，窗口为零）。

    ``parents`` 为该文档的 ``chunk_parents`` 行（设计 §19.2-A）：给出时同时物化
    ``parent_chunk`` 对象行，使 small-to-big 回填的父块正文进入第 12 环判定范围。
    """
    rows, stats = build_object_rows(doc, points, parents=parents)

    async def _run(sess: Any) -> None:
        stats["written"] = await _upsert_rows(sess, rows)

    if session is not None:
        await _run(session)
    else:
        from app.db.postgres import get_db_session

        async with get_db_session() as sess:
            await _run(sess)

    if push_payload:
        await push_payload_async(rows)
    return stats


async def sync_doc_row(
    document_id: uuid.UUID | str,
    *,
    session: Any = None,
    security_level: int | None = None,
    visibility_mode: str | None = None,
    project_ids: Sequence[str] | None = None,
    acl_allow: Sequence[str] | None = None,
    acl_deny: Sequence[str] | None = None,
    excluded: bool | None = None,
) -> dict:
    """
    文档权限变更：**同步**改 ``documents`` 与全部派生 ``document_objects`` 行（收紧方向）.

    只对显式传入的字段做修改（``None`` = 不动）。派生对象的
    ``effective_security_level`` 重新取 ``max(doc, 自身)``，``excluded`` /
    ``acl_deny`` 向下传染 —— 与 ``derive_child_fields`` 同源。

    返回 ``{"document_rows": n, "object_rows": n, "acl_sync_state": ...}``。
    """
    from sqlalchemy import select, update

    from app.db.models import Document
    from app.db.security_models import DocumentObject

    doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id

    async def _run(sess: Any) -> dict:
        doc = (
            await sess.execute(select(Document).where(Document.id == doc_uuid))
        ).scalar_one_or_none()
        if doc is None:
            return {"document_rows": 0, "object_rows": 0, "acl_sync_state": ACL_SYNC_STALE}

        doc_updates: dict[str, Any] = {}
        if security_level is not None:
            doc_updates["security_level"] = int(security_level)
        if visibility_mode is not None:
            doc_updates["visibility_mode"] = visibility_mode
        if project_ids is not None:
            doc_updates["project_ids"] = sorted({str(p) for p in project_ids if _norm(p)})
        if acl_allow is not None:
            doc_updates["acl_allow"] = sorted({str(a) for a in acl_allow if _norm(a)})
        if acl_deny is not None:
            doc_updates["acl_deny"] = sorted({str(a) for a in acl_deny if _norm(a)})
        # 任何权限变更都把同步水位打成 pending，payload 推送成功后置 synced
        doc_updates["acl_sync_state"] = ACL_SYNC_PENDING
        await sess.execute(update(Document).where(Document.id == doc_uuid).values(**doc_updates))

        # 重新读一遍权威值（含未显式传入、但需要向下传染的字段）
        await sess.refresh(doc)
        new_level = _as_int(getattr(doc, "security_level", None), DEFAULT_SECURITY_LEVEL) or 0
        new_deny = _as_list(getattr(doc, "acl_deny", None))
        new_projects = _as_list(getattr(doc, "project_ids", None))
        new_visibility = _norm(getattr(doc, "visibility_mode", None)) or "tier"

        # doc 镜像行
        doc_obj_row = {
            "security_level": new_level,
            "effective_security_level": new_level,
            "parent_security_level": None,
            "visibility_mode": new_visibility,
            "project_ids": new_projects,
            "acl_allow": _as_list(getattr(doc, "acl_allow", None)),
            "acl_deny": new_deny,
            "acl_sync_state": ACL_SYNC_PENDING,
        }
        # FIX-C（T5 预发布）：``excluded`` 未显式传入时**保留原值**（不覆写为 False）。
        # 用 ``excluded is not None`` 区分「未传」与「显式传了 False」两态；否则任何
        # 一次其它权限变更都会把已剔除的文档静默撤销（权限回退）。
        if excluded is not None:
            doc_obj_row["excluded"] = bool(excluded)
        await sess.execute(
            update(DocumentObject)
            .where(DocumentObject.object_id == str(doc_uuid))
            .values(**doc_obj_row)
        )

        # 全部派生行：有效密级取 max（只收紧），deny / excluded 向下传染
        derived_rows = (
            await sess.execute(
                select(DocumentObject).where(
                    DocumentObject.document_id == doc_uuid,
                    DocumentObject.object_id != str(doc_uuid),
                )
            )
        ).scalars().all()

        obj_count = 1  # doc 镜像行
        for obj in derived_rows:
            src_like = {
                "security_level": new_level,
                "effective_security_level": new_level,
                "acl_deny": new_deny,
                # FIX-C：未传 excluded 时沿用该派生对象的原值（避免撤销剔除）。
                "excluded": bool(excluded) if excluded is not None else bool(obj.excluded),
            }
            fields = derive_child_fields(
                {
                    "security_level": obj.security_level,
                    "effective_security_level": obj.effective_security_level,
                    "acl_deny": obj.acl_deny,
                    "excluded": obj.excluded,
                },
                src_like,
            )
            # 项目维度：只加闸不放宽（沿用父文档的 project 集合）
            fields["visibility_mode"] = (
                "project" if new_visibility == "project" else obj.visibility_mode
            )
            await sess.execute(
                update(DocumentObject)
                .where(DocumentObject.object_id == obj.object_id)
                .values(**fields)
            )
            obj_count += 1

        return {
            "document_rows": 1,
            "object_rows": obj_count,
            "acl_sync_state": ACL_SYNC_PENDING,
        }

    if session is not None:
        result = await _run(session)
    else:
        from app.db.postgres import get_db_session

        async with get_db_session() as sess:
            result = await _run(sess)

    # payload 副本异步追平（失败置 stale，不阻断请求）
    await _push_document_payload(document_id, session=session)
    return result


async def cascade_image_derived(
    document_id: uuid.UUID | str,
    image_object_id: str,
    *,
    session: Any = None,
    src_level: int | None = None,
    src_excluded: bool | None = None,
) -> dict:
    """
    图片被提级 / 剔除后，**同步**重算其全部 OCR 派生对象（设计 §7.4）.

    ``image_object_id`` = ``make_object_id(document_id, image_id, object_type="image")``。
    ``src_level`` / ``src_excluded`` 为 ``None`` 时以 image 行的 PG 真值为准。

    返回 ``{"derived_rows": n, "image_rows": n}``。
    """
    from sqlalchemy import select, update

    from app.db.security_models import DocumentObject

    doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id

    async def _run(sess: Any) -> dict:
        image = (
            await sess.execute(
                select(DocumentObject).where(DocumentObject.object_id == image_object_id)
            )
        ).scalar_one_or_none()
        if image is None:
            logger.warning(
                "cascade_image_derived: image object row missing (object_id=%s) — "
                "nothing to cascade (fail-closed: 调用方应确保行已物化)",
                image_object_id,
            )
            return {"derived_rows": 0, "image_rows": 0}

        # 先按传入值更新 image 行本身（提级 / 剔除）
        image_updates: dict[str, Any] = {}
        if src_level is not None:
            image_updates["security_level"] = int(src_level)
            image_updates["effective_security_level"] = int(src_level)
        if src_excluded is not None:
            image_updates["excluded"] = bool(src_excluded)
        if image_updates:
            image_updates["acl_sync_state"] = ACL_SYNC_PENDING
            await sess.execute(
                update(DocumentObject)
                .where(DocumentObject.object_id == image_object_id)
                .values(**image_updates)
            )
            await sess.refresh(image)

        src = {
            "security_level": image.security_level,
            "effective_security_level": image.effective_security_level,
            "acl_deny": image.acl_deny,
            "excluded": image.excluded,
        }

        derived = (
            await sess.execute(
                select(DocumentObject).where(
                    DocumentObject.parent_object_id == image_object_id
                )
            )
        ).scalars().all()

        for child in derived:
            fields = derive_child_fields(
                {
                    "security_level": child.security_level,
                    "effective_security_level": child.effective_security_level,
                    "acl_deny": child.acl_deny,
                    "excluded": child.excluded,
                },
                src,
            )
            await sess.execute(
                update(DocumentObject)
                .where(DocumentObject.object_id == child.object_id)
                .values(**fields)
            )

        return {"derived_rows": len(derived), "image_rows": 1}

    if session is not None:
        result = await _run(session)
    else:
        from app.db.postgres import get_db_session

        async with get_db_session() as sess:
            result = await _run(sess)

    await _push_document_payload(document_id, session=session)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ③ payload 副本推送（异步追平；失败置 stale，绝不阻断主流程）
# ═══════════════════════════════════════════════════════════════════════════════


async def push_payload_async(rows: Sequence[Mapping[str, Any]]) -> int:
    """
    把行里内嵌的 ``_payload`` 推给对应 Qdrant 点（逐点 ``set_payload``）.

    失败只告警：payload 是 PG 的副本，推失败不该让入库 / 权限变更整体失败
    （改由第 11 环的 PG 权威复核兜底）。
    """
    pushed = 0
    for row in rows:
        point_id = row.get("_point_id")
        payload = row.get("_payload")
        if not point_id or not payload:
            continue
        try:
            if await _set_point_payload(str(point_id), dict(payload)):
                pushed += 1
        except Exception:      # noqa: BLE001 — 副本推送失败不阻断
            logger.exception("push_payload_async: set_payload failed for point %s", point_id)
    return pushed


async def _set_point_payload(point_id: str, payload: Mapping[str, Any]) -> bool:
    from app.config import get_settings
    from app.db.qdrant import get_qdrant_client

    settings = get_settings()
    client = get_qdrant_client()
    await client.set_payload(
        collection_name=settings.QDRANT_COLLECTION,
        payload=dict(payload),
        points=[point_id],
    )
    return True


async def _push_document_payload(
    document_id: uuid.UUID | str, *, session: Any = None
) -> int:
    """把某文档**全部**对象的 payload 推给 Qdrant（按 object_id → point_id 反查）。"""
    from sqlalchemy import select

    from app.db.security_models import DocumentObject

    doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id

    async def _load(sess: Any) -> list[dict]:
        rows = (
            await sess.execute(
                select(DocumentObject).where(DocumentObject.document_id == doc_uuid)
            )
        ).scalars().all()
        out: list[dict] = []
        for r in rows:
            view = ObjectACLView.from_row(r)
            payload = view.to_payload()
            # 点 id：非 doc 对象的 object_id 是 `{doc}::{point_id}`，取冒号后段
            point_id = _point_id_from_object_id(r.object_id, str(doc_uuid))
            if point_id:
                out.append({"point_id": point_id, "payload": payload})
        return out

    try:
        if session is not None:
            items = await _load(session)
        else:
            from app.db.postgres import get_db_session

            async with get_db_session() as sess:
                items = await _load(sess)
    except Exception:      # noqa: BLE001
        logger.exception("_push_document_payload: load failed for %s", document_id)
        return 0

    pushed = 0
    for item in items:
        try:
            if await _set_point_payload(item["point_id"], item["payload"]):
                pushed += 1
        except Exception:      # noqa: BLE001
            logger.warning(
                "_push_document_payload: set_payload failed for point %s", item["point_id"]
            )
    return pushed


def _point_id_from_object_id(object_id: str, document_id: str) -> str | None:
    """``{document_id}::{point_id}`` → ``point_id``；doc 行返回 ``None``（无独立点）。"""
    from app.db.security_models import OBJECT_ID_SEPARATOR

    if object_id == document_id:
        return None
    marker = f"{document_id}{OBJECT_ID_SEPARATOR}"
    if object_id.startswith(marker):
        return object_id[len(marker):]
    return object_id     # OBJECT_ID_MODE='raw' 时 object_id 即 point_id


# ═══════════════════════════════════════════════════════════════════════════════
# ④ 视图解析（第 11 / 12 环与引用点击复用；PG 是权威源）
# ═══════════════════════════════════════════════════════════════════════════════


def document_view(doc: Mapping[str, Any] | Any) -> ObjectACLView:
    """
    由 ``documents`` 行构造 **doc 镜像视图**.

    为什么不能直接用 ``ObjectACLView.from_row(doc)``：``Document`` 模型没有
    ``object_id`` 属性，``from_row`` 会把 ``object_id`` 取成空串，进而被
    :func:`allows` 的第 0 步 fail-closed 拒掉 —— 一个"文档级判定永远为假"的
    静默缺陷。这里显式补上 ``object_id = str(doc.id)`` 与 ``object_type='doc'``。
    """
    from dataclasses import replace

    view = ObjectACLView.from_row(doc)
    return replace(
        view,
        object_id=str(_doc_get(doc, "id") or ""),
        object_type=OBJECT_TYPE_DOC,
    )


async def load_object_views(
    object_ids: Sequence[str], *, session: Any = None
) -> dict[str, ObjectACLView]:
    """按 ``object_id`` 批量取权威视图（缺失的对象不出现在返回字典里）。"""
    ids = [str(o) for o in object_ids if _norm(o)]
    if not ids:
        return {}

    from sqlalchemy import select

    from app.db.security_models import DocumentObject

    async def _load(sess: Any) -> dict[str, ObjectACLView]:
        rows = (
            await sess.execute(
                select(DocumentObject).where(DocumentObject.object_id.in_(ids))
            )
        ).scalars().all()
        return {r.object_id: ObjectACLView.from_row(r) for r in rows}

    try:
        if session is not None:
            return await _load(session)
        from app.db.postgres import get_db_session

        async with get_db_session() as sess:
            return await _load(sess)
    except Exception:      # noqa: BLE001 — 视图解析失败 → fail-closed（空字典）
        logger.exception("load_object_views: query failed — fail-closed (empty)")
        return {}


async def load_document_views(
    document_id: uuid.UUID | str, *, session: Any = None
) -> dict[str, ObjectACLView]:
    """某文档全部对象的权威视图（按 ``object_id`` 索引）。"""
    doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id

    from sqlalchemy import select

    from app.db.security_models import DocumentObject

    async def _load(sess: Any) -> dict[str, ObjectACLView]:
        rows = (
            await sess.execute(
                select(DocumentObject).where(DocumentObject.document_id == doc_uuid)
            )
        ).scalars().all()
        return {r.object_id: ObjectACLView.from_row(r) for r in rows}

    try:
        if session is not None:
            return await _load(session)
        from app.db.postgres import get_db_session

        async with get_db_session() as sess:
            return await _load(sess)
    except Exception:      # noqa: BLE001
        logger.exception("load_document_views: query failed — fail-closed (empty)")
        return {}


async def load_document_view_index(
    document_id: uuid.UUID | str, *, session: Any = None
) -> dict[str, ObjectACLView]:
    """
    某文档的"引用 / 上下文消费侧"视图索引（键为**规范化 chunk 键**）.

    检索产物（``RetrievedChunk``）**不携带 Qdrant point_id**，因此不能用
    ``object_id`` 直接映射。这里改用两条稳定键：

        ``"doc"``                 → doc 镜像行
        ``"ci:{chunk_index}"``    → text/table/code（含 OCR 派生）行
        ``"img:{image_id}"``      → 图片对象行

    ``uq_dobj_doc_chunk``（(document_id, chunk_index) 部分唯一索引）保证
    ``ci:`` 键在文档内唯一；图片对象行 ``chunk_index`` 为 NULL，用 ``image_id`` 定位。
    """
    doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id

    from sqlalchemy import select

    from app.db.security_models import DocumentObject

    async def _load(sess: Any) -> dict[str, ObjectACLView]:
        rows = (
            await sess.execute(
                select(DocumentObject).where(DocumentObject.document_id == doc_uuid)
            )
        ).scalars().all()
        index: dict[str, ObjectACLView] = {}
        for r in rows:
            view = ObjectACLView.from_row(r)
            if r.object_id == str(doc_uuid):
                index["doc"] = view
            elif r.object_type == OBJECT_TYPE_IMAGE and r.image_id:
                index[f"img:{r.image_id}"] = view
            elif r.object_type == OBJECT_TYPE_PARENT_CHUNK:
                _raw = _raw_from_object_id(r.object_id, str(doc_uuid))
                if _raw:
                    index[f"pc:{_raw}"] = view
            if r.chunk_index is not None:
                index[f"ci:{int(r.chunk_index)}"] = view
        return index

    try:
        if session is not None:
            return await _load(session)
        from app.db.postgres import get_db_session

        async with get_db_session() as sess:
            return await _load(sess)
    except Exception:      # noqa: BLE001 — 解析失败 → fail-closed（空索引）
        logger.exception("load_document_view_index: query failed — fail-closed (empty)")
        return {}


async def document_is_materialized(
    document_id: uuid.UUID | str, *, session: Any = None
) -> bool:
    """
    该文档是否已物化对象权限行.

    ⚠️ 语义边界：``False`` 表示"这份文档从未物化过"（T1 回填尚未覆盖到的存量文档）
    —— 此时引用点击校验应**回退到文档级判定**，而不是把整份文档判为不可见
    （否则回填上线前所有存量文档的原图 / 原文预览会集体变 404，这是功能退化）。
    ``True`` 表示对象级权限是**权威且完整**的，缺失某对象行即 fail-closed。
    """
    doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id

    from sqlalchemy import func, select

    from app.db.security_models import DocumentObject

    async def _load(sess: Any) -> bool:
        count = (
            await sess.execute(
                select(func.count())
                .select_from(DocumentObject)
                .where(DocumentObject.document_id == doc_uuid)
            )
        ).scalar()
        return bool(count)

    try:
        if session is not None:
            return await _load(session)
        from app.db.postgres import get_db_session

        async with get_db_session() as sess:
            return await _load(sess)
    except Exception:      # noqa: BLE001 — 判定不了就当未物化（回退文档级，不误伤）
        logger.exception("document_is_materialized: query failed — assume not materialized")
        return False


async def load_view_indexes(
    document_ids: Sequence[uuid.UUID | str], *, session: Any = None
) -> tuple[dict[str, dict[str, ObjectACLView]], dict[str, bool]]:
    """
    **批量版** :func:`load_document_view_index` + :func:`document_is_materialized`.

    为什么需要它：master_graph 的三个 assemble 调用点（context_builder /
    multimodal_context / generate 兜底）要在**同一次请求**里为若干份文档解析对象视图。
    逐份调用单文档版是 **N+1**（每份文档两条 SQL）；本函数把 N 份文档收敛成**一条**
    ``select(...).where(document_id.in_(...))``。

    与单文档版**逐条同构**的键规则（改一处必须改另一处）：

        ``"doc"``                 → 文档镜像行（``object_id == str(document_id)``）
        ``"img:{image_id}"``      → 图片对象行（``object_type == image`` 且 ``image_id`` 非空）
        ``"ci:{chunk_index}"``    → 文本/表格/代码（含 OCR 派生）行（``chunk_index`` 非空）

    返回值第二项为**每个被请求的** ``document_id`` 都给一个条目：命中 ≥1 行 → ``True``，
    否则 ``False``（镜像 :func:`document_is_materialized` 的「无行 ⇒ False ⇒ 从未物化
    ⇒ 缺行回退允许」语义，避免存量文档集体不可见）。

    失败时 fail-closed 返回 ``({}, {})``（与相邻函数一致），调用方据此整段跳过或按
    缺行处理，绝不降级为"不过滤"。
    """
    uuids: list[uuid.UUID] = []
    requested: dict[str, uuid.UUID] = {}
    for raw in document_ids:
        try:
            u = raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw))
        except (TypeError, ValueError):
            continue
        uuids.append(u)
        requested[str(u)] = u

    view_index: dict[str, dict[str, ObjectACLView]] = {k: {} for k in requested}
    materialized: dict[str, bool] = {k: False for k in requested}
    if not uuids:
        return view_index, materialized

    from sqlalchemy import select

    from app.db.security_models import DocumentObject

    async def _load(sess: Any) -> None:
        rows = (
            await sess.execute(
                select(DocumentObject).where(DocumentObject.document_id.in_(uuids))
            )
        ).scalars().all()
        for r in rows:
            doc_key = str(r.document_id)
            # IN 查询已限定范围；防御性兜底，避免把未请求的文档混进返回集。
            if doc_key not in view_index:
                view_index[doc_key] = {}
            materialized[doc_key] = True
            view = ObjectACLView.from_row(r)
            if r.object_id == doc_key:
                view_index[doc_key]["doc"] = view
            elif r.object_type == OBJECT_TYPE_IMAGE and r.image_id:
                view_index[doc_key][f"img:{r.image_id}"] = view
            if r.chunk_index is not None:
                view_index[doc_key][f"ci:{int(r.chunk_index)}"] = view

    try:
        if session is not None:
            await _load(session)
            return view_index, materialized
        from app.db.postgres import get_db_session

        async with get_db_session() as sess:
            await _load(sess)
        return view_index, materialized
    except Exception:      # noqa: BLE001 — 解析失败 → fail-closed（空）
        logger.exception("load_view_indexes: query failed — fail-closed (empty)")
        return {}, {}


async def expire_acl_grants() -> int:
    """
    把已过期的 ``acl_grants`` 置为 ``expired``（P1-1；调度位置复用既有定时任务的挂载点）.

    仅更新状态，不动 `document_objects`（物化副本的回收由 T5 的 grant 流程负责）。
    """
    from sqlalchemy import update

    from app.db.postgres import get_db_session
    from app.db.security_models import (
        GRANT_STATUS_APPROVED,
        GRANT_STATUS_EXPIRED,
        AclGrant,
    )

    now = datetime.now(timezone.utc)
    try:
        async with get_db_session() as sess:
            result = await sess.execute(
                update(AclGrant)
                .where(
                    AclGrant.status == GRANT_STATUS_APPROVED,
                    AclGrant.expires_at.is_not(None),
                    AclGrant.expires_at <= now,
                )
                .values(status=GRANT_STATUS_EXPIRED)
            )
        return int(result.rowcount or 0)
    except Exception:      # noqa: BLE001
        logger.exception("expire_acl_grants: failed")
        return 0


__all__ = [
    "build_object_rows",
    "cascade_image_derived",
    "derive_child_fields",
    "document_is_materialized",
    "document_view",
    "expire_acl_grants",
    "load_document_view_index",
    "load_document_views",
    "load_object_views",
    "load_view_indexes",
    "materialize_document_objects",
    "push_payload_async",
    "sync_doc_row",
]
