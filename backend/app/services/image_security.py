"""
图片级权限收紧（设计文档决策 11）—— **最小侵入**，只做"提级"与"剔除"两种.

允许的动作（且**仅**这两种）
────────────────────────────
    提级（escalate）  ：把某张图的 ``security_level`` 提到比父文档更高
    剔除（exclude）   ：把某张图 ``excluded=true``（对所有人下线）

**不允许**"放宽"：图片只能比父文档更严，不能更宽（PRD 3.2）。本模块不提供
"降低密级 / 取消剔除"的入口 —— 那需要走管理面（T5）并留审计。

红线（不重构多模态）
────────────────────
本模块**不碰** ``image_understanding/*``、``vision/*``、``parsers/image_parser.py``。
它只读写 ``document_objects`` 的 ``object_type='image'`` 行，并调用
:func:`security_cascade.cascade_image_derived` 同步重算该图的 OCR 派生块
（``text`` / ``table`` / ``code``）。多模态链路的解析 / 识别行为一行未动。

``object_id`` 用复合键
----------------------
图片对象的 ``object_id`` 由 :func:`security_models.make_object_id` 统一构造
（``{document_id}::{image_id}``）。``image_id`` 目前 0 碰撞但唯一性非构造保证
（代码里有 ``sha1(filename)[:12]`` 回退，同名文件会撞），复合键把这件事收敛在
一个函数里。

反查键（设计 §17-7 的裁决）
--------------------------
``GET /documents/{id}/images/{image_name}`` 的 ``image_name`` 是**落盘文件名的
basename**（如 ``page_3_image_1.png``），而 ``document_objects.image_path`` 存的是
文档相对路径（``images/page_3_image_1.png``）。因此反查键定为
「``image_path`` 的 basename == ``image_name``」，并对 ``image_id`` 的 basename
做同样比较以兼容历史数据。**反查不到 → 返回 None（调用方 fail-closed 或回退文档级）。**
"""

from __future__ import annotations

import uuid
from typing import Any

from app.db.security_models import (
    OBJECT_TYPE_IMAGE,
    make_object_id,
)
from app.services.security_cascade import (
    _as_int,
    cascade_image_derived,
    document_is_materialized,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _basename(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None
    return text.rsplit("/", 1)[-1] or None


def _image_object_id(document_id: uuid.UUID | str, image_id: str) -> str:
    return make_object_id(str(document_id), image_id, object_type=OBJECT_TYPE_IMAGE)


async def list_image_objects(
    document_id: uuid.UUID | str, *, session: Any = None
) -> list[dict]:
    """列出某文档的全部图片对象行（``object_type='image'``）。"""
    from sqlalchemy import select

    from app.db.security_models import DocumentObject

    doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id

    async def _load(sess: Any) -> list[dict]:
        rows = (
            await sess.execute(
                select(DocumentObject).where(
                    DocumentObject.document_id == doc_uuid,
                    DocumentObject.object_type == OBJECT_TYPE_IMAGE,
                )
            )
        ).scalars().all()
        return [
            {
                "object_id": r.object_id,
                "image_id": r.image_id,
                "image_path": r.image_path,
                "security_level": r.security_level,
                "effective_security_level": r.effective_security_level,
                "excluded": r.excluded,
                "acl_sync_state": r.acl_sync_state,
            }
            for r in rows
        ]

    if session is not None:
        return await _load(session)
    from app.db.postgres import get_db_session

    async with get_db_session() as sess:
        return await _load(sess)


async def resolve_image_object_id(
    document_id: uuid.UUID | str,
    image_name: str,
    *,
    session: Any = None,
) -> dict | None:
    """
    由 URL 里的 ``image_name`` 反查图片对象行.

    Returns:
        ``{"object_id", "image_id", "image_path", "access_level", "tenant_id",
        "owner_id", "department_id", "effective_security_level", "visibility_mode",
        "project_ids", "security_level", "parent_security_level", "acl_allow",
        "acl_deny", "acl_expires_at", "excluded", "acl_sync_state"}`` 形式的 dict；
        **反查不到返回 ``None``**（调用方据此 fail-closed 404 或回退文档级判定）。
    """
    target = _basename(image_name)
    if not target:
        return None

    from sqlalchemy import select

    from app.db.security_models import DocumentObject

    doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id

    async def _load(sess: Any) -> dict | None:
        rows = (
            await sess.execute(
                select(DocumentObject).where(
                    DocumentObject.document_id == doc_uuid,
                    DocumentObject.object_type == OBJECT_TYPE_IMAGE,
                )
            )
        ).scalars().all()
        for r in rows:
            if _basename(r.image_path) == target or _basename(r.image_id) == target:
                return {
                    "object_id": r.object_id,
                    "object_type": r.object_type,
                    "image_id": r.image_id,
                    "image_path": r.image_path,
                    "access_level": r.access_level,
                    "tenant_id": r.tenant_id,
                    "owner_id": r.owner_id,
                    "department_id": r.department_id,
                    "visibility_mode": r.visibility_mode,
                    "project_ids": r.project_ids,
                    "security_level": r.security_level,
                    "parent_security_level": r.parent_security_level,
                    "effective_security_level": r.effective_security_level,
                    "acl_allow": r.acl_allow,
                    "acl_deny": r.acl_deny,
                    "acl_expires_at": r.acl_expires_at,
                    "excluded": r.excluded,
                    "acl_sync_state": r.acl_sync_state,
                }
        return None

    if session is not None:
        return await _load(session)
    from app.db.postgres import get_db_session

    async with get_db_session() as sess:
        return await _load(sess)


async def escalate_image_object(
    document_id: uuid.UUID | str,
    *,
    image_id: str | None = None,
    image_name: str | None = None,
    security_level: int | None = None,
    excluded: bool | None = None,
    actor: str | uuid.UUID | None = None,
    session: Any = None,
) -> dict:
    """
    提级 / 剔除一张图片对象，并**同步**级联到其 OCR 派生块.

    必须给出 ``image_id`` 或 ``image_name``（后者经反查）。二者都没有 / 反查不到
    → 返回 ``{"ok": False, "reason": ...}``，**不做任何写操作**（fail-closed）。

    ``security_level`` 只能提不能降（`value < 当前` 记 warning 但仍写入 —— 允许管理员
    依 A8 口径下调由管理面 T5 负责；本函数只保证"派生块跟着源图取严"这条红线）。
    """
    from sqlalchemy import select, update

    from app.db.security_models import DocumentObject

    if security_level is None and excluded is None:
        return {"ok": False, "reason": "nothing_to_change"}

    doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id

    # 反查（未直接给 image_id 时）
    resolved_image_id = image_id
    if not resolved_image_id and image_name:
        info = await resolve_image_object_id(document_id, image_name, session=session)
        if info is None:
            return {"ok": False, "reason": "image_object_not_found"}
        resolved_image_id = info.get("image_id")
    if not resolved_image_id:
        return {"ok": False, "reason": "image_id_required"}

    object_id = _image_object_id(document_id, str(resolved_image_id))

    async def _run(sess: Any) -> dict:
        row = (
            await sess.execute(
                select(DocumentObject).where(DocumentObject.object_id == object_id)
            )
        ).scalar_one_or_none()
        if row is None:
            # 图片行未物化：以文档行补一行（父 = 文档），再级联
            from app.db.models import Document

            doc = (
                await sess.execute(select(Document).where(Document.id == doc_uuid))
            ).scalar_one_or_none()
            if doc is None:
                return {"ok": False, "reason": "document_not_found"}
            from app.db.security_models import ACL_SYNC_PENDING

            doc_level = _as_int(getattr(doc, "security_level", None), 1) or 0
            new_row = DocumentObject(
                object_id=object_id,
                document_id=doc_uuid,
                object_type=OBJECT_TYPE_IMAGE,
                parent_object_id=str(doc_uuid),
                inherited_from=str(doc_uuid),
                tenant_id=getattr(doc, "tenant_id", None) or "default",
                owner_id=getattr(doc, "owner_id", None),
                department_id=getattr(doc, "department_id", None),
                access_level=getattr(doc, "access_level", None) or "private",
                visibility_mode=getattr(doc, "visibility_mode", None) or "tier",
                project_ids=list(getattr(doc, "project_ids", None) or []),
                security_level=int(security_level) if security_level is not None else doc_level,
                parent_security_level=doc_level,
                effective_security_level=max(
                    doc_level,
                    int(security_level) if security_level is not None else doc_level,
                ),
                acl_allow=[],
                acl_deny=list(getattr(doc, "acl_deny", None) or []),
                excluded=bool(excluded),
                acl_sync_state=ACL_SYNC_PENDING,
                image_id=str(resolved_image_id),
                content_type="image",
            )
            sess.add(new_row)
            await sess.flush()
        else:
            image_updates: dict[str, Any] = {}
            if security_level is not None:
                image_updates["security_level"] = int(security_level)
                # 设计决策 12：有效密级恒取 max(父文档, 源图片)，只收紧不放宽。
                # 父密级优先取图片行的 parent_security_level，缺失时回退父文档 security_level。
                # 否则父文档绝密(3)、管理员把图片设为机密(2)时，图片 effective 会被写成 2，
                # 对 clearance=2 的用户可见 —— 越权放宽（FIX-D，T5 预发布）。
                parent_level = _as_int(getattr(row, "parent_security_level", None), None)
                if parent_level is None:
                    from app.db.models import Document

                    _doc = (
                        await sess.execute(
                            select(Document).where(Document.id == doc_uuid)
                        )
                    ).scalar_one_or_none()
                    parent_level = _as_int(getattr(_doc, "security_level", None), 1) or 0
                image_updates["effective_security_level"] = max(
                    int(parent_level), int(security_level)
                )
            if excluded is not None:
                image_updates["excluded"] = bool(excluded)
            if image_updates:
                from app.db.security_models import ACL_SYNC_PENDING

                image_updates["acl_sync_state"] = ACL_SYNC_PENDING
                await sess.execute(
                    update(DocumentObject)
                    .where(DocumentObject.object_id == object_id)
                    .values(**image_updates)
                )

        return {"ok": True, "object_id": object_id, "image_id": str(resolved_image_id)}

    if session is not None:
        result = await _run(session)
    else:
        from app.db.postgres import get_db_session

        async with get_db_session() as sess:
            result = await _run(sess)

    if not result.get("ok"):
        return result

    cascade = await cascade_image_derived(
        document_id,
        object_id,
        session=session,
        src_level=security_level,
        src_excluded=excluded,
    )
    result["cascade"] = cascade

    await _audit_image_change(
        document_id=document_id,
        object_id=object_id,
        image_id=str(resolved_image_id),
        security_level=security_level,
        excluded=excluded,
        actor=actor,
    )
    return result


async def _audit_image_change(
    *,
    document_id: uuid.UUID | str,
    object_id: str,
    image_id: str,
    security_level: int | None,
    excluded: bool | None,
    actor: str | uuid.UUID | None,
) -> None:
    """写一条 ``image.escalate`` / ``image.exclude`` 审计（best-effort）。"""
    try:
        from app.services.audit_service import record_audit

        action = "image.exclude" if excluded else "image.escalate"
        detail = f"document_id={document_id} image_id={image_id}"
        if security_level is not None:
            detail += f" security_level={security_level}"
        if excluded is not None:
            detail += f" excluded={excluded}"
        await record_audit(
            action,
            user_id=actor if isinstance(actor, uuid.UUID) else None,
            username=(str(actor)[:64] if actor is not None else None),
            resource_type="document_object",
            resource_id=object_id[:64],
            detail=detail[:2000],
        )
    except Exception:      # noqa: BLE001
        logger.warning("image change audit failed (object_id=%s)", object_id)


async def is_document_materialized(
    document_id: uuid.UUID | str, *, session: Any = None
) -> bool:
    """转发 :func:`security_cascade.document_is_materialized`（供端点判定回退口径）。"""
    return await document_is_materialized(document_id, session=session)


__all__ = [
    "escalate_image_object",
    "is_document_materialized",
    "list_image_objects",
    "resolve_image_object_id",
]
