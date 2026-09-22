"""
Audit logging service (企业落地第一阶段).

Writes one AuditLog row per auditable action. Best-effort by design: an
audit failure is logged and swallowed so it can never break the primary
operation (upload/query/delete).
"""

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from app.db.postgres import get_db_session
from app.db.user_models import AuditLog
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _audit_entry(
    action: str,
    *,
    user_id: uuid.UUID | str | None = None,
    username: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    detail: str | None = None,
    ip: str | None = None,
) -> dict:
    """
    把一次审计的入参规范化成 ``AuditLog`` 的列值（**唯一**的截断实现点）.

    单条 :func:`record_audit` 与批量 :func:`record_audit_many` 共用本函数 ——
    两边各写一份列宽截断，迟早会出现"批量能过、单条被截"这类只在某条路径上
    复现的脏数据。
    """
    if isinstance(user_id, str):
        try:
            user_id = uuid.UUID(user_id)
        except ValueError:
            user_id = None
    return {
        "id": uuid.uuid4(),
        "user_id": user_id,
        "username": (username or "")[:64] or None,
        "action": action[:64],
        "resource_type": (resource_type or "")[:64] or None,
        "resource_id": (resource_id or "")[:64] or None,
        "detail": (detail or "")[:2000] or None,
        "ip": (ip or "")[:64] or None,
    }


async def record_audit(
    action: str,
    *,
    user_id: uuid.UUID | str | None = None,
    username: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    detail: str | None = None,
    ip: str | None = None,
) -> None:
    """Persist one audit entry; never raises."""
    try:
        async with get_db_session() as session:
            session.add(AuditLog(**_audit_entry(
                action, user_id=user_id, username=username,
                resource_type=resource_type, resource_id=resource_id,
                detail=detail, ip=ip,
            )))
        logger.debug("audit: %s user=%s resource=%s/%s", action, username, resource_type, resource_id)
    except Exception as exc:
        logger.warning("Failed to write audit log for action=%s: %s", action, exc)


async def record_audit_many(entries: Sequence[Mapping[str, Any]]) -> None:
    """
    批量写审计：**一条事务、一次 commit**，每项仍单独成行.

    为什么需要：第 11 环一次检索可能剔除几十个对象（权限刚收紧时最多，而那正是
    最该快的时候）。逐条 ``await record_audit`` = 几十次串行事务，全压在用户请求
    路径上 —— 写库次数与剔除数量线性相关。``add_all`` 之后与数量**解耦**。

    ``entries`` 是 :func:`record_audit` 的关键字参数（每项须含 ``action``）。
    与单条版同为 best-effort：写失败只告警，绝不抛。
    """
    if not entries:
        return
    try:
        async with get_db_session() as session:
            session.add_all([
                AuditLog(**_audit_entry(
                    str(e.get("action") or ""),
                    user_id=e.get("user_id"),
                    username=e.get("username"),
                    resource_type=e.get("resource_type"),
                    resource_id=e.get("resource_id"),
                    detail=e.get("detail"),
                    ip=e.get("ip"),
                ))
                for e in entries
            ])
        logger.debug("audit: wrote %d entries in one batch", len(entries))
    except Exception as exc:
        logger.warning("Failed to write %d audit log(s) in batch: %s", len(entries), exc)


def _acl_drop_detail(
    stage: str,
    *,
    document_id: str | None = None,
    gate: str | None = None,
    reason: str | None = None,
    scope_fingerprint: str | None = None,
    acl_sync_state: str | None = None,
    extra: str | None = None,
) -> str | None:
    """``acl.drop`` 审计的 detail 文本（单条 / 批量共用，格式只写一遍）."""
    parts: list[str] = [f"stage={stage}"]
    if document_id:
        parts.append(f"document_id={document_id}")
    if gate:
        parts.append(f"gate={gate}")
    if reason:
        parts.append(f"reason={reason}")
    if scope_fingerprint:
        parts.append(f"scope_fingerprint={scope_fingerprint}")
    if acl_sync_state:
        parts.append(f"acl_sync_state={acl_sync_state}")
    if extra:
        parts.append(extra)
    return " ".join(parts) or None


async def record_acl_drops(
    stage: str,
    drops: Sequence[Mapping[str, Any]],
    *,
    user_id: uuid.UUID | str | None = None,
    username: str | None = None,
    scope_fingerprint: str | None = None,
) -> None:
    """
    批量写 ``acl.drop.<stage>`` 审计（**一条事务写完同批全部剔除**）.

    每条剔除**仍单独成行**（``resource_id`` = 对象 id，可逐对象追溯 —— PRD P0-10
    的硬要求不变），只是不再各占一次事务。``drops`` 每项：
    ``{"object_id", "document_id", "reason", "gate", "acl_sync_state", "extra"}``
    （后三项可缺省）。

    调用方是第 11 环的对象级复核（一次检索可能剔除几十个候选）；第 12 环的剔除
    数量受上下文条数约束，走单条 :func:`record_acl_drop` 即可。
    """
    if not drops:
        return
    await record_audit_many([
        {
            "action": f"acl.drop.{stage}",
            "user_id": user_id,
            "username": username,
            "resource_type": "document_object",
            "resource_id": (str(d.get("object_id") or d.get("document_id") or "")[:64] or None),
            "detail": _acl_drop_detail(
                stage,
                document_id=(str(d["document_id"]) if d.get("document_id") else None),
                gate=d.get("gate"),
                reason=d.get("reason"),
                scope_fingerprint=scope_fingerprint,
                acl_sync_state=d.get("acl_sync_state"),
                extra=d.get("extra"),
            ),
        }
        for d in drops
    ])


async def record_acl_drop(
    stage: str,
    *,
    object_id: str | None = None,
    document_id: str | None = None,
    reason: str | None = None,
    gate: str | None = None,
    user_id: uuid.UUID | str | None = None,
    username: str | None = None,
    scope_fingerprint: str | None = None,
    acl_sync_state: str | None = None,
    extra: str | None = None,
) -> None:
    """
    Record one ACL-drop audit entry (`acl.drop.<stage>`).

    权限剔除**必须留痕**（PRD P0-10 / A4）：否则"用户说搜不到、运维查不出为什么"
    会成为常态。三处剔除共用本函数，只靠 ``stage`` 区分：

        stage ∈ {"prefilter", "postcheck", "final_check", "citation_open"}
            → action = f"acl.drop.{stage}"

    与 :func:`record_audit` 一样是 best-effort：审计写失败绝不阻断主流程。
    单条版是 :func:`record_acl_drops` 的薄包装（detail 格式同源，不分叉）。

    ⚠️ ``AuditLog.resource_id`` 为 ``String(64)``、``detail`` 截断 2000 字符 ——
    这里统一按列宽截断，避免长对象 id 把写入搞崩（详见设计 §15-13）。
    """
    await record_acl_drops(
        stage,
        [{
            "object_id": object_id,
            "document_id": document_id,
            "reason": reason,
            "gate": gate,
            "acl_sync_state": acl_sync_state,
            "extra": extra,
        }],
        user_id=user_id,
        username=username,
        scope_fingerprint=scope_fingerprint,
    )
