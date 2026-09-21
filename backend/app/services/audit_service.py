"""
Audit logging service (企业落地第一阶段).

Writes one AuditLog row per auditable action. Best-effort by design: an
audit failure is logged and swallowed so it can never break the primary
operation (upload/query/delete).
"""

import uuid

from app.db.postgres import get_db_session
from app.db.user_models import AuditLog
from app.utils.logging import get_logger

logger = get_logger(__name__)


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
        if isinstance(user_id, str):
            try:
                user_id = uuid.UUID(user_id)
            except ValueError:
                user_id = None

        async with get_db_session() as session:
            session.add(
                AuditLog(
                    id=uuid.uuid4(),
                    user_id=user_id,
                    username=(username or "")[:64] or None,
                    action=action[:64],
                    resource_type=(resource_type or "")[:64] or None,
                    resource_id=(resource_id or "")[:64] or None,
                    detail=(detail or "")[:2000] or None,
                    ip=(ip or "")[:64] or None,
                )
            )
        logger.debug("audit: %s user=%s resource=%s/%s", action, username, resource_type, resource_id)
    except Exception as exc:
        logger.warning("Failed to write audit log for action=%s: %s", action, exc)


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

    ⚠️ ``AuditLog.resource_id`` 为 ``String(64)``、``detail`` 截断 2000 字符 ——
    这里统一按列宽截断，避免长对象 id 把写入搞崩（详见设计 §15-13）。
    """
    parts: list[str] = []
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
    detail = " ".join(parts) or None

    await record_audit(
        f"acl.drop.{stage}",
        user_id=user_id,
        username=username,
        resource_type="document_object",
        resource_id=(str(object_id)[:64] if object_id else None),
        detail=(detail[:2000] if detail else None),
    )
