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
