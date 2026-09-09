"""
Legacy data backfill (多用户改造后的旧数据初始化).

多用户改造前上传的 Document / Conversation 行 owner_id 为 NULL：
  - 判重全局命中，但对非管理员不可见 → "第一次上传却提示已索引过"
  - 旧会话/文档谁也管理不了

本模块在应用启动时把无主记录划给**第一个管理员**，使其重新可见、
可管理。幂等：只处理 owner_id IS NULL 的行；没有管理员时跳过。
"""

import uuid

from sqlalchemy import select, update

from app.db.models import Document
from app.db.conversation_models import Conversation
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def assign_legacy_records_to_admin() -> None:
    """把 owner 为 NULL 的旧文档/会话划给第一个管理员。幂等，可每次启动调用。"""
    async with get_db_session() as session:
        admin_id: uuid.UUID | None = await session.scalar(
            select(User.id)
            .where(User.role == User.ROLE_ADMIN, User.is_active.is_(True))
            .order_by(User.created_at.asc())
            .limit(1)
        )
        if admin_id is None:
            logger.debug("legacy backfill: no admin yet — skip")
            return

        docs_result = await session.execute(
            update(Document)
            .where(Document.owner_id.is_(None))
            .values(owner_id=admin_id)
        )
        convs_result = await session.execute(
            update(Conversation)
            .where(Conversation.owner_id.is_(None))
            .values(owner_id=admin_id)
        )

    if docs_result.rowcount or convs_result.rowcount:
        logger.info(
            "legacy backfill: assigned %d document(s) and %d conversation(s) "
            "to admin %s",
            docs_result.rowcount,
            convs_result.rowcount,
            admin_id,
        )
