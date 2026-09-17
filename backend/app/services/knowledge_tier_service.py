"""
三层知识库的"发布 / 层级变更"服务.

    企业知识库
      ├─ 个人知识库  张三自己可访问          access_level=private（默认）
      ├─ 部门知识库  技术部所有人按权限访问   access_level=department + department_id
      └─ 公司知识库  全公司按权限访问         access_level=tenant

本模块是**层级变更的唯一实现点**：无论是"有权限的人直接发布"，还是"普通员工
申请共享被审核通过后自动发布"，都走同一个 ``set_document_access_level``，
因此 PG 行、向量载荷、审计日志三处永远同步。

能力判定也放在这里（而不是散落在路由里），前端拿到的按钮状态与后端真正执行的
校验来自同一份逻辑 —— 不会出现"按钮能点但接口 403"。
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db.models import Document
from app.db.postgres import get_db_session
from app.services.audit_service import record_audit
from app.services.tenancy import (
    ACCESS_DEPARTMENT,
    # `required_permission_message(ACCESS_PRIVATE)` 用到它。此前没导入，而该分支
    # 恰好被 document_management 的 `if required and ...` 短路挡住（private 的
    # publish_requirement 是 None），于是**一直没暴露**：谁写个"为什么不能收回
    # 个人库"的接口就会 NameError → 500。pyflakes 扫出来的，见 _audit_916。
    ACCESS_PRIVATE,
    ACCESS_TENANT,
    access_label,
    normalize_access_level,
    publish_requirement,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class TierError(Exception):
    """层级变更失败（携带用户可读中文信息与 HTTP 状态码）。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def resolve_department_for_level(
    level: str,
    *,
    user_department_id: str | None,
    doc_department_id: str | None,
) -> str | None:
    """
    目标层级的归属部门。

    部门库必须带部门：优先用文档原有部门，其次用操作者所在部门。
    公司库/个人库一律清空部门归属（避免旧部门残留导致日后误判）。
    """
    if level == ACCESS_DEPARTMENT:
        return (doc_department_id or "").strip() or user_department_id
    return None


async def set_document_access_level(
    document_id: uuid.UUID,
    *,
    level: str,
    department_id: str | None,
    actor_id: uuid.UUID | None = None,
    actor_username: str | None = None,
    action: str = "document.publish",
    detail_extra: str = "",
) -> Document:
    """
    更新文档层级：PG 行 + 向量 ACL 载荷 + 审计，一次到位。

    Raises:
        TierError(404): 文档不存在。
    """
    normalized = normalize_access_level(level)

    async with get_db_session() as session:
        doc = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()
        if doc is None:
            raise TierError("文档不存在或已被删除", status_code=404)

        previous = normalize_access_level(doc.access_level)
        doc.access_level = normalized
        doc.department_id = (
            (department_id or "").strip() or None
            if normalized == ACCESS_DEPARTMENT
            else None
        )
        await session.flush()
        await session.refresh(doc)

    # 向量载荷与 PG 同步（失败只告警，不阻断——PG 是判定权威）
    from app.services.vector_service import update_document_access_payload

    applied = await update_document_access_payload(
        str(document_id),
        access_level=normalized,
        department_id=doc.department_id,
    )
    if not applied:
        # 返回值 0 说明**向量点还不存在**（上传后立刻发布 —— 正常操作路径）或
        # 同步调用失败。此时 PG 已经改好、列表接口立刻可见，但 Qdrant 的 ACL
        # 副本还是上传时的 private，而检索的 ACL 前置过滤读的正是那一份 →
        # 除 owner 本人谁都检索不到。这不是要在这里补救（点还不存在，无从补），
        # 而是必须让入库收尾来追平（resync_document_acl_payload）。
        logger.warning(
            "Document %s set to %s but no vectors matched (ingestion still running?) "
            "— vector-side ACL will be aligned at ingestion completion",
            document_id, normalized,
        )

    detail = (
        f"{access_label(previous)}→{access_label(normalized)}"
        f"; department={doc.department_id or '-'}"
    )
    if detail_extra:
        detail = f"{detail}; {detail_extra}"

    await record_audit(
        action,
        user_id=actor_id,
        username=actor_username,
        resource_type="document",
        resource_id=str(document_id),
        detail=detail,
    )

    logger.info(
        "Document %s access level: %s → %s (dept=%s) by %s",
        document_id, previous, normalized, doc.department_id, actor_username,
    )
    return doc


async def resync_document_acl_payload(
    document_id: uuid.UUID | str,
    *,
    reason: str = "ingestion_completion",
    expected: tuple[str | None, str | None] | None = None,
) -> int:
    """
    按 PostgreSQL 真值重写该文档全部向量点的 ACL 载荷（幂等）.

    为什么必须有这个函数 —— **层级变更可能发生在向量点写入之前**：

        上传是异步的（``POST /upload`` 受理即返回 202，管线丢后台），上传那一刻
        ``access_level`` 就固定成 private；而"传完立刻点发布"正是前端与验收脚本
        的正常路径。发布走 :func:`set_document_access_level` → Qdrant
        ``set_payload``，按 ``document_id`` 过滤更新：**此时向量点往往还没写入**，
        于是更新匹配 0 个点并静默成功（旧实现不校验匹配数，照样记一条
        "Updated vector ACL payload" 的成功日志）。随后入库又用上传时刻的快照
        private 把点写进去，两份事实永久分叉：

            PostgreSQL  access_level=department    ← 列表接口、详情、删除都看这份
            Qdrant      access_level=private       ← 检索的 ACL 前置过滤看这份

        表现极具误导性：**列表里看得见、点得开、权限判断全对，但提问时谁都
        检索不到**（owner 除外，因为 private 对 owner 本来就放行）。排查时容易
        一路去怀疑检索、阈值、证据门控，而真正的分叉在发布那一刻。

    在**入库收尾**（所有 upsert 完成之后、标记 COMPLETED 之前）调用本函数，
    是唯一"必然正确"的时点：此前发生的任何发布/收回都会被追平，此后发生的
    发布必然能匹配到点。这是一个**读时对齐 + 写时兜底**的收口，不依赖时序运气。

    Args:
        document_id: 文档 UUID（str 也可以）。
        reason:      仅用于日志，标明本次对齐的触发场景。
        expected:    入库时使用的 ``(access_level, department_id)`` 快照。
            两者与 PG 真值一致时**直接跳过**（99% 的文档从未发布，省掉一次
            无意义的 count + set_payload）。传 ``None`` 表示强制对齐 ——
            存量数据修复脚本用它。

    Returns:
        实际被改写的向量点数（跳过时为 0）。
    """
    doc_uuid = (
        document_id if isinstance(document_id, uuid.UUID) else uuid.UUID(str(document_id))
    )

    async with get_db_session() as session:
        row = (
            await session.execute(
                select(Document.access_level, Document.department_id).where(
                    Document.id == doc_uuid
                )
            )
        ).first()

    if row is None:
        logger.warning("resync ACL payload (%s): document %s not found", reason, doc_uuid)
        return 0

    level = normalize_access_level(row[0])
    # 与 set_document_access_level 同一口径：只有部门库保留部门归属，
    # 个人库/公司库一律清空，避免旧部门值残留导致日后误判。
    dept = row[1] if level == ACCESS_DEPARTMENT else None

    if expected is not None:
        exp_level = normalize_access_level(expected[0]) if expected[0] else None
        exp_dept = expected[1] if exp_level == ACCESS_DEPARTMENT else None
        if (level, dept) == (exp_level, exp_dept):
            logger.debug(
                "resync ACL payload (%s): document=%s already aligned (level=%s) — skipped",
                reason, doc_uuid, level,
            )
            return 0

    from app.services.vector_service import update_document_access_payload

    applied = await update_document_access_payload(
        str(doc_uuid), access_level=level, department_id=dept
    )
    if not applied:
        logger.warning(
            "resync ACL payload (%s): document=%s level=%s but no vectors matched — "
            "PostgreSQL and the vector store still disagree (retrieval will miss it)",
            reason, doc_uuid, level,
        )
    else:
        logger.info(
            "resync ACL payload (%s): document=%s level=%s dept=%s points=%d",
            reason, doc_uuid, level, dept, applied,
        )
    return applied


def publish_capability(user, doc) -> dict:
    """
    描述 *user* 对 *doc* 的层级操作能力（前端按钮 + 后端校验共用）.

    返回：
        {
          "is_owner": bool,
          "current_level": "private|department|tenant",
          "current_label": "个人|部门|公司",
          "can_publish_department": bool,   # 是否可**直接**发布到部门库
          "can_publish_company": bool,      # 是否可**直接**发布到公司库
          "needs_share_request": bool,      # 是否必须走「申请共享」
          "can_delete": bool,
          "delete_denied_reason": str,
          "publish_denied_reason": str,
        }
    """
    from app.services.permissions import has_permission
    from app.services.tenancy import (
        can_request_delete,
        delete_permission_for,
        normalize_access_level,
    )

    level = normalize_access_level(getattr(doc, "access_level", None))
    is_owner = bool(
        getattr(doc, "owner_id", None) is not None and user is not None
        and doc.owner_id == user.id
    )

    can_dept = has_permission(user, "document.publish.department")
    can_company = has_permission(user, "document.publish.company")

    # 个人库文档只有本人（或无权限者）需要申请；已发布到部门/公司的文档
    # 不再需要申请，除非要把层级再往上提。
    needs_request = is_owner and not can_dept and not can_company

    allowed, reason = delete_permission_for(doc, user) if user is not None else (False, "未登录")
    # 自己没有删除权但看得见 → 走「申请删除」，由上级同意或拒绝
    request_delete = can_request_delete(doc, user) if user is not None else False

    if not (can_dept or can_company):
        publish_reason = (
            "你的角色暂无发布权限：可点击「申请共享」提交给部门负责人或知识库管理员审核"
        )
    else:
        publish_reason = ""

    return {
        "is_owner": is_owner,
        "current_level": level,
        "current_label": access_label(level),
        "can_publish_department": can_dept,
        "can_publish_company": can_company,
        "needs_share_request": needs_request,
        "can_delete": allowed,
        "delete_denied_reason": "" if allowed else reason,
        "can_request_delete": request_delete,
        "publish_denied_reason": publish_reason,
    }


def required_permission_message(level: str) -> str:
    """发布到 *level* 所需权限缺失时的中文提示。"""
    permission = publish_requirement(level)
    if level == ACCESS_TENANT:
        return "发布到公司知识库需要「知识库管理员」及以上权限，请提交「申请共享」"
    if level == ACCESS_DEPARTMENT:
        return "发布到部门知识库需要「部门负责人」及以上权限，请提交「申请共享」"
    if level == ACCESS_PRIVATE:
        return "仅文档所有者可将文档收回个人知识库"
    return f"缺少权限：{permission}"


__all__ = [
    "TierError",
    "set_document_access_level",
    "resync_document_acl_payload",
    "publish_capability",
    "required_permission_message",
    "resolve_department_for_level",
]
