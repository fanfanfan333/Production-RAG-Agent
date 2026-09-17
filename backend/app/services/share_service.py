"""
「申请共享」闭环服务：个人文档 → 部门库 / 公司库.

权限矩阵里"发布到部门库/公司库"的那一列，对普通员工是**申请制**：他们不能
直接发布，只能提交申请，由上一级权限持有者审核。

    目标层级    审核人                              批准后的效果
    部门库      本部门负责人（dept_manager）        文档转为部门库并写入部门
    公司库      知识库管理员 / 企业管理员           文档转为公司库，全公司可见

审核通过后调用 ``knowledge_tier_service.set_document_access_level`` 完成
真正的层级变更（PG + 向量载荷 + 审计一次性同步），本模块不自己改文档字段 ——
避免出现第二处"改层级"的实现。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select

from app.db.models import Document
from app.db.postgres import get_db_session
from app.db.share_models import ShareRequest
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.services.knowledge_tier_service import (
    TierError,
    set_document_access_level,
)
from app.services.permissions import has_permission, role_label
from app.services.tenancy import (
    ACCESS_DEPARTMENT,
    ACCESS_TENANT,
    access_label,
    access_scope_name,
    effective_department_id,
    effective_tenant_id,
    normalize_tenant_id,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ShareError(Exception):
    """共享申请流程失败（用户可读中文 + HTTP 状态码）。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# 申请意图 → 中文（前端列表 / 审核按钮 / 审计日志共用一份）
INTENT_LABELS: dict[str, str] = {
    ShareRequest.INTENT_PUBLISH: "申请共享",
    ShareRequest.INTENT_DELETE: "申请删除",
}


# ── 审核人判定（单一实现点）───────────────────────────────────────────────────

def review_scope(user: User) -> str | None:
    """
    当前用户能审核到哪一级：``"company"`` / ``"department"`` / ``None``.

    公司级审核权（知识库管理员、企业管理员、平台管理员）天然覆盖部门级申请。
    """
    if user is None:
        return None
    if user.is_admin or has_permission(user, "share.review.company"):
        return "company"
    if has_permission(user, "share.review.department"):
        return "department"
    return None


def can_review(request: ShareRequest, user: User) -> bool:
    """单条申请是否轮得到 *user* 审核（同时排除自己的申请）。"""
    scope = review_scope(user)
    if scope is None:
        return False
    if request.requester_id == user.id:
        return False  # 不能自审
    # 公司边界：平台管理员是唯一例外（它要给所有公司配管理层、审所有公司的申请）。
    # 这里必须显式放行 —— 否则 admin 会在待办列表里"看得见但点不动"。
    if not user.is_admin and normalize_tenant_id(request.tenant_id) != effective_tenant_id(user):
        return False
    if scope == "company":
        return True
    # 部门级审核人：只看得到本部门的部门库申请
    if request.target_level != ACCESS_DEPARTMENT:
        return False
    dept = effective_department_id(user)
    return bool(dept) and dept == (request.target_department_id or "").strip()


# ── 序列化 ────────────────────────────────────────────────────────────────────

def serialize(request: ShareRequest, *, viewer: User | None = None) -> dict:
    """统一的出参结构（前端"查看申请"页直接用，无需二次映射）。"""
    intent = (getattr(request, "intent", None) or ShareRequest.INTENT_PUBLISH).strip()
    return {
        "id": str(request.id),
        # 删除申请批准后文档已不存在 → 保持 None，前端据此显示「文档已删除」
        "document_id": str(request.document_id) if request.document_id else None,
        "document_name": request.document_name,
        "intent": intent,
        "intent_label": INTENT_LABELS.get(intent, "申请共享"),
        "requester_username": request.requester_username,
        "requester_department_id": request.requester_department_id,
        "target_level": request.target_level,
        "target_label": access_scope_name(request.target_level),
        "target_department_id": request.target_department_id,
        "reason": request.reason,
        "status": request.status,
        "reviewer_username": request.reviewer_username,
        "review_comment": request.review_comment,
        "created_at": request.created_at.isoformat() if request.created_at else None,
        "reviewed_at": request.reviewed_at.isoformat() if request.reviewed_at else None,
        "requester_seen": request.requester_seen,
        "is_mine": bool(viewer and request.requester_id == viewer.id),
        "can_review": bool(viewer and can_review(request, viewer)),
    }


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── 申请 ──────────────────────────────────────────────────────────────────────

async def create_share_request(
    user: User,
    document_id: uuid.UUID,
    target_level: str,
    reason: str | None = None,
) -> ShareRequest:
    """提交一份共享申请（仅文档归属人可提，且只能往上提一层语义）。"""
    if target_level not in ShareRequest.VALID_TARGETS:
        raise ShareError("目标层级不合法：只能申请发布到「部门知识库」或「公司知识库」")

    permission = (
        "document.publish.company"
        if target_level == ACCESS_TENANT
        else "document.publish.department"
    )
    if has_permission(user, permission):
        raise ShareError(
            f"你当前角色（{role_label(user.role)}）已可直接发布到"
            f"{access_scope_name(target_level)}，无需申请",
            status_code=409,
        )

    async with get_db_session() as session:
        doc = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()
        if doc is None:
            raise ShareError("文档不存在或已被删除", status_code=404)

        if normalize_tenant_id(doc.tenant_id) != effective_tenant_id(user):
            # 跨公司：不暴露存在性
            raise ShareError("文档不存在或无权访问", status_code=404)

        if doc.owner_id != user.id:
            raise ShareError(
                "只有文档的归属人可以申请共享他人的查看权限；"
                "如需共享同事的文档，请让 TA 本人提交申请",
                status_code=403,
            )

        if (doc.access_level or "private") == target_level:
            raise ShareError(
                f"该文档已经在{access_scope_name(target_level)}中，无需重复申请"
            )

        # 同文档同目标只允许一份待审申请（防重复点击刷屏审核队列）
        duplicate = await session.scalar(
            select(ShareRequest).where(
                ShareRequest.document_id == document_id,
                ShareRequest.target_level == target_level,
                ShareRequest.status == ShareRequest.STATUS_PENDING,
            ).limit(1)
        )
        if duplicate is not None:
            raise ShareError(
                f"该文档已有一份待审核的「{access_scope_name(target_level)}」申请，"
                "请耐心等待审核结果",
                status_code=409,
            )

        request = ShareRequest(
            id=uuid.uuid4(),
            tenant_id=effective_tenant_id(user),
            document_id=document_id,
            document_name=doc.filename,
            requester_id=user.id,
            requester_username=user.username,
            requester_department_id=effective_department_id(user),
            target_level=target_level,
            target_department_id=(
                effective_department_id(user)
                if target_level == ACCESS_DEPARTMENT
                else None
            ),
            reason=(reason or "").strip() or None,
            status=ShareRequest.STATUS_PENDING,
        )
        session.add(request)
        await session.flush()
        await session.refresh(request)

    await record_audit(
        "share.request.create",
        user_id=user.id,
        username=user.username,
        resource_type="document",
        resource_id=str(document_id),
        detail=f"target={access_scope_name(target_level)}; reason={request.reason or '-'}",
    )
    logger.info(
        "Share request created: doc=%s target=%s by=%s",
        doc.filename, target_level, user.username,
    )
    return request


# ── 申请删除 ──────────────────────────────────────────────────────────────────

async def create_delete_request(
    user: User,
    document_id: uuid.UUID,
    reason: str | None = None,
) -> ShareRequest:
    """
    提交「申请删除」—— 自己没有删除权、但看得见某份部门库/公司库文档时使用.

    与「申请共享」共用一张表与同一套审核范围判定：
        文档在部门库 → 本部门负责人审
        文档在公司库 → 知识库管理员 / 企业管理员审

    拒绝的三种情况刻意分开表述，避免用户反复试：
        * 看不见该文档      → 404（不泄漏存在性）
        * 本来就能直接删    → 409 + "你可以直接删除"
        * 个人库归属人      → 同上（归属人本就随时可删）
    """
    from app.services.tenancy import (
        can_request_delete,
        delete_permission_for,
        normalize_access_level,
    )

    async with get_db_session() as session:
        doc = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()
        if doc is None:
            raise ShareError("文档不存在或已被删除", status_code=404)

        if normalize_tenant_id(doc.tenant_id) != effective_tenant_id(user):
            raise ShareError("文档不存在或无权访问", status_code=404)

        allowed, deny_reason = delete_permission_for(doc, user)
        if allowed:
            raise ShareError(
                "你有权直接删除该文档，无需提交申请", status_code=409
            )

        if not can_request_delete(doc, user):
            # 个人库他人文档：连可见性都没有，不能拿它当探测接口
            raise ShareError("文档不存在或无权访问", status_code=404)

        level = normalize_access_level(doc.access_level)

        duplicate = await session.scalar(
            select(ShareRequest).where(
                ShareRequest.document_id == document_id,
                ShareRequest.intent == ShareRequest.INTENT_DELETE,
                ShareRequest.status == ShareRequest.STATUS_PENDING,
            ).limit(1)
        )
        if duplicate is not None:
            raise ShareError(
                "该文档已有一份待审核的删除申请，请耐心等待审核结果",
                status_code=409,
            )

        request = ShareRequest(
            id=uuid.uuid4(),
            tenant_id=effective_tenant_id(user),
            document_id=document_id,
            document_name=doc.filename,
            intent=ShareRequest.INTENT_DELETE,
            requester_id=user.id,
            requester_username=user.username,
            requester_department_id=effective_department_id(user),
            # 删除申请的 target_level 记录文档**当前**层级：它决定谁来审
            target_level=level,
            target_department_id=(
                (doc.department_id or "").strip() or None
                if level == ACCESS_DEPARTMENT
                else None
            ),
            reason=(reason or "").strip() or None,
            status=ShareRequest.STATUS_PENDING,
        )
        session.add(request)
        await session.flush()
        await session.refresh(request)

    await record_audit(
        "share.request.delete.create",
        user_id=user.id,
        username=user.username,
        resource_type="document",
        resource_id=str(document_id),
        detail=(
            f"intent=delete; level={level}; doc={doc.filename}; "
            f"reason={request.reason or '-'}; deny={deny_reason or '-'}"
        ),
    )
    logger.info(
        "Delete request created: doc=%s level=%s by=%s",
        doc.filename, level, user.username,
    )
    return request


# ── 查询 ──────────────────────────────────────────────────────────────────────

async def list_my_requests(
    user: User,
    *,
    status: str | None = None,
    limit: int = 100,
) -> list[ShareRequest]:
    async with get_db_session() as session:
        stmt = (
            select(ShareRequest)
            .where(ShareRequest.requester_id == user.id)
            .order_by(ShareRequest.created_at.desc())
            .limit(limit)
        )
        if status:
            stmt = stmt.where(ShareRequest.status == status)
        return list((await session.execute(stmt)).scalars().all())


async def list_inbox(user: User, *, limit: int = 100) -> list[ShareRequest]:
    """待我审核（按我的审核范围裁剪，含已审记录以便回溯）。"""
    scope = review_scope(user)
    if scope is None:
        return []

    async with get_db_session() as session:
        stmt = (
            select(ShareRequest)
            .where(
                ShareRequest.tenant_id == effective_tenant_id(user),
                ShareRequest.requester_id != user.id,
            )
            .order_by(ShareRequest.created_at.desc())
            .limit(limit)
        )
        if scope == "department":
            dept = effective_department_id(user)
            if not dept:
                return []
            stmt = stmt.where(
                ShareRequest.target_level == ACCESS_DEPARTMENT,
                ShareRequest.target_department_id == dept,
            )
        return list((await session.execute(stmt)).scalars().all())


async def summary(user: User) -> dict:
    """主界面角标数据：待我审核 / 我待出结果 / 有结论但未读。"""
    scope = review_scope(user)
    inbox_count = 0
    if scope is not None:
        async with get_db_session() as session:
            stmt = select(func.count()).select_from(ShareRequest).where(
                ShareRequest.tenant_id == effective_tenant_id(user),
                ShareRequest.requester_id != user.id,
                ShareRequest.status == ShareRequest.STATUS_PENDING,
            )
            if scope == "department":
                dept = effective_department_id(user)
                if not dept:
                    # 部门负责人尚未归属部门时看不到任何部门级申请
                    return {
                        "pending_for_me": 0,
                        "my_pending": 0,
                        "my_decided_unseen": 0,
                        "review_scope": scope,
                        "can_review": True,
                        "total_badge": 0,
                    }
                stmt = stmt.where(
                    ShareRequest.target_level == ACCESS_DEPARTMENT,
                    # 括号不能省：`== dept or ""` 会被解析成 `(== dept) or ""`，
                    # 把空字符串塞进 where() → 500。
                    ShareRequest.target_department_id == (dept or ""),
                )
            inbox_count = int((await session.execute(stmt)).scalar_one() or 0)

    async with get_db_session() as session:
        mine_pending = int(
            (
                await session.execute(
                    select(func.count()).select_from(ShareRequest).where(
                        ShareRequest.requester_id == user.id,
                        ShareRequest.status == ShareRequest.STATUS_PENDING,
                    )
                )
            ).scalar_one()
            or 0
        )
        decided_unseen = int(
            (
                await session.execute(
                    select(func.count()).select_from(ShareRequest).where(
                        ShareRequest.requester_id == user.id,
                        ShareRequest.status != ShareRequest.STATUS_PENDING,
                        ShareRequest.requester_seen.is_(False),
                    )
                )
            ).scalar_one()
            or 0
        )

    return {
        "pending_for_me": inbox_count,
        "my_pending": mine_pending,
        "my_decided_unseen": decided_unseen,
        "review_scope": scope,
        "can_review": scope is not None,
        "total_badge": inbox_count + decided_unseen,
    }


# ── 审核 ──────────────────────────────────────────────────────────────────────

async def review_request(
    reviewer: User,
    request_id: uuid.UUID,
    *,
    approve: bool,
    comment: str | None = None,
) -> tuple[ShareRequest, dict | None, dict | None]:
    """
    同意 / 拒绝一份申请（发布与删除共用入口）.

    Returns:
        ``(申请行, 发布结果, 删除结果)``
        —— 批准"发布"申请时第二个值非空；批准"删除"申请时第三个值非空；
           拒绝时两个都是 None。
    """
    async with get_db_session() as session:
        request = await session.get(ShareRequest, request_id)
        if request is None:
            raise ShareError("申请不存在", status_code=404)
        if not can_review(request, reviewer):
            raise ShareError(
                "你没有权限审核该申请（可能是跨公司、跨部门，或这是你自己的申请）",
                status_code=403,
            )
        if request.status != ShareRequest.STATUS_PENDING:
            raise ShareError(
                f"该申请已处理（当前状态：{request.status}），无需重复操作",
                status_code=409,
            )

        document_id = request.document_id
        intent = (
            getattr(request, "intent", None) or ShareRequest.INTENT_PUBLISH
        ).strip()
        target_level = request.target_level
        target_department_id = request.target_department_id
        requester_username = request.requester_username
        document_name = request.document_name

    published: dict | None = None
    deleted: dict | None = None
    if approve and intent == ShareRequest.INTENT_DELETE:
        # 批准删除：真正删掉文档（Qdrant 向量 → PG 行）。
        # 先删除、后改状态：删除失败（Qdrant 不可用 / 无权限）时**不**把申请
        # 标成"已通过"，否则会出现"显示已批准但文档还在"的假象。
        from app.services.document_query_service import (
            PermissionDenied as _DocPermissionDenied,
            delete_document as _delete_document,
        )

        if document_id is None:
            raise ShareError("该文档已被删除，无需审核", status_code=409)
        try:
            await _delete_document(document_id, actor=reviewer)
            deleted = {"document_id": str(document_id), "document_name": document_name}
        except KeyError:
            # 文档已不在（别人先删了 / 重复审核）——按"目的已达成"处理，
            # 仍把申请关闭，避免它永远挂在自己的待办里。
            deleted = {
                "document_id": str(document_id),
                "document_name": document_name,
                "already_gone": True,
            }
        except _DocPermissionDenied as exc:
            raise ShareError(
                f"你没有权限删除该文档：{exc}", status_code=403
            ) from None
    elif approve:
        # 批准 = 真正发布到目标层级（PG + 向量载荷 + 审计）
        doc = await set_document_access_level(
            document_id,
            level=target_level,
            department_id=target_department_id,
            actor_id=reviewer.id,
            actor_username=reviewer.username,
            action="share.request.approve",
            detail_extra=f"申请={request_id}; 申请人={requester_username}",
        )
        published = {
            "document_id": str(doc.id),
            "access_level": doc.access_level,
            "access_label": access_label(doc.access_level),
            "department_id": doc.department_id,
        }

    async with get_db_session() as session:
        request = await session.get(ShareRequest, request_id)
        if request is None or request.status != ShareRequest.STATUS_PENDING:
            raise ShareError("该申请已被处理", status_code=409)
        request.status = (
            ShareRequest.STATUS_APPROVED if approve else ShareRequest.STATUS_REJECTED
        )
        request.reviewer_id = reviewer.id
        request.reviewer_username = reviewer.username
        request.review_comment = (comment or "").strip() or None
        request.reviewed_at = _now()
        request.requester_seen = False     # 申请人下次打开"查看申请"看到未读
        await session.flush()
        await session.refresh(request)

    await record_audit(
        "share.request.review",
        user_id=reviewer.id,
        username=reviewer.username,
        resource_type="document",
        # 删除申请批准后 document_id 可能已为 NULL（SET NULL），退回申请行 id
        resource_id=str(request.document_id or request.id),
        detail=(
            f"{'approved' if approve else 'rejected'}; "
            f"intent={intent}; "
            f"target={access_scope_name(request.target_level)}; "
            f"requester={requester_username}; doc={document_name}; "
            f"comment={request.review_comment or '-'}"
        ),
    )
    logger.info(
        "Share request %s (%s) %s by %s (doc=%s)",
        request_id, intent, "approved" if approve else "rejected",
        reviewer.username, document_name,
    )
    return request, published, deleted


async def cancel_request(user: User, request_id: uuid.UUID) -> ShareRequest:
    """申请人主动撤回待审申请。"""
    async with get_db_session() as session:
        request = await session.get(ShareRequest, request_id)
        if request is None:
            raise ShareError("申请不存在", status_code=404)
        if request.requester_id != user.id:
            raise ShareError("只能撤回自己提交的申请", status_code=403)
        if request.status != ShareRequest.STATUS_PENDING:
            raise ShareError("该申请已处理，无法撤回", status_code=409)

        request.status = ShareRequest.STATUS_CANCELLED
        request.reviewed_at = _now()
        request.requester_seen = True
        await session.flush()
        await session.refresh(request)

    await record_audit(
        "share.request.cancel",
        user_id=user.id,
        username=user.username,
        resource_type="document",
        resource_id=str(request.document_id),
        detail=f"申请={request_id}",
    )
    return request


async def mark_my_requests_seen(user: User) -> int:
    """把"我的申请"里的审核结论标记为已读（清角标）。"""
    async with get_db_session() as session:
        rows = list(
            (
                await session.execute(
                    select(ShareRequest).where(
                        ShareRequest.requester_id == user.id,
                        ShareRequest.status != ShareRequest.STATUS_PENDING,
                        ShareRequest.requester_seen.is_(False),
                    )
                )
            ).scalars().all()
        )
        for row in rows:
            row.requester_seen = True
        await session.flush()
    return len(rows)


__all__ = [
    "ShareError",
    "INTENT_LABELS",
    "create_share_request",
    "create_delete_request",
    "list_my_requests",
    "list_inbox",
    "summary",
    "review_request",
    "cancel_request",
    "mark_my_requests_seen",
    "serialize",
    "can_review",
    "review_scope",
    "TierError",
]
