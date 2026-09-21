"""
安全隔离管理面 API（``/security``）—— T5，设计文档决策 13 / 14 / §4.5 / §15-13.

    方法   路径                                          权限
    ────   ────────────────────────────────────────────  ─────────────────────────
    POST   /security/grants                              security.grant（kb_admin+）
    GET    /security/grants                              security.grant
    POST   /security/grants/{gid}/review                 security.review.grant
    POST   /security/grants/{gid}/revoke                 security.review.grant
    PATCH  /security/documents/{did}                     security.escalate
    PATCH  /security/documents/{did}/objects/{oid}       security.escalate
    GET    /security/settings                            平台管理员

**禁止自我授予**在这里被真正拦住：``POST /security/grants`` 先判主体是否等于
调用者本人，是则 **403 + 审计**（服务层还有第二道，见 ``grant_service``）。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.deps import get_current_user, require_admin
from app.config import get_settings
from app.db.security_models import DEFAULT_SECURITY_LEVEL
from app.db.user_models import User
from app.schemas.security import (
    DocumentSecurityUpdate,
    GrantCreate,
    GrantItem,
    GrantReview,
    ObjectEscalate,
    SecuritySettings,
)
from app.services.audit_service import record_audit
from app.services.grant_service import (
    GrantError,
    counted_null_effective_levels,
    escalate_object,
    is_self_grant,
    list_grants,
    request_grant,
    review_grant,
    revoke_grant,
    set_document_security,
)
from app.services.permissions import require_permission

router = APIRouter(prefix="/security", tags=["Security Isolation"])

_require_grant = require_permission("security.grant")
_require_review = require_permission("security.review.grant")
_require_escalate = require_permission("security.escalate")


def _raise(exc: GrantError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None


def _serialize_grant(grant) -> dict:
    return GrantItem(
        grant_id=str(grant.id),
        document_id=str(grant.document_id),
        object_id=str(grant.object_id),
        subject=str(grant.subject),
        effect=str(grant.effect),
        status=str(grant.status),
        granted_by=str(grant.granted_by) if grant.granted_by else None,
        reviewer_id=str(grant.reviewer_id) if grant.reviewer_id else None,
        reason=grant.reason,
        expires_at=grant.expires_at,
        created_at=grant.created_at,
        reviewed_at=grant.reviewed_at,
    ).model_dump()


# ── need-to-know 授予 ─────────────────────────────────────────────────────────


@router.post(
    "/grants",
    status_code=status.HTTP_201_CREATED,
    summary="提交 need-to-know 授予申请（security.grant）",
    description=(
        "对象级例外授予，必须带有效期、必须走审批链。**禁止自我授予**"
        "（主体设成自己 → 403 + 审计）。主体格式：``user:<uuid>`` / ``dept:<id>`` / "
        "``role:<role>`` / ``project:<id>`` / ``group:<id>``。\n\n"
        "只能在 ``doc`` / ``image`` 对象上授予（派生对象不得通过 acl_allow 获得"
        "父之外的可见性）。``pending`` 期间**不写** ``acl_allow``，被授权人读不到，"
        "批准后才生效。"
    ),
)
async def create_grant_endpoint(
    body: GrantCreate,
    actor: Annotated[User, Depends(_require_grant)],
) -> dict:
    # ── 禁止自我授予（API 层第一道拦截，写审计）──────────────────────────────
    if is_self_grant(body.subject, getattr(actor, "id", None)):
        await record_audit(
            "security.grant.request",
            user_id=getattr(actor, "id", None),
            username=getattr(actor, "username", None),
            resource_type="document",
            resource_id=str(body.document_id)[:64],
            detail=f"denied=self_grant(api); subject={body.subject}",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="禁止自我授予：不能把 need-to-know 例外授予给自己",
        )

    try:
        grant = await request_grant(
            actor,
            body.document_id,
            body.subject,
            effect=body.effect,
            reason=body.reason,
            expires_at=body.expires_at,
            object_id=body.object_id,
        )
    except GrantError as exc:
        _raise(exc)
    return _serialize_grant(grant)


@router.get(
    "/grants",
    summary="列出授予（security.grant）",
    description="可按 ``document_id`` / ``status`` 过滤；非平台管理员只见本公司的授予。",
)
async def list_grants_endpoint(
    user: Annotated[User, Depends(_require_grant)],
    document_id: str | None = None,
    status_filter: str | None = None,
) -> dict:
    rows = await list_grants(user, document_id=document_id, status=status_filter)
    items = [_serialize_grant(g) for g in rows]
    return {"items": items, "total": len(items)}


@router.post(
    "/grants/{grant_id}/review",
    summary="审批授予（security.review.grant）",
    description=(
        "批准即把对象级例外物化进 ``document_objects.acl_allow``（取**最早**到期时间），"
        "被授权人此后可读；拒绝则不改任何可见性。**审批人不得审批授予给自己的申请**。"
    ),
)
async def review_grant_endpoint(
    grant_id: Annotated[str, Path(description="授予 id")],
    body: GrantReview,
    reviewer: Annotated[User, Depends(_require_review)],
) -> dict:
    try:
        grant = await review_grant(
            reviewer, grant_id, approve=body.approve, comment=body.comment
        )
    except GrantError as exc:
        _raise(exc)
    return _serialize_grant(grant)


@router.post(
    "/grants/{grant_id}/revoke",
    summary="撤销授予（security.review.grant）",
    description="撤销 待审/已批准 的授予，并立即重算物化副本（被授权人随之失去例外）。",
)
async def revoke_grant_endpoint(
    grant_id: Annotated[str, Path(description="授予 id")],
    actor: Annotated[User, Depends(_require_review)],
) -> dict:
    try:
        grant = await revoke_grant(actor, grant_id)
    except GrantError as exc:
        _raise(exc)
    return _serialize_grant(grant)


# ── 密级 / 可见性 / 项目维度 ──────────────────────────────────────────────────


@router.patch(
    "/documents/{document_id}",
    summary="设置文档密级 / 可见性 / 项目集合（security.escalate）",
    description=(
        "只修改显式传入的字段。变更会触发 T4 的**取严级联**：派生对象有效密级"
        "取 ``max(文档, 自身)``，源图片收紧时其 OCR 派生块同步取严；项目维度"
        "同步到全部对象行；need-to-know 副本按权威源重新物化。\n\n"
        "⚠️ ``access_level``（private/department/tenant）**不在此接口内** —— "
        "三值语义一行未改，归属变更仍走既有的共享 / 层级链路。"
    ),
)
async def set_document_security_endpoint(
    document_id: Annotated[str, Path(description="文档 id")],
    body: DocumentSecurityUpdate,
    actor: Annotated[User, Depends(_require_escalate)],
) -> dict:
    try:
        result = await set_document_security(
            actor,
            document_id,
            security_level=body.security_level,
            visibility_mode=body.visibility_mode,
            project_ids=body.project_ids,
        )
    except GrantError as exc:
        _raise(exc)
    return result


@router.patch(
    "/documents/{document_id}/objects/{object_id}",
    summary="对象级提级 / 剔除（security.escalate）",
    description=(
        "对单个对象提级或剔除。**图片对象**会触发 OCR 派生取严级联："
        "其派生的 text/table/code 块同步变严，堵住「图看不了、字还能搜」的泄密面。"
    ),
)
async def escalate_object_endpoint(
    document_id: Annotated[str, Path(description="文档 id")],
    object_id: Annotated[str, Path(description="对象 id（document_objects.object_id）")],
    body: ObjectEscalate,
    actor: Annotated[User, Depends(_require_escalate)],
) -> dict:
    try:
        result = await escalate_object(
            actor,
            document_id,
            object_id,
            security_level=body.security_level,
            excluded=body.excluded,
        )
    except GrantError as exc:
        _raise(exc)
    return result


# ── 全局开关（运维）──────────────────────────────────────────────────────────


@router.get(
    "/settings",
    response_model=SecuritySettings,
    summary="安全隔离全局开关与前置条件自检（平台管理员）",
    description=(
        "回显 ``SECURITY_STRICT_MODE`` 与 ``ACL_SECURITY_PREFILTER_STRICT``，并给出"
        "开启前置过滤的**前置条件**：``document_objects.effective_security_level IS NULL`` "
        "的对象数必须为 0（回填已执行）。满足前置条件时可安全开启"
        "``ACL_SECURITY_PREFILTER_STRICT=true``（在 ``backend/.env`` 设置并重启后端）。"
    ),
)
async def security_settings_endpoint(
    _actor: Annotated[User, Depends(require_admin)],
) -> SecuritySettings:
    settings = get_settings()
    null_count = await counted_null_effective_levels()
    precondition_met = null_count == 0
    note = (
        "回填已完成，可安全开启 ACL_SECURITY_PREFILTER_STRICT（在 backend/.env 设置后重启）。"
        if precondition_met
        else (
            f"尚有 {null_count} 个对象的 effective_security_level 为 NULL —— "
            "请先运行 scripts/backfill_security_level.py 再开启严格前置过滤，"
            "否则存量对象会被整体排除。"
        )
    )
    return SecuritySettings(
        security_strict_mode=bool(settings.SECURITY_STRICT_MODE),
        acl_security_prefilter_strict=bool(settings.ACL_SECURITY_PREFILTER_STRICT),
        default_security_level=int(DEFAULT_SECURITY_LEVEL),
        project_enabled=bool(getattr(settings, "PROJECT_ENABLED", True)),
        null_effective_level_count=null_count,
        prefilter_strict_precondition_met=precondition_met,
        note=note,
    )


__all__ = ["router"]
