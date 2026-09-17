"""
共享申请 API（「申请共享」+「查看申请」）.

    POST   /share-requests                  提交申请（个人文档 → 部门库/公司库）
    GET    /share-requests/mine             我的申请（是否通过一目了然）
    GET    /share-requests/inbox            待我审核（含我范围内已处理的记录）
    GET    /share-requests/summary          角标汇总（主界面入口用）
    POST   /share-requests/mark-seen        把审核结论标记为已读（清角标）
    POST   /share-requests/{id}/review      同意 / 拒绝
    POST   /share-requests/{id}/cancel      撤回自己的待审申请

权限：提交需要 ``share.request``；审核范围由 ``share_service.review_scope``
按角色推导（部门负责人 → 部门范围，知识库管理员/企业管理员 → 公司范围），
路由本身不再重复实现范围判断。
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.db.user_models import User
from app.services.permissions import require_permission
from app.services.share_service import (
    ShareError,
    cancel_request,
    create_delete_request,
    create_share_request,
    list_inbox,
    list_my_requests,
    mark_my_requests_seen,
    review_request,
    serialize,
    summary,
)

router = APIRouter(prefix="/share-requests", tags=["Share Requests"])


class CreateShareRequest(BaseModel):
    """
    提交申请。

    ``intent="publish"``（默认）：申请把文档发布到更高层级，必须给 ``target_level``。
    ``intent="delete"``：申请删除一份自己没有删除权的部门库/公司库文档，
        ``target_level`` 由后端按文档当前层级推导（决定谁来审核），无需前端提供。

    默认值让老前端（只传 document_id/target_level/reason）行为完全不变。
    """

    document_id: uuid.UUID
    intent: Literal["publish", "delete"] = "publish"
    target_level: Literal["department", "tenant"] | None = Field(
        None,
        description="目标层级（intent=publish 时必填）：department=部门知识库，tenant=公司知识库",
    )
    reason: str | None = Field(None, max_length=1000)


class ReviewRequest(BaseModel):
    approve: bool
    comment: str | None = Field(None, max_length=1000)


def _raise(exc: ShareError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None


@router.post("", summary="提交共享 / 删除申请", status_code=status.HTTP_201_CREATED)
async def create_endpoint(
    body: CreateShareRequest,
    user: Annotated[User, Depends(require_permission("share.request"))],
) -> dict:
    try:
        if body.intent == "delete":
            request = await create_delete_request(
                user, body.document_id, body.reason
            )
            message = "删除申请已提交，等待上级审核"
        else:
            if not body.target_level:
                raise ShareError("申请共享必须指定目标层级（部门库或公司库）")
            request = await create_share_request(
                user, body.document_id, body.target_level, body.reason
            )
            message = "申请已提交，等待审核"
    except ShareError as exc:
        _raise(exc)
    return {
        "request": serialize(request, viewer=user),
        "message": message,
    }


@router.get("/mine", summary="我的申请（查看是否通过）")
async def my_requests_endpoint(
    user: Annotated[User, Depends(get_current_user)],
    status_filter: Annotated[
        Literal["pending", "approved", "rejected", "cancelled"] | None,
        Query(alias="status"),
    ] = None,
) -> dict:
    rows = await list_my_requests(user, status=status_filter)
    return {
        "items": [serialize(row, viewer=user) for row in rows],
        "total": len(rows),
    }


@router.get("/inbox", summary="待我审核的申请")
async def inbox_endpoint(
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    rows = await list_inbox(user)
    return {
        "items": [serialize(row, viewer=user) for row in rows],
        "total": len(rows),
        "pending": sum(1 for row in rows if row.status == "pending"),
    }


@router.get("/summary", summary="申请角标汇总")
async def summary_endpoint(
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    return await summary(user)


@router.post("/mark-seen", summary="清除我这边的新结果角标")
async def mark_seen_endpoint(
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    cleared = await mark_my_requests_seen(user)
    return {"cleared": cleared}


@router.post("/{request_id}/review", summary="同意 / 拒绝共享或删除申请")
async def review_endpoint(
    request_id: uuid.UUID,
    body: ReviewRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    try:
        request, published, deleted = await review_request(
            user, request_id, approve=body.approve, comment=body.comment
        )
    except ShareError as exc:
        _raise(exc)
    is_delete = request.intent == "delete"
    if not body.approve:
        message = "已拒绝，申请人会看到你的意见"
    elif is_delete:
        message = f"已批准，「{request.document_name}」已删除"
    else:
        message = "已批准，文档已发布到目标知识库"
    return {
        "request": serialize(request, viewer=user),
        "published": published,
        "deleted": deleted,
        "message": message,
    }


@router.post("/{request_id}/cancel", summary="撤回自己的待审申请")
async def cancel_endpoint(
    request_id: uuid.UUID,
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    try:
        request = await cancel_request(user, request_id)
    except ShareError as exc:
        _raise(exc)
    return {"request": serialize(request, viewer=user), "message": "申请已撤回"}
