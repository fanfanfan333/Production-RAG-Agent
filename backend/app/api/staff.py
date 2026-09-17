"""
企业身份验证与企业管理员后台 API.

    GET    /staff/me                    个人主页身份详情（姓名/账号/公司/部门/职位）
    GET    /staff/summary               角标汇总 + 我的身份状态
    GET    /staff/config                当前用户可授予的角色 / 是否可审核（下拉框用）

    POST   /staff/requests              提交「身份验证」申请（公司/部门/职责）
    GET    /staff/requests/mine         我的申请（含结论与负责人）
    GET    /staff/requests/inbox        待我审核（按公司隔离，可从下方越级审核）
    POST   /staff/requests/{id}/review  同意 / 拒绝（须填负责人 = 职务 + 名称）
    POST   /staff/requests/{id}/cancel  撤回自己的待审申请
    POST   /staff/requests/mark-seen    清除我的未读结论角标

    GET    /staff/members               成员列表（企业管理后台，公司隔离）
    GET    /staff/companies             公司清单（平台管理员看全部）
    PATCH  /staff/members/{user_id}     更换成员职责 / 部门 / 启用状态
    GET    /staff/members/{id}/deletion-preview  删除前的影响预检（会删什么/留什么）
    DELETE /staff/members/{user_id}     删除成员账号（保留其部门/公司知识库文档）

权限口径与 ``staff_service`` 的等级链一致，路由层不重复实现判定：
  * 提交申请：任何登录用户（否则未验证用户无法自救）
  * 审核：``staff.review``（部门负责人及以上）
  * 后台成员管理：``staff.admin``（知识库管理员及以上）
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.db.user_models import User
from app.services.permissions import require_permission
from app.services.staff_service import (
    GRANTABLE_ROLES,
    StaffError,
    cancel_staff_request,
    create_staff_request,
    delete_member,
    detail_for,
    grantable_roles_for,
    list_companies,
    list_inbox,
    list_members,
    list_my_requests,
    mark_my_requests_seen,
    preview_member_deletion,
    review_staff_request,
    serialize,
    set_member_identity,
    summary,
)

router = APIRouter(prefix="/staff", tags=["Staff Identity"])

RoleName = Literal["kb_admin", "dept_manager", "employee", "viewer"]


class CreateStaffRequest(BaseModel):
    """「身份验证」弹窗的三行输入。"""

    company_name: str = Field(..., max_length=128, description="公司名称")
    department_name: str = Field(..., max_length=128, description="公司部门")
    duty: str = Field(..., max_length=128, description="部门职责")


class ReviewStaffRequest(BaseModel):
    """
    审核请求。

    ``reviewer_title`` / ``reviewer_name`` 是**必填**：产品要求通过或拒绝后
    在申请记录后方显示「负责人 = 职务 + 名称」，这条信息必须来自审核人本人。
    """

    approve: bool
    reviewer_title: str = Field(..., max_length=64, description="你的职务")
    reviewer_name: str = Field(..., max_length=64, description="你的姓名")
    role: RoleName | None = Field(
        None, description="批准时授予的职责（默认 employee）"
    )
    department_name: str | None = Field(
        None, max_length=128, description="批准时实际分配的部门（可与申请不同）"
    )
    duty: str | None = Field(
        None, max_length=128, description="批准时实际职责（可与申请不同）"
    )
    comment: str | None = Field(None, max_length=1000)


class UpdateMemberRequest(BaseModel):
    """更换成员职责（企业管理后台）。"""

    role: RoleName | None = None
    department_name: str | None = Field(None, max_length=128)
    job_title: str | None = Field(None, max_length=128)
    is_active: bool | None = None


def _raise(exc: StaffError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None


# ── 我的身份 ──────────────────────────────────────────────────────────────────

@router.get("/me", summary="个人主页身份详情")
async def me_endpoint(
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    return await detail_for(user)


@router.get("/summary", summary="身份状态与审核角标")
async def summary_endpoint(
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    return await summary(user)


@router.get(
    "/config",
    summary="我可授予的角色 / 可享有的能力（前端下拉框直接用）",
)
async def config_endpoint(
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    from app.services.permissions import role_label

    return {
        "grantable_roles": [
            {"value": value, "label": role_label(value)}
            for value in grantable_roles_for(user)
        ],
        "can_review": user.is_admin or bool(grantable_roles_for(user)),
        "identity_status": (await summary(user))["identity_status"],
    }


# ── 身份验证申请 ──────────────────────────────────────────────────────────────

@router.post(
    "/requests",
    summary="提交身份验证申请",
    status_code=status.HTTP_201_CREATED,
)
async def create_request_endpoint(
    body: CreateStaffRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    try:
        request = await create_staff_request(
            user,
            company_name=body.company_name,
            department_name=body.department_name,
            duty=body.duty,
        )
    except StaffError as exc:
        _raise(exc)
    return {
        "request": serialize(request, viewer=user),
        "message": "申请已提交，等待上级审核",
    }


@router.get("/requests/mine", summary="我的身份验证申请")
async def my_requests_endpoint(
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    items = await list_my_requests(user)
    return {"items": items, "total": len(items)}


@router.get("/requests/inbox", summary="待我审核的身份申请")
async def inbox_endpoint(
    user: Annotated[User, Depends(require_permission("staff.review"))],
) -> dict:
    items = await list_inbox(user)
    return {
        "items": items,
        "total": len(items),
        "pending": sum(1 for item in items if item["status"] == "pending"),
    }


@router.post("/requests/mark-seen", summary="清除我的未读结论角标")
async def mark_seen_endpoint(
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    return {"cleared": await mark_my_requests_seen(user)}


@router.post(
    "/requests/{request_id}/review",
    summary="同意 / 拒绝身份验证申请",
)
async def review_endpoint(
    request_id: uuid.UUID,
    body: ReviewStaffRequest,
    user: Annotated[User, Depends(require_permission("staff.review"))],
) -> dict:
    try:
        request, profile = await review_staff_request(
            user,
            request_id,
            approve=body.approve,
            reviewer_title=body.reviewer_title,
            reviewer_name=body.reviewer_name,
            role=body.role,
            department_name=body.department_name,
            duty=body.duty,
            comment=body.comment,
        )
    except StaffError as exc:
        _raise(exc)
    return {
        "request": serialize(request, viewer=user),
        "profile": profile,
        "message": (
            f"已批准，{request.applicant_username} 已加入"
            f"{request.company_name} · {request.department_name}"
            if body.approve
            else "已拒绝，申请人会看到你的意见"
        ),
    }


@router.post("/requests/{request_id}/cancel", summary="撤回我的待审申请")
async def cancel_endpoint(
    request_id: uuid.UUID,
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    try:
        request = await cancel_staff_request(user, request_id)
    except StaffError as exc:
        _raise(exc)
    return {"request": serialize(request, viewer=user), "message": "申请已撤回"}


# ── 企业管理后台 ──────────────────────────────────────────────────────────────

@router.get("/members", summary="成员列表（企业管理员后台）")
async def members_endpoint(
    user: Annotated[User, Depends(require_permission("staff.admin"))],
    company_id: Annotated[str | None, Query(description="仅平台管理员可指定")] = None,
    keyword: Annotated[str | None, Query(description="按账号或姓名搜索")] = None,
) -> dict:
    try:
        items = await list_members(user, company_id=company_id, keyword=keyword)
    except StaffError as exc:
        _raise(exc)
    return {"items": items, "total": len(items)}


@router.get("/companies", summary="公司清单（企业管理员后台）")
async def companies_endpoint(
    user: Annotated[User, Depends(require_permission("staff.admin"))],
) -> dict:
    try:
        items = await list_companies(user)
    except StaffError as exc:
        _raise(exc)
    return {"items": items, "total": len(items)}


@router.patch("/members/{user_id}", summary="更换成员职责 / 部门 / 启用状态")
async def update_member_endpoint(
    user_id: uuid.UUID,
    body: UpdateMemberRequest,
    user: Annotated[User, Depends(require_permission("staff.admin"))],
) -> dict:
    if (
        body.role is None
        and body.department_name is None
        and body.job_title is None
        and body.is_active is None
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="至少需要提供 role / department_name / job_title / is_active 之一",
        )
    try:
        member = await set_member_identity(
            user,
            user_id,
            role=body.role,
            department_name=body.department_name,
            job_title=body.job_title,
            is_active=body.is_active,
        )
    except StaffError as exc:
        _raise(exc)
    return {"member": member, "message": "职责已更新"}


@router.get(
    "/members/{user_id}/deletion-preview",
    summary="删除成员前的数据影响预检（将删除什么 / 将保留什么）",
)
async def member_deletion_preview_endpoint(
    user_id: uuid.UUID,
    user: Annotated[User, Depends(require_permission("staff.admin"))],
) -> dict:
    """
    确认弹窗的数据来源。

    权限与可删范围判定完全复用 ``delete_member`` 的那一套（``_deletable_member``），
    因此"预览能打开、点删除却 403"这种前后不一致不可能出现。
    """
    try:
        return await preview_member_deletion(user, user_id)
    except StaffError as exc:
        _raise(exc)


@router.delete(
    "/members/{user_id}",
    summary="删除成员账号（保留其部门/公司知识库文档）",
)
async def delete_member_endpoint(
    user_id: uuid.UUID,
    user: Annotated[User, Depends(require_permission("staff.admin"))],
) -> dict:
    """
    注销成员：账号 + 个人数据（会话/个人库文档/文档集合）一并删除，
    他上传到**部门库 / 公司库**的文档保留（仅解除归属关系），
    审核与审计留痕保留。平台管理员账号不可被删除，也不能删自己。
    """
    try:
        return await delete_member(user, user_id)
    except StaffError as exc:
        _raise(exc)


__all__ = ["router", "GRANTABLE_ROLES"]
