"""
Platform user and role management endpoints (RBAC).

All endpoints are platform-admin only. Role updates are audited and validate
against the centrally declared User.VALID_ROLES set. At least one active admin
must remain to prevent a self-inflicted tenant lockout.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select

from app.db.postgres import get_db_session
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.services.permissions import require_platform_admin

router = APIRouter(prefix="/admin", tags=["Administration"])

RoleName = Literal["admin", "manager", "editor", "viewer", "user"]


class UserAdminOut(BaseModel):
    id: str
    username: str
    role: str
    is_active: bool


class UpdateUserAccessRequest(BaseModel):
    role: RoleName | None = None
    is_active: bool | None = None


@router.get("/users", response_model=list[UserAdminOut], summary="List users and roles")
async def list_users(
    _: Annotated[User, Depends(require_platform_admin)],
) -> list[UserAdminOut]:
    async with get_db_session() as session:
        users = list((await session.execute(select(User).order_by(User.created_at))).scalars())
    return [
        UserAdminOut(
            id=str(user.id), username=user.username, role=user.role, is_active=user.is_active
        )
        for user in users
    ]


@router.patch("/users/{user_id}", response_model=UserAdminOut, summary="Change a user's role or active state")
async def update_user_access(
    user_id: uuid.UUID,
    body: UpdateUserAccessRequest,
    admin: Annotated[User, Depends(require_platform_admin)],
) -> UserAdminOut:
    if body.role is None and body.is_active is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="至少需要提供 role 或 is_active")

    async with get_db_session() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")

        # Do not allow removal/deactivation of the last active administrator.
        will_remove_admin = user.is_admin and (
            (body.role is not None and body.role != User.ROLE_ADMIN)
            or body.is_active is False
        )
        if will_remove_admin:
            admins = await session.scalar(
                select(func.count()).select_from(User).where(
                    User.role == User.ROLE_ADMIN,
                    User.is_active.is_(True),
                )
            )
            if int(admins or 0) <= 1:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="系统至少需要保留一个启用中的管理员",
                )

        if body.role is not None:
            user.role = body.role
        if body.is_active is not None:
            user.is_active = body.is_active
        await session.flush()
        await session.refresh(user)
        out = UserAdminOut(
            id=str(user.id), username=user.username, role=user.role, is_active=user.is_active
        )

    await record_audit(
        "admin.user_access.update",
        user_id=admin.id,
        username=admin.username,
        resource_type="user",
        resource_id=str(user_id),
        detail=f"role={out.role}; is_active={out.is_active}",
    )
    return out
