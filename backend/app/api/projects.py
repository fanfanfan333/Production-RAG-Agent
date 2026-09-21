"""
项目维度管理 API（``/projects``）—— T5，设计文档 §4.4 / 决策 4.

项目是五维权限里的**横向**维度：它只增加可见性，**不改** ``access_level`` 三值。

    方法   路径                                  权限
    ────   ────────────────────────────────────  ─────────────────────
    GET    /projects                              已登录（只见自己公司/可见项目）
    POST   /projects                              security.escalate（kb_admin 及以上）
    GET    /projects/{pid}                        已登录（跨租户 404）
    PATCH  /projects/{pid}                        security.escalate
    DELETE /projects/{pid}                        security.escalate
    GET    /projects/{pid}/members                已登录
    POST   /projects/{pid}/members                security.escalate
    DELETE /projects/{pid}/members/{user_id}      security.escalate

**项目归属唯一租户、不跨租户**；成员**可跨部门**（跨部门项目组只看成员身份）。
成员带 ``expires_at`` 即 P1-2 临时成员：到期后自动不再进入 ``UserScope.project_ids``。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.deps import get_current_user
from app.db.user_models import User
from app.schemas.project import (
    MemberAdd,
    MemberItem,
    ProjectCreate,
    ProjectItem,
    ProjectRename,
)
from app.services.permissions import require_permission
from app.services.project_service import (
    ProjectError,
    add_member,
    create_project,
    delete_project,
    get_project,
    list_members,
    list_projects,
    member_counts,
    member_is_active,
    remove_member,
    rename_project,
    resolve_usernames,
    validate_project_id,
)

router = APIRouter(prefix="/projects", tags=["Projects"])

_require_project_admin = require_permission("security.escalate")


def _raise(exc: ProjectError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from None


def _serialize_project(project, *, member_count: int = 0) -> dict:
    return ProjectItem(
        project_id=project.id,
        tenant_id=project.tenant_id,
        name=project.name,
        created_by=str(project.created_by) if project.created_by else None,
        created_at=project.created_at,
        member_count=int(member_count or 0),
    ).model_dump()


# ── 项目 CRUD ─────────────────────────────────────────────────────────────────


@router.get(
    "/projects",
    summary="我可管理的项目清单",
    description=(
        "返回调用者**可见范围**内的项目。普通成员只见**自己公司**的项目；"
        "平台管理员可见全部公司。项目是横向维度，不改变三层知识库的可见范围。"
    ),
)
async def list_projects_endpoint(
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    projects = await list_projects(user)
    counts = await member_counts([p.id for p in projects])
    items = [_serialize_project(p, member_count=counts.get(p.id, 0)) for p in projects]
    return {"items": items, "total": len(items)}


@router.post(
    "/projects",
    status_code=status.HTTP_201_CREATED,
    summary="创建项目（security.escalate）",
    description=(
        "新建一个项目，**归属唯一租户**（不跨租户）。``tenant_id`` 仅平台管理员可指定；"
        "其余角色强制落在自己公司。``project_id`` 缺省时随机生成。"
    ),
)
async def create_project_endpoint(
    body: ProjectCreate,
    actor: Annotated[User, Depends(_require_project_admin)],
) -> dict:
    try:
        project = await create_project(
            actor, body.name, project_id=body.project_id, tenant_id=body.tenant_id
        )
    except ProjectError as exc:
        _raise(exc)
    return _serialize_project(project, member_count=0)


@router.get(
    "/projects/{project_id}",
    summary="取单个项目（跨租户 404）",
)
async def get_project_endpoint(
    project_id: Annotated[str, Path(description="项目标识")],
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    try:
        project = await get_project(user, project_id)
    except ProjectError as exc:
        _raise(exc)
    counts = await member_counts([project.id])
    return _serialize_project(project, member_count=counts.get(project.id, 0))


@router.patch(
    "/projects/{project_id}",
    summary="改项目名（security.escalate）",
)
async def rename_project_endpoint(
    project_id: Annotated[str, Path(description="项目标识")],
    body: ProjectRename,
    actor: Annotated[User, Depends(_require_project_admin)],
) -> dict:
    try:
        project = await rename_project(actor, project_id, body.name)
    except ProjectError as exc:
        _raise(exc)
    return {"project_id": project.id, "name": project.name, "message": "已改名"}


@router.delete(
    "/projects/{project_id}",
    summary="删项目（security.escalate，成员一并删除）",
)
async def delete_project_endpoint(
    project_id: Annotated[str, Path(description="项目标识")],
    actor: Annotated[User, Depends(_require_project_admin)],
) -> dict:
    try:
        removed = await delete_project(actor, project_id)
    except ProjectError as exc:
        _raise(exc)
    return {"project_id": project_id, "members_removed": removed, "message": "项目已删除"}


# ── 成员管理 ──────────────────────────────────────────────────────────────────


async def _serialize_members(rows) -> list[dict]:
    names = await resolve_usernames([r.user_id for r in rows])
    return [
        MemberItem(
            user_id=str(r.user_id),
            username=names.get(str(r.user_id)),
            expires_at=r.expires_at,
            active=member_is_active(r),
            added_by=str(r.added_by) if r.added_by else None,
            added_at=r.added_at,
        ).model_dump()
        for r in rows
    ]


@router.get(
    "/projects/{project_id}/members",
    summary="项目成员清单",
    description="默认返回**全部**成员（含已过期）；``active_only=true`` 时只返回当前有效成员。",
)
async def list_members_endpoint(
    project_id: Annotated[str, Path(description="项目标识")],
    user: Annotated[User, Depends(get_current_user)],
    active_only: bool = False,
) -> dict:
    try:
        rows = await list_members(user, project_id, active_only=active_only)
    except ProjectError as exc:
        _raise(exc)
    items = await _serialize_members(rows)
    return {"items": items, "total": len(items)}


@router.post(
    "/projects/{project_id}/members",
    summary="加成员 / 更新成员有效期（security.escalate）",
    description=(
        "成员**可跨部门**，但必须与项目**同租户**（项目不跨租户）。"
        "``expires_at`` 非空即 P1-2 临时成员：到期后自动失效（不再进入 project_ids）。"
    ),
)
async def add_member_endpoint(
    project_id: Annotated[str, Path(description="项目标识")],
    body: MemberAdd,
    actor: Annotated[User, Depends(_require_project_admin)],
) -> dict:
    try:
        member, created = await add_member(
            actor, validate_project_id(project_id), body.user_id,
            expires_at=body.expires_at,
        )
    except ProjectError as exc:
        _raise(exc)
    items = await _serialize_members([member])
    return {"member": items[0], "created": created}


@router.delete(
    "/projects/{project_id}/members/{user_id}",
    summary="移除成员（security.escalate）",
)
async def remove_member_endpoint(
    project_id: Annotated[str, Path(description="项目标识")],
    user_id: Annotated[str, Path(description="成员用户 id")],
    actor: Annotated[User, Depends(_require_project_admin)],
) -> dict:
    try:
        await remove_member(actor, validate_project_id(project_id), user_id)
    except ProjectError as exc:
        _raise(exc)
    return {"project_id": project_id, "user_id": user_id, "message": "成员已移除"}


__all__ = ["router"]
