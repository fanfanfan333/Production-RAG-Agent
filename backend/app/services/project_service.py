"""
项目维度管理服务（T5，设计文档 §4.4 / 决策 4 / 共享知识 5）.

项目是五维权限里的**横向**维度：它**只增加**可见性（作为 ``_source_gate`` 的第四个
OR 分支），不替换 ``access_level`` 的三值语义，也与三层知识库正交。本模块只负责
``projects`` / ``project_members`` 两张表的读写，不碰判定内核
（``security_policy`` / ``security_scope``）。

三条不可动摇的口径
──────────────────
1. **项目归属唯一租户，不跨租户**（PRD Q3 / §4.4）：项目有且只有一个 ``tenant_id``；
   成员也必须属于该租户 —— 跨租户的"项目成员"在判定侧本来就会被 ``_tenant_gate``
   拦下（非 private 对象的租户闸门），允许它只会制造"看起来能授权其实授不出"的假象。
   跨**部门**成员完全合法：跨部门项目组只看成员身份，部门维度不参与判定。
2. **成员有效期 = P1-2 临时成员**：``project_members.expires_at`` 非空且已过期 ⇒
   该成员**不再**进入 ``UserScope.project_ids``（``security_scope._resolve_project_ids``
   的 SQL 已经带上 ``expires_at > now`` 条件，本模块只负责把有效期原样落库）。
3. **管理动作留痕**：建/改/删项目与成员变更都写审计（``project.member.change`` 等）。

为什么把"纯函数"和"DB 访问"分开
────────────────────────────────
``member_is_active`` / ``project_ids_of`` 是**不碰 DB** 的纯函数 —— 有效期语义
（"过期即失效"）因此有可执行、可回归的断言，而不是埋在 SQL 的 where 里。
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from app.db.postgres import get_db_session
from app.db.security_models import Project, ProjectMember
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.services.tenancy import effective_tenant_id, is_platform_admin
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: 项目标识与名称的长度/字符约束（与 DDL 的 VARCHAR(64)/VARCHAR(128) 对齐）
PROJECT_ID_MAX_LEN = 64
PROJECT_NAME_MAX_LEN = 128
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class ProjectError(Exception):
    """项目操作失败（用户可读中文 + HTTP 状态码）。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# ═══════════════════════════════════════════════════════════════════════════════
# 纯函数（不碰 DB / 不碰网络）—— 有效期语义的唯一落点
# ═══════════════════════════════════════════════════════════════════════════════


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    """把可能 naive 的时间当成 UTC（Qdrant/前端都可能回传无时区的 ISO）。"""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def validate_project_id(project_id: str | None) -> str:
    """校验并归一化项目标识（防目录/注入类字符；与 tenant_id 同一套口子）。"""
    pid = (project_id or "").strip()
    if not pid or not _PROJECT_ID_RE.match(pid):
        raise ProjectError(
            "项目标识只能包含字母/数字/下划线/连字符，长度 1-64"
        )
    return pid


def normalize_name(name: str | None) -> str:
    """校验并归一化项目名称（非空、不超长）。"""
    value = (name or "").strip()
    if not value:
        raise ProjectError("项目名称不能为空")
    if len(value) > PROJECT_NAME_MAX_LEN:
        raise ProjectError(f"项目名称过长（≤{PROJECT_NAME_MAX_LEN} 字符）")
    return value


def member_is_active(member: Any, now: datetime | None = None) -> bool:
    """
    成员是否**当前有效**（P1-2 临时成员到期自动失效）.

    ``expires_at is None`` = 长期成员（一直有效）；非空且 ``<= now`` = 已失效。
    与 ``security_scope._resolve_project_ids`` 的 SQL 条件
    （``expires_at IS NULL OR expires_at > now``）**逐条同构**。
    """
    exp = _aware(getattr(member, "expires_at", None))
    if exp is None:
        return True
    return exp > (now or _now())


def project_ids_of(members: Any, now: datetime | None = None) -> frozenset[str]:
    """
    一组成员行 → 该用户**当前有效**的项目集合（纯函数，供测试与审计复用）.

    ⚠️ 运行时的权威入口是 ``security_scope._resolve_project_ids``（同样带上有效期
    条件）。本函数是**语义镜像**，用来把"过期即失效"钉成可回归断言。
    """
    now = now or _now()
    return frozenset(
        str(getattr(m, "project_id", ""))
        for m in (members or [])
        if getattr(m, "project_id", None) and member_is_active(m, now)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DB 访问（可选注入 session：None 时自建独立事务）
# ═══════════════════════════════════════════════════════════════════════════════


async def _with_session(session: Any, factory: Any) -> Any:
    """在传入的 session 上执行，或自建一个独立事务执行（与 security_cascade 同形）。"""
    if session is not None:
        return await factory(session)
    async with get_db_session() as sess:
        return await factory(sess)


def _assert_project_visible(actor: User, project: Project) -> None:
    """
    跨租户不可见（含平台管理员之外的任何角色）.

    * 平台管理员是**唯一**的跨租户身份（与公司注册表一致）——它给所有公司配项目；
    * 其余角色只能看到/管理**自己公司**的项目；跨公司一律 **404**（不泄露存在性）。
    """
    if is_platform_admin(actor):
        return
    if str(project.tenant_id) != effective_tenant_id(actor):
        raise ProjectError("项目不存在", status_code=404)


# ── 项目 CRUD ─────────────────────────────────────────────────────────────────


async def create_project(
    actor: User,
    name: str,
    *,
    project_id: str | None = None,
    tenant_id: str | None = None,
    session: Any = None,
) -> Project:
    """
    建一个项目（归属唯一租户）.

    ``tenant_id`` 只对平台管理员生效（它给指定公司建项目）；其余角色强制落在
    自己的 ``effective_tenant_id``。``project_id`` 缺省时随机生成 ``p_<hex12>``。
    """
    display = normalize_name(name)
    actor_tenant = effective_tenant_id(actor)
    target_tenant = actor_tenant
    if tenant_id and is_platform_admin(actor):
        target_tenant = str(tenant_id).strip() or actor_tenant
    pid = validate_project_id(project_id) if project_id else f"p_{uuid.uuid4().hex[:12]}"

    async def _do(sess: Any) -> Project:
        existing = await sess.get(Project, pid)
        if existing is not None:
            raise ProjectError(f"项目 {pid} 已存在", status_code=409)
        project = Project(
            id=pid,
            tenant_id=target_tenant,
            name=display,
            created_by=getattr(actor, "id", None),
            created_at=_now(),
        )
        sess.add(project)
        await sess.flush()
        return project

    project = await _with_session(session, _do)
    await record_audit(
        "project.create",
        user_id=getattr(actor, "id", None),
        username=getattr(actor, "username", None),
        resource_type="project",
        resource_id=project.id,
        detail=f"name={display}; tenant={target_tenant}",
    )
    logger.info("Project created: %s (%s) tenant=%s", project.id, display, target_tenant)
    return project


async def list_projects(actor: User, *, session: Any = None) -> list[Project]:
    """列出调用者可见的项目（非管理员只看自己公司；管理员看全部）。"""

    async def _do(sess: Any) -> list[Project]:
        stmt = select(Project)
        if not is_platform_admin(actor):
            stmt = stmt.where(Project.tenant_id == effective_tenant_id(actor))
        stmt = stmt.order_by(Project.created_at.desc())
        return list((await sess.scalars(stmt)).all())

    return await _with_session(session, _do)


async def member_counts(
    project_ids: list[str], *, session: Any = None
) -> dict[str, int]:
    """批量统计各项目的成员数（一次聚合查询，避免 N+1）。"""
    ids = [str(p) for p in project_ids if str(p).strip()]
    if not ids:
        return {}

    async def _do(sess: Any) -> dict[str, int]:
        stmt = (
            select(ProjectMember.project_id, func.count())
            .where(ProjectMember.project_id.in_(ids))
            .group_by(ProjectMember.project_id)
        )
        rows = (await sess.execute(stmt)).all()
        return {str(pid): int(count) for pid, count in rows}

    return await _with_session(session, _do)


async def get_project(actor: User, project_id: str, *, session: Any = None) -> Project:
    """取单个项目（跨租户 → 404）。"""
    pid = validate_project_id(project_id)

    async def _do(sess: Any) -> Project | None:
        return await sess.get(Project, pid)

    project = await _with_session(session, _do)
    if project is None:
        raise ProjectError("项目不存在", status_code=404)
    _assert_project_visible(actor, project)
    return project


async def rename_project(
    actor: User, project_id: str, name: str, *, session: Any = None
) -> Project:
    """改项目名（归属与成员一个不动）。"""
    display = normalize_name(name)

    async def _do(sess: Any) -> Project:
        project = await sess.get(Project, validate_project_id(project_id))
        if project is None:
            raise ProjectError("项目不存在", status_code=404)
        _assert_project_visible(actor, project)
        project.name = display
        await sess.flush()
        return project

    project = await _with_session(session, _do)
    await record_audit(
        "project.rename",
        user_id=getattr(actor, "id", None),
        username=getattr(actor, "username", None),
        resource_type="project",
        resource_id=project.id,
        detail=f"name={display}",
    )
    return project


async def delete_project(actor: User, project_id: str, *, session: Any = None) -> int:
    """删项目（成员随 ``ON DELETE CASCADE`` 一起删；返回被删成员数）。"""
    pid = validate_project_id(project_id)

    async def _do(sess: Any) -> int:
        project = await sess.get(Project, pid)
        if project is None:
            raise ProjectError("项目不存在", status_code=404)
        _assert_project_visible(actor, project)
        members = list(
            (await sess.scalars(
                select(ProjectMember).where(ProjectMember.project_id == pid)
            )).all()
        )
        for m in members:
            await sess.delete(m)
        await sess.delete(project)
        await sess.flush()
        return len(members)

    removed = await _with_session(session, _do)
    await record_audit(
        "project.delete",
        user_id=getattr(actor, "id", None),
        username=getattr(actor, "username", None),
        resource_type="project",
        resource_id=pid,
        detail=f"members_removed={removed}",
    )
    logger.info("Project deleted: %s (members=%d)", pid, removed)
    return removed


# ── 成员 CRUD ─────────────────────────────────────────────────────────────────


async def list_members(
    actor: User, project_id: str, *, session: Any = None, active_only: bool = False
) -> list[ProjectMember]:
    """列出项目成员（``active_only=True`` 时只返回**当前有效**的成员）。"""
    pid = validate_project_id(project_id)

    async def _do(sess: Any) -> list[ProjectMember]:
        project = await sess.get(Project, pid)
        if project is None:
            raise ProjectError("项目不存在", status_code=404)
        _assert_project_visible(actor, project)
        rows = list(
            (await sess.scalars(
                select(ProjectMember).where(ProjectMember.project_id == pid)
            )).all()
        )
        if active_only:
            rows = [r for r in rows if member_is_active(r)]
        return rows

    return await _with_session(session, _do)


async def add_member(
    actor: User,
    project_id: str,
    user_id: str | uuid.UUID,
    *,
    expires_at: datetime | None = None,
    session: Any = None,
) -> tuple[ProjectMember, bool]:
    """
    加成员 / 更新成员有效期（幂等 upsert）.

    Returns:
        ``(成员行, created)``：``created=False`` 表示"该成员已存在，只更新了有效期"。

    成员必须与项目**同租户**（项目不跨租户）；**跨部门完全允许**
    （跨部门项目组只看成员身份）。过期时间允许在过去（用于测试/纠错），
    判定侧以"是否已过期"为准 —— 已过期即不进入 ``project_ids``。
    """
    pid = validate_project_id(project_id)
    uid = _as_uuid(user_id)
    if uid is None:
        raise ProjectError("成员 user_id 非法", status_code=400)
    exp = _aware(expires_at)

    async def _do(sess: Any) -> tuple[ProjectMember, bool]:
        project = await sess.get(Project, pid)
        if project is None:
            raise ProjectError("项目不存在", status_code=404)
        _assert_project_visible(actor, project)

        user = await sess.get(User, uid)
        if user is None or not getattr(user, "is_active", True):
            raise ProjectError("目标成员不存在或已停用", status_code=404)
        if effective_tenant_id(user) != str(project.tenant_id):
            raise ProjectError(
                "成员必须与项目属于同一公司（项目不跨租户）", status_code=400
            )

        member = await sess.get(ProjectMember, (pid, uid))
        if member is not None:
            member.expires_at = exp
            member.added_by = getattr(actor, "id", None)
            await sess.flush()
            return member, False

        member = ProjectMember(
            project_id=pid,
            user_id=uid,
            expires_at=exp,
            added_by=getattr(actor, "id", None),
            added_at=_now(),
        )
        sess.add(member)
        await sess.flush()
        return member, True

    member, created = await _with_session(session, _do)
    await record_audit(
        "project.member.change",
        user_id=getattr(actor, "id", None),
        username=getattr(actor, "username", None),
        resource_type="project",
        resource_id=pid,
        detail=(
            f"action={'add' if created else 'update'}; member={uid}; "
            f"expires_at={exp.isoformat() if exp else '-'}"
        ),
    )
    return member, created


async def remove_member(
    actor: User, project_id: str, user_id: str | uuid.UUID, *, session: Any = None
) -> None:
    """移除成员（不存在 → 404）。"""
    pid = validate_project_id(project_id)
    uid = _as_uuid(user_id)
    if uid is None:
        raise ProjectError("成员 user_id 非法", status_code=400)

    async def _do(sess: Any) -> None:
        project = await sess.get(Project, pid)
        if project is None:
            raise ProjectError("项目不存在", status_code=404)
        _assert_project_visible(actor, project)
        member = await sess.get(ProjectMember, (pid, uid))
        if member is None:
            raise ProjectError("该成员不在项目中", status_code=404)
        await sess.delete(member)
        await sess.flush()

    await _with_session(session, _do)
    await record_audit(
        "project.member.change",
        user_id=getattr(actor, "id", None),
        username=getattr(actor, "username", None),
        resource_type="project",
        resource_id=pid,
        detail=f"action=remove; member={uid}",
    )


async def resolve_usernames(
    user_ids: list[str | uuid.UUID], *, session: Any = None
) -> dict[str, str]:
    """批量把 ``user_id`` 解析成展示名（一次查询，避免 N+1）。"""
    uuids: list[uuid.UUID] = []
    for value in user_ids:
        parsed = _as_uuid(value)
        if parsed is not None:
            uuids.append(parsed)
    if not uuids:
        return {}

    async def _do(sess: Any) -> dict[str, str]:
        stmt = select(User.id, User.username, User.display_name).where(
            User.id.in_(uuids)
        )
        rows = (await sess.execute(stmt)).all()
        return {
            str(uid): (display_name or username or str(uid))
            for uid, username, display_name in rows
        }

    return await _with_session(session, _do)


def _as_uuid(value: str | uuid.UUID | None) -> uuid.UUID | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


__all__ = [
    "PROJECT_ID_MAX_LEN",
    "PROJECT_NAME_MAX_LEN",
    "ProjectError",
    "add_member",
    "create_project",
    "delete_project",
    "get_project",
    "list_members",
    "list_projects",
    "member_counts",
    "member_is_active",
    "normalize_name",
    "project_ids_of",
    "remove_member",
    "rename_project",
    "resolve_usernames",
    "validate_project_id",
]
