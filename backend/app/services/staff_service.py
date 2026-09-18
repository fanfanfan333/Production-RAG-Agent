"""
企业身份验证与层级授权服务（企业管理后台的单一策略实现点）.

## 等级链（越级审核的判定依据）

    admin (100)  >  company_admin (90)  >  kb_admin (70)
                 >  dept_manager (60)   >  employee (30)

产品要求：
  * 企业管理员是最高身份，且**全局唯一**（就是 admin 账号）
  * 企业管理员可设置**所有公司**的知识库管理员、部门负责人
  * 部门负责人设置普通员工
  * **上级可以越级审核下级**（不必逐级上报）
  * 上级拥有更换下级职责的权利

本模块把上述规则收敛成两个纯判定函数：

    ``can_review_staff(request, reviewer, applicant_role)``
        能不能审这一份 —— 等级更高 + 同公司（admin 除外）+ 不能自审
    ``can_grant_role(actor, role)``
        能不能授予某个角色 —— 只能授予**严格低于自己**的角色

其余函数（创建 / 审核 / 成员管理）都只调用这两个判定，不各自实现一遍。

## 公司隔离

申请的 ``company_id`` 与用户的 ``tenant_id`` 是同一个命名空间（都来自**公司注册表**
``companies.tenant_id``，由 ``company_registry`` 解析），因此"审核人只能看本公司申请"
直接复用第一层租户隔离，不引入第二套公司概念。平台管理员（admin）是唯一的例外：
它要管理**自己创建**的测试公司，所以可见范围收敛为 ``created_by == 它自己``。

## 身份状态

个人主页 / 首页徽标需要显示「去验证 / 审核中 / 已通过」。状态由**最新一条
申请**推导（另建状态列会与申请记录产生第二个事实来源，迟早不一致）。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, func, or_, select, update

from app.db.badcase_models import BadCase
from app.db.conversation_models import Conversation, Message
from app.db.feedback_models import AnswerFeedback
from app.db.models import Document
from app.db.postgres import get_db_session
from app.db.share_models import ShareRequest
from app.db.staff_models import StaffRequest
from app.db.user_models import Collection, User
from app.services.audit_service import record_audit
from app.services.permissions import has_permission, role_label
from app.services.tenancy import (
    ACCESS_DEPARTMENT,
    ACCESS_PRIVATE,
    ACCESS_TENANT,
    DEFAULT_TENANT_ID,
    company_display_name,
    department_display_name,
    department_id_from_name,
    effective_department_id,
    effective_tenant_id,
    normalize_tenant_id,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class StaffError(Exception):
    """身份验证流程失败（用户可读中文 + HTTP 状态码）。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# ── 等级链 ────────────────────────────────────────────────────────────────────

# 数值越大权限越高。历史角色与业务角色同权（与 permissions.py 的等价映射一致）。
ROLE_RANK: dict[str, int] = {
    User.ROLE_ADMIN: 100,
    User.ROLE_COMPANY_ADMIN: 90,
    User.ROLE_KB_ADMIN: 70,
    User.ROLE_DEPT_MANAGER: 60,
    User.ROLE_MANAGER: 60,
    User.ROLE_EMPLOYEE: 30,
    User.ROLE_EDITOR: 30,
    User.ROLE_USER: 30,
    User.ROLE_VIEWER: 10,
}

# 可被"授予"的角色（admin 被刻意排除：企业管理员全局唯一，不能由任何
# 接口授出，只能由注册时"第一个用户成为管理员"这条引导路径产生）。
GRANTABLE_ROLES: tuple[str, ...] = (
    User.ROLE_KB_ADMIN,
    User.ROLE_DEPT_MANAGER,
    User.ROLE_EMPLOYEE,
    User.ROLE_VIEWER,
)

# 具备"审核下级身份申请"资格的最低等级（部门负责人及以上）
REVIEWER_MIN_RANK = ROLE_RANK[User.ROLE_DEPT_MANAGER]


def role_rank(role: str | None) -> int:
    """角色的等级值；未知角色按普通员工处理（保守：不会被判成高权限）。"""
    return ROLE_RANK.get((role or "").strip(), ROLE_RANK[User.ROLE_EMPLOYEE])


def can_grant_role(actor: User, role: str | None) -> bool:
    """
    能否把 *role* 授予他人：**严格低于**自己的等级，且不是 admin.

    严格低于（而不是 <=）是为了杜绝"知识库管理员再造就一个知识库管理员"
    这类平级扩散 —— 管理层级因此始终收敛，且每条授权链都可追溯到 admin。
    """
    value = (role or "").strip()
    if value not in GRANTABLE_ROLES:
        return False
    return role_rank(actor.role) > role_rank(value)


def grantable_roles_for(actor: User) -> list[str]:
    """当前用户可授予的角色列表（前端下拉框直接用，避免"选了才 403"）。"""
    return [r for r in GRANTABLE_ROLES if can_grant_role(actor, r)]


def is_administer(actor: User | None) -> bool:
    """能否进入企业管理后台（成员列表 / 换职责）：知识库管理员及以上。"""
    if actor is None:
        return False
    return actor.is_admin or has_permission(actor, "staff.admin")


def can_review_any(actor: User | None) -> bool:
    """是否具备审核资格（部门负责人及以上，含平台管理员）。"""
    if actor is None:
        return False
    return actor.is_admin or role_rank(actor.role) >= REVIEWER_MIN_RANK


def can_review_staff(
    request: StaffRequest,
    reviewer: User,
    *,
    applicant_role: str | None = None,
) -> bool:
    """
    单份申请能否由 *reviewer* 处理.

    规则（全部满足）：
      1. 申请处于待审状态，且不是自己的申请（不能自审自批）
      2. reviewer 具备审核资格（部门负责人及以上）
      3. 公司一致 —— 平台管理员例外（它要管理所有公司）
      4. 等级严格高于申请人当前等级（这就是"越级审核"：高等级可直接审
         低等级，不必先由直属上级过一遍）
    """
    if request is None or reviewer is None:
        return False
    if request.status != StaffRequest.STATUS_PENDING:
        return False
    if request.applicant_id is not None and request.applicant_id == reviewer.id:
        return False
    if not can_review_any(reviewer):
        return False
    if not reviewer.is_admin:
        if normalize_tenant_id(request.company_id) != effective_tenant_id(reviewer):
            return False
    # 申请人当前等级（新注册用户是默认 employee 级）
    applicant_rank = role_rank(applicant_role)
    if role_rank(reviewer.role) <= applicant_rank:
        return False
    return True


# ── 身份状态 ──────────────────────────────────────────────────────────────────

_STATUS_TO_IDENTITY = {
    StaffRequest.STATUS_PENDING: StaffRequest.IDENTITY_PENDING,
    StaffRequest.STATUS_APPROVED: StaffRequest.IDENTITY_APPROVED,
    StaffRequest.STATUS_REJECTED: StaffRequest.IDENTITY_REJECTED,
}


def _derive_identity(user: User, latest: StaffRequest | None) -> str:
    """
    身份状态推导（顺序即优先级）:

        1. 平台管理员 → 已通过（它本身就是身份体系的管理者）
        2. 有申请记录 → 按最新一条映射（撤回视为未验证，可重新提交）
        3. 已有公司/部门归属的存量账号 → 已通过（升级兼容，不打扰老用户）
        4. 其余（新注册的普通账号）→ 未验证
    """
    if user.is_admin:
        return StaffRequest.IDENTITY_APPROVED
    if latest is not None:
        return _STATUS_TO_IDENTITY.get(latest.status, StaffRequest.IDENTITY_NONE)
    if effective_department_id(user) or effective_tenant_id(user) != DEFAULT_TENANT_ID:
        return StaffRequest.IDENTITY_APPROVED
    return StaffRequest.IDENTITY_NONE


async def identity_status(user: User) -> str:
    """当前用户的身份验证状态（none / pending / approved / rejected）。"""
    if user.is_admin:
        return StaffRequest.IDENTITY_APPROVED
    latest = await _latest_request(user.id)
    return _derive_identity(user, latest)


async def _latest_request(user_id: uuid.UUID) -> StaffRequest | None:
    async with get_db_session() as session:
        return await session.scalar(
            select(StaffRequest)
            .where(StaffRequest.applicant_id == user_id)
            .order_by(StaffRequest.created_at.desc())
            .limit(1)
        )


async def is_identity_verified(user: User | None) -> bool:
    """
    能否访问业务功能（上传 / 提问 / 申请共享）.

    产品要求"没有注册和职责的不能进入"：未通过身份验证的账号可以登录
    （否则无法提交验证申请，形成死锁），但进不了知识库业务。管理员天然通过。
    """
    if user is None:
        return False
    if user.is_admin:
        return True
    if not effective_department_id(user):
        # 没有部门 = 还没有被收进任何组织结构
        latest = await _latest_request(user.id)
        return _derive_identity(user, latest) == StaffRequest.IDENTITY_APPROVED
    return True


# ── 序列化 ────────────────────────────────────────────────────────────────────

def serialize(
    request: StaffRequest,
    *,
    viewer: User | None = None,
    can_review: bool = False,
) -> dict:
    """统一下发结构（前端申请列表 / 审核队列直接渲染，无需二次映射）。"""
    return {
        "id": str(request.id),
        "applicant_username": request.applicant_username,
        "applicant_display_name": request.applicant_display_name,
        "applicant_label": (
            request.applicant_display_name or request.applicant_username
        ),
        "company_name": request.company_name,
        "company_id": request.company_id,
        "department_name": request.department_name,
        "department_id": request.department_id,
        "duty": request.duty,
        "status": request.status,
        "reviewer_username": request.reviewer_username,
        "reviewer_title": request.reviewer_title,
        "reviewer_name": request.reviewer_name,
        # 「负责人 = 职务 + 名称」——产品要求展示在申请记录后方
        "reviewer_label": request.reviewer_label,
        "review_comment": request.review_comment,
        "granted_role": request.granted_role,
        "granted_role_label": (
            role_label(request.granted_role) if request.granted_role else None
        ),
        "created_at": request.created_at.isoformat() if request.created_at else None,
        "reviewed_at": request.reviewed_at.isoformat() if request.reviewed_at else None,
        "applicant_seen": request.applicant_seen,
        "is_mine": bool(viewer and request.applicant_id == viewer.id),
        "can_review": can_review,
    }


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


async def _applicant_roles(session, ids: list[uuid.UUID | None]) -> dict:
    """批量取申请人的当前角色（用于 can_review 的等级比较，避免 N+1）。"""
    clean = [i for i in ids if i is not None]
    if not clean:
        return {}
    rows = await session.execute(
        select(User.id, User.role).where(User.id.in_(clean))
    )
    return {uid: role for uid, role in rows.all()}


# ── 提交身份验证申请 ──────────────────────────────────────────────────────────

def _clean(value: str | None, label: str, *, max_len: int = 128) -> str:
    text = (value or "").strip()
    if not text:
        raise StaffError(f"请填写{label}")
    if len(text) > max_len:
        raise StaffError(f"{label}过长（最多 {max_len} 个字符）")
    return text


async def create_staff_request(
    user: User,
    *,
    company_id: str | None = None,
    company_name: str | None = None,
    department_name: str,
    duty: str,
) -> StaffRequest:
    """
    提交企业身份验证申请（首页弹窗 / 个人主页「身份验证」都走这里）.

    公司归属**从注册表解析**（不再用 ``company_id_from_name`` —— 那是名称哈希，
    与「改名零迁移」冲突）：

        * 给了 ``company_id``（下拉值）→ ``get_company``；未登记 → 400
        * 只给了 ``company_name``（旧入口）→ ``find_by_name``（按当前 ``name_key``
          解析，因此**改名后新名可解析、旧名失效**）；未登记 → 400

    未登记一律 400「请先由管理员注册该公司」：公司是**一等实体**，必须先在
    注册表存在，用户才能提交归属申请。

    同一时刻只允许一份待审申请：重复提交只会让审核队列出现多份一样的记录，
    申请人也拿不到更多信息。想改填内容就先撤回再提交。
    """
    if user.is_admin:
        raise StaffError(
            "平台管理员无需进行身份验证",
            status_code=409,
        )

    department = _clean(department_name, "公司部门")
    job_duty = _clean(duty, "部门职责")

    from app.services.company_registry import (
        MSG_NOT_REGISTERED,
        find_by_name,
        get_company,
    )

    company = None
    if company_id and company_id.strip():
        company = await get_company(company_id.strip())
    elif company_name and company_name.strip():
        company = await find_by_name(company_name.strip())
    if company is None:
        # 公司未登记（含"旧名已被改名"）：按设计明确引导管理员先注册
        raise StaffError(MSG_NOT_REGISTERED, status_code=400)

    resolved_company_id = company.tenant_id
    resolved_company_name = company.display_name

    department_id = department_id_from_name(department)
    if department_id is None:
        raise StaffError("公司部门不合法，请填写可识别的部门名称")

    async with get_db_session() as session:
        pending = await session.scalar(
            select(StaffRequest).where(
                StaffRequest.applicant_id == user.id,
                StaffRequest.status == StaffRequest.STATUS_PENDING,
            ).limit(1)
        )
        if pending is not None:
            raise StaffError(
                "你已经有一份待审核的身份验证申请，请耐心等待审核结果",
                status_code=409,
            )

        request = StaffRequest(
            id=uuid.uuid4(),
            applicant_id=user.id,
            applicant_username=user.username,
            applicant_display_name=user.display_name,
            applicant_company_id=effective_tenant_id(user),
            company_name=resolved_company_name,
            company_id=resolved_company_id,
            department_name=department,
            department_id=department_id,
            duty=job_duty,
            status=StaffRequest.STATUS_PENDING,
        )
        session.add(request)
        await session.flush()
        await session.refresh(request)

    await record_audit(
        "staff.request.create",
        user_id=user.id,
        username=user.username,
        resource_type="user",
        resource_id=str(user.id),
        detail=(
            f"company={resolved_company_name}({resolved_company_id}); "
            f"department={department}; duty={job_duty}"
        ),
    )
    logger.info(
        "Staff request created: user=%s company=%s department=%s",
        user.username, resolved_company_name, department,
    )
    return request


async def cancel_staff_request(user: User, request_id: uuid.UUID) -> StaffRequest:
    """申请人撤回自己的待审申请（撤回后可以重新提交）。"""
    async with get_db_session() as session:
        request = await session.get(StaffRequest, request_id)
        if request is None:
            raise StaffError("申请不存在", status_code=404)
        if request.applicant_id != user.id:
            raise StaffError("只能撤回自己提交的申请", status_code=403)
        if request.status != StaffRequest.STATUS_PENDING:
            raise StaffError("该申请已处理，无法撤回", status_code=409)

        request.status = StaffRequest.STATUS_CANCELLED
        request.reviewed_at = _now()
        request.applicant_seen = True
        await session.flush()
        await session.refresh(request)

    await record_audit(
        "staff.request.cancel",
        user_id=user.id,
        username=user.username,
        resource_type="user",
        resource_id=str(user.id),
        detail=f"申请={request_id}",
    )
    return request


# ── 查询 ──────────────────────────────────────────────────────────────────────

async def list_my_requests(user: User, *, limit: int = 100) -> list[dict]:
    """我的身份验证申请（含历史，便于回看谁在什么职务上批的）。"""
    async with get_db_session() as session:
        rows = list(
            (
                await session.execute(
                    select(StaffRequest)
                    .where(StaffRequest.applicant_id == user.id)
                    .order_by(StaffRequest.created_at.desc())
                    .limit(limit)
                )
            ).scalars().all()
        )
    return [serialize(row, viewer=user, can_review=False) for row in rows]


async def list_inbox(reviewer: User, *, limit: int = 200) -> list[dict]:
    """
    待我审核的身份申请.

    公司隔离：非平台管理员只看得到本公司的申请（跨公司的一律不出现在列表里，
    避免"看得见但点不动"）。平台管理员跨公司可见 —— 它要负责给各公司配管理层。
    """
    if not can_review_any(reviewer):
        return []

    async with get_db_session() as session:
        stmt = (
            select(StaffRequest)
            .where(StaffRequest.applicant_id != reviewer.id)
            .order_by(StaffRequest.created_at.desc())
            .limit(limit)
        )
        if not reviewer.is_admin:
            stmt = stmt.where(
                StaffRequest.company_id == effective_tenant_id(reviewer)
            )
        rows = list((await session.execute(stmt)).scalars().all())
        roles = await _applicant_roles(session, [row.applicant_id for row in rows])

    return [
        serialize(
            row,
            viewer=reviewer,
            can_review=can_review_staff(
                row, reviewer, applicant_role=roles.get(row.applicant_id)
            ),
        )
        for row in rows
    ]


async def summary(reviewer: User) -> dict:
    """首页 / 导航栏角标：待我审核、我的待出结果、有新结论未读、身份状态。"""
    status = await identity_status(reviewer)

    pending_for_me = 0
    if can_review_any(reviewer):
        async with get_db_session() as session:
            stmt = select(func.count()).select_from(StaffRequest).where(
                StaffRequest.applicant_id != reviewer.id,
                StaffRequest.status == StaffRequest.STATUS_PENDING,
            )
            if not reviewer.is_admin:
                stmt = stmt.where(
                    StaffRequest.company_id == effective_tenant_id(reviewer)
                )
            pending_for_me = int((await session.execute(stmt)).scalar_one() or 0)

    async with get_db_session() as session:
        my_pending = int(
            (
                await session.execute(
                    select(func.count()).select_from(StaffRequest).where(
                        StaffRequest.applicant_id == reviewer.id,
                        StaffRequest.status == StaffRequest.STATUS_PENDING,
                    )
                )
            ).scalar_one()
            or 0
        )
        decided_unseen = int(
            (
                await session.execute(
                    select(func.count()).select_from(StaffRequest).where(
                        StaffRequest.applicant_id == reviewer.id,
                        StaffRequest.status != StaffRequest.STATUS_PENDING,
                        StaffRequest.applicant_seen.is_(False),
                    )
                )
            ).scalar_one()
            or 0
        )

    return {
        "identity_status": status,
        "pending_for_me": pending_for_me,
        "my_pending": my_pending,
        "my_decided_unseen": decided_unseen,
        "can_review": can_review_any(reviewer),
        "can_administer": is_administer(reviewer),
        "total_badge": pending_for_me + decided_unseen,
    }


async def mark_my_requests_seen(user: User) -> int:
    """把审核结论标记为已读（清个人主页的未读提示）。"""
    async with get_db_session() as session:
        rows = list(
            (
                await session.execute(
                    select(StaffRequest).where(
                        StaffRequest.applicant_id == user.id,
                        StaffRequest.status != StaffRequest.STATUS_PENDING,
                        StaffRequest.applicant_seen.is_(False),
                    )
                )
            ).scalars().all()
        )
        for row in rows:
            row.applicant_seen = True
        await session.flush()
    return len(rows)


# ── 审核 ──────────────────────────────────────────────────────────────────────

async def review_staff_request(
    reviewer: User,
    request_id: uuid.UUID,
    *,
    approve: bool,
    reviewer_title: str,
    reviewer_name: str,
    role: str | None = None,
    department_name: str | None = None,
    duty: str | None = None,
    comment: str | None = None,
) -> tuple[StaffRequest, dict | None]:
    """
    同意 / 拒绝一份身份验证申请.

    产品要求「通过或拒绝后，要在该页面后方填写负责人，负责人为职务+名称」，
    因此 ``reviewer_title`` / ``reviewer_name`` 是**必填**（两个动作都要）：
    每条审核结论都要能追到具体的人与职务，而不是一个登录名。

    批准时可以就地修正三件事（这就是"上级拥有更换职责的权利"）：
      * ``role``            授予的权限角色，默认 employee（产品流程图的要求）
      * ``department_name`` 实际分配的部门（可与申请人填的不同）
      * ``duty``            实际职责（可与申请人填的不同）

    Returns:
        ``(申请行, 身份落地摘要)`` —— 批准时第二个值非空。
    """
    title = _clean(reviewer_title, "你的职务", max_len=64)
    name = _clean(reviewer_name, "你的姓名", max_len=64)

    async with get_db_session() as session:
        request = await session.get(StaffRequest, request_id)
        if request is None:
            raise StaffError("申请不存在", status_code=404)
        applicant = (
            await session.get(User, request.applicant_id)
            if request.applicant_id
            else None
        )
        if not can_review_staff(
            request, reviewer, applicant_role=applicant.role if applicant else None
        ):
            raise StaffError(
                "你没有权限审核该申请（可能不是本公司、等级不高于申请人，"
                "或这是你自己的申请）",
                status_code=403,
            )
        if request.status != StaffRequest.STATUS_PENDING:
            raise StaffError(
                f"该申请已处理（当前状态：{request.status}），无需重复操作",
                status_code=409,
            )
        snapshot = {
            "applicant_username": request.applicant_username,
            "company_name": request.company_name,
            "company_id": request.company_id,
            "department_name": request.department_name,
            "duty": request.duty,
        }

    granted = None
    profile: dict | None = None

    if approve:
        if applicant is None:
            raise StaffError("申请人账号已不存在，无法批准", status_code=410)

        target_role = (role or User.ROLE_EMPLOYEE).strip()
        if target_role == User.ROLE_ADMIN:
            raise StaffError(
                "企业管理员全局唯一，不能被授予。请选择其他职责。",
                status_code=403,
            )
        if not can_grant_role(reviewer, target_role):
            raise StaffError(
                f"你的等级（{role_label(reviewer.role)}）无法授予"
                f"「{role_label(target_role)}」：只能授予低于自己的职责",
                status_code=403,
            )

        final_department_name = (department_name or snapshot["department_name"]).strip()
        final_department_id = department_id_from_name(final_department_name)
        if final_department_id is None:
            raise StaffError("部门名称不合法")
        final_duty = (duty or snapshot["duty"]).strip() or snapshot["duty"]

        # 批准后的归属以**注册表最新展示名**为准：申请快照可能停留在改名之前，
        # 落库时若照抄旧名，成员面板会出现一个已经不存在的公司名。
        from app.services.company_registry import get_company as _get_company

        company = await _get_company(snapshot["company_id"])
        resolved_company_id = (
            company.tenant_id if company is not None
            else normalize_tenant_id(snapshot["company_id"])
        )
        resolved_company_name = (
            company.display_name if company is not None
            else snapshot["company_name"]
        )

        async with get_db_session() as session:
            applicant = await session.get(User, request.applicant_id)
            if applicant is None:
                raise StaffError("申请人账号已不存在，无法批准", status_code=410)
            applicant.tenant_id = resolved_company_id
            applicant.company_name = resolved_company_name
            applicant.department_id = final_department_id
            applicant.department_name = final_department_name
            applicant.job_title = final_duty
            applicant.role = target_role
            await session.flush()

        granted = target_role
        profile = {
            "user_id": str(applicant.id),
            "company_name": resolved_company_name,
            "company_id": resolved_company_id,
            "department_name": final_department_name,
            "department_id": final_department_id,
            "duty": final_duty,
            "role": target_role,
            "role_label": role_label(target_role),
        }

    async with get_db_session() as session:
        request = await session.get(StaffRequest, request_id)
        if request is None or request.status != StaffRequest.STATUS_PENDING:
            raise StaffError("该申请已被处理", status_code=409)
        request.status = (
            StaffRequest.STATUS_APPROVED if approve else StaffRequest.STATUS_REJECTED
        )
        request.reviewer_id = reviewer.id
        request.reviewer_username = reviewer.username
        request.reviewer_title = title
        request.reviewer_name = name
        request.review_comment = (comment or "").strip() or None
        request.granted_role = granted
        request.reviewed_at = _now()
        request.applicant_seen = False
        await session.flush()
        await session.refresh(request)

    await record_audit(
        "staff.request.review",
        user_id=reviewer.id,
        username=reviewer.username,
        resource_type="user",
        resource_id=str(request.applicant_id) if request.applicant_id else None,
        detail=(
            f"{'approved' if approve else 'rejected'}; "
            f"applicant={snapshot['applicant_username']}; "
            f"company={snapshot['company_name']}; "
            f"department={snapshot['department_name']}; "
            f"role={granted or '-'}; "
            f"reviewer={title} {name}; comment={request.review_comment or '-'}"
        ),
    )
    logger.info(
        "Staff request %s %s by %s (%s %s) for %s",
        request_id, "approved" if approve else "rejected",
        reviewer.username, title, name, snapshot["applicant_username"],
    )
    return request, profile


# ── 企业管理后台：成员与职责 ──────────────────────────────────────────────────

async def list_members(
    actor: User,
    *,
    company_id: str | None = None,
    keyword: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """
    成员列表（管理后台）

    公司隔离：
      * 平台管理员       —— 默认全公司；传 ``company_id`` 可只看某一家
      * 知识库管理员等   —— 强制只看本公司（参数被忽略，不能越权）
    """
    if not is_administer(actor):
        raise StaffError("你没有管理成员身份的权限", status_code=403)

    async with get_db_session() as session:
        stmt = select(User).order_by(User.created_at)
        if actor.is_admin:
            if company_id:
                stmt = stmt.where(User.tenant_id == normalize_tenant_id(company_id))
        else:
            stmt = stmt.where(User.tenant_id == effective_tenant_id(actor))
        if keyword:
            like = f"%{keyword.strip()}%"
            stmt = stmt.where(
                User.username.ilike(like) | User.display_name.ilike(like)
            )
        stmt = stmt.limit(limit)
        users = list((await session.execute(stmt)).scalars().all())
        latest_map = await _latest_status_map(session, [u.id for u in users])

    return [_member_payload(u, latest_map.get(u.id)) for u in users]


async def _latest_status_map(session, ids: list[uuid.UUID]) -> dict:
    """
    批量取每人最新一条申请的状态.

    用窗口函数取每组第一条，避免"每人一次查询"的 N+1（成员列表可能有几百人）。
    """
    if not ids:
        return {}
    ranked = (
        select(
            StaffRequest.applicant_id,
            StaffRequest.status,
            func.row_number()
            .over(
                partition_by=StaffRequest.applicant_id,
                order_by=StaffRequest.created_at.desc(),
            )
            .label("rn"),
        )
        .where(StaffRequest.applicant_id.in_(ids))
        .subquery()
    )
    rows = await session.execute(
        select(ranked.c.applicant_id, ranked.c.status).where(ranked.c.rn == 1)
    )
    return {uid: status for uid, status in rows.all()}


def _member_payload(user: User, latest_status: str | None) -> dict:
    """成员条目（个人主页与成员表格共用一份字段口径）。"""
    if user.is_admin:
        identity = StaffRequest.IDENTITY_APPROVED
    elif latest_status is not None:
        identity = _STATUS_TO_IDENTITY.get(latest_status, StaffRequest.IDENTITY_NONE)
    elif effective_department_id(user) or effective_tenant_id(user) != DEFAULT_TENANT_ID:
        identity = StaffRequest.IDENTITY_APPROVED
    else:
        identity = StaffRequest.IDENTITY_NONE

    return {
        "id": str(user.id),
        "username": user.username,
        "display_name": user.display_name,
        "name": user.display_name or user.username,
        "role": user.role,
        "role_label": role_label(user.role),
        "is_admin": user.is_admin,
        "is_active": user.is_active,
        "company_id": effective_tenant_id(user),
        "company_name": company_display_name(user),
        "department_id": effective_department_id(user),
        "department_name": department_display_name(user),
        "job_title": user.job_title,
        "identity_status": identity,
        "auth_source": user.auth_source or "local",
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


async def list_companies(actor: User) -> list[dict]:
    """
    公司清单（管理后台的左侧过滤 + 成员面板公司下拉）

    平台管理员：只看到**自己创建**的公司（``created_by == actor.id``，注册表为准）
    —— 跨公司信息（哪怕只是"存在这家公司"）也不应泄漏；其自建公司 ``is_test=True``。
    其他管理员：只看到自己所属公司。

    平台管理员本人**不计入任何公司**：它是「全平台」身份，不属于某家公司。
    """
    if not is_administer(actor):
        raise StaffError("你没有管理成员身份的权限", status_code=403)

    from app.services.company_registry import (
        list_registered_companies,
        tenant_ids_created_by,
    )

    if actor.is_admin:
        allowed_ids = await tenant_ids_created_by(actor.id)
    else:
        allowed_ids = frozenset({effective_tenant_id(actor)})

    if not allowed_ids:
        return []

    async with get_db_session() as session:
        count_stmt = (
            select(User.tenant_id, func.count())
            .where(User.role != User.ROLE_ADMIN)
            .where(User.tenant_id.in_(sorted(allowed_ids)))
            .group_by(User.tenant_id)
        )
        rows = (await session.execute(count_stmt)).all()
        counts = {normalize_tenant_id(tid): int(count) for tid, count in rows}
        # 未登记到注册表的历史租户：退回 users.company_name 作展示名
        fallback_rows = await session.execute(
            select(User.tenant_id, func.min(User.company_name))
            .where(User.role != User.ROLE_ADMIN)
            .where(User.tenant_id.in_(sorted(allowed_ids)))
            .group_by(User.tenant_id)
        )
        fallback_names = {
            normalize_tenant_id(tid): cname for tid, cname in fallback_rows.all()
        }

    registry = {c.tenant_id: c for c in await list_registered_companies()}

    items: list[dict] = []
    for tid in sorted(allowed_ids):
        company = registry.get(tid)
        if company is not None:
            company_name = company.display_name
            is_test = bool(company.is_test)
        else:
            company_name = fallback_names.get(tid) or normalize_tenant_id(tid)
            is_test = False
        items.append(
            {
                "company_id": tid,
                "company_name": company_name,
                "is_test": is_test,
                "member_count": counts.get(tid, 0),
            }
        )
    return sorted(
        items, key=lambda item: (-item["member_count"], item["company_id"])
    )


async def set_member_identity(
    actor: User,
    target_id: uuid.UUID,
    *,
    role: str | None = None,
    department_name: str | None = None,
    job_title: str | None = None,
    is_active: bool | None = None,
) -> dict:
    """
    「更换职责」：管理员直接改成员的角色 / 部门 / 职责 / 启用状态.

    这是审核之外的第二条路径 —— 已入职成员调整岗位不必再走一遍申请。
    约束与审核一致：只能授予严格低于自己的角色；不能跨公司；不能改自己
    （否则等于自助提权）。admin 角色永远不能被授予。
    """
    if not is_administer(actor):
        raise StaffError("你没有管理成员身份的权限", status_code=403)

    async with get_db_session() as session:
        target = await session.get(User, target_id)
        if target is None:
            raise StaffError("成员不存在", status_code=404)
        if target.id == actor.id:
            raise StaffError(
                "不能修改自己的职责与权限，请让上级管理员操作",
                status_code=403,
            )
        if target.is_admin:
            raise StaffError("企业管理员账号不可被修改", status_code=403)
        if not actor.is_admin and effective_tenant_id(target) != effective_tenant_id(actor):
            raise StaffError("该成员不属于你所在的公司", status_code=403)
        if not actor.is_admin and not target.is_active and is_active is None:
            # 停用本公司成员需要更高等级
            pass

        changes: list[str] = []

        if role is not None:
            value = role.strip()
            if value == User.ROLE_ADMIN:
                raise StaffError(
                    "企业管理员全局唯一，不能被授予。请选择其他职责。",
                    status_code=403,
                )
            if not can_grant_role(actor, value):
                raise StaffError(
                    f"你的等级（{role_label(actor.role)}）无法授予"
                    f"「{role_label(value)}」：只能授予低于自己的职责",
                    status_code=403,
                )
            if role_rank(target.role) >= role_rank(actor.role):
                raise StaffError(
                    "不能修改与自己同级或更高等级的成员",
                    status_code=403,
                )
            changes.append(f"role={target.role}->{value}")
            target.role = value

        if department_name is not None:
            dept_name = department_name.strip()
            if not dept_name:
                changes.append(f"department={target.department_id}->None")
                target.department_id = None
                target.department_name = None
            else:
                dept_id = department_id_from_name(dept_name)
                if dept_id is None:
                    raise StaffError("部门名称不合法")
                changes.append(f"department={target.department_id}->{dept_id}")
                target.department_id = dept_id
                target.department_name = dept_name

        if job_title is not None:
            value = job_title.strip() or None
            changes.append(f"duty={target.job_title}->{value}")
            target.job_title = value

        if is_active is not None and is_active != target.is_active:
            changes.append(f"is_active={target.is_active}->{is_active}")
            target.is_active = is_active

        if not changes:
            raise StaffError("没有需要修改的内容")

        await session.flush()
        await session.refresh(target)
        payload = _member_payload(target, None)

    await record_audit(
        "staff.member.update",
        user_id=actor.id,
        username=actor.username,
        resource_type="user",
        resource_id=str(target_id),
        detail="; ".join(changes),
    )
    logger.info("Staff member updated by %s: %s", actor.username, "; ".join(changes))
    return payload


# ── 企业管理后台：删除成员（账号注销） ────────────────────────────────────────

# 删除账号时**保留**的文档层级：部门库与公司库已经是组织资产 —— 同事的问答、
# 报告都在引用它，跟着一个离职账号一起消失，别人只会在某天提问时突然发现依据
# 没了。因此删账号前先把归属人解绑（owner_id → NULL），文档本体、可见层级、
# 部门归属与向量索引原样留给组织。
_KEEP_DOC_LEVELS = (ACCESS_DEPARTMENT, ACCESS_TENANT)


async def _deletable_member(actor: User, target_id: uuid.UUID) -> User:
    """
    取回一个**可被 actor 删除**的成员；不可删时抛 StaffError.

    判定顺序（与 ``set_member_identity`` 同口径，避免两处规则漂移）：
      1. actor 具备成员管理权限（``staff.admin``）
      2. 不能删自己 —— 否则等于自助注销，且操作链上没有第二个人可追溯
      3. 平台管理员账号不可被删除（全局唯一，删掉即失去最高权限的来源）
      4. 只能删**等级严格低于**自己的成员
      5. 非平台管理员只能删本公司成员（跨公司按"不存在"处理，不泄漏存在性）
    """
    if not is_administer(actor):
        raise StaffError("你没有管理成员身份的权限", status_code=403)

    async with get_db_session() as session:
        target = await session.get(User, target_id)
        if target is None:
            raise StaffError("成员不存在", status_code=404)
        if target.id == actor.id:
            raise StaffError(
                "不能删除自己的账号，请让另一位管理员操作", status_code=403
            )
        if target.is_admin:
            raise StaffError("平台管理员账号不可被删除", status_code=403)
        if role_rank(target.role) >= role_rank(actor.role):
            raise StaffError(
                f"不能删除与自己同级或更高等级的成员"
                f"（对方等级：{role_label(target.role)}）",
                status_code=403,
            )
        if not actor.is_admin and effective_tenant_id(target) != effective_tenant_id(actor):
            raise StaffError("该成员不属于你所在的公司", status_code=403)
    return target


def _member_snapshot(user: User) -> dict:
    """被删成员的身份快照（写进审计日志，删完就查不到了）。"""
    return {
        "id": str(user.id),
        "username": user.username,
        "name": user.display_name or user.username,
        "company_name": company_display_name(user),
        "department_name": department_display_name(user),
        "job_title": user.job_title,
        "role": user.role,
        "role_label": role_label(user.role),
    }


async def _collect_member_assets(session, target_id: uuid.UUID) -> dict:
    """
    清点删除该成员会动到哪些数据（预检与执行共用同一份口径）.

    ``personal_documents`` 是**待删除**的个人库文档 ``(id, tenant_id)`` 列表
    （租户用于定位磁盘目录）；其余计数用于把"删了什么 / 留了什么"如实写进
    审计日志与前端确认弹窗 —— 界面上说的和代码做的是同一组数字。
    """
    personal_rows = (
        await session.execute(
            select(Document.id, Document.tenant_id).where(
                Document.owner_id == target_id,
                or_(
                    Document.access_level == ACCESS_PRIVATE,
                    Document.access_level.is_(None),  # 迁移前的老数据按个人库处理
                ),
            )
        )
    ).all()

    async def _count(stmt) -> int:
        return int((await session.execute(stmt)).scalar_one() or 0)

    return {
        "personal_documents": [(doc_id, tenant_id) for doc_id, tenant_id in personal_rows],
        "shared_documents": await _count(
            select(func.count())
            .select_from(Document)
            .where(
                Document.owner_id == target_id,
                Document.access_level.in_(_KEEP_DOC_LEVELS),
            )
        ),
        "conversations": await _count(
            select(func.count())
            .select_from(Conversation)
            .where(Conversation.owner_id == target_id)
        ),
        # 会话被删时消息由外键 CASCADE 一并带走，因此这里统计的是"该成员全部
        # 会话下的消息"（含 user_id 为空的老消息），而不是仅按 user_id 命中。
        "messages": await _count(
            select(func.count())
            .select_from(Message)
            .where(
                or_(
                    Message.user_id == target_id,
                    Message.conversation_id.in_(
                        select(Conversation.id).where(Conversation.owner_id == target_id)
                    ),
                )
            )
        ),
        "collections": await _count(
            select(func.count())
            .select_from(Collection)
            .where(Collection.owner_id == target_id)
        ),
        "feedback": await _count(
            select(func.count())
            .select_from(AnswerFeedback)
            .where(AnswerFeedback.user_id == target_id)
        ),
        # 以下三类是**组织留痕**：删账号只解绑归属人（外键 SET NULL），记录本身保留
        "bad_cases": await _count(
            select(func.count()).select_from(BadCase).where(BadCase.user_id == target_id)
        ),
        "share_requests": await _count(
            select(func.count())
            .select_from(ShareRequest)
            .where(
                or_(
                    ShareRequest.requester_id == target_id,
                    ShareRequest.reviewer_id == target_id,
                )
            )
        ),
        "staff_requests": await _count(
            select(func.count())
            .select_from(StaffRequest)
            .where(
                or_(
                    StaffRequest.applicant_id == target_id,
                    StaffRequest.reviewer_id == target_id,
                )
            )
        ),
    }


def _impact_payload(assets: dict, member: dict | None = None) -> dict:
    """把清点结果整理成「将删除 / 将保留」两组（前端直接渲染，无需二次口径）。"""
    payload: dict = {
        "deleted": {
            "personal_documents": len(assets["personal_documents"]),
            "conversations": assets["conversations"],
            "messages": assets["messages"],
            "collections": assets["collections"],
            "feedback": assets["feedback"],
        },
        "kept": {
            "shared_documents": assets["shared_documents"],
            "staff_requests": assets["staff_requests"],
            "share_requests": assets["share_requests"],
            "bad_cases": assets["bad_cases"],
        },
    }
    if member is not None:
        payload["member"] = member
    return payload


async def preview_member_deletion(actor: User, target_id: uuid.UUID) -> dict:
    """
    删除影响预检：不动任何数据，只回答"删掉会失去什么 / 会留下什么".

    存在的意义是让确认弹窗里的数字来自数据库而不是前端硬编码文案 ——
    "删除后个人文档消失、部门与公司文档保留"这条产品约定，必须在点下去
    之前就能被看到。
    """
    target = await _deletable_member(actor, target_id)
    async with get_db_session() as session:
        assets = await _collect_member_assets(session, target_id)
    return _impact_payload(assets, _member_snapshot(target))


async def delete_member(actor: User, target_id: uuid.UUID) -> dict:
    """
    删除成员账号（不可恢复）及其**个人**数据，保留组织资产.

    删除范围（先在 PG 里按与个人数据无关的规则切分，再统一执行）：

        删除   账号本体、他的全部会话与消息、他的文档集合（个人收藏）、
               他的回答反馈、他上传到**个人库**的文档（含 Qdrant 向量与
               uploads/{tenant}/{doc}/ 下的原文件与原图）
        保留   他上传到**部门库 / 公司库**的文档（仅归属人解绑为 NULL，
               可见层级与部门归属不变，同事与全公司仍可检索）、
               身份申请记录、共享申请记录、疑难案例记录（组织留痕），
               审计日志

    执行顺序经过刻意安排：

      1. **向量先行**。Qdrant 不可用时整体中止、账号与 PG 数据一行不动，
         用户可稍后原样重试 —— 与 ``delete_document`` 的失败语义一致；
         反过来（先删 PG 再删向量）会留下无人认领的向量，被检索命中后
         变成"引用了一份已经不存在的文档"。
      2. **先解绑、后删账号**。``documents.owner_id`` 的外键是
         ``ON DELETE CASCADE``：不先把部门库/公司库文档的归属人置空，
         删 users 行会把它们一起带走，"保留部门与公司文档"就成了空话。
      3. **同一事务**内完成解绑 + 清理 + 删账号，避免出现"文档没了但账号还在"
         这种中间态。

    磁盘与向量清理是 best-effort 的收尾步骤，失败只记日志 —— 用户按下的
    "删除该成员"这个动作已经完成，不应因为一个孤儿图片文件而报错。
    """
    target = await _deletable_member(actor, target_id)
    snapshot = _member_snapshot(target)

    async with get_db_session() as session:
        assets = await _collect_member_assets(session, target_id)
    personal_docs: list[tuple[uuid.UUID, str]] = assets["personal_documents"]

    # ── 1. 向量先行：失败即中止，账号与 PG 数据保持原样 ────────────────────────
    if personal_docs:
        from app.services.vector_service import delete_by_document_id

        for doc_id, _tenant in personal_docs:
            try:
                await delete_by_document_id(str(doc_id))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Member deletion aborted — vector purge failed for document_id=%s: %s",
                    doc_id, exc,
                )
                raise StaffError(
                    "个人知识库文档的向量索引删除失败，本次删除已中止"
                    "（账号与数据均未变动），请稍后重试",
                    status_code=503,
                ) from exc

    # ── 2. 单事务：解绑组织文档 → 清个人数据 → 删账号本体 ──────────────────────
    async with get_db_session() as session:
        detached = await session.execute(
            update(Document)
            .where(
                Document.owner_id == target_id,
                Document.access_level.in_(_KEEP_DOC_LEVELS),
            )
            .values(owner_id=None)
        )
        if personal_docs:
            await session.execute(
                delete(Document).where(
                    Document.id.in_([doc_id for doc_id, _ in personal_docs])
                )
            )
        conversations = await session.execute(
            delete(Conversation).where(Conversation.owner_id == target_id)
        )
        collections = await session.execute(
            delete(Collection).where(Collection.owner_id == target_id)
        )
        feedback = await session.execute(
            delete(AnswerFeedback).where(AnswerFeedback.user_id == target_id)
        )
        # 账号本体最后删：staff_requests / share_requests / bad_cases 的归属人
        # 外键是 SET NULL，审核与留痕不会被带走。
        await session.execute(delete(User).where(User.id == target_id))
        # 影响行数在事务内取出（会话关闭后不再读结果对象）
        affected = {
            "conversations": int(conversations.rowcount or 0),
            "collections": int(collections.rowcount or 0),
            "feedback": int(feedback.rowcount or 0),
            "shared_documents": int(detached.rowcount or 0),
        }

    # ── 3. 磁盘收尾（best-effort）：原文件 / 抽取出的图片 ──────────────────────
    if personal_docs:
        from app.services.storage import delete_document_images

        for doc_id, tenant_id in personal_docs:
            try:
                delete_document_images(str(doc_id), tenant_id=tenant_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to purge on-disk assets for document_id=%s: %s", doc_id, exc
                )

    payload = _impact_payload(assets, snapshot)
    # 用执行时的真实影响行数覆盖预检计数（预检到执行之间可能有人新传了文档）
    payload["deleted"] = {
        "personal_documents": len(personal_docs),
        "conversations": affected["conversations"],
        "messages": assets["messages"],
        "collections": affected["collections"],
        "feedback": affected["feedback"],
    }
    payload["kept"]["shared_documents"] = affected["shared_documents"]
    payload["message"] = (
        f"已删除成员 {snapshot['name']}（{snapshot['username']}）的账号与个人数据；"
        f"{payload['kept']['shared_documents']} 份部门/公司知识库文档已保留"
        f"（仅解除归属关系），审核与审计记录一并留存"
    )

    await record_audit(
        "staff.member.delete",
        user_id=actor.id,
        username=actor.username,
        resource_type="user",
        resource_id=str(target_id),
        detail=(
            f"username={snapshot['username']}; name={snapshot['name']}; "
            f"company={snapshot['company_name']}; department={snapshot['department_name']}; "
            f"role={snapshot['role']}; "
            f"deleted[personal_docs={payload['deleted']['personal_documents']}, "
            f"conversations={payload['deleted']['conversations']}, "
            f"messages={payload['deleted']['messages']}, "
            f"collections={payload['deleted']['collections']}, "
            f"feedback={payload['deleted']['feedback']}]; "
            f"kept[shared_docs={payload['kept']['shared_documents']}]"
        ),
    )
    logger.info(
        "Member deleted by %s: username=%s personal_docs=%d conversations=%d kept_docs=%d",
        actor.username, snapshot["username"],
        payload["deleted"]["personal_documents"],
        payload["deleted"]["conversations"],
        payload["kept"]["shared_documents"],
    )
    return payload


async def detail_for(user: User) -> dict:
    """
    个人主页所需的完整身份信息（姓名 / 账号 / 公司 / 部门 / 职位 + 验证状态）.

    前端把它直接铺到个人主页上，因此"公司名 / 部门名"用的是可读原文而不是
    内部 ID —— 用户不应该在界面上看到 ``c1a2b3...`` 这种映射结果。
    """
    status = await identity_status(user)
    latest = await _latest_request(user.id)

    return {
        "id": str(user.id),
        "username": user.username,
        "name": user.display_name or user.username,
        "display_name": user.display_name,
        "role": user.role,
        "role_label": role_label(user.role),
        "is_admin": user.is_admin,
        "is_active": user.is_active,
        "company_id": effective_tenant_id(user),
        "company_name": company_display_name(user),
        "department_id": effective_department_id(user),
        "department_name": department_display_name(user),
        "job_title": user.job_title,
        "auth_source": user.auth_source or "local",
        "identity_status": status,
        "grantable_roles": grantable_roles_for(user),
        "can_review": can_review_any(user),
        "can_administer": is_administer(user),
        "latest_request": (
            serialize(latest, viewer=user) if latest is not None else None
        ),
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


__all__ = [
    "StaffError",
    "ROLE_RANK",
    "GRANTABLE_ROLES",
    "role_rank",
    "can_grant_role",
    "grantable_roles_for",
    "is_administer",
    "can_review_any",
    "can_review_staff",
    "identity_status",
    "is_identity_verified",
    "create_staff_request",
    "cancel_staff_request",
    "list_my_requests",
    "list_inbox",
    "summary",
    "mark_my_requests_seen",
    "review_staff_request",
    "list_members",
    "list_companies",
    "set_member_identity",
    "preview_member_deletion",
    "delete_member",
    "detail_for",
    "serialize",
]
