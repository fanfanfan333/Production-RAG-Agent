"""
Central RBAC policy for enterprise RAG resources.

角色与能力矩阵（与产品给出的三层知识库权限表一一对应）:

    角色            上传  个人库  发布到部门库  发布到公司库   删除他人文档
    普通员工        ✓     ✓      申请(需审核)    ✗             ✗
    部门负责人      ✓     ✓      ✓              申请(需审核)   本部门范围
    知识库管理员    ✓     ✓      ✓              ✓             全公司
    企业管理员      ✓     ✓      ✓              ✓             全公司
    平台管理员      ✓     ✓      所有公司        所有公司        所有公司
                                                的部门库        的部门库

三条贯穿全矩阵的硬规则（改动权限时先看这三条）:

  1. **个人库（private）对任何人都不开放**，只有上传者本人可见 ——
     平台管理员也不行。"其他人的个人文档看不到"是产品红线，不是 UI 文案。
  2. **公司边界**：非平台管理员的一切读写都被锁在 ``users.tenant_id``
     之内；平台管理员是唯一的跨公司身份，身份上记为「全平台」。
  3. 能力矩阵（本文件）决定"能不能做这个动作"，``tenancy.scope_for``
     决定"这个动作能看到哪些数据"，两者一起才是一条完整的授权。

权限名统一为 ``resource.action`` 字符串，便于审计聚合。需要"范围"语义的
能力（删除部门内他人文档 / 删除全公司文档 / 跨公司只读）刻意拆成不同权限名，
而不是靠调用点各自判断 —— 谁能删什么只有一个地方可以改。

历史角色（manager/editor/viewer/user）等价映射到上面的能力集：升级后旧账号
行为不变（user/editor ≈ 普通员工，manager ≈ 部门负责人）。
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, status

from app.db.user_models import User
from app.services.tenancy import TENANT_WIDE_READER_ROLES

# 注意：``get_current_user`` 刻意**不**在模块级导入。
# ``app.api.__init__`` 会导入各 API 模块，而 API 模块又要导入本模块的
# require_permission —— 若本模块在导入期反向依赖 app.api.deps，就会形成
# "permissions → app.api → documents → permissions" 的循环，只有在"被
# 第一个导入的恰好是 permissions"时才暴露为 ImportError（极其难排查）。
# 延迟到函数体内导入即可彻底断开这条环。

# ── 角色中文名（前端展示 / 审计日志 / 申请审核说明共用一份）───────────────────
ROLE_LABELS: dict[str, str] = {
    User.ROLE_COMPANY_ADMIN: "企业管理员",
    User.ROLE_KB_ADMIN: "知识库管理员",
    User.ROLE_DEPT_MANAGER: "部门负责人",
    User.ROLE_EMPLOYEE: "普通员工",
    User.ROLE_ADMIN: "平台管理员",
    User.ROLE_MANAGER: "部门负责人",
    User.ROLE_EDITOR: "普通员工",
    User.ROLE_USER: "普通员工",
    User.ROLE_VIEWER: "只读成员",
}

# 角色中文名的单一来源（tenancy / 申请服务 / API 出参都从这里取）
def role_label(role: str | None) -> str:
    return ROLE_LABELS.get((role or "").strip(), role or "成员")


# ── 能力集 ────────────────────────────────────────────────────────────────────

# 普通员工：上传 + 个人知识库；发布到部门库要申请，发布到公司库无权限。
_EMPLOYEE_PERMISSIONS = frozenset(
    {
        "knowledge.read", "knowledge.write", "knowledge.delete",
        "document.read", "document.write", "document.delete.own",
        "conversation.read", "conversation.write", "conversation.delete",
        "feedback.write",
        "share.request",                 # 申请把自己的个人文档共享出去
    }
)

# 部门负责人：+ 直接发布到部门库、审核部门内申请、删除本部门他人文档。
#   staff.review —— 审核**下级成员**的身份验证申请（产品要求"部门负责人设置
#   普通员工"）。注意它与 share.review.department 是两件事：前者审"人"，
#   后者审"文档"。
_DEPT_MANAGER_PERMISSIONS = _EMPLOYEE_PERMISSIONS | frozenset(
    {
        "document.publish.department",
        "document.delete.department",
        "share.review.department",
        "staff.review",
        "audit.read",
    }
)

# 知识库管理员：+ 直接发布到公司库、审核公司级申请、删除全公司文档。
#   staff.admin —— 进入企业管理后台：查看全公司成员、更换成员职责、
#   设置部门负责人（产品要求"企业管理人可以设置知识库管理员与部门负责人"）。
_KB_ADMIN_PERMISSIONS = _DEPT_MANAGER_PERMISSIONS | frozenset(
    {
        "document.publish.company",
        "document.delete.tenant",
        "document.read.all",
        "share.review.company",
        "staff.admin",
        # audit.write —— 重置质量统计（清 quality_events / 进程内计数器）。
        # 刻意**不放**在部门负责人层：quality_events 不带 tenant_id，重置是
        # 一次全局且不可逆的删除；让部门负责人（只能看指标的角色）握有抹掉
        # 全公司历史指标的权力属于过度授权。可见性归 audit.read，重置归
        # audit.write，二者分层的意义正在于此。
        "audit.write",
    }
)

# 企业管理员：本公司范围内全部业务权限（用户/审计另由平台管理员把关）。
_COMPANY_ADMIN_PERMISSIONS = _KB_ADMIN_PERMISSIONS | frozenset(
    {
        "knowledge.admin",
        "feedback.read",
    }
)

ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    # 平台管理员（跨租户）
    User.ROLE_ADMIN: frozenset({"*"}),
    User.ROLE_COMPANY_ADMIN: _COMPANY_ADMIN_PERMISSIONS,
    User.ROLE_KB_ADMIN: _KB_ADMIN_PERMISSIONS,
    User.ROLE_DEPT_MANAGER: _DEPT_MANAGER_PERMISSIONS,
    User.ROLE_EMPLOYEE: _EMPLOYEE_PERMISSIONS,
    # ── 历史角色 ──────────────────────────────────────────────────────────────
    User.ROLE_MANAGER: _DEPT_MANAGER_PERMISSIONS,
    User.ROLE_EDITOR: _EMPLOYEE_PERMISSIONS,
    User.ROLE_USER: _EMPLOYEE_PERMISSIONS,
    User.ROLE_VIEWER: frozenset(
        {
            "knowledge.read", "document.read", "conversation.read",
            "conversation.write", "feedback.write",
        }
    ),
}


def has_permission(user: User, permission: str) -> bool:
    """Return whether *user* owns an exact named permission or wildcard."""
    permissions = ROLE_PERMISSIONS.get(user.role, frozenset())
    return "*" in permissions or permission in permissions


def permissions_of(user: User) -> list[str]:
    """Sorted capability list — surfaced to the UI so buttons match the API."""
    permissions = ROLE_PERMISSIONS.get(user.role, frozenset())
    return sorted(permissions)


def can_access_all_documents_in_tenant(user: User) -> bool:
    """
    是否可读**本租户内**全部部门库 / 公司库文档.

    企业管理员 / 知识库管理员具备 —— 他们需要审核共享申请、处理合规删除。
    普通员工与部门负责人只能看本部门。

    **不含他人个人库**：个人库对任何人都不开放（包括平台管理员），
    角色集合与 tenancy.TENANT_WIDE_READER_ROLES 共用同一份定义。
    """
    return user is not None and user.role in TENANT_WIDE_READER_ROLES


def can_access_all_documents_platform(user: User) -> bool:
    """
    是否可读**所有公司**的部门库 / 公司库文档（平台管理员专有）.

    注意它同样不包含他人的个人库 —— 平台管理员的跨公司能力只作用于
    部门层与公司层。真正的可见范围由 ``tenancy.scope_for(user)`` 组装。
    """
    return user is not None and user.is_admin


def require_permission(permission: str) -> Callable:
    """
    Build a FastAPI dependency enforcing one RBAC permission.

    在权限判定**之后**再叠加一道"身份已验证"闸门（``REQUIRE_IDENTITY_VERIFICATION``
    为 True 时）：未通过企业身份验证的账号拿不到任何业务权限，从而自动覆盖
    上传 / 提问 / 共享 / 文档管理等全部入口，不需要在每个路由上重复挂依赖。
    认证类（/auth/*）与身份验证类（/staff/me、/staff/requests）刻意不走
    ``require_permission``，因此未验证用户仍能自救。
    """
    from app.api.deps import get_current_user

    async def dependency(user: User = Depends(get_current_user)) -> User:
        if not has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"当前角色（{role_label(user.role)}）无权执行此操作"
                    f"（需要权限：{permission}）"
                ),
            )

        from app.config import get_settings

        if get_settings().REQUIRE_IDENTITY_VERIFICATION:
            from app.services.staff_service import is_identity_verified

            if not await is_identity_verified(user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        "你还未通过企业身份验证，暂时无法使用知识库功能。"
                        "请在首页提交「身份验证」申请，等待上级审核通过。"
                    ),
                )
        return user

    return dependency


def require_verified_identity() -> Callable:
    """
    身份验证闸门：只有通过「企业身份验证」的账号才能使用知识库业务
    （上传 / 提问 / 申请共享）。

    与产品要求"没有注册和职责的不能进入"对应。刻意**不拦**认证类接口
    （/auth/*）与身份验证接口（/staff/*）—— 否则未验证用户连提交申请都做不到，
    会形成死锁。

    同样延迟导入：``staff_service`` 反向依赖本模块的 ``has_permission`` /
    ``role_label``，模块级导入会形成循环。
    """
    from app.api.deps import get_current_user

    async def dependency(user: User = Depends(get_current_user)) -> User:
        from app.services.staff_service import is_identity_verified

        if not await is_identity_verified(user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "你还未通过企业身份验证，暂时无法使用知识库功能。"
                    "请在首页提交「身份验证」申请，等待上级审核通过。"
                ),
            )
        return user

    return dependency


def require_platform_admin() -> Callable:
    """
    Build a dependency for platform-only operations.

    与 ``require_permission`` 同理做成工厂：依赖必须延迟到调用点所在模块
    （API 模块）导入时才解析 ``get_current_user``。若写成模块级
    ``async def require_platform_admin(user=Depends(get_current_user))``，
    ``get_current_user`` 会在本模块被导入的那一刻求值，落回上面注明的循环
    依赖里。用法：``Depends(require_platform_admin())``。
    """
    from app.api.deps import get_current_user

    async def dependency(user: User = Depends(get_current_user)) -> User:
        if not user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="该操作需要平台管理员权限",
            )
        return user

    return dependency
