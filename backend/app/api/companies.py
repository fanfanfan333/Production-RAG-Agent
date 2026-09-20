"""
公司注册表 API 路由（``/companies``）.

四个端点，两个语义域，刻意不复用（决策 7 / §10.6）：

  1. ``GET /companies``            已登录          全部已注册公司（身份验证下拉候选）
  2. ``GET /companies/accessible`` ``document.read`` 调用者**可见范围**内的公司 + ``doc_count``
                                     （文档页公司筛选；候选与列表/检索**同源**）
  3. ``POST /companies``           平台管理员      创建公司（唯一名，重名 409）
  4. ``PATCH /companies/{id}``     平台管理员（须自建）改名（``tenant_id`` 不变）
  5. ``GET /companies/{id}/deletion-preview`` 平台管理员  删除前的影响预检
  6. ``DELETE /companies/{id}``    平台管理员      删除整家公司（不可恢复）

``GET /companies/accessible`` 的 ``doc_count`` 必须与 ``GET /documents?company_id=...``
的条数逐条一致 —— 两者共用 ``document_query_service.count_by_tenant`` /
``list_documents`` 背后的**同一个** ``document_scope_clause`` 入口。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.deps import get_current_user, require_admin
from app.db.user_models import User
from app.schemas.company import (
    AccessibleCompanyItem,
    CompanyCreate,
    CompanyItem,
    CompanyRename,
)
from app.services.company_registry import (
    CompanyError,
    company_display_names,
    create_company,
    list_registered_companies,
    rename_company,
)
from app.services.document_query_service import count_by_tenant
from app.services.permissions import require_permission
from app.services.staff_service import (
    StaffError,
    delete_company,
    preview_company_deletion,
)
from app.services.tenancy import (
    DEFAULT_TENANT_ID,
    is_platform_admin,
    request_scope,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Companies"])

# ``default`` 租户在**平台管理员**视角下的展示别名。companies 表里没有 ``default``
# 行（它是"无公司"的历史兜底租户，见 ``tenancy.DEFAULT_TENANT_ID``），显示名原本
# 会回退成裸 tenant_id，管理员就在下拉里看到一个无法解释的内部标识。改成中文
# 别名只是**显示层**换名：tenant_id 仍是 ``default``，文档归属与检索 scope 一律
# 不动（零数据迁移）。
ADMIN_TENANT_DISPLAY_NAME = "管理员"


def _http_from_company_error(exc: CompanyError) -> HTTPException:
    """把 :class:`CompanyError` 映射成携带其 HTTP 语义的响应。"""
    return HTTPException(status_code=exc.status_code, detail=str(exc))


# ── GET /companies ────────────────────────────────────────────────────────────

@router.get(
    "/companies",
    summary="已注册公司清单（身份验证下拉候选）",
    description=(
        "返回注册表中的**全部**已注册公司（``company_id`` + ``company_name``）。"
        "任何已登录用户可用 —— 身份验证弹窗需要先让申请人选得到公司；"
        "提交的身份验证申请再由审核链路校验归属。"
    ),
)
async def list_companies_endpoint(
    user: Annotated[User, Depends(get_current_user)],
) -> dict:
    companies = await list_registered_companies()
    items = [
        CompanyItem(company_id=c.tenant_id, company_name=c.display_name).model_dump()
        for c in companies
    ]
    return {"items": items, "total": len(items)}


# ── GET /companies/accessible ─────────────────────────────────────────────────

@router.get(
    "/companies/accessible",
    response_model=list[AccessibleCompanyItem],
    summary="我可访问的公司 + 文档数（文档页公司筛选）",
    description=(
        "返回调用者**可见范围**内的公司清单，以及每家公司下**可见**文档数。\n\n"
        "- 平台管理员：其**所属租户 ∪ 自建测试公司集合**（二者皆无文档时返回空列表）；\n"
        "- 普通成员：其所属公司。\n\n"
        "``doc_count`` 与文档列表过滤结果同源（同一 ``document_scope_clause``），"
        "保证「筛选项里的计数 == 选中该公司的列表结果数」。\n\n"
        "``default``（历史兜底租户）**只有平台管理员看得到**：管理员的那个选项显示"
        "为「管理员」，其余账号的清单里该项被剔除。"
    ),
)
async def list_accessible_companies_endpoint(
    user: Annotated[User, Depends(require_permission("document.read"))],
) -> list[AccessibleCompanyItem]:
    # 候选与列表/检索**同源**：同一 request_scope → 同一 tenant_ids 集合
    scope = await request_scope(user)
    tenant_ids = scope.tenant_ids or frozenset()  # None(诊断) 在此端点按空处理
    if not tenant_ids:
        return []

    platform_admin = is_platform_admin(user)
    if not platform_admin:
        # ``default`` 是"尚未归属公司"的内部兜底租户，在上传/筛选的下拉里毫无语义
        # —— 只有平台管理员（其 ``effective_tenant_id`` 就是它）需要用它代表自己
        # 的归属。老账号（未设公司）也会落到该租户，若不在源头剔除，普通成员就会
        # 在候选里看到一个既不对应任何注册公司、又说不清是什么的 "default" 项。
        # 剔除只影响**候选清单**：文档可见范围（``request_scope``）原样保留。
        tenant_ids = tenant_ids - {DEFAULT_TENANT_ID}
        if not tenant_ids:
            return []

    names = await company_display_names(tenant_ids)
    counts = await count_by_tenant(
        owner_id=scope.owner_id,
        tenant_ids=tenant_ids,
        owns_tenant_ids=scope.owns_tenant_ids,
        department_id=scope.department_id,
        tenant_wide=scope.tenant_wide,
    )
    return [
        AccessibleCompanyItem(
            company_id=t,
            display_name=_display_name_for(t, names, platform_admin),
            doc_count=counts.get(t, 0),
        )
        for t in sorted(tenant_ids)
    ]


def _display_name_for(
    tenant_id: str,
    names: dict[str, str],
    platform_admin: bool,
) -> str:
    """候选项的展示名：注册表里有行就用注册名，否则回落 ``tenant_id``."""
    if platform_admin and tenant_id == DEFAULT_TENANT_ID:
        return ADMIN_TENANT_DISPLAY_NAME
    return names.get(tenant_id, tenant_id)


# ── POST /companies ───────────────────────────────────────────────────────────

@router.post(
    "/companies",
    status_code=status.HTTP_201_CREATED,
    summary="创建公司（平台管理员）",
    description=(
        "平台管理员注册一家新公司。公司名唯一（大小写/空格变体视为重名 → 409）；"
        "``company_id``（``tenant_id``）随机生成且与名称解耦 —— 改名不换 id。\n\n"
        "创建者即平台管理员本人，该公司随即进入其「自建测试公司集合」。"
    ),
)
async def create_company_endpoint(
    body: CompanyCreate,
    actor: Annotated[User, Depends(require_admin)],
) -> dict:
    try:
        company = await create_company(actor, body.display_name)
    except CompanyError as exc:
        raise _http_from_company_error(exc) from exc

    await _audit_company(
        "company.create", actor, company.tenant_id,
        f"name={company.display_name}",
    )
    return {
        "company_id": company.tenant_id,
        "company_name": company.display_name,
        "created_at": company.created_at.isoformat() if company.created_at else None,
    }


# ── PATCH /companies/{company_id} ─────────────────────────────────────────────

@router.patch(
    "/companies/{company_id}",
    summary="公司改名（平台管理员，须自建或无主）",
    description=(
        "修改公司展示名。平台管理员可改**自己创建的**或**无主（创建者为空）的历史"
        "公司**；他人创建的公司 → 403。重名（含大小写/空格变体）→ 409。\n\n"
        "``tenant_id`` 与全部文档/向量归属**不动**：改名只改 ``companies.display_name``"
        " / ``name_key``，并同步本租户成员的 ``users.company_name``（展示副本）；"
        "**不回填 ``created_by``**（无主公司改名后仍保持无主，不进入文档可见范围）。"
    ),
)
async def rename_company_endpoint(
    company_id: Annotated[str, Path(description="公司标识（tenant_id）")],
    body: CompanyRename,
    actor: Annotated[User, Depends(require_admin)],
) -> dict:
    try:
        company = await rename_company(actor, company_id, body.display_name)
    except CompanyError as exc:
        raise _http_from_company_error(exc) from exc

    return {
        "company_id": company.tenant_id,
        "company_name": company.display_name,
        "updated_at": company.updated_at.isoformat() if company.updated_at else None,
        "message": "已改名",
    }


# ── GET /companies/{company_id}/deletion-preview ──────────────────────────────

@router.get(
    "/companies/{company_id}/deletion-preview",
    summary="删除公司前的影响预检（将删除什么 / 将保留什么）",
    description=(
        "平台管理员专用。返回删除该公司**会失去什么 / 会留下什么**的逐项计数，"
        "供确认弹窗展示 —— 数字全部来自数据库，与真正执行的删除**同一份口径**。\n\n"
        "将删除：员工账号（连带个人数据）、该公司 ``tenant_id`` 下的**三级文档**"
        "（个人 / 部门 / 公司，含分块与向量索引）、全部会话与消息、公司注册行。\n\n"
        "将保留：审计日志与审核留痕（身份验证申请 / 共享申请 / 疑难案例）。\n\n"
        "只有平台管理员（``is_admin``）可调用；``default`` 兜底租户不在注册表里，"
        "预检直接 404，因此管理员删不掉自己的落脚点。"
    ),
)
async def company_deletion_preview_endpoint(
    company_id: Annotated[str, Path(description="公司标识（tenant_id）")],
    actor: Annotated[User, Depends(require_admin)],
) -> dict:
    try:
        return await preview_company_deletion(actor, company_id)
    except StaffError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


# ── DELETE /companies/{company_id} ────────────────────────────────────────────

@router.delete(
    "/companies/{company_id}",
    summary="删除整家公司（平台管理员，不可恢复）",
    description=(
        "平台管理员删除一家公司，**不可恢复**：\n\n"
        "- 该公司下**全部员工账号**一并注销（连同其个人数据），员工需重新注册；\n"
        "- 该公司 ``tenant_id`` 下的**全部文档**（个人 / 部门 / 公司三级）连同分块、"
        "倒排索引、Qdrant 向量与磁盘原文件（``uploads/{tenant_id}/``）一起删除；\n"
        "- 全部会话与消息删除；公司注册行删除。\n\n"
        "**保留**：审计日志与审核留痕（身份验证申请 / 共享申请 / 疑难案例），用于"
        "企业合规回溯。\n\n"
        "只有平台管理员（``is_admin``）可调用；公司必须已注册（否则 404）—— "
        "``default``（历史兜底租户）不在注册表里，因此**不可删除**。\n\n"
        "失败语义：先清向量索引，Qdrant 不可用时整体中止、数据库**一行不动**，"
        "可稍后原样重试。"
    ),
)
async def delete_company_endpoint(
    company_id: Annotated[str, Path(description="公司标识（tenant_id）")],
    actor: Annotated[User, Depends(require_admin)],
) -> dict:
    try:
        return await delete_company(actor, company_id)
    except StaffError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


async def _audit_company(action: str, actor: User, tenant_id: str, detail: str) -> None:
    """记录公司级审计（改名已在注册表服务内落审计，这里只补创建）。"""
    from app.services.audit_service import record_audit

    await record_audit(
        action,
        user_id=getattr(actor, "id", None),
        username=getattr(actor, "username", None),
        resource_type="company",
        resource_id=tenant_id,
        detail=detail,
    )


__all__ = ["router"]
