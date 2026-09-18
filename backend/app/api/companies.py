"""
公司注册表 API 路由（``/companies``）.

四个端点，两个语义域，刻意不复用（决策 7 / §10.6）：

  1. ``GET /companies``            已登录          全部已注册公司（身份验证下拉候选）
  2. ``GET /companies/accessible`` ``document.read`` 调用者**可见范围**内的公司 + ``doc_count``
                                     （文档页公司筛选；候选与列表/检索**同源**）
  3. ``POST /companies``           平台管理员      创建公司（唯一名，重名 409）
  4. ``PATCH /companies/{id}``     平台管理员（须自建）改名（``tenant_id`` 不变）

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
from app.services.tenancy import request_scope
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Companies"])


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
        "- 平台管理员：其**自建测试公司集合**（无自建公司时返回空列表）；\n"
        "- 普通成员：其所属公司。\n\n"
        "``doc_count`` 与文档列表过滤结果同源（同一 ``document_scope_clause``），"
        "保证「筛选项里的计数 == 选中该公司的列表结果数」。"
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
            display_name=names.get(t, t),
            doc_count=counts.get(t, 0),
        )
        for t in sorted(tenant_ids)
    ]


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
    summary="公司改名（平台管理员，须自建）",
    description=(
        "修改公司展示名。**只有创建该公司的平台管理员**可改名（非自建 → 403）；"
        "重名（含大小写/空格变体）→ 409。\n\n"
        "``tenant_id`` 与全部文档/向量归属**不动**：改名只改 ``companies.display_name``"
        " / ``name_key``，并同步本租户成员的 ``users.company_name``（展示副本）。"
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
