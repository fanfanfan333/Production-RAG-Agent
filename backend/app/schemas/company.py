"""
公司注册表 Pydantic schema.

两类清单、两个端点，避免混用（决策 8 / §10.6）：

  1. ``GET /companies``            → ``CompanyItem``        全部已注册公司
                                     （任何已登录用户可用，供身份验证下拉）
  2. ``GET /companies/accessible`` → ``AccessibleCompanyItem``
                                     调用者**可见范围**内的公司 + 文档数
                                     （``document.read`` 权限，候选与列表/检索同源）
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompanyCreate(BaseModel):
    """创建公司请求。"""

    display_name: str = Field(..., min_length=1, max_length=128)


class CompanyRename(BaseModel):
    """公司改名请求（只改展示名，``tenant_id`` 不变）。"""

    display_name: str = Field(..., min_length=1, max_length=128)


class CompanyItem(BaseModel):
    """``GET /companies`` 的单项：身份验证下拉候选。"""

    company_id: str
    company_name: str


class VisibleCompanyItem(BaseModel):
    """``GET /staff/companies`` 的单项（成员管理过滤用）。"""

    company_id: str
    company_name: str
    is_test: bool = False
    member_count: int = 0


class AccessibleCompanyItem(BaseModel):
    """
    ``GET /companies/accessible`` 的单项：文档页公司筛选候选。

    字段名用 ``display_name``（team-lead 指定），并带 ``doc_count`` ——
    ``doc_count`` 必须走 ``document_scope_clause``（同一入口）统计，
    从结构上保证「筛选项里的计数 == 选中该公司的列表结果数」。
    """

    company_id: str
    display_name: str
    doc_count: int = 0


__all__ = [
    "CompanyCreate",
    "CompanyRename",
    "CompanyItem",
    "VisibleCompanyItem",
    "AccessibleCompanyItem",
]
