"""
项目维度 API 的 Pydantic schema（T5）.

项目是横向维度：它只**增加**可见性（``_source_gate`` 的第四个 OR 分支），
不替换 ``access_level`` 的三值语义。schema 只做形状与范围校验，"谁能管"由
API 依赖（``security.escalate``）把关。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    """创建项目请求。``tenant_id`` 只对平台管理员生效（给指定公司建项目）。"""

    name: str = Field(..., min_length=1, max_length=128)
    project_id: str | None = Field(
        None, max_length=64, description="可选；缺省时随机生成 p_<hex12>"
    )
    tenant_id: str | None = Field(
        None, max_length=64, description="仅平台管理员可指定"
    )


class ProjectRename(BaseModel):
    """改项目名。"""

    name: str = Field(..., min_length=1, max_length=128)


class MemberAdd(BaseModel):
    """加成员 / 更新成员有效期（``expires_at`` 为空 = 长期成员）。"""

    user_id: str = Field(..., max_length=64)
    expires_at: datetime | None = Field(
        None, description="P1-2 临时成员：到期后自动失效（不再进入 project_ids）"
    )


class ProjectItem(BaseModel):
    """项目出参。"""

    project_id: str
    tenant_id: str
    name: str
    created_by: str | None = None
    created_at: datetime | None = None
    member_count: int = 0


class MemberItem(BaseModel):
    """成员出参（``active`` 由 ``expires_at`` 与当前时间比较得出）。"""

    user_id: str
    username: str | None = None
    expires_at: datetime | None = None
    active: bool = True
    added_by: str | None = None
    added_at: datetime | None = None


__all__ = [
    "MemberAdd",
    "MemberItem",
    "ProjectCreate",
    "ProjectItem",
    "ProjectRename",
]
