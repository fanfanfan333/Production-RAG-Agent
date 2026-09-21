"""
安全隔离管理面（need-to-know 授予 + 密级/可见性 + 全局开关）的 Pydantic schema.

对应 ``docs/system_design_security_isolation.md`` 决策 13 / §4.5 / §15-13。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── need-to-know 授予 ─────────────────────────────────────────────────────────


class GrantCreate(BaseModel):
    """
    提交 need-to-know 授予申请（``pending``；pending 期间不写 ``acl_allow``）.

    三条硬约束由服务层强制（schema 只表达形状）：
      * 必须带 ``expires_at``（有效期）；
      * 只能在 ``doc`` / ``image`` 对象上授予；
      * **禁止自我授予**（主体设成自己 → 403）。
    """

    document_id: str
    subject: str = Field(
        ...,
        description='主体：user:<uuid> | dept:<id> | role:<role> | project:<id> | group:<id>',
        max_length=128,
    )
    effect: Literal["allow", "deny"] = "allow"
    reason: str | None = Field(None, max_length=1000)
    expires_at: datetime = Field(..., description="必填：例外到期时间（Q5=A）")
    object_id: str | None = Field(
        None,
        max_length=128,
        description="对象级授予目标；缺省 = 文档本身（str(document_id)）",
    )


class GrantReview(BaseModel):
    """审批一条授予申请。"""

    approve: bool
    comment: str | None = Field(None, max_length=1000)


class GrantItem(BaseModel):
    """授予出参。"""

    grant_id: str
    document_id: str
    object_id: str
    subject: str
    effect: str
    status: str
    granted_by: str | None = None
    reviewer_id: str | None = None
    reason: str | None = None
    expires_at: datetime | None = None
    created_at: datetime | None = None
    reviewed_at: datetime | None = None


# ── 密级 / 可见性 / 项目维度 ──────────────────────────────────────────────────


class DocumentSecurityUpdate(BaseModel):
    """
    设置文档密级 / 可见性模式 / 项目集合（只改显式传入的字段）.

    * ``security_level``：0=公开 / 1=内部 / 2=机密 / 3=绝密（空 = 不改）；
    * ``visibility_mode``：``tier``（走三层知识库）/ ``project``（走项目成员）；
    * ``project_ids``：``visibility_mode=project`` 时命中的项目集合。
    """

    security_level: int | None = Field(None, ge=0, le=3)
    visibility_mode: Literal["tier", "project"] | None = None
    project_ids: list[str] | None = None


class ObjectEscalate(BaseModel):
    """对单个对象提级 / 剔除（图片对象会触发 OCR 派生取严级联）。"""

    security_level: int | None = Field(None, ge=0, le=3)
    excluded: bool | None = None


class SecuritySettings(BaseModel):
    """安全隔离的全局开关与前置条件自检（运维视角）。"""

    security_strict_mode: bool
    acl_security_prefilter_strict: bool
    default_security_level: int
    project_enabled: bool
    null_effective_level_count: int
    prefilter_strict_precondition_met: bool
    note: str


__all__ = [
    "DocumentSecurityUpdate",
    "GrantCreate",
    "GrantItem",
    "GrantReview",
    "ObjectEscalate",
    "SecuritySettings",
]
