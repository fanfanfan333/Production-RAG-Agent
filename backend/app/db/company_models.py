"""
公司注册表 ORM（公司从「推导概念」升级为一等注册实体）.

背景
────
改造前「公司」不是实体 —— 它是 ``GROUP BY users.tenant_id`` 推导出来的，
因此既没有「创建公司」这个动作，也没有「改名」能力。本表把公司物化为
可管理的一等实体：

    tenant_id    稳定标识（主键，改名不变）—— 权限与绑定一律用它
    display_name 展示名（可改）
    name_key     归一化键（NFKC → 去空白 → casefold），唯一约束是重名最后一道闸
    created_by   创建者（ON DELETE SET NULL）—— admin 的「自建测试公司」按它定位
    is_test      是否测试公司（创建者为平台管理员）

为什么 ``tenant_id`` 与名称解耦：这是「改名不改 tenant_id、零迁移」的前提。
新公司的 ``tenant_id`` 由 ``company_registry.generate_tenant_id()`` 随机生成
（``"c" + uuid4().hex[:12]``），**不再**由名称哈希派生（后者改名即换 id）。

为什么把 ``is_test`` 物化成一列：``created_by`` 是 ``ON DELETE SET NULL``，
一旦管理员账号被删，若靠 ``created_by IS NOT NULL`` 判测试公司会静默漂移；
``is_test`` 把分类钉死。

本模块只声明模型；``main.py`` import 它，``Base.metadata.create_all`` 便会
自动建表（与 alembic 迁移并存，两边都用 IF NOT EXISTS 幂等）。
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base


class Company(Base):
    """一家已注册公司（``tenant_id`` 为稳定主键，展示名可改）。"""

    __tablename__ = "companies"
    __table_args__ = (
        UniqueConstraint("name_key", name="uq_companies_name_key"),
        Index("ix_companies_created_by", "created_by"),
    )

    # 稳定标识：改名不变；权限/绑定一律用它（绝不能拿展示名做绑定或比较）。
    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 展示名（可由 admin 改名；改名只改这一列 + name_key，不动 documents.tenant_id）。
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    # 归一化键（NFKC → 去全部空白 → casefold），唯一约束保证「公司名唯一」。
    name_key: Mapped[str] = mapped_column(String(128), nullable=False)
    # 创建者：平台管理员创建的公司即其「自建测试公司」；A/B 公司为 NULL。
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    # 是否测试公司（创建者为平台管理员）——物化成列，避免 created_by 被置空后漂移。
    is_test: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"<Company tenant_id={self.tenant_id!r} "
            f"display_name={self.display_name!r} is_test={self.is_test}>"
        )
