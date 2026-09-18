"""add companies registry (公司从推导概念升级为一等注册实体)

本轮把「公司」从 ``GROUP BY users.tenant_id`` 推导的概念，升级为可管理的
一等实体 ``companies``。这是「改名不改 tenant_id、零迁移」与「admin 只在自己
创建的公司内工作」两项需求的事实源。

表结构（决策 3）
────────────────
    tenant_id     VARCHAR(64)  PRIMARY KEY          -- 稳定标识（改名不变）
    display_name  VARCHAR(128) NOT NULL             -- 展示名（可改）
    name_key      VARCHAR(128) NOT NULL             -- 归一化键（唯一性判定）
    created_by    UUID NULL REFERENCES users(id) ON DELETE SET NULL
    is_test       BOOLEAN NOT NULL DEFAULT false     -- 创建者为平台管理员
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    CONSTRAINT uq_companies_name_key UNIQUE (name_key)
    CREATE INDEX ix_companies_created_by ON companies(created_by)

写法与既有迁移一致：全部 ``IF NOT EXISTS``，与启动时 ``Base.metadata.create_all``
两种建表机制任一先跑都不冲突（幂等）。

Revision ID: p1j2k3l4m5n6
Revises: o0i1j2k3l4m5
Create Date: 2026-09-18 18:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "p1j2k3l4m5n6"
down_revision: Union[str, None] = "o0i1j2k3l4m5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS companies (
            tenant_id    VARCHAR(64)  PRIMARY KEY,
            display_name VARCHAR(128) NOT NULL,
            name_key     VARCHAR(128) NOT NULL,
            created_by   UUID         NULL
                         REFERENCES users(id) ON DELETE SET NULL,
            is_test      BOOLEAN      NOT NULL DEFAULT false,
            created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
            CONSTRAINT uq_companies_name_key UNIQUE (name_key)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_companies_created_by ON companies(created_by)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_companies_created_by")
    op.execute("DROP TABLE IF EXISTS companies")
