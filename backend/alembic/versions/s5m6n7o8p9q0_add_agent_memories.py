"""add agent_memories (用户级长期记忆)

为什么必须补这一条迁移，而不是只靠 `Base.metadata.create_all`：

本项目同时存在两条建表路径 —— 启动时 `create_all`（补齐缺失表）与
`alembic upgrade head`（版本化演进）。新表只加进 ORM、不写迁移时，
两条路径会分叉：

* 走 `create_all` 的环境（容器/开发机）有表，功能看起来正常；
* 走 `alembic upgrade head` 的环境（按版本部署、或 CI 起干净库跑迁移）
  **没有表**，第一次读记忆就 500 —— 而应用启动日志一切正常，极难定位；
* `alembic downgrade` 也无法回收它（版本号里根本没有这一步）。

所以这里补上迁移，并镜像 ORM 的真实形状（不额外加 `server_default`，
避免 `create_all` 与迁移产出两份不完全相同的 DDL）。

表语义（见 `app/db/agent_memory_models.py`）：
  - 隔离键是 `(tenant_id, user_id)` 两列**同时**等值匹配，没有管理员分支；
  - 撤销是**置位**不是删除：`revoked_at` 非空后不再参与任何读取，
    行本身保留用于审计追溯。

Revision ID: s5m6n7o8p9q0
Revises: r3l4m5n6o7p8
Create Date: 2026-10-28 13:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 's5m6n7o8p9q0'
down_revision: Union[str, None] = 'r3l4m5n6o7p8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # IF NOT EXISTS：与 create_all 共存的环境（容器里表已由启动钩子建好）
    # 再跑一次迁移不该报错，否则"补迁移"本身会把部署卡死。
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_memories (
            id UUID NOT NULL,
            tenant_id VARCHAR(64) NOT NULL DEFAULT 'default',
            user_id UUID,
            kind VARCHAR(32) NOT NULL,
            content TEXT NOT NULL,
            source VARCHAR(32) NOT NULL,
            source_conversation_id UUID,
            version INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            revoked_at TIMESTAMPTZ,
            revoked_by VARCHAR(128),
            revoke_reason TEXT,
            meta JSONB NOT NULL DEFAULT '{}'::jsonb,
            PRIMARY KEY (id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_memories_tenant_id "
        "ON agent_memories (tenant_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_memories_user_id "
        "ON agent_memories (user_id)"
    )
    # 读取路径永远是 `tenant + user + revoked_at IS NULL ORDER BY updated_at DESC`，
    # 复合索引让"取最新 N 条"不必回表排序。
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_memories_scope "
        "ON agent_memories (tenant_id, user_id, revoked_at)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_memories")
