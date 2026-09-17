"""add multi-tenant isolation columns (三层隔离)

三层隔离的库存储：

    第一层 Tenant Isolation
        users.tenant_id / documents.tenant_id / conversations.tenant_id /
        messages.tenant_id —— 存量数据统一回填 'default'，单机部署行为不变。

    第二层 Document ACL
        documents.access_level ('private'|'department'|'tenant') +
        documents.department_id + users.department_id。
        存量文档回填 'private'（保持"仅本人可见"的旧语义，升级不意外共享）。

    第三层 User/Conversation Isolation
        messages.user_id —— 消息写入时的真实用户；存量为 NULL，
        读取侧按"继承所属 Conversation"处理。

注意：Qdrant 里的存量向量 payload 没有 tenant_id，检索前置过滤会查不到
它们 —— 升级后需执行 scripts/migrate_tenant_payload.py 回填 payload。

Revision ID: h3c4d5e6f7g8
Revises: g2b3c4d5e6f7
Create Date: 2026-09-13 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'h3c4d5e6f7g8'
down_revision: Union[str, None] = 'g2b3c4d5e6f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── users：租户 + 部门 ──────────────────────────────────────────────────
    op.add_column(
        'users',
        sa.Column('tenant_id', sa.String(length=64), nullable=False,
                  server_default='default'),
    )
    op.add_column(
        'users',
        sa.Column('department_id', sa.String(length=64), nullable=True),
    )
    op.create_index('ix_users_tenant_id', 'users', ['tenant_id'])
    op.create_index('ix_users_department_id', 'users', ['department_id'])

    # ── documents：租户 + ACL ───────────────────────────────────────────────
    op.add_column(
        'documents',
        sa.Column('tenant_id', sa.String(length=64), nullable=False,
                  server_default='default'),
    )
    # 存量文档按 private 回填：升级前的语义就是"仅本人可见"，不能默认放开。
    op.add_column(
        'documents',
        sa.Column('access_level', sa.String(length=20), nullable=False,
                  server_default='private'),
    )
    op.add_column(
        'documents',
        sa.Column('department_id', sa.String(length=64), nullable=True),
    )
    op.create_index('ix_documents_tenant_id', 'documents', ['tenant_id'])
    op.create_index('ix_documents_access_level', 'documents', ['access_level'])
    op.create_index('ix_documents_department_id', 'documents', ['department_id'])

    # ── conversations：租户 ─────────────────────────────────────────────────
    op.add_column(
        'conversations',
        sa.Column('tenant_id', sa.String(length=64), nullable=False,
                  server_default='default'),
    )
    op.create_index('ix_conversations_tenant_id', 'conversations', ['tenant_id'])

    # ── messages：真实用户 + 租户（存量 NULL = 继承会话）─────────────────────
    op.add_column(
        'messages',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        'messages',
        sa.Column('tenant_id', sa.String(length=64), nullable=True),
    )
    op.create_index('ix_messages_user_id', 'messages', ['user_id'])
    op.create_index('ix_messages_tenant_id', 'messages', ['tenant_id'])


def downgrade() -> None:
    op.drop_index('ix_messages_tenant_id', table_name='messages')
    op.drop_index('ix_messages_user_id', table_name='messages')
    op.drop_column('messages', 'tenant_id')
    op.drop_column('messages', 'user_id')

    op.drop_index('ix_conversations_tenant_id', table_name='conversations')
    op.drop_column('conversations', 'tenant_id')

    op.drop_index('ix_documents_department_id', table_name='documents')
    op.drop_index('ix_documents_access_level', table_name='documents')
    op.drop_index('ix_documents_tenant_id', table_name='documents')
    op.drop_column('documents', 'department_id')
    op.drop_column('documents', 'access_level')
    op.drop_column('documents', 'tenant_id')

    op.drop_index('ix_users_department_id', table_name='users')
    op.drop_index('ix_users_tenant_id', table_name='users')
    op.drop_column('users', 'department_id')
    op.drop_column('users', 'tenant_id')
