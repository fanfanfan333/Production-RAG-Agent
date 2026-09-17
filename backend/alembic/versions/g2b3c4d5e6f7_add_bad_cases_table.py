"""add bad_cases table (Bad Case 回流队列)

持续监控闭环的落地表：自动回流（引用不被支持 / 证据门控拒答 / 输出净化命中）
与人工反馈（用户 👎）共用一张表，reason 字段区分来源。

字段设计围绕**可复现**：question + answer + sources_snapshot + detail 四者齐全时，
审阅者不需要翻对话历史就能复现这次问答的完整上下文。

注：应用启动时 `Base.metadata.create_all` 也会建这张表；本迁移用于**存量库**
（已经跑过 alembic upgrade 的库）补齐，两者殊途同归。

Revision ID: g2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-09-13 09:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'g2b3c4d5e6f7'
down_revision: Union[str, None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'bad_cases',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('username', sa.String(length=64), nullable=True),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('reason', sa.String(length=32), nullable=False),
        sa.Column('severity', sa.String(length=16), nullable=False),
        sa.Column('intent', sa.String(length=64), nullable=True),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('detail', sa.Text(), nullable=True),
        sa.Column('sources_snapshot', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('resolution', sa.Text(), nullable=True),
        sa.Column('tags', sa.String(length=256), nullable=True),
        sa.Column(
            'created_at', sa.DateTime(timezone=True),
            server_default=sa.text('now()'), nullable=False,
        ),
        sa.Column(
            'updated_at', sa.DateTime(timezone=True),
            server_default=sa.text('now()'), nullable=False,
        ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(
            ['conversation_id'], ['conversations.id'], ondelete='SET NULL'
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_bad_cases_reason', 'bad_cases', ['reason'])
    op.create_index('ix_bad_cases_severity', 'bad_cases', ['severity'])
    op.create_index('ix_bad_cases_intent', 'bad_cases', ['intent'])
    op.create_index('ix_bad_cases_status', 'bad_cases', ['status'])
    op.create_index('ix_bad_cases_created_at', 'bad_cases', ['created_at'])
    op.create_index('ix_bad_cases_user_id', 'bad_cases', ['user_id'])
    op.create_index('ix_bad_cases_conversation_id', 'bad_cases', ['conversation_id'])


def downgrade() -> None:
    op.drop_index('ix_bad_cases_conversation_id', table_name='bad_cases')
    op.drop_index('ix_bad_cases_user_id', table_name='bad_cases')
    op.drop_index('ix_bad_cases_created_at', table_name='bad_cases')
    op.drop_index('ix_bad_cases_status', table_name='bad_cases')
    op.drop_index('ix_bad_cases_intent', table_name='bad_cases')
    op.drop_index('ix_bad_cases_severity', table_name='bad_cases')
    op.drop_index('ix_bad_cases_reason', table_name='bad_cases')
    op.drop_table('bad_cases')
