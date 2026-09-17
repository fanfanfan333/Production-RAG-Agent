"""add messages.meta (回答依据快照)

messages 表此前只存 ``content`` —— 一条回答"依据了什么"（引用来源、
引用校验结论、证据门控、输出合规、生成的文档、路由意图）只随 SSE 事件
发到浏览器内存里。

后果：切页 / 切窗口 / 重开标签页之后，历史会话只剩正文，用户看到的是
"上一次提问的数据来源不见了"。加一列 JSONB 把这份快照随回答一起落库，
重新打开历史即可完整还原。

存储约定：
  - 只有 assistant 消息写入；user 消息与存量数据为 NULL。
  - 前端遇到 NULL 时按"老数据没有依据快照"降级（只显示正文），
    因此不需要回填，升级过程对存量会话无影响。

Revision ID: i4d5e6f7g8h9
Revises: h3c4d5e6f7g8
Create Date: 2026-09-13 17:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'i4d5e6f7g8h9'
down_revision: Union[str, None] = 'h3c4d5e6f7g8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'messages',
        sa.Column('meta', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('messages', 'meta')
