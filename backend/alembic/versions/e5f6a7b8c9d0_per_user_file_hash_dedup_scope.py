"""per-user file_hash dedup scope

Revision ID: e5f6a7b8c9d0
Revises: c8d41f2a7b30
Create Date: 2026-09-08 12:00:00.000000

修复"第一次上传却提示已被索引过"的 BUG：
  - file_hash 的全局唯一索引 → 普通索引
  - 新增复合唯一约束 (owner_id, file_hash)：判重按用户范围，
    不同用户上传同一文件各自独立索引；owner_id 为 NULL 的旧数据
    互不冲突（PostgreSQL 中 NULL != NULL）。

幂等写法，与 create_all 兼容。
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, None] = 'c8d41f2a7b30'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) 拆掉全局唯一索引（SQLAlchemy 对 unique=True, index=True 的默认命名）
    op.execute("DROP INDEX IF EXISTS ix_documents_file_hash")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_documents_file_hash ON documents (file_hash)"
    )
    # 2) 建立按用户的复合唯一约束
    op.execute(
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS uq_documents_owner_hash"
    )
    op.execute(
        "ALTER TABLE documents "
        "ADD CONSTRAINT uq_documents_owner_hash UNIQUE (owner_id, file_hash)"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE documents DROP CONSTRAINT IF EXISTS uq_documents_owner_hash"
    )
    op.execute("DROP INDEX IF EXISTS ix_documents_file_hash")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_file_hash "
        "ON documents (file_hash)"
    )
