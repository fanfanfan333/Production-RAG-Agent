"""multi-user auth, ownership, kb collections, audit logs

Revision ID: c8d41f2a7b30
Revises: b35f99144db8
Create Date: 2026-09-06 12:00:00.000000

企业落地第一阶段：
  - users / collections / audit_logs 表
  - documents.owner_id / documents.collection_id
  - conversations.owner_id

All DDL is idempotent (IF NOT EXISTS) because the app also runs
Base.metadata.create_all on startup — whichever runs first wins and the
other is a no-op.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = 'c8d41f2a7b30'
down_revision: Union[str, None] = 'b35f99144db8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY,
            username VARCHAR(64) NOT NULL,
            password_hash VARCHAR(128) NOT NULL,
            role VARCHAR(20) NOT NULL DEFAULT 'user',
            is_active BOOLEAN NOT NULL DEFAULT true,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON users (username)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            id UUID PRIMARY KEY,
            name VARCHAR(128) NOT NULL,
            description VARCHAR(512),
            owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_collections_owner_name UNIQUE (owner_id, name)
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_collections_name ON collections (name)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_collections_owner_id ON collections (owner_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id UUID PRIMARY KEY,
            user_id UUID,
            username VARCHAR(64),
            action VARCHAR(64) NOT NULL,
            resource_type VARCHAR(64),
            resource_id VARCHAR(64),
            detail TEXT,
            ip VARCHAR(64),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_audit_logs_user_id ON audit_logs (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_audit_logs_action ON audit_logs (action)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_audit_logs_created_at ON audit_logs (created_at)")

    op.execute("""
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES users(id) ON DELETE CASCADE
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_documents_owner_id ON documents (owner_id)")

    op.execute("""
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS collection_id UUID REFERENCES collections(id) ON DELETE SET NULL
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_documents_collection_id ON documents (collection_id)")

    op.execute("""
        ALTER TABLE conversations
        ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES users(id) ON DELETE CASCADE
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_conversations_owner_id ON conversations (owner_id)")


def downgrade() -> None:
    op.execute("ALTER TABLE conversations DROP COLUMN IF EXISTS owner_id")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS collection_id")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS owner_id")
    op.execute("DROP TABLE IF EXISTS audit_logs")
    op.execute("DROP TABLE IF EXISTS collections")
    op.execute("DROP TABLE IF EXISTS users")
