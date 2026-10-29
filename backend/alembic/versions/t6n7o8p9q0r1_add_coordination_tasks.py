"""add coordination tasks and durable event log

Revision ID: t6n7o8p9q0r1
Revises: s5m6n7o8p9q0
Create Date: 2026-10-28 19:50:00.000000

The table is also registered with ``Base.metadata`` so ``create_all`` and a
versioned deployment converge. ``IF NOT EXISTS`` keeps the migration safe for a
database where the table was already created by the startup hook.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "t6n7o8p9q0r1"
down_revision: Union[str, None] = "s5m6n7o8p9q0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS coordination_tasks (
            id UUID NOT NULL,
            tenant_id VARCHAR(64) NOT NULL,
            user_id UUID NOT NULL,
            idempotency_key VARCHAR(128) NOT NULL,
            request_text TEXT NOT NULL,
            requested_kind VARCHAR(24) NOT NULL DEFAULT 'auto',
            status VARCHAR(32) NOT NULL DEFAULT 'queued',
            plan JSONB NOT NULL DEFAULT '{}'::jsonb,
            result JSONB NOT NULL DEFAULT '{}'::jsonb,
            error TEXT,
            execution_id VARCHAR(128),
            conversation_id UUID,
            stop_requested BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ,
            PRIMARY KEY (id),
            CONSTRAINT uq_coordination_tasks_owner_idempotency
                UNIQUE (tenant_id, user_id, idempotency_key)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_coordination_tasks_tenant_id "
        "ON coordination_tasks (tenant_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_coordination_tasks_user_id "
        "ON coordination_tasks (user_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_coordination_tasks_owner_created "
        "ON coordination_tasks (tenant_id, user_id, created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_coordination_tasks_owner_status "
        "ON coordination_tasks (tenant_id, user_id, status)"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS coordination_task_events (
            id BIGSERIAL NOT NULL,
            task_id UUID NOT NULL REFERENCES coordination_tasks(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL,
            event_type VARCHAR(48) NOT NULL,
            data JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (id),
            CONSTRAINT uq_coordination_task_events_seq UNIQUE (task_id, seq)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_coordination_task_events_task_id "
        "ON coordination_task_events (task_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS coordination_task_events")
    op.execute("DROP TABLE IF EXISTS coordination_tasks")
