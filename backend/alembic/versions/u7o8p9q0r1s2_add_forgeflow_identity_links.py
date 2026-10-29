"""Add revocable, credential-free ForgeFlow account links."""
from alembic import op

revision = "u7o8p9q0r1s2"
down_revision = "t6n7o8p9q0r1"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE IF NOT EXISTS forgeflow_identity_links (
        tenant_id VARCHAR(64) NOT NULL,
        user_id UUID NOT NULL,
        forgeflow_tenant_id VARCHAR(128) NOT NULL,
        forgeflow_user_id VARCHAR(128) NOT NULL,
        PRIMARY KEY (tenant_id, user_id),
        CONSTRAINT uq_forgeflow_identity_link UNIQUE (forgeflow_tenant_id, forgeflow_user_id)
    )""")


def downgrade():
    op.execute("DROP TABLE IF EXISTS forgeflow_identity_links")
