"""Add graph provenance and training review records; preserve existing data."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "r3l4m5n6o7p8"
down_revision = "q2k3l4m5n6o7"
branch_labels = depends_on = None


def upgrade():
    bind = op.get_bind()
    if not sa.inspect(bind).has_table("graph_indexes"):
        op.create_table("graph_indexes",
            sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("revision", sa.String(64), nullable=False), sa.Column("status", sa.String(16), nullable=False),
            sa.Column("cursor", sa.Integer(), nullable=False), sa.Column("total", sa.Integer(), nullable=False),
            sa.Column("error", sa.String(256)), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()))
    if not sa.inspect(bind).has_table("graph_entities"):
        op.create_table("graph_entities",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
            sa.Column("revision", sa.String(64), nullable=False), sa.Column("kind", sa.String(32), nullable=False),
            sa.Column("name", sa.String(128), nullable=False),
            sa.UniqueConstraint("document_id", "revision", "kind", "name", name="uq_graph_entity"))
        op.create_index("ix_graph_entities_document_id", "graph_entities", ["document_id"])
        op.create_index("ix_graph_entities_name", "graph_entities", ["name"])
    if not sa.inspect(bind).has_table("graph_relations"):
        op.create_table("graph_relations",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
            sa.Column("revision", sa.String(64), nullable=False),
            sa.Column("subject_id", UUID(as_uuid=True), sa.ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False),
            sa.Column("object_id", UUID(as_uuid=True), sa.ForeignKey("graph_entities.id", ondelete="CASCADE"), nullable=False),
            sa.Column("predicate", sa.String(64), nullable=False), sa.Column("chunk_index", sa.Integer(), nullable=False),
            sa.Column("point_id", sa.String(64), nullable=False), sa.Column("evidence", sa.Text(), nullable=False),
            sa.Column("source_hash", sa.String(64), nullable=False),
            sa.UniqueConstraint("document_id", "revision", "chunk_index", "subject_id", "object_id", "predicate", name="uq_graph_relation"))
        op.create_index("ix_graph_relations_document_id", "graph_relations", ["document_id"])
    if not sa.inspect(bind).has_table("training_examples"):
        from sqlalchemy.dialects.postgresql import JSONB
        op.create_table("training_examples",
            sa.Column("id", UUID(as_uuid=True), primary_key=True), sa.Column("prompt", sa.Text(), nullable=False),
            sa.Column("evidence", JSONB(), nullable=False), sa.Column("chosen", sa.Text(), nullable=False), sa.Column("rejected", sa.Text()),
            sa.Column("source_group", sa.String(256), nullable=False), sa.Column("source_kind", sa.String(16), nullable=False),
            sa.Column("license", sa.String(128), nullable=False), sa.Column("status", sa.String(16), nullable=False),
            sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("reviewed_by", UUID(as_uuid=True), sa.ForeignKey("users.id")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()))
        op.create_index("ix_training_examples_source_group", "training_examples", ["source_group"])
    if "message_id" not in {column["name"] for column in sa.inspect(bind).get_columns("answer_feedback")}:
        op.add_column("answer_feedback", sa.Column("message_id", UUID(as_uuid=True), nullable=True))
        op.create_foreign_key("fk_feedback_message", "answer_feedback", "messages", ["message_id"], ["id"], ondelete="SET NULL")


def downgrade():
    bind = op.get_bind()
    if "message_id" in {column["name"] for column in sa.inspect(bind).get_columns("answer_feedback")}:
        for constraint in sa.inspect(bind).get_foreign_keys("answer_feedback"):
            if constraint["constrained_columns"] == ["message_id"]:
                op.drop_constraint(constraint["name"], "answer_feedback", type_="foreignkey")
        op.drop_column("answer_feedback", "message_id")
    for table in ("training_examples", "graph_relations", "graph_entities", "graph_indexes"):
        op.drop_table(table)
