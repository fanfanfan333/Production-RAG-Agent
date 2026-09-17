"""three-tier knowledge base + share requests（三层知识库 + 申请共享 + Keycloak 联邦身份）

本次升级的三件事：

1. **三层知识库**（个人 / 部门 / 公司）
   ``documents.access_level`` 与 ``department_id`` 在 h3c4d5e6f7g8 已经存在，
   本迁移不需要改表结构；变化在语义层：新上传默认落在 **个人库**（private），
   共享必须显式发布或走「申请共享」。存量数据不动 —— 已经发布到部门/公司的
   文档保持原样，避免升级瞬间把别人的可见范围改掉。

2. **共享申请**：新建 ``share_requests`` 表（申请人、目标层级、审核人、
   审核意见、状态机、申请人已读标记）。

3. **Keycloak 联邦身份**：``users`` 增加
   - ``keycloak_sub``   —— realm 内唯一且稳定的 sub，用于绑定远端身份
   - ``auth_source``    —— local | keycloak
   - ``display_name``   —— 展示名（中文名）
   并把 ``password_hash`` 改为可空：联邦账号没有本地密码。

幂等性说明：容器启动时 ``Base.metadata.create_all`` 可能抢先建出 share_requests /
新列（create_all 不会给**已存在**的表补列，但会建缺失的表）。因此本迁移在
升级前先用 inspector 探测，已存在则跳过，避免出现 "DuplicateTable" 导致
alembic 卡住（历史踩过的坑）。

Revision ID: j5e6f7g8h9i0
Revises: i4d5e6f7g8h9
Create Date: 2026-09-13 19:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'j5e6f7g8h9i0'
down_revision: Union[str, None] = 'i4d5e6f7g8h9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(bind, table: str) -> set[str]:
    inspector = sa.inspect(bind)
    if table not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table)}


def _tables(bind) -> set[str]:
    return set(sa.inspect(bind).get_table_names())


def upgrade() -> None:
    bind = op.get_bind()

    # ── 1. users：Keycloak 联邦身份 ─────────────────────────────────────────
    existing = _columns(bind, "users")
    if "display_name" not in existing:
        op.add_column("users", sa.Column("display_name", sa.String(128), nullable=True))
    if "keycloak_sub" not in existing:
        op.add_column("users", sa.Column("keycloak_sub", sa.String(128), nullable=True))
        op.create_index("ix_users_keycloak_sub", "users", ["keycloak_sub"], unique=True)
    if "auth_source" not in existing:
        op.add_column(
            "users",
            sa.Column(
                "auth_source", sa.String(16),
                server_default="local", nullable=False,
            ),
        )
    # 联邦账号没有本地密码 → 该列必须可空
    op.alter_column("users", "password_hash", existing_type=sa.String(128), nullable=True)

    # ── 2. share_requests：申请共享 ────────────────────────────────────────
    if "share_requests" not in _tables(bind):
        op.create_table(
            "share_requests",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", sa.String(64), nullable=False, server_default="default"),
            sa.Column(
                "document_id", postgresql.UUID(as_uuid=True),
                sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False,
            ),
            sa.Column("document_name", sa.String(512), nullable=False),
            sa.Column(
                "requester_id", postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
            ),
            sa.Column("requester_username", sa.String(64), nullable=False),
            sa.Column("requester_department_id", sa.String(64), nullable=True),
            sa.Column("target_level", sa.String(20), nullable=False),
            sa.Column("target_department_id", sa.String(64), nullable=True),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
            sa.Column(
                "reviewer_id", postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
            ),
            sa.Column("reviewer_username", sa.String(64), nullable=True),
            sa.Column("review_comment", sa.Text(), nullable=True),
            sa.Column(
                "requester_seen", sa.Boolean(),
                nullable=False, server_default=sa.text("false"),
            ),
            sa.Column(
                "created_at", sa.DateTime(timezone=True),
                nullable=False, server_default=sa.func.now(),
            ),
            sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_share_requests_tenant_id", "share_requests", ["tenant_id"])
        op.create_index("ix_share_requests_document_id", "share_requests", ["document_id"])
        op.create_index("ix_share_requests_requester_id", "share_requests", ["requester_id"])
        op.create_index("ix_share_requests_status", "share_requests", ["status"])
        op.create_index("ix_share_requests_created_at", "share_requests", ["created_at"])
        op.create_index(
            "ix_share_requests_doc_status", "share_requests", ["document_id", "status"]
        )
        op.create_index(
            "ix_share_requests_tenant_status", "share_requests", ["tenant_id", "status"]
        )

    # ── 3. 存量文档的层级可视化：把 NULL 归一成 private ─────────────────────
    # 老数据的 access_level 可能是 NULL（按 private 处理）。显式落成 'private'
    # 让"个人/部门/公司"三态在数据库里就有确定值，前端与审计日志不再需要
    # 到处判空。
    op.execute(
        "UPDATE documents SET access_level = 'private' WHERE access_level IS NULL"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if "share_requests" in _tables(bind):
        op.drop_table("share_requests")
    existing = _columns(bind, "users")
    if "auth_source" in existing:
        op.drop_column("users", "auth_source")
    if "keycloak_sub" in existing:
        op.drop_index("ix_users_keycloak_sub", table_name="users")
        op.drop_column("users", "keycloak_sub")
    if "display_name" in existing:
        op.drop_column("users", "display_name")
    op.alter_column("users", "password_hash", existing_type=sa.String(128), nullable=False)
