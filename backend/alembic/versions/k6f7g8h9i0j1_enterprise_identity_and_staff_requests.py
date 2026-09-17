"""enterprise identity verification + staff approval（企业身份验证与层级审核）

本次升级的三件事（对应产品需求的功能 1/3/4）：

1. **staff_requests**：新建「企业身份验证申请」表。
   用户注册后提交（公司名称 / 公司部门 / 部门职责）→ 进入待审核 → 上级
   同意或拒绝 → 批准时写入公司、部门、职责并授予权限角色（默认 employee）。
   审核落库同时记录 ``reviewer_title`` + ``reviewer_name``：产品要求申请
   记录后方显示「负责人 = 职务 + 名称」。

2. **users 增加身份展示字段**（不影响任何权限判定，纯展示与检索用）：
   - ``company_name``    —— 公司名称原文（用户填的中文名，tenant_id 不能存中文）
   - ``department_name`` —— 部门名称原文
   - ``job_title``       —— 部门职责 / 职务（区别于权限角色 role）

3. **存量账号回填**：已有 ``department_id`` / ``tenant_id`` 的账号把名称字段
   回填成占位值，保证个人主页「公司 / 部门」两行永远有内容可显示（否则会
   出现空白行）。真正的名称会在下次身份验证通过时被覆盖。

幂等性说明：容器启动时 ``Base.metadata.create_all`` 会抢先建出 staff_requests
与缺失的表（但**不会**给已存在的 users 表补列）。因此本迁移先用 inspector
探测再执行，重复运行不会抛 DuplicateTable 卡住 alembic（历史踩过的坑）。

Revision ID: k6f7g8h9i0j1
Revises: j5e6f7g8h9i0
Create Date: 2026-09-14 19:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'k6f7g8h9i0j1'
down_revision: Union[str, None] = 'j5e6f7g8h9i0'
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

    # ── 1. users：身份展示字段 ──────────────────────────────────────────────
    existing = _columns(bind, "users")
    if "company_name" not in existing:
        op.add_column("users", sa.Column("company_name", sa.String(128), nullable=True))
    if "department_name" not in existing:
        op.add_column("users", sa.Column("department_name", sa.String(128), nullable=True))
    if "job_title" not in existing:
        op.add_column("users", sa.Column("job_title", sa.String(128), nullable=True))

    # 存量回填：让「公司 / 部门」两行不至于显示空白。
    # 只填还没值的行，不覆盖用户已经填过的真实名称。
    op.execute(
        "UPDATE users SET company_name = tenant_id "
        "WHERE company_name IS NULL AND tenant_id IS NOT NULL"
    )
    op.execute(
        "UPDATE users SET department_name = department_id "
        "WHERE department_name IS NULL AND department_id IS NOT NULL"
    )

    # ── 2. staff_requests：企业身份验证申请 ─────────────────────────────────
    if "staff_requests" not in _tables(bind):
        op.create_table(
            "staff_requests",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "applicant_id", postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
            ),
            sa.Column("applicant_username", sa.String(64), nullable=False),
            sa.Column("applicant_display_name", sa.String(128), nullable=True),
            sa.Column(
                "applicant_company_id", sa.String(64),
                nullable=False, server_default="default",
            ),
            sa.Column("company_name", sa.String(128), nullable=False),
            sa.Column("company_id", sa.String(64), nullable=False),
            sa.Column("department_name", sa.String(128), nullable=False),
            sa.Column("department_id", sa.String(64), nullable=False),
            sa.Column("duty", sa.String(128), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
            sa.Column(
                "reviewer_id", postgresql.UUID(as_uuid=True),
                sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
            ),
            sa.Column("reviewer_username", sa.String(64), nullable=True),
            sa.Column("reviewer_title", sa.String(128), nullable=True),
            sa.Column("reviewer_name", sa.String(128), nullable=True),
            sa.Column("review_comment", sa.Text(), nullable=True),
            sa.Column("granted_role", sa.String(20), nullable=True),
            sa.Column(
                "applicant_seen", sa.Boolean(),
                nullable=False, server_default=sa.text("false"),
            ),
            sa.Column(
                "created_at", sa.DateTime(timezone=True),
                nullable=False, server_default=sa.func.now(),
            ),
            sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_staff_requests_company_id", "staff_requests", ["company_id"])
        op.create_index("ix_staff_requests_applicant_id", "staff_requests", ["applicant_id"])
        op.create_index("ix_staff_requests_status", "staff_requests", ["status"])
        op.create_index("ix_staff_requests_created_at", "staff_requests", ["created_at"])
        op.create_index(
            "ix_staff_requests_company_status", "staff_requests", ["company_id", "status"]
        )
        op.create_index(
            "ix_staff_requests_applicant_status",
            "staff_requests",
            ["applicant_id", "status"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    if "staff_requests" in _tables(bind):
        op.drop_table("staff_requests")
    existing = _columns(bind, "users")
    for column in ("job_title", "department_name", "company_name"):
        if column in existing:
            op.drop_column("users", column)
