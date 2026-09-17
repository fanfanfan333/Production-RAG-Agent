"""share_requests.intent —— 申请删除（与申请共享对称的另一条申请链路）

产品要求：

    删除文档功能，个人只能删除个人的文档；部门 RAG 文档由部门负责人及以上
    权限删除；公司 RAG 文档由知识库管理员或企业管理员删除；
    **不包含在内的需要提交申请，由上层进行同意或拒绝**。

原实现里 ``delete_permission_for`` 是"归属人永远能删"，于是文档一旦发布到
部门库/公司库，作者仍可一键删除 —— 而同事的问答与报告正在引用它。本次把
删除权改为跟着**文档所在层级**走（见 tenancy.delete_permission_for），并补上
「申请删除」这条升级路径。

实现方式是在既有的 ``share_requests`` 表上加一列 ``intent``：

    intent = publish   —— 原语义：申请把文档发布到更高层级
    intent = delete    —— 新语义：申请删除一份自己没有删除权的文档

复用同一张表而不是新开一张，是因为"审核范围"的判定完全一致
（部门级申请 → 本部门负责人；公司级申请 → 知识库管理员 / 企业管理员），
审核队列、角标、撤回、已读回执这些周边逻辑因此零改动。

存量行一律按 ``publish`` 回填，语义不变。

Revision ID: l7g8h9i0j1k2
Revises: k6f7g8h9i0j1
Create Date: 2026-09-14 20:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'l7g8h9i0j1k2'
down_revision: Union[str, None] = 'k6f7g8h9i0j1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(bind, table: str) -> set[str]:
    inspector = sa.inspect(bind)
    if table not in inspector.get_table_names():
        return set()
    return {col["name"] for col in inspector.get_columns(table)}


def upgrade() -> None:
    bind = op.get_bind()
    if "share_requests" not in sa.inspect(bind).get_table_names():
        # 表还没建（全新库）——create_all 会按模型定义直接带上 intent 列
        return

    if "intent" not in _columns(bind, "share_requests"):
        op.add_column(
            "share_requests",
            sa.Column(
                "intent", sa.String(16),
                nullable=False, server_default="publish",
            ),
        )
    # 存量行显式回填，避免 server_default 在不同 PG 版本上的行为差异
    op.execute("UPDATE share_requests SET intent = 'publish' WHERE intent IS NULL")

    # document_id 由 CASCADE 改为 SET NULL 且可空：删除申请批准后文档消失，
    # 但这条申请记录必须留下（审计凭证 + 申请人要看到"已通过"）。
    op.execute(
        "ALTER TABLE share_requests DROP CONSTRAINT IF EXISTS "
        "share_requests_document_id_fkey"
    )
    op.execute(
        "ALTER TABLE share_requests ALTER COLUMN document_id DROP NOT NULL"
    )
    op.execute(
        "ALTER TABLE share_requests ADD CONSTRAINT share_requests_document_id_fkey "
        "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if "intent" in _columns(bind, "share_requests"):
        op.drop_column("share_requests", "intent")
