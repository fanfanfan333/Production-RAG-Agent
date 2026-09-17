"""add image object metadata (image_count)

部分1+部分2：内嵌图片不再是纯文本 —— 每张图片被抽成独立对象并落盘，
文档元数据里记录识别到的图片数量，便于上传回执与前端展示。

Revision ID: f1a2b3c4d5e6
Revises: e5f6a7b8c9d0
Create Date: 2026-09-12 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('image_count', sa.Integer(), server_default='0', nullable=False),
    )
    op.add_column(
        'documents',
        sa.Column('image_object_count', sa.Integer(), server_default='0', nullable=False),
    )


def downgrade() -> None:
    op.drop_column('documents', 'image_object_count')
    op.drop_column('documents', 'image_count')
