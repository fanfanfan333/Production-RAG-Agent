"""
Schemas for document listing and management endpoints (Phase 3).

Kept separate from app.schemas.document (Phase 2) to honour the
'do not modify previous code' constraint.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.db.models import DocumentStatus


class DocumentSummary(BaseModel):
    """Lightweight document row — used in paginated list responses."""

    document_id: uuid.UUID
    filename: str
    status: DocumentStatus
    page_count: int
    chunk_count: int
    file_size_bytes: int
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    collection_id: uuid.UUID | None = None

    # ── 入库进度 ──────────────────────────────────────────────────────────────
    # POST /upload 现在只做"受理"就返回 202，真正的解析/向量化在后台跑；前端
    # 靠这三个字段把"处理中"讲清楚（在解析、还是在向量化第 3/12 批），而不是
    # 干瘪地转一个圈。
    current_stage: str | None = None
    total_chunks: int | None = None
    embedded_chunks: int | None = None

    # ── 图片（三层图片处理是核心卖点，列表里必须看得见）──────────────────────
    # image_count       = 解析出的内嵌图片总数
    # image_object_count = 其中真正成为**独立检索对象**的图片块数
    # 之前列表接口不返回这两个字段，前端只能显示"0"，验收脚本也因此打印假的 0
    # —— 明明库里有 20 张图，报表上却是 0，等于把功能藏起来了。
    image_count: int = 0
    image_object_count: int = 0

    # ── 三层知识库（个人 / 部门 / 公司）───────────────────────────────────────
    # access_label 直接给前端渲染中文标注，避免前端再维护一份 level→中文 的映射。
    access_level: str = "private"
    access_label: str = "个人"
    tenant_id: str = "default"
    department_id: str | None = None

    # ── 权限能力（前端按钮显隐与后端校验同源）─────────────────────────────────
    owner_id: uuid.UUID | None = None
    owner_username: str | None = None
    is_owner: bool = False
    can_delete: bool = False
    delete_denied_reason: str = ""
    # 自己没有删除权、但看得见该文档 → 可提交「申请删除」由上级审核
    can_request_delete: bool = False
    can_publish_department: bool = False
    can_publish_company: bool = False
    needs_share_request: bool = False
    publish_denied_reason: str = ""
    pending_share_request: bool = False

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    total: int = Field(..., description="Total documents matching the filter")
    page: int = Field(..., description="Current page (1-indexed)")
    limit: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")
    documents: list[DocumentSummary]

    model_config = {
        "json_schema_extra": {
            "example": {
                "total": 42,
                "page": 1,
                "limit": 20,
                "pages": 3,
                "documents": [],
            }
        }
    }


class DocumentDeleteResponse(BaseModel):
    """Confirmation returned after a successful document deletion."""

    document_id: uuid.UUID
    filename: str
    message: str = "Document and associated vectors deleted successfully."

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "filename": "annual_report.pdf",
                "message": "Document and associated vectors deleted successfully.",
            }
        }
    }
