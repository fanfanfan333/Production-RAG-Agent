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
    # ── 三层标注的展示名（T03）───────────────────────────────────────────────
    # 后端只给**原始字段**，拼接（"公司 · 部门" / 仅"个人"）放前端 —— 但后端保证：
    # 能取到就是干净字符串，取不到就是 None，**绝不产出** "None"/"undefined" 之类。
    #   tenant_name     公司展示名，来自 companies 注册表 display_name；
    #                   未注册（如 default 历史租户）为 None。
    #   department_name 该公司内该 department_id 对应的部门名，
    #                   取自 users.department_name 的成员归属聚合（不依赖 owner）。
    tenant_name: str | None = None
    department_name: str | None = None

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
    # 申请能力**按目标层级**下发：能直接发的那层不给申请入口，不能直接发的层给。
    # 两者对不同层级可同时为真 —— 部门负责人就是这种形态（发部门 / 申请公司）。
    can_request_department: bool = False
    can_request_company: bool = False
    needs_share_request: bool = False
    # 公司级管理者（企业管理员 / 知识库管理员 / 平台管理员）可把已共享的文档
    # **改归到指定部门** —— 界面据此渲染「转为部门文档」入口。
    # 与 can_publish_department 是两件事：后者发的是"我自己的部门"，无选择。
    can_transfer_department: bool = False
    transfer_denied_reason: str = ""
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
