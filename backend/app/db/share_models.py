"""
共享申请（三级知识库的"申请共享"闭环）.

业务故事：普通员工把自己的文档放在**个人知识库**里（默认不可见），想让它
被同事看到时，在文档上点「申请共享」→ 选择目标层级（部门库 / 公司库）→
填理由 → 提交。

    目标层级        审核人（同意后自动发布）
    部门库          本部门负责人（dept_manager / manager）
    公司库          知识库管理员 / 企业管理员（kb_admin / company_admin）

审核人批准后，文档的 ``access_level`` 就地升级（部门库同时写入 department_id），
向量库 payload 同步更新，申请人可在「查看申请」看到"已通过"。

状态机：

    pending ──approve──▶ approved
            └─reject───▶ rejected
            └─cancel───▶ cancelled      （申请人主动撤回）

被拒绝/已通过的申请保留记录（审计与复查），审批动作写 audit_logs。
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.postgres import Base


class ShareRequest(Base):
    __tablename__ = "share_requests"
    __table_args__ = (
        # "某文档是否已有待审申请" 是最热的查询（列表页要据此决定按钮状态）
        Index(
            "ix_share_requests_doc_status",
            "document_id",
            "status",
        ),
        Index(
            "ix_share_requests_tenant_status",
            "tenant_id",
            "status",
        ),
    )

    STATUS_PENDING = "pending"
    STATUS_APPROVED = "approved"
    STATUS_REJECTED = "rejected"
    STATUS_CANCELLED = "cancelled"

    # 目标层级只允许这两种（个人库无需申请）——取自 tenancy 的常量语义
    TARGET_DEPARTMENT = "department"
    TARGET_COMPANY = "tenant"
    VALID_TARGETS = {TARGET_DEPARTMENT, TARGET_COMPANY}

    # 申请意图：把文档**发布**到更高层级，或**删除**一份自己没有删除权的文档。
    # 两者共用一张表与同一套"审核范围"判定（部门级申请 → 部门负责人；
    # 公司级申请 → 知识库管理员/企业管理员），因此审核队列、角标、
    # 撤回、已读回执这些周边逻辑不用各写一遍。
    INTENT_PUBLISH = "publish"
    INTENT_DELETE = "delete"
    VALID_INTENTS = {INTENT_PUBLISH, INTENT_DELETE}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64), default="default", server_default="default",
        nullable=False, index=True,
    )
    # 允许为空：删除申请被批准后文档就没了，而这条申请记录必须**留下来**
    # 作为审计凭证（谁在什么时候基于什么理由批准删除了哪份文档）。早期用
    # ON DELETE CASCADE，删文档会把申请行一起带走 —— 申请人再也看不到
    # "已通过"，审计链也断了。
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # 文档名冗余存一份：文档删除后审计记录仍能说清"删的是哪一份"
    document_name: Mapped[str] = mapped_column(String(512), nullable=False)

    # 申请人
    requester_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    requester_username: Mapped[str] = mapped_column(String(64), nullable=False)
    requester_department_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 申请内容
    #   intent = publish：target_level 是**目标**层级
    #   intent = delete ：target_level 是文档**当前**层级（决定谁来审、按什么范围审）
    intent: Mapped[str] = mapped_column(
        String(16), default=INTENT_PUBLISH, server_default=INTENT_PUBLISH,
        nullable=False,
    )
    target_level: Mapped[str] = mapped_column(String(20), nullable=False)
    target_department_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(
        String(20), default=STATUS_PENDING, server_default=STATUS_PENDING,
        nullable=False, index=True,
    )

    # 审核人
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    reviewer_username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    review_comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 申请人是否已查看过审核结果（"查看申请"里的未读小红点）
    requester_seen: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<ShareRequest doc={self.document_name!r} target={self.target_level} "
            f"status={self.status}>"
        )
