"""
企业身份验证申请（StaffRequest）—— 企业管理后台的"入伙审核"闭环.

业务故事（与产品流程图一致）：

    用户注册 → 进入待审核 → 上级审核（同意 / 拒绝）→ 加入部门
             → 系统自动赋予 role（默认 Employee）

与 ``share_requests`` 的区别（为什么另建一张表而不是复用）：

    share_requests   审核的对象是**文档**，批准的效果是改文档可见层级
    staff_requests   审核的对象是**人**，批准的效果是给这个人定身份
                     （公司 + 部门 + 职责 + 权限角色）

两者的审核人推导、状态机、以及"批准后做什么"完全不同，硬塞进一张表会出现
大量"这一列只在某一种情况下有意义"的字段。分表后各自的 NOT NULL 约束才
说得清楚。

## 层级授权（越级审核）

角色权限等级（``staff_service.ROLE_RANK``）：

    admin > kb_admin > dept_manager > employee

审核规则：**审核人的等级必须严格高于申请人**，同公司内可越级。
即知识库管理员能直接审普通员工与部门负责人的申请，不必逐级上报；
平台管理员（admin，全局唯一）可审任意公司的申请。

## 公司隔离

每份申请都带 ``company_id``（租户）。除平台管理员外，审核人与管理后台
只能看到本公司（``effective_tenant_id`` 相同）的申请 —— 与检索链路上的
第一层隔离共用同一套 tenant_id，不引入第二套公司概念。

## 责任留痕：负责人 = 职务 + 名称

产品要求在通过 / 拒绝的申请后面显示"负责人"（如「研发部负责人 李四」）。
因此审核落库时同时记录 ``reviewer_title``（职务）与 ``reviewer_name``
（姓名），而不是只存用户名 —— 用户账号名往往是 ``zhangsan`` 这类登录名，
不满足"职务 + 名称"的展示要求。
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.postgres import Base


class StaffRequest(Base):
    __tablename__ = "staff_requests"
    __table_args__ = (
        # 管理后台最热的两个查询：「本公司待审队列」「某人最新一条申请」
        Index("ix_staff_requests_company_status", "company_id", "status"),
        Index("ix_staff_requests_applicant_status", "applicant_id", "status"),
    )

    STATUS_PENDING = "pending"
    STATUS_APPROVED = "approved"
    STATUS_REJECTED = "rejected"
    STATUS_CANCELLED = "cancelled"

    # 身份状态（前端个人主页 / 首页徽标用）——由最新一条申请推导
    IDENTITY_NONE = "none"            # 从未提交过 → 显示「去验证」
    IDENTITY_PENDING = "pending"      # 已提交待审 → 显示「审核中」
    IDENTITY_APPROVED = "approved"    # 已通过   → 显示「已通过」
    IDENTITY_REJECTED = "rejected"    # 被拒绝   → 显示「去验证」（可重新提交）

    VALID_STATUSES = {STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED, STATUS_CANCELLED}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )

    # ── 申请人 ────────────────────────────────────────────────────────────────
    applicant_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    applicant_username: Mapped[str] = mapped_column(String(64), nullable=False)
    applicant_display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # 提交时的所属公司（未验证用户通常是 default；已通过后再次申请则是原公司）
    applicant_company_id: Mapped[str] = mapped_column(
        String(64), default="default", server_default="default", nullable=False,
    )

    # ── 申请内容（身份验证表单三行）───────────────────────────────────────────
    # 公司名称：用户填的原文（可中文），company_id 是它的安全映射。
    company_name: Mapped[str] = mapped_column(String(128), nullable=False)
    company_id: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
    )
    # 公司部门
    department_name: Mapped[str] = mapped_column(String(128), nullable=False)
    department_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # 部门职责（如「嵌入式软件工程师」）——审核人可在批准时就地修正
    duty: Mapped[str] = mapped_column(String(128), nullable=False)

    # ── 审核结果 ──────────────────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        String(20), default=STATUS_PENDING, server_default=STATUS_PENDING,
        nullable=False, index=True,
    )
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    reviewer_username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 「负责人」= 职务 + 名称（产品要求展示在申请记录后方）
    reviewer_title: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewer_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 批准时授予的权限角色（默认 employee）；拒绝时为空。
    granted_role: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # 申请人是否已看过审核结论（个人主页上的未读提示）
    applicant_seen: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    @property
    def reviewer_label(self) -> str | None:
        """「研发部负责人 李四」—— 展示用；两者都缺失时退回用户名。"""
        title = (self.reviewer_title or "").strip()
        name = (self.reviewer_name or "").strip() or (self.reviewer_username or "").strip()
        if not title and not name:
            return None
        return f"{title} {name}".strip() if title else name

    def __repr__(self) -> str:
        return (
            f"<StaffRequest user={self.applicant_username!r} "
            f"company={self.company_id!r} status={self.status}>"
        )
