"""
Document management API router (Phase 3 / 企业落地第一阶段).

Endpoints:
  GET    /documents                    — paginated document list (owner-scoped)
  PATCH  /documents/{id}/collection    — assign/unassign a KB collection
  DELETE /documents/{id}               — delete document + vectors (owner-scoped)

All endpoints require authentication; non-admin users only ever see and
touch their own documents.
"""

import mimetypes
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import get_current_user, get_current_user_media
from app.services.permissions import require_permission
from app.db.models import Document, DocumentStatus
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.schemas.document_management import (
    DocumentDeleteResponse,
    DocumentListResponse,
)
from app.services.audit_service import record_audit
from app.services.document_query_service import (
    PermissionDenied,
    delete_document,
    get_document_chunks,
    list_documents,
)
from app.services.kb_collection_service import assign_document
from app.services.tenancy import (
    can_access_document,
    effective_department_id,
    effective_tenant_id,
    request_scope,
)
from app.services.security_scope import request_security_scope
from app.utils.errors import clean_message
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Document Management"])

# 【T4 / 决策 16】"文档不存在或无权访问"的**唯一常量**：
# 文本来源、图片来源、以及任何对象级剔除，都必须返回**逐字一致**的 404 文案 ——
# 任何措辞差异都等于确认"该文档 / 图片确实存在"，可被批量探测利用。
_DOCUMENT_404_DETAIL = "文档不存在或无权访问"


# ── GET /documents ─────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List uploaded documents (owner-scoped)",
    description=(
        "Returns a paginated list of documents, sorted by upload date "
        "descending. Optionally filter by processing status or knowledge-base "
        "collection. Non-admin users only see their own documents."
    ),
)
async def list_documents_endpoint(
    user: Annotated[User, Depends(require_permission("document.read"))],
    page: Annotated[
        int,
        Query(ge=1, description="Page number (1-indexed)."),
    ] = 1,
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="Items per page (max 100)."),
    ] = 20,
    status: Annotated[
        DocumentStatus | None,
        Query(description="Filter by document status."),
    ] = None,
    collection_id: Annotated[
        uuid.UUID | None,
        Query(description="Filter by knowledge-base collection."),
    ] = None,
    access_level: Annotated[
        str | None,
        Query(
            pattern="^(private|department|tenant)$",
            description="只列出某一层知识库：private=个人 / department=部门 / tenant=公司",
        ),
    ] = None,
    company_id: Annotated[
        str | None,
        Query(
            description=(
                "只列出归属某家公司的文档（文档页公司筛选）。取值来自 "
                "`GET /companies/accessible`；超出可见范围的取值返回空集。"
            )
        ),
    ] = None,
) -> DocumentListResponse:
    # 可见范围由安全 Scope 一处组装（五维：租户 + 层级 + 个人库 + 密级 + 项目）。
    # 平台管理员 = 自建测试公司集合，仍看不到别公司文档与他人个人库；其他人 =
    # 本公司内个人 + 本部门 + 公司库；V-03 修复：列表元数据现在同样受密级 /
    # 项目 / deny / excluded 约束，不再泄露高密级 / 项目外文档。
    scope = await request_security_scope(user)

    return await list_documents(
        page=page,
        limit=limit,
        status=status,
        owner_id=scope.base.owner_id,
        collection_id=collection_id,
        # 三层隔离：列表与检索同规则（公司集合 + ACL），共享文档按层级可见
        tenant_ids=scope.base.tenant_ids,
        user_department_id=scope.base.department_id,
        # 企业/知识库管理员可读本公司全部部门库（审核共享申请需要）
        tenant_wide=scope.base.tenant_wide,
        owns_tenant_ids=scope.base.owns_tenant_ids,
        # 文档页公司筛选：只保留归属该公司的文档（仍受可见范围上限约束）
        company_id=company_id,
        # 逐条算出 is_owner / can_delete / 发布能力，前端按钮与后端校验同源
        viewer=user,
        access_level=access_level,
        # V-03：五维 Scope 下推（密级 / 项目 / deny / excluded），优先于上面的
        # 三维标量参数（当 scope 非 None 时内层改用 to_sql(pred, Document)）。
        scope=scope,
    )


# ── PATCH /documents/{id}/collection ──────────────────────────────────────────

class CollectionAssignRequest(BaseModel):
    collection_id: uuid.UUID | None = None   # null = unassign


@router.patch(
    "/documents/{document_id}/collection",
    summary="Assign or unassign a document to a knowledge-base collection",
)
async def assign_collection_endpoint(
    document_id: Annotated[uuid.UUID, Path(description="UUID of the document.")],
    body: CollectionAssignRequest,
    user: Annotated[User, Depends(require_permission("document.write"))],
) -> dict:
    try:
        await assign_document(user.id, document_id, body.collection_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=clean_message(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    await record_audit(
        "document.assign_collection",
        user_id=user.id,
        username=user.username,
        resource_type="document",
        resource_id=str(document_id),
        detail=f"collection_id={body.collection_id}",
    )
    return {
        "document_id": str(document_id),
        "collection_id": body.collection_id and str(body.collection_id),
    }


# ── DELETE /documents/{id} ────────────────────────────────────────────────────

@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a document and all its vectors",
    description=(
        "Permanently removes the document record from PostgreSQL and all "
        "associated vector embeddings from Qdrant. This operation is "
        "**irreversible**. Qdrant vectors are deleted first so the PostgreSQL "
        "record is preserved on vector-store failure, enabling safe retries. "
        "Non-admin users can only delete their own documents."
    ),
)
async def delete_document_endpoint(
    document_id: Annotated[
        uuid.UUID,
        Path(description="UUID of the document to delete."),
    ],
    user: Annotated[User, Depends(require_permission("document.read"))],
) -> DocumentDeleteResponse:
    scope = await request_scope(user)
    try:
        result = await delete_document(
            document_id,
            owner_id=scope.owner_id,
            tenant_ids=scope.tenant_ids,
            owns_tenant_ids=scope.owns_tenant_ids,
            actor=user,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=clean_message(exc),
        ) from exc
    except PermissionDenied as exc:
        # 跨公司 → 404（不泄漏存在性）；权限不足 → 403 + 中文原因
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND if exc.not_found else status.HTTP_403_FORBIDDEN
            ),
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    await record_audit(
        "document.delete",
        user_id=user.id,
        username=user.username,
        resource_type="document",
        resource_id=str(document_id),
        detail=f"filename={result.filename}",
    )
    return result


# ── GET /documents/{id}/chunks ────────────────────────────────────────────────

@router.get(
    "/documents/{document_id}/chunks",
    summary="Get all chunks of a document (原文预览)",
    description=(
        "Returns the document's chunks in reading order, each with its page "
        "number and full text.  Used by the citation '定位原文' feature to "
        "show the exact source passage behind an answer.  Non-admin users "
        "can only preview their own documents."
    ),
)
async def get_document_chunks_endpoint(
    document_id: uuid.UUID,
    user: Annotated[User, Depends(require_permission("document.read"))],
) -> dict:
    # 【T4 / 决策 16】两件事都做：
    #   ① 沿用 request_scope 的三维 kwargs（保持既有 SQL 组装点不变）；
    #   ② 额外**重新签发** UserScope 并取 predicate()，供逐 chunk 对象级再校验。
    # 重新签发发生在新的一次 HTTP 请求入口（api/**），符合决策 10-④。
    scope = await request_scope(user)
    sec = await request_security_scope(user)
    try:
        return await get_document_chunks(
            document_id,
            owner_id=scope.owner_id,
            tenant_ids=scope.tenant_ids,
            owns_tenant_ids=scope.owns_tenant_ids,
            user_department_id=scope.department_id,
            tenant_wide=scope.tenant_wide,
            pred=sec.predicate(),
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=clean_message(exc),
        ) from exc


# ── PATCH /documents/{id}/visibility ──────────────────────────────────────────
#
# 三层知识库的"直接发布"入口：权限足够的角色（部门负责人 → 部门库；
# 知识库管理员/企业管理员 → 公司库）不必走申请，直接改层级。
# 普通员工调用会被 403 拦下并得到"请提交申请共享"的中文提示。

class VisibilityUpdateRequest(BaseModel):
    access_level: Literal["private", "department", "tenant"] = Field(
        ...,
        description="目标层级：private=个人知识库 / department=部门知识库 / tenant=公司知识库",
    )


@router.patch(
    "/documents/{document_id}/visibility",
    summary="发布 / 收回文档到个人、部门或公司知识库",
    description=(
        "把文档在三层知识库之间移动：\n"
        "- `private`（个人）：仅归属人可见，本人可自由操作；\n"
        "- `department`（部门）：需要 `document.publish.department` 权限（部门负责人及以上）；\n"
        "- `tenant`（公司）：需要 `document.publish.company` 权限（知识库管理员及以上）。\n\n"
        "层级变更会同时更新 PostgreSQL 行、Qdrant 向量 ACL 载荷并写入审计日志。"
    ),
)
async def update_document_visibility_endpoint(
    document_id: Annotated[uuid.UUID, Path(description="UUID of the document.")],
    body: VisibilityUpdateRequest,
    user: Annotated[User, Depends(require_permission("document.write"))],
) -> dict:
    from app.services.knowledge_tier_service import (
        TierError,
        publish_capability,
        required_permission_message,
        resolve_department_for_level,
        set_document_access_level,
    )
    from app.services.permissions import has_permission
    from app.services.tenancy import (
        ACCESS_PRIVATE,
        ACCESS_TENANT,
        access_label,
        is_downgrade,
        publish_requirement,
    )

    # 先取文档做可见性 + 能力判定（404 不泄漏存在性）
    scope = await request_scope(user)
    async with get_db_session() as session:
        doc = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()
    if doc is None or not can_access_document(
        doc, user,
        tenant_ids=scope.tenant_ids,
        owns_tenant_ids=scope.owns_tenant_ids,
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在或无权访问"
        )

    required = publish_requirement(body.access_level)
    if required and not has_permission(user, required):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=required_permission_message(body.access_level),
        )

    # 只有归属人（或有管理权限者）能改层级，避免"别人把我的文档降级回收"
    if doc.owner_id != user.id and not has_permission(user, "document.publish.company"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有文档归属人可以调整它的知识库层级",
        )

    # 层级只能**向上或平级**。这条比申请链路那条更关键：直接发布不需要任何人
    # 审批，一次 PATCH 就能把公司库文档降成部门库，其他部门同事静默失去访问权，
    # 而审计日志里只是一条正常的"层级变更"。能力字段（publish_capability）已经
    # 把降级按钮收窄掉了，但**按钮隐藏不是权限控制** —— 直接构造请求照样能过，
    # 所以接口必须独立兜底。
    #
    # 收回个人库不在拦截范围：它是归属人的正当操作（前端有独立的「收回」入口，
    # 文案也明示了"已共享的成员将无法再检索到"）。
    if (
        body.access_level != ACCESS_PRIVATE
        and is_downgrade(doc.access_level, body.access_level)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"这份文档当前在{access_label(doc.access_level)}，不能再变更到更低的"
                f"「{access_label(body.access_level)}」；如需仅自己可见请点「收回至"
                "个人知识库」"
            ),
        )

    try:
        updated = await set_document_access_level(
            document_id,
            level=body.access_level,
            department_id=resolve_department_for_level(
                body.access_level,
                user_department_id=effective_department_id(user),
                doc_department_id=doc.department_id,
            ),
            actor_id=user.id,
            actor_username=user.username,
            action="document.visibility.update",
        )
    except TierError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return {
        "document_id": str(updated.id),
        "access_level": updated.access_level,
        "access_label": access_label(updated.access_level),
        "department_id": updated.department_id,
        "capability": publish_capability(
            user, updated,
            tenant_ids=scope.tenant_ids,
            owns_tenant_ids=scope.owns_tenant_ids,
        ),
        "message": (
            f"已{'发布到' if body.access_level != 'private' else '收回至'}"
            f"{access_label(body.access_level)}知识库"
        ),
    }


# ── 公司文档 → 部门文档（公司 HR / 知识库管理员的"改归部门"） ─────────────────
#
# 与上面 `/visibility` 的分工：那条管**层级升降**（个人/部门/公司之间），这条管
# **归属哪个部门** —— 目标部门由操作者在"公司已有部门"里选定，而不是隐含地取
# 操作者自己的部门（后者是「发布到部门知识库」的语义）。
#
# 为什么单独开一个端点，而不是给 visibility 加个 department_id 参数：这条路径
# 携带一次**降级豁免**。公司库 → 部门库在其他所有路径上都被 is_downgrade 兜底
# 拦掉（上一轮补的补丁，见 update_document_visibility_endpoint），因为那会让
# 其他部门同事静默失去访问权；而公司 HR 主动把文档下沉到指定部门，恰恰是这次
# 要的功能。把豁免严格圈在一个独立端点里，"谁能绕开防降级"是一眼可查的；
# 塞进 visibility 则要把那条通用兜底改成"看角色再决定拦不拦"，防降级的语义
# 立刻变得可以协商 —— 那正是最容易出越权的地方。


class TransferDepartmentRequest(BaseModel):
    department_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="目标部门 ID（必须出现在该公司已有部门清单中）",
    )
    note: str | None = Field(
        None, max_length=200, description="可选说明，一并写入审计日志",
    )


@router.get(
    "/documents/{document_id}/transfer-targets",
    summary="转为部门文档：可选的部门清单",
    description=(
        "返回这份文档可以转入的部门清单（该公司内已有成员归属的部门），"
        "以及当前是否具备「转为部门文档」的能力。仅企业管理员 / 知识库管理员 / "
        "平台管理员可用 —— 普通员工与部门负责人调用会被 403 拦下。"
    ),
)
async def document_transfer_targets_endpoint(
    document_id: Annotated[uuid.UUID, Path(description="UUID of the document.")],
    user: Annotated[User, Depends(require_permission("document.read"))],
) -> dict:
    from app.services.knowledge_tier_service import (
        list_department_options,
        publish_capability,
    )
    from app.services.permissions import has_permission

    # 可见性先行：跨公司/看不见的文档一律 404（不泄漏存在性），再谈权限。
    scope = await request_scope(user)
    async with get_db_session() as session:
        doc = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()
    if doc is None or not can_access_document(
        doc, user,
        tenant_ids=scope.tenant_ids,
        owns_tenant_ids=scope.owns_tenant_ids,
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在或无权访问"
        )

    if not has_permission(user, "document.read.all"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "「转为部门文档」需要企业管理员或知识库管理员权限；"
                "如需把文档共享给某个部门，请使用「申请共享」"
            ),
        )

    capability = publish_capability(
        user, doc,
        tenant_ids=scope.tenant_ids,
        owns_tenant_ids=scope.owns_tenant_ids,
    )
    options = await list_department_options(getattr(doc, "tenant_id", None))
    current_dept = (doc.department_id or "").strip()

    return {
        "document_id": str(document_id),
        "current_level": capability["current_level"],
        "current_label": capability["current_label"],
        "department_id": doc.department_id,
        # 当前部门的中文名（公司库文档为 None —— 它不属于任何部门）
        "department_name": next(
            (o["department_name"] for o in options if o["department_id"] == current_dept),
            None,
        ),
        "can_transfer_department": capability["can_transfer_department"],
        "transfer_denied_reason": capability["transfer_denied_reason"],
        "options": options,
    }


@router.post(
    "/documents/{document_id}/transfer-department",
    summary="把公司文档转为指定部门的部门文档",
    description=(
        "公司 HR / 知识库管理员把一份已共享的文档**改归到指定部门**：\n"
        "- 公司库 → 部门库：其他部门同事将无法再检索到这份文档；\n"
        "- 部门库 → 另一部门：部门归属被改写，原部门同事将无法再检索到。\n\n"
        "目标部门必须出现在该公司已有部门清单中（见 transfer-targets），"
        "否则 400 —— 防的是把文档塞进不存在的部门，那等于一次无痕迹的软删除。"
    ),
)
async def transfer_document_department_endpoint(
    document_id: Annotated[uuid.UUID, Path(description="UUID of the document.")],
    body: TransferDepartmentRequest,
    user: Annotated[User, Depends(require_permission("document.write"))],
) -> dict:
    from app.services.knowledge_tier_service import (
        TierError,
        list_department_options,
        publish_capability,
        transfer_document_to_department,
    )
    from app.services.permissions import has_permission
    from app.services.tenancy import access_label

    scope = await request_scope(user)
    async with get_db_session() as session:
        doc = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()
    if doc is None or not can_access_document(
        doc, user,
        tenant_ids=scope.tenant_ids,
        owns_tenant_ids=scope.owns_tenant_ids,
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在或无权访问"
        )

    if not has_permission(user, "document.read.all"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "只有企业管理员 / 知识库管理员可以把文档转为部门文档；"
                "部门负责人只能发布到本部门知识库"
            ),
        )

    capability = publish_capability(
        user, doc,
        tenant_ids=scope.tenant_ids,
        owns_tenant_ids=scope.owns_tenant_ids,
    )
    if not capability["can_transfer_department"]:
        # 个人库文档：先共享再谈归属（见 publish_capability 的判定注释）
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                capability["transfer_denied_reason"]
                or "当前状态下不能把这份文档转为部门文档"
            ),
        )

    try:
        updated = await transfer_document_to_department(
            document_id,
            target_department_id=body.department_id,
            actor_id=user.id,
            actor_username=user.username,
            note=body.note or "",
        )
    except TierError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    # 只为把部门中文名写进给用户看的消息 —— 事实校验由服务层自己做了一遍
    # （transfer_document_to_department 内部同样查询清单），这里不承担校验职责。
    options = await list_department_options(getattr(doc, "tenant_id", None))
    target_name = next(
        (o["department_name"] for o in options
         if o["department_id"] == (updated.department_id or "")),
        body.department_id,
    )

    return {
        "document_id": str(updated.id),
        "access_level": updated.access_level,
        "access_label": access_label(updated.access_level),
        "department_id": updated.department_id,
        "department_name": target_name,
        "capability": publish_capability(
            user, updated,
            tenant_ids=scope.tenant_ids,
            owns_tenant_ids=scope.owns_tenant_ids,
        ),
        "message": f"已转为「{target_name}」的部门文档，仅该部门成员可检索到",
    }


# ── GET /documents/{id}/images/{name} ─────────────────────────────────────────
#
# 部分2「返回原始图片」的服务端入口：检索命中图片对象后，前端引用卡片直接
# 用这个 URL 渲染原始图片。浏览器 <img> 无法携带 Authorization 头，因此
# get_current_user_media 额外接受 ?token=<jwt>；资源归属仍按 owner 校验。

@router.get(
    "/documents/{document_id}/images/{image_name}",
    summary="Serve an extracted document image (原始图片回显)",
    description=(
        "Returns the original picture extracted from a document during "
        "ingestion. Images live under uploads/{document_id}/images/ and are "
        "only served to the document's owner (admins see all)."
    ),
)
async def get_document_image_endpoint(
    document_id: uuid.UUID,
    image_name: str,
    user: Annotated[User, Depends(get_current_user_media)],
) -> FileResponse:
    from sqlalchemy import select

    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.services.security_policy import ObjectACLView, allows
    from app.services.security_cascade import document_view
    from app.services.security_scope import request_security_scope
    from app.services.storage import resolve_image_path
    from app.services.storage.image_store import IMAGES_SUBDIR

    # 三层隔离的可见性检查（404 without leaking foreign documents）：
    # 第一、二层由 can_access_document 判定（本人 / 公司集合内且过 ACL /
    # 平台管理员在自建测试公司内；**他人个人库对任何人都 404**，admin 在
    # 自建集合内可读但不可删）。
    scope = await request_scope(user)
    async with get_db_session() as session:
        query = select(Document).where(Document.id == document_id)
        doc = (await session.execute(query)).scalar_one_or_none()
    if doc is None or not can_access_document(
        doc, user,
        tenant_ids=scope.tenant_ids,
        owns_tenant_ids=scope.owns_tenant_ids,
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_DOCUMENT_404_DETAIL,
        )

    # ── 【T4 / 决策 16】图片级对象校验（不能只凭"文档可见"就放行任意一张图）──────
    pred = (await request_security_scope(user)).predicate()
    # ① 文档级五维判定（密级 / 项目 / deny / excluded）
    if not allows(pred, document_view(doc)).allowed:
        await _audit_citation_open_drop(user, str(document_id), str(document_id), "document_level", "security")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_DOCUMENT_404_DETAIL,
        )
    # ② 图片对象级判定：由 image_name 反查图片对象行再 allows()
    from app.services.image_security import resolve_image_object_id
    from app.services.security_cascade import document_is_materialized

    info = await resolve_image_object_id(document_id, image_name)
    if info is None:
        # 反查不到行：
        #   - 文档**已物化** → fail-closed 404（对象权限是权威且完整的，没有行就是不该有）
        #   - 文档**从未物化**（回填未覆盖的存量文档）→ 回退文档级判定（已在上面通过），
        #     仅继续校验文件是否存在，避免存量文档的图片集体变 404（功能退化）。
        if await document_is_materialized(document_id):
            await _audit_citation_open_drop(
                user, str(document_id), f"{document_id}::{image_name}",
                "image_object_not_found", "tenant",
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=_DOCUMENT_404_DETAIL,
            )
    else:
        decision = allows(pred, ObjectACLView.from_row(info))
        if not decision.allowed:
            await _audit_citation_open_drop(
                user, str(document_id), info.get("object_id"), decision.reason, decision.gate,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=_DOCUMENT_404_DETAIL,
            )

    resolved = resolve_image_path(
        str(document_id), f"{IMAGES_SUBDIR}/{image_name}",
        tenant_id=getattr(doc, "tenant_id", None),
    )
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="图片不存在",
        )

    media_type = mimetypes.guess_type(resolved.name)[0] or "image/png"
    return FileResponse(
        resolved,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )


async def _audit_citation_open_drop(
    user: User,
    document_id: str,
    object_id: str | None,
    reason: str | None,
    gate: str | None,
) -> None:
    """引用点击回源被对象级 ACL 剔除时的审计（best-effort，绝不阻断请求）."""
    try:
        from app.services.audit_service import record_acl_drop

        await record_acl_drop(
            "citation_open",
            object_id=object_id,
            document_id=document_id,
            reason=reason,
            gate=gate,
            user_id=getattr(user, "id", None),
            username=getattr(user, "username", None),
        )
    except Exception:      # noqa: BLE001
        logger.warning("citation_open audit failed (doc=%s)", document_id, exc_info=True)


# ── GET /documents/generated/{filename} ───────────────────────────────────────
#
# Document Agent 生成的 Word 下载入口。文件名由服务端生成（时间戳 + 随机后缀），
# 不接受用户输入作为路径，因此天然免疫目录穿越。
#
# ⚠️ 仅靠"文件名不可猜"是不够的：产物内含 N 条来源片段与原始图片，一旦落到
# 别人手里，检索前租户过滤 / 检索后 ACL / LLM 输入前校验整条隔离链路都被绕过。
# 因此这里做了**归属 + 权限**双重判定（document_agent_service.authorize_
# generated_file）：调用者必须是产物所有者，或者对产物引用的**全部**源文档都
# 可访问；归属无法确定时 fail-closed。

# 未授权与"文件不存在"必须返回**逐字一致**的响应 —— 「无权访问」会确认
# "该文件存在"，本身即信息泄露，且可被批量探测。两者共用这一个常量，杜绝漂移。
_GENERATED_404_DETAIL = "文件不存在或已过期"


@router.get(
    "/documents/generated/{filename}",
    summary="Download a generated Word document (Document Agent 产物)",
    description=(
        "Downloads a .docx produced by the Document Agent. Access is granted "
        "only when the caller is the artifact's owner, or can access **every** "
        "source document cited by the artifact. File names are server-generated. "
        "Unauthorized requests are indistinguishable from a missing file."
    ),
)
async def download_generated_document_endpoint(
    filename: str,
    user: Annotated[User, Depends(get_current_user_media)],
) -> FileResponse:
    # 注意：Authorization 头与 ?token= 两条通道都收敛到 get_current_user_media
    # → _resolve_user，拿到的是同一个 user 对象；归属判定只依赖这个 user，因此
    # 两条通道同源，不存在"带 token 就绕过"。
    from app.services.document_agent_service import authorize_generated_file

    resolved, decision = await authorize_generated_file(filename, user)
    if resolved is None:
        logger.warning(
            "Generated download denied: file=%s user=%s reason=%s",
            filename,
            getattr(user, "username", None),
            decision,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=_GENERATED_404_DETAIL,
        )

    media_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(
        resolved,
        media_type=media_type,
        filename=resolved.name,
        headers={"Cache-Control": "private, max-age=60"},
    )
