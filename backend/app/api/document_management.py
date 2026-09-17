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
    scope_for,
)
from app.utils.errors import clean_message
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Document Management"])


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
) -> DocumentListResponse:
    # 可见范围由 tenancy.scope_for 一处组装（平台管理员 = 全平台，但仍看不到
    # 别人的个人库；其他人 = 本公司内个人 + 本部门 + 公司库）。
    scope = scope_for(user)

    return await list_documents(
        page=page,
        limit=limit,
        status=status,
        owner_id=scope.owner_id,
        collection_id=collection_id,
        # 三层隔离：列表与检索同规则（公司 + ACL），共享文档按层级可见
        tenant_id=scope.tenant_id,
        user_department_id=scope.department_id,
        # 企业/知识库管理员可读本公司全部部门库（审核共享申请需要）
        read_all=scope.tenant_wide,
        platform_wide=scope.platform_wide,
        # 逐条算出 is_owner / can_delete / 发布能力，前端按钮与后端校验同源
        viewer=user,
        access_level=access_level,
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
    scope = scope_for(user)
    try:
        result = await delete_document(
            document_id,
            owner_id=scope.owner_id,
            tenant_id=scope.tenant_id,
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
    scope = scope_for(user)
    try:
        return await get_document_chunks(
            document_id,
            owner_id=scope.owner_id,
            tenant_id=scope.tenant_id,
            user_department_id=scope.department_id,
            read_all=scope.tenant_wide,
            platform_wide=scope.platform_wide,
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
        ACCESS_TENANT,
        access_label,
        publish_requirement,
    )

    # 先取文档做可见性 + 能力判定（404 不泄漏存在性）
    async with get_db_session() as session:
        doc = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()
    if doc is None or not can_access_document(doc, user):
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
        "capability": publish_capability(user, updated),
        "message": (
            f"已{'发布到' if body.access_level != 'private' else '收回至'}"
            f"{access_label(body.access_level)}知识库"
        ),
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
    from app.services.storage import resolve_image_path
    from app.services.storage.image_store import IMAGES_SUBDIR

    # 三层隔离的可见性检查（404 without leaking foreign documents）：
    # 第一、二层由 can_access_document 判定（本人 / 同公司且过 ACL /
    # 平台管理员跨公司；**他人个人库对任何人都 404**）。
    async with get_db_session() as session:
        query = select(Document).where(Document.id == document_id)
        doc = (await session.execute(query)).scalar_one_or_none()
    if doc is None or not can_access_document(doc, user):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在或无权访问",
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


# ── GET /documents/generated/{filename} ───────────────────────────────────────
#
# Document Agent 生成的 Word 下载入口。文件名由服务端生成（时间戳 + 随机后缀），
# 不接受用户输入作为路径，因此天然免疫目录穿越。

@router.get(
    "/documents/generated/{filename}",
    summary="Download a generated Word document (Document Agent 产物)",
    description=(
        "Downloads a .docx produced by the Document Agent. Any authenticated "
        "user may download a generated document; file names are server-generated."
    ),
)
async def download_generated_document_endpoint(
    filename: str,
    user: Annotated[User, Depends(get_current_user_media)],
) -> FileResponse:
    from app.services.document_agent_service import resolve_generated_file

    resolved = resolve_generated_file(filename)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在或已过期",
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
