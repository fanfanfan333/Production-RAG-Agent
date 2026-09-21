"""
Document query & management service (Phase 3).

Provides:
  - Paginated listing of Document rows from PostgreSQL
  - Hard-delete: removes the PostgreSQL row AND all associated Qdrant vectors

This module is intentionally separate from document_service.py (Phase 2 upload
pipeline) so that Phase 2 code is never touched.
"""

import math
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, func, select

from app.db.models import Document, DocumentStatus
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.schemas.document_management import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentSummary,
)
from app.services.vector_service import delete_by_document_id
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PermissionDenied(Exception):
    """
    能力矩阵拒绝了一个动作（携带用户可读的中文原因）。

    ``not_found=True`` 表示按"不泄漏存在性"的约定应当对外表现为 404
    （跨公司访问），而不是 403 —— 否则攻击者可用 403/404 的差异探测
    别家公司有哪些文档。
    """

    def __init__(self, message: str, *, not_found: bool = False):
        super().__init__(message)
        self.not_found = not_found


def _opt_int(value) -> int | None:
    """宽松地把 payload 里的位置信息转成 int；缺失/非法一律返回 None."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _opt_bbox(value) -> list[float] | None:
    """宽松地把 payload 里的 bbox 转成 4 元素 list；缺失/非法返回 None."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _opt_dict(value) -> dict:
    """宽松地把 payload 里的嵌套报告字段转成 dict；缺失/非法返回 {}."""
    return dict(value) if isinstance(value, dict) else {}


async def count_by_tenant(
    *,
    owner_id: uuid.UUID | None = None,
    tenant_ids: frozenset[str] | None = None,
    owns_tenant_ids: frozenset[str] = frozenset(),
    department_id: str | None = None,
    tenant_wide: bool = False,
) -> dict[str, int]:
    """
    按公司分组统计**当前用户可见**的文档数（``/companies/accessible`` 的 ``doc_count``）.

    与 :func:`list_documents` **共用同一个** ``document_scope_clause`` 入口 ——
    这是「筛选项里的计数 == 选中该公司的列表结果数」的结构性保证，杜绝
    「筛选项显示 3 家但点进去空」。

    参数：

        owner_id:        个人库归属人（恒为本人 id）。
        tenant_ids:      待统计的公司集合（``None``/空集 → 返回空字典，不泄漏全平台）。
        owns_tenant_ids: admin 自建测试公司集合（可见其中他人私库）。
        department_id:   第二层部门条件。
        tenant_wide:     本租户内部门库全通。

    Returns:
        ``{tenant_id: count}``（仅含 ``tenant_ids`` 里的公司；无文档的公司不出现在键中，
        调用方按 0 兜底）。
    """
    if tenant_ids is None or not tenant_ids:
        return {}

    from app.services.tenancy import document_scope_clause

    async with get_db_session() as session:
        stmt = (
            select(Document.tenant_id, func.count())
            .where(Document.tenant_id.in_(sorted(tenant_ids)))
            .where(
                document_scope_clause(
                    owner_id=owner_id,
                    department_id=department_id,
                    tenant_ids=tenant_ids,
                    owns_tenant_ids=owns_tenant_ids,
                    tenant_wide=tenant_wide,
                )
            )
            .group_by(Document.tenant_id)
        )
        rows = (await session.execute(stmt)).all()
    return {str(t): int(c) for t, c in rows}


async def list_documents(
    page: int = 1,
    limit: int = 20,
    status: DocumentStatus | None = None,
    owner_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
    tenant_ids: frozenset[str] | None = None,
    user_department_id: str | None = None,
    tenant_wide: bool = False,
    owns_tenant_ids: frozenset[str] = frozenset(),
    company_id: str | None = None,
    viewer: User | None = None,
    access_level: str | None = None,
    scope: "UserScope | None" = None,
) -> DocumentListResponse:
    """
    Return a paginated list of Document rows, optionally filtered by status,
    owner (multi-user isolation) and knowledge-base collection.

    Args:
        page:   1-indexed page number.
        limit:  Rows per page (1–100).
        status: Optional filter on DocumentStatus.
        owner_id:  个人库归属人 —— **恒为本人 id**（含平台管理员）。
                   None = 不返回任何个人库文档。
        collection_id: Restrict to one KB collection (None = all documents).
        tenant_ids: 第一层公司过滤**集合**（None = 不限制诊断；空集 = fail-closed）。
        tenant_wide: 本租户内部门库全通（企业/知识库管理员 / 平台管理员）。
        owns_tenant_ids: admin 的自建测试公司集合（可见其中他人私库）。
        company_id: 只列出归属该公司的文档（文档页公司筛选）。**不**放宽可见
                   范围 —— 它只在前面的 scope 条件之上再收窄；取值不在可见
                   范围内时结果自然为空（fail-safe）。
        viewer: 传入用户后可逐条算出 is_owner / can_delete / 发布能力。
        access_level: 只列出某一层（个人/部门/公司）的文档。
    """
    offset = (page - 1) * limit

    async with get_db_session() as session:
        # ── Base query ────────────────────────────────────────────────────────
        base_q = select(Document)
        count_q = select(func.count()).select_from(Document)

        if status is not None:
            base_q = base_q.where(Document.status == status)
            count_q = count_q.where(Document.status == status)

        # ── 三层隔离：唯一 SQL 组装点（公司边界 ∪ 自己个人库，与检索同规则）────
        # V-03：传入五维 ``UserScope`` 时改用 ``to_sql(pred, Document)`` 下推
        # （三维 ``document_scope_clause`` 已被其内部复用，并追加密级 / 项目 /
        # deny / excluded 四个维度）。编译失败 fail-closed 返回空，绝不降级放行。
        from app.services.tenancy import document_scope_clause

        if scope is not None:
            from app.services.security_policy import to_sql

            try:
                sec_filter = to_sql(scope.predicate(), Document)
            except Exception:      # noqa: BLE001 — 编译失败不得静默放行
                logger.exception(
                    "list_documents: to_sql(pred) 编译失败 — fail-closed 返回空"
                )
                return DocumentListResponse(items=[], total=0, page=page, limit=limit)
            base_q = base_q.where(sec_filter)
            count_q = count_q.where(sec_filter)
        elif (
            owner_id is not None
            or tenant_ids is not None
            or tenant_wide
            or owns_tenant_ids
        ):
            acl = document_scope_clause(
                owner_id=owner_id,
                department_id=user_department_id,
                tenant_ids=tenant_ids,
                owns_tenant_ids=owns_tenant_ids,
                tenant_wide=tenant_wide,
            )
            base_q = base_q.where(acl)
            count_q = count_q.where(acl)

        if collection_id is not None:
            base_q = base_q.where(Document.collection_id == collection_id)
            count_q = count_q.where(Document.collection_id == collection_id)
        if company_id:
            # 文档页公司筛选：只保留归属该公司的文档（在可见范围之上再收窄，
            # 因此取值来自 /companies/accessible 之外的集合只会得到空列表）。
            base_q = base_q.where(Document.tenant_id == company_id)
            count_q = count_q.where(Document.tenant_id == company_id)
        if access_level:
            from app.services.tenancy import normalize_access_level

            base_q = base_q.where(Document.access_level == normalize_access_level(access_level))
            count_q = count_q.where(Document.access_level == normalize_access_level(access_level))

        # ── Total count ───────────────────────────────────────────────────────
        total: int = (await session.execute(count_q)).scalar_one()

        # ── Paginated fetch ───────────────────────────────────────────────────
        rows_result = await session.execute(
            base_q
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        rows: list[Document] = list(rows_result.scalars().all())

        # 归属人用户名（一次批量查，避免 N+1）
        owner_names: dict[uuid.UUID, str] = {}
        owner_ids = {r.owner_id for r in rows if r.owner_id}
        if owner_ids:
            from app.db.user_models import User as _User

            name_rows = await session.execute(
                select(_User.id, _User.username).where(_User.id.in_(owner_ids))
            )
            owner_names = {uid: name for uid, name in name_rows.all()}

        # 待审申请（一次批量查，前端据此把"申请共享"按钮切成"审核中"）
        pending_docs: set[uuid.UUID] = set()
        doc_ids = [r.id for r in rows]
        if doc_ids:
            from app.db.share_models import ShareRequest

            pending_rows = await session.execute(
                select(ShareRequest.document_id).where(
                    ShareRequest.document_id.in_(doc_ids),
                    ShareRequest.status == ShareRequest.STATUS_PENDING,
                )
            )
            pending_docs = {did for (did,) in pending_rows.all()}

    pages = max(1, math.ceil(total / limit))

    # ── 三层标注展示名（一次性批量取，杜绝循环里逐条查库 = N+1）──────────────
    # 公司展示名：走 companies 注册表（唯一权威源），按本页 distinct tenant_id 取。
    # 部门展示名：按本页 distinct (tenant_id, department_id) 聚合 users.department_name，
    #             **不依赖 owner**（owner 解绑为 NULL 时同样能取到）。
    tenant_name_map: dict[str, str] = {}
    dept_name_map: dict[tuple[str, str], str] = {}
    if rows:
        from app.services.knowledge_tier_service import department_display_names
        from app.services.company_registry import company_display_names

        tenant_name_map = await company_display_names(
            frozenset({str(r.tenant_id) for r in rows if r.tenant_id})
        )
        dept_name_map = await department_display_names(
            {
                (str(r.tenant_id), str(r.department_id))
                for r in rows
                if r.tenant_id and r.department_id
            }
        )

    summaries: list[DocumentSummary] = []
    for row in rows:
        from app.services.knowledge_tier_service import publish_capability
        from app.services.tenancy import access_label, normalize_access_level

        level = normalize_access_level(row.access_level)
        capability = (
            publish_capability(
                viewer, row,
                tenant_ids=tenant_ids,
                owns_tenant_ids=owns_tenant_ids,
            )
            if viewer is not None
            else {
                "is_owner": False,
                "can_delete": False,
                "delete_denied_reason": "",
                "can_publish_department": False,
                "can_publish_company": False,
                "can_request_department": False,
                "can_request_company": False,
                "needs_share_request": False,
                "can_transfer_department": False,
                "transfer_denied_reason": "",
                "publish_denied_reason": "",
            }
        )
        summaries.append(
            DocumentSummary(
                document_id=row.id,
                filename=row.filename,
                status=row.status,
                page_count=row.page_count,
                chunk_count=row.chunk_count,
                file_size_bytes=row.file_size,
                error=row.error_message,
                created_at=row.created_at,
                updated_at=row.updated_at,
                collection_id=row.collection_id,
                access_level=level,
                access_label=access_label(level),
                tenant_id=row.tenant_id or "default",
                department_id=row.department_id,
                # 展示名（原始字段，取不到即 None）—— 拼接放前端，后端不产出 "None"
                tenant_name=(
                    tenant_name_map.get(str(row.tenant_id)) if row.tenant_id else None
                ),
                department_name=(
                    dept_name_map.get((str(row.tenant_id), str(row.department_id)))
                    if row.tenant_id and row.department_id
                    else None
                ),
                owner_id=row.owner_id,
                owner_username=owner_names.get(row.owner_id) if row.owner_id else None,
                is_owner=bool(capability.get("is_owner")),
                can_delete=bool(capability.get("can_delete")),
                delete_denied_reason=str(capability.get("delete_denied_reason") or ""),
                can_request_delete=bool(capability.get("can_request_delete")),
                can_publish_department=bool(capability.get("can_publish_department")),
                can_publish_company=bool(capability.get("can_publish_company")),
                can_request_department=bool(capability.get("can_request_department")),
                can_request_company=bool(capability.get("can_request_company")),
                needs_share_request=bool(capability.get("needs_share_request")),
                can_transfer_department=bool(
                    capability.get("can_transfer_department")
                ),
                transfer_denied_reason=str(
                    capability.get("transfer_denied_reason") or ""
                ),
                publish_denied_reason=str(capability.get("publish_denied_reason") or ""),
                pending_share_request=row.id in pending_docs,
                # 异步入库的进度（前端据此显示"解析中 / 向量化 3/12"）
                current_stage=getattr(row, "current_stage", None),
                total_chunks=getattr(row, "total_chunks", None),
                embedded_chunks=getattr(row, "embedded_chunks", None),
                # 图片计数：图片是独立检索对象，列表里要能看出"这份文档带几张图"
                image_count=getattr(row, "image_count", 0) or 0,
                image_object_count=getattr(row, "image_object_count", 0) or 0,
            )
        )

    logger.info(
        "list_documents → total=%d page=%d/%d limit=%d status=%s",
        total, page, pages, limit, status,
    )

    return DocumentListResponse(
        total=total,
        page=page,
        limit=limit,
        pages=pages,
        documents=summaries,
    )


async def delete_document(
    document_id: uuid.UUID,
    owner_id: uuid.UUID | None = None,
    *,
    tenant_ids: frozenset[str] | None = None,
    owns_tenant_ids: frozenset[str] = frozenset(),
    actor: User | None = None,
) -> DocumentDeleteResponse:
    """
    Hard-delete a document: remove its Qdrant vectors first, then the PG row.

    Qdrant vectors are deleted before the PG row so that a partial failure
    (Qdrant down) leaves the PG record intact and the operation can be retried.

    Args:
        document_id: UUID of the document to delete.
        owner_id:    Legacy owner scoping — only used when *actor* is absent.
        tenant_ids:  第一层公司集合；``delete_permission_for`` 据此判定"跨公司
                     一律按不存在返回"（不泄漏存在性）。``None`` = 按身份推导。
        owns_tenant_ids: admin 的自建测试公司集合（可读他人私库，但**不可删**）。
        actor:       删除操作者。传入后按**能力矩阵**判定
                     （本人 / 部门负责人本部门 / 知识库管理员与企业管理员全公司 /
                     平台管理员在自建测试公司内），而不是简单地"必须是我上传的"。
                     无权限时抛 PermissionDenied（含中文原因）。
                     **他人个人库文档谁也删不了** —— 个人库是归属人专属。

    Returns:
        DocumentDeleteResponse on success.

    Raises:
        KeyError: 文档不存在。
        PermissionDenied: 角色能力不足以删除该层级 / 该归属的文档（跨公司时为 404）。
    """
    # ── 1. Fetch document metadata ────────────────────────────────────────────
    # 修复（问题2）：以前这里用 `owner_id = 当前用户` 过滤，于是"列表里看得见、
    # 但归属人不是我"的共享文档会走到 `doc is None`，抛出英文 KeyError →
    # 前端弹出 "Document '...' not found."。用户看到的现象是"文档明明在列表里，
    # 删除却说找不到"。
    #
    # Rev2：公司边界不再写在这条 SQL 里（`tenant_id =` 单值判定已废弃），改由
    # `delete_permission_for` 统一判定 —— 跨公司返回「不存在」（不泄漏存在性），
    # 个人库私库分支先于公司边界返回（admin 自建集合内他人私库可读、**不可删**）。
    async with get_db_session() as session:
        query = select(Document).where(Document.id == document_id)
        if actor is None and owner_id is not None:
            # 兼容旧调用方（无身份上下文时退化为"仅本人"）
            query = query.where(Document.owner_id == owner_id)
        doc: Document | None = (await session.execute(query)).scalar_one_or_none()

    if doc is None:
        raise KeyError("文档不存在或无权访问")

    if actor is not None:
        from app.services.tenancy import delete_permission_for

        allowed, reason = delete_permission_for(
            doc,
            actor,
            tenant_ids=tenant_ids,
            owns_tenant_ids=owns_tenant_ids,
        )
        if not allowed:
            # 跨公司时 reason 就是"文档不存在"（不泄漏），按 404 处理
            raise PermissionDenied(reason, not_found="不存在" in reason)

    filename = doc.filename
    doc_id_str = str(document_id)
    doc_tenant_id = doc.tenant_id

    logger.info("Deleting document id=%s filename='%s'", doc_id_str, filename)

    # ── 2. Delete vectors from Qdrant (fail-safe first) ───────────────────────
    try:
        await delete_by_document_id(doc_id_str)
        logger.info("Qdrant vectors deleted for document_id=%s", doc_id_str)
    except Exception as exc:
        # Log and re-raise; PG row intentionally NOT deleted on Qdrant failure
        logger.error(
            "Failed to delete Qdrant vectors for document_id=%s: %s — aborting delete",
            doc_id_str,
            exc,
        )
        raise RuntimeError(
            f"Vector deletion failed ({exc}). PostgreSQL record preserved for retry."
        ) from exc

    # ── 3. Delete PostgreSQL row ──────────────────────────────────────────────
    async with get_db_session() as session:
        await session.execute(
            delete(Document).where(Document.id == document_id)
        )

    # ── 4. Purge on-disk artifacts（部分2：原图 / 原文档归档）──────────────────
    # 必须在 PG 行删除之后做，且为 best-effort：磁盘清理失败不应让"删除文档"
    # 这个用户动作报错，但也不能不做——否则每删一份带图文档都会永久泄漏
    # uploads/{document_id}/ 下的原图与归档文件。
    try:
        from app.services.storage import delete_document_images

        delete_document_images(doc_id_str, tenant_id=doc_tenant_id)
        logger.info("On-disk artifacts purged for document_id=%s", doc_id_str)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to purge on-disk artifacts for document_id=%s: %s",
            doc_id_str,
            exc,
        )

    logger.info("Document id=%s deleted successfully", doc_id_str)

    return DocumentDeleteResponse(
        document_id=document_id,
        filename=filename,
    )


async def get_document_chunks(
    document_id: uuid.UUID,
    owner_id: uuid.UUID | None = None,
    *,
    tenant_ids: frozenset[str] | None = None,
    owns_tenant_ids: frozenset[str] = frozenset(),
    user_department_id: str | None = None,
    tenant_wide: bool = False,
    pred=None,
) -> dict:
    """
    Fetch the full ordered chunk list of one document for原文预览.

    Metadata (filename, page_count) comes from PostgreSQL; chunk text is
    scrolled from the Qdrant payload and sorted by chunk_index.

    Args:
        document_id: UUID of the document.
        owner_id:    个人库归属人（恒为本人 id）；None = 看不到任何个人库。
        tenant_ids:  第一层公司集合（与检索/列表同规则）；None = 按身份推导。
        owns_tenant_ids: admin 自建测试公司集合（可见其中他人私库）。
        tenant_wide: 企业/知识库管理员可跨部门预览本公司的部门库与公司库文档。
                     **他人个人库文档对他们同样不可预览**（审核共享申请时只
                     看申请单上的文件名/申请人/目标层级，不展示正文）。
        pred:        【T4】**可选**的 :class:`ScopePredicate`（决策 16：引用点击再校验）。
                    给出时：① 文档级先过一次 :func:`allows`（不通过 → KeyError 404）；
                    ② 返回前**逐 chunk** 查 ``document_objects`` 再 :func:`allows`，
                    被剔除的 chunk **不出现在返回列表里**（不返回占位符），并写
                    ``acl.drop.citation_open`` 审计。不给则行为**逐字不变**。

    Returns:
        {"document_id", "filename", "page_count", "total", "chunks": [...]}
        where each chunk is {"chunk_index", "page_number", "text"}.
    """
    # ── 1. Metadata (404 without existence leak) ──────────────────────────────
    async with get_db_session() as session:
        query = select(Document).where(Document.id == document_id)
        from app.services.tenancy import document_scope_clause

        if (
            owner_id is not None
            or tenant_ids is not None
            or tenant_wide
            or owns_tenant_ids
        ):
            # 三层隔离：公司边界 ∪ 自己的个人库（唯一 SQL 组装点，与检索/列表同规则）
            query = query.where(
                document_scope_clause(
                    owner_id=owner_id,
                    department_id=user_department_id,
                    tenant_ids=tenant_ids,
                    owns_tenant_ids=owns_tenant_ids,
                    tenant_wide=tenant_wide,
                )
            )
        doc: Document | None = (await session.execute(query)).scalar_one_or_none()

    if doc is None:
        raise KeyError("文档不存在或无权访问")

    # ── 1.5【T4】文档级五维判定（密级 / 项目 / deny / excluded 的对象级复核前置）──
    # 三层（tenant/department/private）已由上面的 document_scope_clause 判定；
    # 这里补上新增的两个维度，保证"文档级不可见"也返回与"不存在"逐字一致的 404。
    if pred is not None:
        from app.services.security_cascade import document_view
        from app.services.security_policy import allows

        if not allows(pred, document_view(doc)).allowed:
            raise KeyError("文档不存在或无权访问")

    # ── 2. Scroll all chunks from Qdrant ──────────────────────────────────────
    from qdrant_client.http import models as qmodels

    from app.config import get_settings
    from app.db.qdrant import get_qdrant_client

    settings = get_settings()
    client = get_qdrant_client()
    scroll_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="document_id",
                match=qmodels.MatchValue(value=str(document_id)),
            )
        ]
    )

    raw: list[tuple] = []
    offset = None
    while True:
        points, next_offset = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            try:
                chunk_index = int(payload.get("chunk_index") or 0)
            except (TypeError, ValueError):
                chunk_index = 0
            try:
                page_number = int(payload.get("page_number") or 1)
            except (TypeError, ValueError):
                page_number = 1
            raw.append((
                str(p.id),
                chunk_index,
                page_number,
                str(payload.get("text", "")),
                _opt_int(payload.get("line_start")),
                _opt_int(payload.get("line_end")),
                # 图片信息：原文预览也能标出"这一段来自第几张图"
                str(payload.get("content_type") or "text"),
                payload.get("image_id"),
                payload.get("image_path"),
                _opt_int(payload.get("position")),
                _opt_bbox(payload.get("bbox")),
                # 产出质检 + 双通道融合（原文预览也标出"这条可不可信"）
                _opt_dict(payload.get("analyze_quality")),
                _opt_dict(payload.get("analyze_fusion")),
            ))
        if next_offset is None:
            break
        offset = next_offset

    # 阅读顺序按 chunk_index（元组里第 2 个字段；point_id 在第 1 位供 ACL 反查）
    raw.sort(key=lambda t: t[1])
    chunks = [
        {
            "chunk_index": ci,
            "page_number": pn,
            "text": text,
            # 位置信息：原文预览同样能标出"这一段在第几行"
            "line_start": ls,
            "line_end": le,
            "content_type": ct,
            "image_id": iid,
            "image_path": ipath,
            # 图片位置：文档内序号 + 页面边界框
            "position": pos,
            "bbox": bbox,
            # 产出质检 + 双通道融合（可验证的事实）
            "analyze_quality": aq,
            "analyze_fusion": af,
        }
        for _pid, ci, pn, text, ls, le, ct, iid, ipath, pos, bbox, aq, af in raw
    ]

    # ── 3.【T4】逐 chunk 对象级再校验（决策 16）───────────────────────────────
    if pred is not None:
        chunks = await _filter_citation_chunks(document_id, pred, raw)

    logger.info(
        "get_document_chunks id=%s filename='%s' → %d chunks",
        document_id, doc.filename, len(chunks),
    )

    return {
        "document_id": str(document_id),
        "filename": doc.filename,
        "page_count": doc.page_count,
        "total": len(chunks),
        "chunks": chunks,
    }


async def _filter_citation_chunks(
    document_id: uuid.UUID, pred, raw: list[tuple]
) -> list[dict]:
    """
    引用点击回源时的**逐 chunk**对象级过滤（第 12 环同源的 ``allows``）.

    - 每个 chunk 的 ``object_id`` 由 :func:`make_object_id` 从
      ``(document_id, qdrant_point_id)`` 构造，与入库物化时**逐字一致**。
    - 被剔除的 chunk **不出现在返回列表里**（不返回占位符 —— 占位符等于承认
      "这里有东西被藏了"，本身就是存在性信息）。
    - 被剔除的对象写一条 ``acl.drop.citation_open`` 审计。
    - **回退口径**：文档**从未物化**（对象行数为 0）时，逐块判定无据可依 ——
      此时按文档级判定兜底（已经过上层 :func:`allows` + ``document_scope_clause``），
      chunks 原样返回，避免回填上线前所有存量文档的原文预览集体变空（功能退化）。
      文档一旦物化，缺失任一对象行即 **fail-closed 剔除**。
    """
    from app.db.security_models import (
        OBJECT_TYPE_IMAGE,
        make_object_id,
        object_type_from_content_type,
    )
    from app.services.security_cascade import (
        document_is_materialized,
        load_object_views,
    )
    from app.services.security_policy import allows

    materialized = await document_is_materialized(document_id)
    if not materialized:
        return [
            {
                "chunk_index": ci,
                "page_number": pn,
                "text": text,
                "line_start": ls,
                "line_end": le,
                "content_type": ct,
                "image_id": iid,
                "image_path": ipath,
                "position": pos,
                "bbox": bbox,
                "analyze_quality": aq,
                "analyze_fusion": af,
            }
            for _pid, ci, pn, text, ls, le, ct, iid, ipath, pos, bbox, aq, af in raw
        ]

    def _object_id(pid: str, content_type: str, image_id: str | None) -> str:
        ctype = (content_type or "text").lower()
        if ctype == "image" and image_id:
            return make_object_id(str(document_id), pid, object_type=OBJECT_TYPE_IMAGE)
        return make_object_id(
            str(document_id), pid, object_type=object_type_from_content_type(ctype)
        )

    object_ids = [
        _object_id(pid, ct, iid) for pid, ci, pn, text, ls, le, ct, iid, ipath, pos, bbox, aq, af in raw
    ]
    views = await load_object_views(object_ids)

    kept: list[dict] = []
    dropped: list[tuple[str, object]] = []
    for row, object_id in zip(raw, object_ids):
        pid, ci, pn, text, ls, le, ct, iid, ipath, pos, bbox, aq, af = row
        view = views.get(object_id)
        if view is None or not allows(pred, view).allowed:
            decision = allows(pred, view) if view is not None else None
            dropped.append((object_id, decision))
            continue
        kept.append({
            "chunk_index": ci,
            "page_number": pn,
            "text": text,
            "line_start": ls,
            "line_end": le,
            "content_type": ct,
            "image_id": iid,
            "image_path": ipath,
            "position": pos,
            "bbox": bbox,
            "analyze_quality": aq,
            "analyze_fusion": af,
        })

    if dropped:
        try:
            from app.services.audit_service import record_acl_drop

            for object_id, decision in dropped:
                await record_acl_drop(
                    "citation_open",
                    object_id=object_id,
                    document_id=str(document_id),
                    reason=(decision.reason if decision is not None else "missing_object_view"),
                    gate=(decision.gate if decision is not None else "tenant"),
                )
        except Exception:      # noqa: BLE001 — 审计失败不阻断
            logger.warning("citation_open audit failed (id=%s)", document_id, exc_info=True)

    return kept

