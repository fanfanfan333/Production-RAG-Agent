"""
Document upload API router (Phase 2 / 企业落地第一阶段).

POST /upload  — Accept 1–N files, run the ingestion pipeline,
                return per-file status.

Authentication required. Uploaded documents are attributed to the current
user and, when provided, grouped into one of their knowledge-base collections.
"""

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.services.permissions import require_permission
from app.config import get_settings
from app.db.models import DocumentStatus
from app.db.user_models import User
from app.schemas.document import DocumentResult, UploadResponse
from app.services.audit_service import record_audit
from app.services.document_service import prepare_upload, schedule_ingestion
from app.services.tenancy import effective_department_id, effective_tenant_id
from app.utils.file_utils import validate_document_upload
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Accept documents for asynchronous ingestion",
    description=(
        "Accept one or more files and return immediately.\n\n"
        "The request only performs the fast work:\n"
        "1. Validated (size limit and supported format)\n"
        "2. Fingerprinted (SHA-256) and de-duplicated against the caller's own "
        "documents\n"
        "3. A `pending` document row is created\n\n"
        "The heavy pipeline — parser, per-image OCR / table recognition / vision, "
        "recursive chunking, BGE embedding, Qdrant + PostgreSQL writes — runs as a "
        "background task. A 13 MB document takes 7+ minutes, so holding the HTTP "
        "request open for it would be wrong: closing the tab would silently discard "
        "the work and intermediate proxies would cut the connection.\n\n"
        "Track progress with `GET /documents` (`status` + `current_stage` + "
        "`embedded_chunks` / `total_chunks`). Terminal states are `completed`, "
        "`failed` and `already_exists`.\n\n"
        "Optionally pass `collection_id` (form field) to group the upload into "
        "one of the caller's knowledge-base collections.\n\n"
        "Returns 202 with a per-file entry: `already_exists` when the identical "
        "file was indexed before, otherwise `pending`."
    ),
)
async def upload_documents(
    files: list[UploadFile] = File(
        ...,
        description="One or more files (max 50 MB each).",
    ),
    collection_id: uuid.UUID | None = Form(
        None,
        description="Optional knowledge-base collection to group the upload into.",
    ),
    access_level: str | None = Form(
        None,
        description=(
            "Document ACL（第二层隔离）：private=仅本人（默认按系统配置）/ "
            "department=同部门 / tenant=全租户共享。"
        ),
    ),
    company_id: str | None = Form(
        None,
        description=(
            "文档归属的公司 tenant_id（平台管理员**必填**，且限其自建测试公司；"
            "其余角色忽略，恒归属本公司）。"
        ),
    ),
    user: User = Depends(require_permission("document.write")),
) -> UploadResponse:
    settings = get_settings()

    # ── Guard: file count ─────────────────────────────────────────────────────
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file must be provided.",
        )

    if len(files) > settings.MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Too many files. Maximum {settings.MAX_FILES_PER_UPLOAD} "
                f"files per request, received {len(files)}."
            ),
        )

    # ── Guard: collection belongs to the caller ───────────────────────────────
    if collection_id is not None:
        from app.db.postgres import get_db_session
        from app.db.user_models import Collection

        async with get_db_session() as session:
            col = await session.get(Collection, collection_id)
        if col is None or col.owner_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="集合不存在或不属于当前用户",
            )

    # ── Validate and buffer all files before starting the pipeline ────────────
    # This ensures we reject invalid uploads immediately, before any DB writes.
    file_payloads: list[tuple[str, bytes]] = []

    for upload in files:
        raw_filename = upload.filename or "unknown.ext"
        # Sanitize against path traversal (handles both / and \ regardless of OS)
        filename = raw_filename.replace("\\", "/").split("/")[-1]
        logger.info("Received upload: '%s' (content-type=%s)", filename, upload.content_type)

        try:
            content = await validate_document_upload(upload, settings)
        except HTTPException:
            raise  # propagate validation errors as-is
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not read file '{filename}': {exc}",
            ) from exc

        file_payloads.append((filename, content))

    # ── 三层知识库：上传层级必须过权限矩阵 ────────────────────────────────────
    # 以前 access_level 是"表单说了算"的：普通员工上传时带上 tenant 就能把
    # 文档直接发布到公司知识库，整张权限矩阵被绕过。这里按同一份矩阵校验，
    # 未授权的层级一律 403 并提示改用「申请共享」。
    from app.services.permissions import has_permission
    from app.services.tenancy import (
        ACCESS_DEPARTMENT,
        ACCESS_TENANT,
        DEFAULT_DOCUMENT_ACCESS_LEVEL,
        access_scope_name,
        is_platform_admin,
        normalize_access_level,
        publish_requirement,
    )

    is_admin = is_platform_admin(user)

    # ── 决策 6：平台管理员上传**必须**指定归属的测试公司 ─────────────────────
    # admin 的 effective_tenant_id 是 "default"（它不属于任何公司），若沿用会把
    # 文档落进一个它自己都看不到的租户。因此 admin 必须显式选一个**自建测试公司**，
    # 文档 tenant_id = 该公司 id。非 admin 一律归属本公司（company_id 被忽略）。
    upload_tenant_id = effective_tenant_id(user)
    if is_admin:
        from app.services.company_registry import tenant_ids_created_by

        owned = await tenant_ids_created_by(user.id)
        if not owned:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="暂无测试公司，请先创建公司",
            )
        chosen = (company_id or "").strip()
        if not chosen:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="请选择归属的测试公司",
            )
        if chosen not in owned:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="只能上传到你**自己创建**的测试公司",
            )
        upload_tenant_id = chosen

    if access_level is None or not str(access_level).strip():
        # admin 默认落「公司库」（测试公司内成员才检索得到素材）；其余角色沿用
        # 系统默认（个人库）。两者都只在上传者未显式指定层级时生效。
        default_level = (
            ACCESS_TENANT if is_admin else DEFAULT_DOCUMENT_ACCESS_LEVEL
        )
        target_level = normalize_access_level(default_level)
    else:
        raw_level = str(access_level).strip().lower()
        if raw_level not in {"private", "department", "tenant"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="知识库层级不合法：只能是 private（个人）/ department（部门）/ tenant（公司）",
            )
        target_level = raw_level

    required_permission = publish_requirement(target_level)
    if required_permission and not has_permission(user, required_permission):
        needs = (
            "部门负责人"
            if target_level == ACCESS_DEPARTMENT
            else "知识库管理员"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"上传到{access_scope_name(target_level)}需要「{needs}」及以上权限。"
                "你可以先上传到个人知识库，再在文档上提交「申请共享」。"
            ),
        )

    # 部门库必须落在上传者的部门里；个人库 / 公司库不需要部门归属
    upload_department_id = (
        effective_department_id(user) if target_level == ACCESS_DEPARTMENT else None
    )

    # ── 快速受理：判重 + 建行（毫秒级），重活丢后台 ───────────────────────────
    # 以前这里 await 整条管线，一份 13 MB 文档要把 HTTP 连接挂住 7 分钟以上；
    # 用户关掉页面就等于什么都没发生，代理/网关也容易在途中把连接掐断。现在
    # 只做"判重 + 落 PENDING 行"，随后立即返回 202，进度由 GET /documents 暴露
    # 的 current_stage 驱动前端轮询（前端本就在轮询处理中的文档）。
    logger.info(
        "Accepted upload for %d file(s) user=%s collection=%s level=%s",
        len(file_payloads), user.username, collection_id, target_level,
    )

    results: list[DocumentResult] = []
    for filename, content in file_payloads:
        prep = await prepare_upload(
            filename,
            content,
            owner_id=user.id,
            collection_id=collection_id,
            # 三层隔离：非 admin 恒归属本公司；admin 归属其选定的自建测试公司
            # （决策 6，见上方 upload_tenant_id）；层级由权限矩阵裁决后的
            # target_level 决定。
            tenant_id=upload_tenant_id,
            department_id=upload_department_id,
            access_level=target_level,
        )
        if prep.done:
            results.append(prep.result)  # type: ignore[arg-type]  # 已存在 / 正在处理中
            continue
        results.append(
            schedule_ingestion(
                prep,
                filename,
                content,
                owner_id=user.id,
                collection_id=collection_id,
            )
        )

    accepted = sum(1 for r in results if r.status == DocumentStatus.PENDING)
    response = UploadResponse(
        total=len(results),
        # 语义：succeeded 现在指"已受理、会在后台跑完"，不再是"已入库完成"。
        # 真实入库结果通过 GET /documents 的 status/current_stage 回看。
        succeeded=accepted,
        failed=0,
        documents=results,
    )

    logger.info(
        "Upload accepted — total=%d scheduled=%d deduplicated=%d",
        response.total, accepted, response.total - accepted,
    )

    await record_audit(
        "document.upload",
        user_id=user.id,
        username=user.username,
        resource_type="document",
        resource_id=",".join(str(r.document_id) for r in response.documents)[:2000],
        detail=(
            f"files={[f for f, _ in file_payloads]}; "
            f"accepted={accepted}; deduplicated={response.total - accepted}"
        ),
    )

    return response
