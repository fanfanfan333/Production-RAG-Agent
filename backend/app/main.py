"""
FastAPI application entry point.

Responsibilities:
  - Bootstrap logging
  - Register lifespan (startup / shutdown hooks)
  - Add middleware (CORS, request-ID)
  - Mount all API routers
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.admin_users import router as admin_users_router
from app.api.auth import router as auth_router
from app.api.badcases import router as badcases_router
from app.api.collections import router as collections_router
from app.api.companies import router as companies_router
from app.api.conversations import router as conversations_router
from app.api.document_management import router as document_management_router
from app.api.documents import router as documents_router
from app.api.evaluation import router as evaluation_router
from app.api.feedback import router as feedback_router
from app.api.health import router as health_router
from app.api.kb_collections import router as kb_collections_router
from app.api.quality import router as quality_router
from app.api.query import router as query_router
from app.api.share_requests import router as share_requests_router
from app.api.staff import router as staff_router
from app.config import RERANK_MIN_SCORE_RATIO_CEILING, get_settings
from app.middleware.gateway import gateway_middleware
from app.db import models  # noqa: F401 — registers Phase 2 tables with Base.metadata
from app.db import conversation_models  # noqa: F401 — registers Phase 4 tables
from app.db import feedback_models  # noqa: F401 — registers answer_feedback table
from app.db import badcase_models  # noqa: F401 — registers bad_cases table (回流队列)
from app.db import quality_models  # noqa: F401 — registers quality_events table (质量监控)
from app.db import eval_models  # noqa: F401 — registers eval_runs table (评测基线，跨重启保留)
from app.db import user_models  # noqa: F401 — registers users/collections/audit_logs
from app.db import share_models  # noqa: F401 — registers share_requests (三层知识库)
from app.db import staff_models  # noqa: F401 — registers staff_requests (企业身份验证)
from app.db import company_models  # noqa: F401 — registers companies (公司注册表)
from app.db.postgres import dispose_engine, get_engine
from app.db.postgres import Base
from app.db.qdrant import close_qdrant_client
from app.services.vector_service import ensure_collection
from app.utils.logging import get_logger, setup_logging

# ── Bootstrap logging immediately (before any other import that logs) ─────────
_settings = get_settings()
setup_logging(level=_settings.LOG_LEVEL, environment=_settings.ENVIRONMENT)  # type: ignore[arg-type]
logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(
        "Starting %s v%s [%s]",
        _settings.APP_NAME,
        _settings.APP_VERSION,
        _settings.ENVIRONMENT,
    )

    if _settings.JWT_SECRET.startswith("dev-insecure-secret-change-me"):
        logger.warning(
            "JWT_SECRET is still the default development value — "
            "SET A STRONG SECRET before any real deployment!"
        )
    if _settings.ENVIRONMENT == "production":
        if "*" in _settings.CORS_ORIGINS:
            logger.error("Production CORS_ORIGINS must not contain '*'")
        if not _settings.GATEWAY_ENFORCE_ORIGIN:
            logger.warning("Production gateway Origin enforcement is disabled")
        if not _settings.TRUSTED_PROXY_IPS:
            logger.info("No trusted proxy configured; X-Forwarded-For is ignored")

    # ── 检索参数护栏（fail-loud，而不是"看着正常其实错了"）────────────────────
    for level, message in check_retrieval_config_guards(_settings):
        getattr(logger, level)(message)

    # ── Startup: create PostgreSQL tables ────────────────────────────────────
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("PostgreSQL tables verified / created")

    # ── Startup: ensure Qdrant collection exists ──────────────────────────────
    await ensure_collection()

    # ── Startup: recover stuck documents ──────────────────────────────────────
    from app.services.document_service import recover_stuck_documents
    await recover_stuck_documents()

    # ── Startup: probe the multimodal (vision) model（部分4）─────────────────
    # "将 Vision 打开，如果没有就先放弃 vision"：这里做一次性可用性探测并
    # 打印明确结论 —— 模型没拉取时不会报错，多模态环节自动降级，
    # 图片仍作为独立对象入库与回显。
    try:
        from app.services.vision import get_vision_service

        vision_ok = await get_vision_service().is_available(refresh=True)
        if vision_ok:
            logger.info(
                "Vision enabled: model=%s", _settings.OLLAMA_VISION_MODEL
            )
        elif _settings.VISION_ENABLED and _settings.OLLAMA_VISION_MODEL:
            logger.warning(
                "VISION_ENABLED=true but model '%s' is unavailable — "
                "image understanding / image Q&A are DISABLED (images are still "
                "indexed & returned as originals). Run: ollama pull %s",
                _settings.OLLAMA_VISION_MODEL,
                _settings.OLLAMA_VISION_MODEL,
            )
        else:
            logger.info("Vision is disabled by configuration (VISION_ENABLED=false)")
    except Exception as exc:      # noqa: BLE001 — never block startup on vision
        logger.warning("Vision startup probe failed: %s", exc)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing resources …")
    await dispose_engine()
    await close_qdrant_client()
    logger.info("Shutdown complete")


# ── 检索参数护栏 ──────────────────────────────────────────────────────────────
#
# 这一组检查存在的理由：它要拦的这类错误**不会让任何请求失败**，只会让检索
# 静默地少给证据 / 误杀合法证据 —— 表象是"模型答得不全"，根因却在配置。
#
# 而配置有两个入口：代码默认值、环境变量；后者会**静默覆盖**前者且不留痕
# （pydantic-settings 的 env > default）。本次实测就踩到了：`backend/.env`
# 里残留的 `RERANK_MIN_SCORE=0.25` 会被 `docker compose` 原样注入容器
# （`docker compose config` 已证实），一个 `up -d` 就能让上一轮修好的 bug
# 原样复活，而 diff 看起来是干净的。
#
# 抽成纯函数而不是内联在 lifespan 里，是为了**可测**：启动路径本身不好在单测
# 里跑，而"配错了要报出来"这件事必须被钉住，否则护栏会变成注释。
def check_retrieval_config_guards(settings) -> list[tuple[str, str]]:
    """
    检查检索链路的配置一致性，返回 ``[(日志级别, 消息)]``（健康时为空列表）.

    级别用 "error" 而不是 "warning"：这两个值配错都不会抛异常，只会让回答
    悄悄退化成"只答一半"或"答没有"。既然故障是静默的，护栏就不能也是静的。
    """
    problems: list[tuple[str, str]] = []

    ratio = float(settings.RERANK_MIN_SCORE_RATIO)
    if ratio >= RERANK_MIN_SCORE_RATIO_CEILING:
        problems.append((
            "error",
            f"RERANK_MIN_SCORE_RATIO={ratio} 已达到/超过实测安全上界 "
            f"{RERANK_MIN_SCORE_RATIO_CEILING} —— 多证据用例中的次要证据会被"
            "相对分带砍掉（答案只答一半，且不报错）。请降到上界以下，或按当前"
            "语料/精排模型重新测量上界（见 backend/eval/golden_v1.json 的 "
            "band_headroom_measured，并在容器内跑 scripts/run_eval_baseline.py 复测）。"
        ))

    if float(settings.RERANK_MIN_SCORE) != float(settings.EVIDENCE_GATE_MIN_TOP_SCORE):
        problems.append((
            "error",
            "两道证据闸门的分数下限不一致："
            f"RERANK_MIN_SCORE={settings.RERANK_MIN_SCORE} vs "
            f"EVIDENCE_GATE_MIN_TOP_SCORE={settings.EVIDENCE_GATE_MIN_TOP_SCORE} "
            "—— 只要有一个停在旧值，更严的那个就会继续把合格证据判为不合格，"
            "修复等于没做（历史上这两个值都曾停在 0.25，导致金标分片被整体"
            "丢弃 → 确定性地答错「库里没有」）。"
        ))

    return problems


# ── Application factory ───────────────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Production-ready RAG Agent API",
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/openapi.json" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Application-edge gateway ──────────────────────────────────────────────
    # Correlation IDs, request-size limits, route-class rate limits, production
    # Origin enforcement and security response headers are centralized here.
    app.middleware("http")(gateway_middleware)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router)            # Phase 1 — public (liveness)
    app.include_router(auth_router)              # Phase 5 — public (login/register)
    app.include_router(documents_router)         # Phase 2 — POST /upload (auth)
    app.include_router(document_management_router)  # Phase 3 — GET/DELETE /documents (auth)
    app.include_router(kb_collections_router)    # Phase 5 — /kb/collections CRUD (auth)
    app.include_router(collections_router)       # Phase 3 — Qdrant admin API (admin)
    app.include_router(query_router)             # Phase 4 — POST /query SSE (auth)
    app.include_router(conversations_router)     # Phase 4 — conversation history (auth)
    app.include_router(feedback_router)          # 反馈闭环 — POST/GET /feedback (auth/admin)
    app.include_router(badcases_router)          # 持续监控 — /badcases 审阅队列 + 指标
    app.include_router(quality_router)           # 质量监控 — /quality/stats 时间窗聚合
    app.include_router(evaluation_router)        # 质量评测 — /eval Recall/MRR/NDCG/引用准确率
    app.include_router(admin_users_router)       # RBAC — platform user/role administration
    app.include_router(share_requests_router)    # 三层知识库 — 申请共享 / 查看申请 / 审核
    app.include_router(staff_router)             # 企业身份 — 身份验证申请 / 层级审核 / 成员管理
    app.include_router(companies_router)         # 公司注册表 — /companies 清单 / 创建 / 改名 / 可访问

    return app


app = create_app()
