"""
Application configuration using pydantic-settings.
All values are loaded from environment variables (or .env file).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    APP_NAME: str = "RAG Agent API"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"         # development | staging | production
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Ollama (chat / generation only) ───────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:8b"                    # chat / generation model
    OLLAMA_NUM_CTX: int = 8192                        # context window passed to llama-server.
                                                     # qwen3 Ollama 默认 num_ctx=40960，KV cache
                                                     # + compute buffer 需 ~1.5GB+ 常驻内存，小显存/
                                                     # 小内存机器会 OOM（llama-server 启动分配失败）。
                                                     # RAG 场景 8k 足够（检索上下文+历史），可下调
                                                     # 至 4096 进一步省内存。

    # ── Document relation analysis (POST /query mode="doc_relations") ──────────
    RELATION_MAX_DOCUMENTS: int = 12          # max documents included in one analysis
    RELATION_CHUNKS_PER_DOC: int = 8          # chunks sampled per document for its digest
    RELATION_DIGEST_CHARS: int = 1200         # max chars of digest text per document

    # ── BGE (local embedding model, replaces Gemini embedding) ─────────────────
    BGE_MODEL_NAME: str = "BAAI/bge-large-zh-v1.5"    # 1024-dim, good quality/speed balance
    EMBEDDING_DIMENSION: int = 1024           # MUST match BGE_MODEL_NAME's output dim:
                                               #   bge-small-zh-v1.5 -> 512
                                               #   bge-base-zh-v1.5  -> 768
                                               #   bge-large-zh-v1.5 -> 1024
                                               #   bge-m3            -> 1024
                                               # Changing this requires recreating the
                                               # Qdrant collection and re-embedding all docs.
    EMBEDDING_BATCH_SIZE: int = 16         # chunks per embed API call (local BGE: no 429s)
    EMBEDDING_BATCH_SIZE_FLOOR: int = 2    # dynamic fallback minimum when a batch fails

    # ── Auth / multi-user (企业落地第一阶段) ─────────────────────────────────────
    # CHANGE THIS IN PRODUCTION — startup logs a warning while the default is used.
    # (Default is a 64-char placeholder to satisfy the 32-byte HS256 minimum;
    #  it is still public knowledge and must be replaced.)
    JWT_SECRET: str = "dev-insecure-secret-change-me-0123456789abcdef0123456789abcdef"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 720                    # 12 h session lifetime
    ALLOW_SELF_REGISTRATION: bool = True             # first user always becomes admin
    PASSWORD_MIN_LENGTH: int = 8

    # ── Rate limiting (simple in-process sliding window) ───────────────────────
    QUERY_RATE_LIMIT: int = 20             # /query calls per user per window
    LOGIN_RATE_LIMIT: int = 10             # /auth/login attempts per IP per window
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── Prompt injection guard ─────────────────────────────────────────────────
    # Rule-based detection runs before query rewrite/retrieval. The guard blocks
    # direct attempts to override model policy, while audit logs retain a short
    # redacted diagnostic for security review.
    PROMPT_GUARD_ENABLED: bool = True
    PROMPT_GUARD_BLOCK_HIGH_RISK: bool = True
    PROMPT_GUARD_MAX_QUERY_CHARS: int = 4096

    # ── RAG retrieval ───────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5                # default chunks to retrieve per query
    RETRIEVAL_MIN_SCORE: float = 0.30        # minimum cosine similarity score threshold (bge-zh good matches score ~0.4-0.6)
    RETRIEVAL_MAX_GAP: float = 0.05          # maximum score difference from top score to keep a candidate
    MAX_HISTORY_PAIRS: int = 3             # conversation turns kept in context window

    # ── Hybrid retrieval (BM25 + vector RRF fusion, 召回质量优化) ────────────────
    HYBRID_SEARCH_ENABLED: bool = True      # fuse BM25 keyword hits with vector search
    HYBRID_MAX_CORPUS_POINTS: int = 10000   # BM25 corpus scan cap per cache build
    HYBRID_RRF_K: int = 60                  # reciprocal-rank-fusion smoothing constant
    HYBRID_CACHE_TTL_SECONDS: float = 300.0 # BM25 index cache lifetime per (collection, owner)

    # ── Query transformation (召回优化 / 抗幻觉第一道防线) ──────────────────────
    QUERY_REWRITE_ENABLED: bool = True        # 历史感知的指代消解改写
    QUERY_REWRITE_TIMEOUT_SECONDS: float = 12.0  # 改写超时——超时直接用原查询
    MULTI_QUERY_ENABLED: bool = True          # 多查询扩展（多路召回 + RRF 融合）
    MULTI_QUERY_VARIANTS: int = 2             # 每个问题生成的检索变体数

    # ── Reranker 精排（粗排→精排两阶段检索的第二阶段） ─────────────────────────
    RERANKER_ENABLED: bool = True             # cross-encoder 精排总开关
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"  # 默认轻量 cross-encoder
                                                     # （~278MB）。bge-reranker-v2-m3 效果更强
                                                     # 但占 ~2.3GB 常驻内存，与 Ollama 同机
                                                     # 部署时容易把 llama-server 挤到 OOM。
    RERANKER_USE_FP16: bool = False           # CUDA 环境可开 True 省一半显存
    RERANKER_MAX_CANDIDATES: int = 20        # 送入精排的粗排候选数上限

    # ── Hallucination guard（幻觉守卫：低置信度直接拒答，不硬答） ───────────────
    HALLUCINATION_GUARD_ENABLED: bool = True # 精排分低于阈值时短路拒答
    RERANK_MIN_SCORE: float = 0.25           # 精排置信度阈值（cross-encoder sigmoid 概率）

    # ── Enhanced chunking (token-aware + 句子滑窗 + 表格/代码块保护 + small-to-big) ──
    # 旧 API 完全向后兼容：新增参数全部带 default；缺省值 = 关闭新特性，沿用旧行为。
    CHUNK_TOKEN_AWARE: bool = True            # 按 token 数切（chars_per_token 估算 token 上限）
    CHARS_PER_TOKEN: float = 1.6              # 中英文混合经验值
    CHUNK_SENTENCE_OVERLAP: bool = True       # 重叠区按句子边界对齐
    CHUNK_PROTECT_BLOCKS: bool = True         # ```代码块``` 和 Markdown 表格不得从中间拆开
    CHUNK_PARENT_SIZE_MULT: float = 2.5       # small-to-big：父块 = 子块目标大小 × 此倍数
    CHUNK_PARENT_OVERLAP: int = 200           # 父块之间重叠字符数
    HIERARCHICAL_RAG_ENABLED: bool = True     # retrieval 时优先回填父块文本

    # ── LLM Query Router（架构图 Query Router 节点）─────────────────────────────
    ROUTER_ENABLED: bool = True               # 启用 LLM 路由；关闭时回退正则
    ROUTER_TIMEOUT_SECONDS: float = 8.0       # 路由 LLM 超时；超时即降级为 knowledge_qa
    ROUTER_USE_CACHE: bool = True             # 同 query 短时间内复用路由结果

    # ── Retrieval Grader（架构图 Retrieval Grader 节点）─────────────────────────
    RETRIEVAL_GRADER_ENABLED: bool = True     # LLM 证据质量评估开关
    RETRIEVAL_GRADER_TIMEOUT_SECONDS: float = 6.0
    RETRIEVAL_GRADER_MIN_RELEVANT: int = 1    # 至少 N 个 chunk 被判 relevant 才算 good

    # ── Retrieval retry loop（架构图 Retry/Rewrite 循环）────────────────────────
    RETRIEVAL_MAX_RETRIES: int = 2            # 检索失败（grader=bad）最大重试次数
    RETRIEVAL_RETRY_ADD_VARIANTS: bool = True # 重试时附带新增查询变体

    # ── Document Summary（架构图 Document Summary 分支）─────────────────────────
    DOC_SUMMARY_ENABLED: bool = True
    DOC_SUMMARY_MAX_CHARS_PER_DOC: int = 12000  # 单文档摘要输入上限
    DOC_SUMMARY_TIMEOUT_SECONDS: float = 60.0

    # ── General Chat（架构图 General Chat 分支）──────────────────────────────────
    GENERAL_CHAT_ENABLED: bool = True
    GENERAL_CHAT_SYSTEM_PROMPT_LANG: str = "zh"

    # ── Embedding Resiliency ──────────────────────────────────────────────────
    MAX_EMBED_RETRIES: int = 5
    INITIAL_BACKOFF: float = 1.0
    MAX_BACKOFF: float = 16.0
    ENABLE_JITTER: bool = True
    MAX_CONCURRENT_EMBEDDINGS: int = 1


    # ── PostgreSQL ─────────────────────────────────────────────────────────────
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "raguser"
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str = "ragdb"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def postgres_dsn_sync(self) -> str:
        """Sync DSN used for health-check ping only."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Qdrant ─────────────────────────────────────────────────────────────────
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None        # optional; required for Qdrant Cloud
    QDRANT_COLLECTION: str = "documents"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    # ── Upload limits ──────────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 50            # per-file size limit
    MAX_FILES_PER_UPLOAD: int = 10

    # ── PDF Chunking ────────────────────────────────────────────────────────────
    MIN_CHUNK_SIZE: int = 500              # target minimum chars per chunk
    MAX_CHUNK_SIZE: int = 2000             # max chars before recursive split
    CHUNK_OVERLAP: int = 200               # chars shared between consecutive chunks

    # ── Embedded-image recognition (问题3) ──────────────────────────────────────
    ENABLE_IMAGE_OCR: bool = True          # OCR images embedded inside PDF/DOCX/PPTX
    MAX_IMAGES_PER_DOCUMENT: int = 20      # hard cap per document (guards runaway docs)
    MAX_IMAGES_PER_PAGE: int = 6           # per-page cap for PDFs
    MIN_IMAGE_DIMENSION: int = 60          # skip icons / separators smaller than this (px)
    MAX_IMAGE_OCR_CHARS: int = 600         # truncate OCR text per image
    # Optional: describe images with a multimodal Ollama model (e.g. "qwen2.5vl:7b").
    # Empty string disables vision captioning — OCR-only recognition is the default.
    OLLAMA_VISION_MODEL: str = ""

    # ── API gateway / edge protection ─────────────────────────────────────────
    # In-process gateway policy. For multi-instance production, pair this with
    # Kong/APISIX/Envoy and a Redis-backed distributed limiter.
    GATEWAY_MAX_REQUEST_BYTES: int = 55 * 1024 * 1024  # allows 50 MB upload + multipart overhead
    GATEWAY_AUTH_RATE_LIMIT: int = 10
    GATEWAY_UPLOAD_RATE_LIMIT: int = 12
    GATEWAY_ADMIN_RATE_LIMIT: int = 60
    GATEWAY_ENFORCE_ORIGIN: bool = False              # set true in production
    TRUSTED_PROXY_IPS: list[str] = []                 # only these may set X-Forwarded-For

    # ── CORS ───────────────────────────────────────────────────────────────────
    # Never use "*" in production. Configure the exact frontend origin(s).
    CORS_ORIGINS: list[str] = ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of Settings."""
    return Settings()  # type: ignore[call-arg]