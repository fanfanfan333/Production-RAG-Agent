"""
Application configuration using pydantic-settings.
All values are loaded from environment variables (or .env file).
"""

import os
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

    # ── 出网代理豁免（企业网络下必备）────────────────────────────────────────────
    # 这里列出的主机在进程启动时被写进 ``NO_PROXY`` / ``no_proxy`` 环境变量。
    #
    # ⚠️ 为什么必须由代码写，而不能只在 .env 里写 NO_PROXY：
    #   1. ``httpx``（Ollama / qdrant-client / Keycloak JWKS 都走它）**只认 NO_PROXY
    #      环境变量**，不读 Windows 注册表的 ``ProxyOverride``。本机代理软件把
    #      ``localhost;127.*`` 写进了注册表例外，``urllib.proxy_bypass()`` 也认，
    #      但 httpx 认为不存在 → 发给本机/内网服务的请求全被送到代理，拿回
    #      **502 空响应体**（表现为 ``responseError('')``、"服务不可用"、
    #      "模型不稳定"，极难反查）。
    #   2. pydantic-settings 读 .env 只填 Settings 对象，**不会**导出到
    #      ``os.environ``；宿主机直接跑 uvicorn / pytest 时 .env 里的 NO_PROXY
    #      等于没写。所以由 get_settings() 统一物化（见 apply_no_proxy_env）。
    #
    # 语义：**只增不减** —— 运维/CI 已显式设置的条目一律保留，本字段只做补充。
    # 想整体关闭就把本字段设为空字符串。
    HTTP_NO_PROXY: str = (
        "localhost,127.0.0.1,::1,"
        "host.docker.internal,"
        "backend,postgres,qdrant,keycloak"
    )

    # ── Ollama (chat / generation only) ───────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:8b"                    # chat / generation model
    OLLAMA_NUM_CTX: int = 8192                        # context window passed to llama-server.
                                                     # qwen3 Ollama 默认 num_ctx=40960，KV cache
                                                     # + compute buffer 需 ~1.5GB+ 常驻内存，小显存/
                                                     # 小内存机器会 OOM（llama-server 启动分配失败）。
                                                     # RAG 场景 8k 足够（检索上下文+历史），可下调
                                                     # 至 4096 进一步省内存。
    OLLAMA_NUM_GPU: int | None = None                 # GPU offload 层数。None=不干预（全 GPU）；
                                                     # 小显存机器（如 6GB 卡装 qwen3:8b）设 0 强制
                                                     # CPU 推理，或设部分层数（如 20）混合 offload。
                                                     #
                                                     # ⚠️ 这个开关**必须逐个 ChatOllama 构造点显式传
                                                     # num_gpu=settings.OLLAMA_NUM_GPU 才生效** ——
                                                     # langchain_ollama 不读同名环境变量，漏传即回退
                                                     # 到"全 GPU"。本项目 7 个构造点曾只有 2 处接了，
                                                     # 于是"设了 OLLAMA_NUM_GPU=0 却仍然崩"（崩的正是
                                                     # 没接线的文档总结链路）。**新增 LLM 构造处务必
                                                     # 一并传**，改完可用下面这行自检：
                                                     #   grep -rn "ChatOllama(" app | wc -l
                                                     #   grep -rn "num_gpu=settings.OLLAMA_NUM_GPU" app | wc -l
                                                     # 两数必须相等。

    # ── Document relation analysis (POST /query mode="doc_relations") ──────────
    RELATION_MAX_DOCUMENTS: int = 12          # max documents included in one analysis
    RELATION_CHUNKS_PER_DOC: int = 8          # chunks sampled per document for its digest
    RELATION_DIGEST_CHARS: int = 1200         # max chars of digest text per document

    # ── BGE (local embedding model) ───────────────────────────────────────────
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
    # FIX-B（T5 预发布）：弱密钥 escape 开关。默认 False —— 当 JWT_SECRET 是默认值 /
    # 已知弱值 / 长度 < 32 时，启动**直接失败**（fail-closed，杜绝自签 admin token）。
    # 仅在本地联调显式置 true（配合醒目 ERROR 日志）。生产部署应改用 ≥32 字符强密钥，
    # 而非开启此开关。
    ALLOW_INSECURE_JWT: bool = False
    ALLOW_SELF_REGISTRATION: bool = True             # first user always becomes admin
    PASSWORD_MIN_LENGTH: int = 8

    # ── Keycloak / OIDC（企业统一身份）──────────────────────────────────────────
    # 身份链路：前端 → Keycloak 登录 → JWT → FastAPI 验签 → 身份信息
    #          (tenant_id/department/roles) → Permission Layer → Retriever。
    #
    # KEYCLOAK_ENABLED=false 时后端只认本地 HS256 令牌（离线开发/单机部署），
    # 打开后同时接受 Keycloak 的 RS256 令牌 —— 两条通路并存，互不影响。
    KEYCLOAK_ENABLED: bool = False
    # 服务端到 Keycloak 的地址（容器网络内用服务名，如 http://keycloak:8080）；
    # 用于拉取 JWKS 公钥集合。
    KEYCLOAK_URL: str = "http://keycloak:8080"
    # 浏览器访问 Keycloak 的地址（前端 PKCE 登录、前端展示用）。
    # 与本机端口映射不一致（内网名 / 外网名）时务必分开配置。
    KEYCLOAK_PUBLIC_URL: str = "http://localhost:8080"
    KEYCLOAK_REALM: str = "rag"
    KEYCLOAK_CLIENT_ID: str = "rag-web"
    KEYCLOAK_AUDIENCE: str = ""              # 留空 = 不校验 aud（Keycloak 默认 aud=account）
    KEYCLOAK_VERIFY_AUDIENCE: bool = False
    KEYCLOAK_VERIFY_ISSUER: bool = True
    KEYCLOAK_JWKS_CACHE_SECONDS: float = 600.0
    # 承载租户/部门的 claim 名（可用 Keycloak 的 User Attribute mapper 输出）
    KEYCLOAK_TENANT_CLAIM: str = "tenant_id"
    KEYCLOAK_DEPARTMENT_CLAIM: str = "department"
    # 首次用 Keycloak 登录时自动建档；关闭则只允许已开通账号登录
    KEYCLOAK_AUTO_PROVISION: bool = True
    # 是否允许把已存在的本地同名账号绑定到 Keycloak 身份
    # （方便"先本地注册的管理员，后接入 Keycloak"的平滑迁移）
    KEYCLOAK_LINK_EXISTING_USERS: bool = True
    # token 里没有任何已知角色时使用的兜底角色
    KEYCLOAK_DEFAULT_ROLE: str = "employee"
    # 是否保留本地账号/密码登录（Keycloak 上线后可以关掉，只留 SSO）
    ALLOW_LOCAL_LOGIN: bool = True

    # ── 企业身份验证闸门 ───────────────────────────────────────────────────────
    # True：通过「企业身份验证」审核之前，账号只能登录与提交验证申请，
    #       不能使用知识库业务（上传/提问/共享）—— 对应产品要求
    #       "没有注册和职责的不能进入"。
    # False：退回旧行为（任何登录账号都能使用全部功能），便于灰度或排障。
    REQUIRE_IDENTITY_VERIFICATION: bool = True

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

    # ── Output guard（架构图 Generate → Citation Check → END；问题3+问题4）──────
    # 在生成答案后做四层合规检查：引用越界 / 系统提示词泄露 / 闲聊分支幻觉措辞 /
    # Agent 工具调用意图。完全确定性、不调 LLM；净化后答案与审计信号落库。
    OUTPUT_GUARD_ENABLED: bool = True
    OUTPUT_GUARD_BLOCK_ON_LEAK: bool = False   # True 则把整段置为拒答；默认仅替换短语

    # ── Evidence Gate（架构图 Grader → Evidence Gate → Generate/Refuse）────────
    # Grader 是 LLM 语义判断（fail-open，超时即放行）；Evidence Gate 是它之后
    # 的**确定性 fail-closed 兜底**：条数 / 最高精排分 / 关键词覆盖率 / 证据长度
    # 任一不达标即拒答，不再把"形态上明显不够"的证据交给模型硬答。
    EVIDENCE_GATE_ENABLED: bool = True
    EVIDENCE_GATE_MIN_CHUNKS: int = 1            # 至少 N 条证据
    # 最高精排分下限 —— **必须与 RERANK_MIN_SCORE 保持一致**：两者是同一件
    # 事的两个执行点（前者在 rag_graph 的路由守卫，后者在证据门控节点）。只要
    # 有一个还停在旧值，更严的那个就会继续把合格证据判为不合格，修复等于没做。
    EVIDENCE_GATE_MIN_TOP_SCORE: float = 0.05    # 与 RERANK_MIN_SCORE 对齐（校准依据见上）
    # 关键词覆盖率是**独立于分数的词面佐证**：分数单独放行时，它负责拦住
    # "高分但与问题毫不相关"的证据。因此分数下限放宽后，拒答能力并没有丢 ——
    # 库里确实没有的查询，覆盖率同样过不去。
    EVIDENCE_GATE_MIN_COVERAGE: float = 0.20     # 问题关键词在证据中的覆盖率下限
    # 证据正文总长度下限 —— 这是一道**空壳检查**（拦"检索返回空/垃圾 payload"），
    # 阈值刻意很低：设高会误杀合法的短结构化块（一张 40 字的小表格本身就能
    # 回答"营收是多少"），证据质量已由分数与覆盖率两个信号把关。
    EVIDENCE_GATE_MIN_CHARS: int = 20

    # ── Citation Verifier（架构图 Generate → Citation Verifier → Output Guard）─
    # 逐条校验五项：引用存在 / 引用位置正确 / 原文支持该结论 / 数字一致 / 日期一致。
    # 纯确定性文本比对，不调 LLM。阈值偏保守 —— 宁可漏判，不可错杀。
    CITATION_VERIFIER_ENABLED: bool = True
    CITATION_MIN_SUPPORT: float = 0.30            # 句子内容词在被引原文中的覆盖率下限
    CITATION_MISATTRIBUTION_MARGIN: float = 0.25  # 别条来源高出多少才判"引用位置错误"
    CITATION_STRIP_UNSUPPORTED: bool = True       # 移除不被原文支持的引用标记
    CITATION_ANNOTATE: bool = True                # 末尾追加一行"引用校验"说明
    # ── 命中句回标（引用卡片"主要是哪几句"）────────────────────────────────
    # 引用卡片此前只给整块切片的行范围（"第 83-105 行"）+ 整段原文，用户看得
    # 到 23 行原文却不知道答案实际用了哪几句。开启后逐句回标命中句（连同
    # 命中句自己的行号），卡片默认只列命中句。
    CITATION_EVIDENCE_HIGHLIGHT: bool = True
    CITATION_EVIDENCE_MIN_RATIO: float = 0.34     # 命中句判定：与答案句的内容词重合率下限
    CITATION_EVIDENCE_MAX_SENTENCES: int = 3      # 每条引用最多回标几句（满屏高亮等于没有高亮）

    # ── 持续监控 + Bad Case 回流 ─────────────────────────────────────────────
    # 运行期指标（证据门控拒答率 / 引用校验通过率 / 输出净化命中率 / 时延）
    # + 自动 Bad Case 回流（引用不支持、门控拒答、输出净化命中自动入库）。
    RAG_MONITORING_ENABLED: bool = True
    BADCASE_AUTO_CAPTURE: bool = True             # 自动把可疑问答写入 Bad Case 队列
    BADCASE_AUTO_CAPTURE_LIMIT: int = 20000       # 队列行数软上限（超出时仅记指标）

    # ── RAG retrieval ───────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5                # default chunks to retrieve per query
    RETRIEVAL_MIN_SCORE: float = 0.30        # minimum cosine similarity score threshold (bge-zh good matches score ~0.4-0.6)
    RETRIEVAL_MAX_GAP: float = 0.05          # maximum score difference from top score to keep a candidate
    # 向量腿过取倍数（召回率防御）。
    #
    # 检索是"先 ANN 取候选、再按 PG 可见性/状态过滤"。两道过滤的口径并不完全一致
    # —— Qdrant 前置过滤是刻意 fail-open 的（老向量 payload 缺字段就不排除，交给
    # PG 兜底），加上"已删除但向量还在"的孤儿点，都会让**取回的候选被大量丢掉**。
    # 实测本仓库（186 个孤儿 document_id / 214 点）：平台管理员 ANN top-20 里只有
    # 4 条属于现存文档，候选池 80% 名额被吃掉 —— 表现为"库里明明有却搜不到"。
    #
    # 过取把"候选池 N 条"的语义从「取 N 条原始候选」纠正为「最终保留 N 条可用候选」，
    # 于是召回不再取决于脏数据比例。代价只是一次 ANN 多返回几十个点的 payload
    # （毫秒级），相比它换回的召回是划算的。
    # 置 1 关闭过取（退化为升级前行为，便于 A/B 对照）。
    ANN_OVERFETCH_FACTOR: int = 4
    ANN_OVERFETCH_MAX: int = 400             # 过取上限，防止异常配置把 ANN 打爆
    MAX_HISTORY_PAIRS: int = 3             # conversation turns kept in context window

    # ── Hybrid retrieval (BM25 + vector RRF fusion, 召回质量优化) ────────────────
    HYBRID_SEARCH_ENABLED: bool = True      # fuse BM25 keyword hits with vector search
    HYBRID_MAX_CORPUS_POINTS: int = 10000   # BM25 corpus scan cap per cache build
    HYBRID_RRF_K: int = 60                  # reciprocal-rank-fusion smoothing constant
    HYBRID_CACHE_TTL_SECONDS: float = 300.0 # BM25 index cache lifetime per (collection, owner)
    HYBRID_VECTOR_WEIGHT: float = 1.0       # 向量腿 RRF 权重（调大更偏语义相似）
    HYBRID_BM25_WEIGHT: float = 1.0         # BM25 腿 RRF 权重（调大更偏精确词/编号命中）

    # ── Query transformation (召回优化 / 抗幻觉第一道防线) ──────────────────────
    QUERY_REWRITE_ENABLED: bool = True        # 历史感知的指代消解改写
    # 关掉 qwen3 thinking 后实测 ~11s（27 字问题、无历史）。原值 12s 只剩 1s 余量，
    # 长问题 / 带历史（prompt 更长、prefill 更慢）/ 产出含子问题（输出更长）都会
    # 直接顶破 —— 而超时的后果是**静默回退原查询**，看起来"功能开着但其实没跑"。
    # 20s 给到约 2 倍余量；真超时仍按设计优雅降级成原查询，不会让提问失败。
    QUERY_REWRITE_TIMEOUT_SECONDS: float = 20.0
    MULTI_QUERY_ENABLED: bool = True          # 多查询扩展（多路召回 + RRF 融合）
    MULTI_QUERY_VARIANTS: int = 2             # 每个问题生成的检索变体数
    # 送进检索的**额外**查询路数上限（子问题优先，其次变体）。
    # 文字上的 MULTI_QUERY_VARIANTS / QUERY_DECOMPOSITION_MAX 各自限死了一类产物的
    # 上限，但它们的**总和**没人管：子问题 3 + 变体 2 = 5 路，每路 = 一次向量 ANN +
    # 一次关键词腿查询，延迟近似线性上升。这个闸门是总的，防止"各自守规矩、加起来
    # 失控"。HyDE 不占这个额度（它只走向量腿，见 QUERY_HYDE_VECTOR_ONLY）。
    MULTI_QUERY_MAX_EXTRA: int = 4

    # ── Reranker 精排（粗排→精排两阶段检索的第二阶段） ─────────────────────────
    RERANKER_ENABLED: bool = True             # cross-encoder 精排总开关
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"  # 默认轻量 cross-encoder
                                                     # （~278MB）。bge-reranker-v2-m3 效果更强
                                                     # 但占 ~2.3GB 常驻内存，与 Ollama 同机
                                                     # 部署时容易把 llama-server 挤到 OOM。
    RERANKER_USE_FP16: bool = False           # CUDA 环境可开 True 省一半显存
    RERANKER_MAX_CANDIDATES: int = 20        # 送入精排的粗排候选数上限（框架图 Candidate Top 20~50）
    RERANKER_MAX_TEXT_CHARS: int = 1200      # 打分前截断候选文本（bge-reranker 窗口 512 token，
                                             # 超长部分本来就无效，先截省 tokenization 与推理）
    RERANKER_USE_QUERY_VARIANTS: bool = False  # 多查询精排：对主查询+变体分别打分取最大。
                                             # 质量更稳但推理次数 × 查询数，CPU 部署建议关闭

    # ── Hallucination guard（幻觉守卫：低置信度直接拒答，不硬答） ───────────────
    HALLUCINATION_GUARD_ENABLED: bool = True # 精排分低于阈值时短路拒答
    # ⚠️ 这个常数只在"候选分**确实**是精排分"时才有意义 —— 前提由
    # retrieval_service 保证（只要有候选就必须进精排，见那里的量纲泄漏说明）。
    #
    # 为什么必须校准、不能照抄：cross-encoder（bge-reranker 系列）用 margin loss
    # 训练，输出的是**无界相关性分**、不是概率；代码里做 sigmoid 只是把量纲统一
    # 到 [0,1] 便于展示与比较 —— sigmoid 后的 0.22 **不等于**"22% 相关概率"。
    #
    # 实测（BAAI/bge-reranker-base + 本仓库中文语料）：
    #     金标分片（真阳性，主证据）  0.22 ~ 0.9999   ← 最弱主证据 = 0.22
    #     金标分片（次要证据）        0.0582         ← 最弱合法证据（多证据用例实测）
    #     "库里根本没有"的问题        ≤ 0.0005       ← 3 条否定对照实测
    # 两者相差约 116 倍（按最弱的 0.0582 算），可分。0.05 落在中间且仍在下限之下，
    # 但**余量薄**（0.0582/0.05 = 1.16 倍）—— 见 RERANK_MIN_SCORE_RATIO 末尾的警示。
    #
    # 旧值 0.25 高于"最弱真阳性 0.22" —— 后果是**金标分片在向量腿排第 1、
    # 精排也排第 1，却因为 0.22 < 0.25 被整体丢弃**，检索返回空 → 拒答节点
    # 告诉用户"知识库里没有"，而答案就在库里。这是最糟的失败：**确定性地
    # 答错"没有"**。（旧值之所以被设得这么高，是因为当时精排分与粗排分混用，
    # 需要高阈值压住"余弦基线偏高"的假阳性 —— 修复量纲后不再需要。）
    RERANK_MIN_SCORE: float = 0.05           # 精排分下限（sigmoid 归一化后的无界分）
    RERANK_MIN_SCORE_FILTER: bool = True     # Relevance Threshold：精排后逐条丢弃低于
                                             # 下限且低于相对分带的候选，不让擦边证据进上下文
    # 相对分带（relevance band）：候选分 ≥ 本次最高分 × 该比例时保留。
    #
    # 定位是**纵深防御 + 非空保证**，不是主判据（主判据是上面校准过的绝对下限）：
    #   · 绝对分的水位会随**模型换版 / 语料语言与体裁**整体漂移。相对分带以
    #     "本次检索的最好结果"为基准，分布漂移时仍能保住同档证据。
    #   · 它**天然保证非空**（最高分恒满足 best ≥ best × ratio，ratio ≤ 1）——
    #     检索层不该因为一个常数而返回空；够不够格进上下文由下游
    #     evidence_gate / 路由守卫判定（那里有独立的分数+覆盖率判据）。
    #
    # 0.05 = 与最高分相差 20 倍以内的候选仍算同档证据。
    #
    # ⚠️ 为什么从 0.10 降到 0.05（这是一次被**实测数据**推翻的取值，不是口味调整）
    # 0.10 是旧金标集下"扫 7 档指标逐位相同"时拍下来的，而旧集 11 例的金标
    # **全是精排第 1 名**（gold == head），所以 gold/head ≡ 1.0 —— 分带风险在
    # 那套集上**恒真地测不出**，0.10 从未被真正验证过。
    # 补 5 条多证据用例后立刻测出：复合问句的次要证据可以弱到头名分的 1/14
    # （实测最弱 0.0582 / 0.832 = 0.07），ratio=0.10 算出的带（0.0832）高于它
    # → **整条合法证据被砍，答案只答一半**（召回 2→1）。降到 0.05 后恢复。
    # 代价实测很小：11 例集上平均返回条数 1.91 → 2.09（+0.18 条/查询），
    # precision@3 不变（0.8182）。而砍过头的代价是**确定性地答错**（见上）。
    #
    # 上界 0.07 由 backend/eval/golden_v1.json 的 thresholds.max_rerank_min_score_ratio
    # 声明，scripts/run_eval_baseline.py 每次会重新测量并要求配置值严格小于它。
    # 同时由本模块的 ``RERANK_MIN_SCORE_RATIO_CEILING``（不可被环境变量覆盖的
    # 模块级常量）在**进程启动时**把关：main.py 的 lifespan 会在此值 ≥ 上界时
    # 打印 ERROR。两条护栏是刻意的冗余 —— 门禁挡 CI，启动检查挡"直接改 .env 上线"。
    RERANK_MIN_SCORE_RATIO: float = 0.05

    # ── Context Compression（架构图 Relevance Threshold → Compression → LLM）────
    # 查询感知的句子级抽取式压缩：与问题无关的句子被裁掉，控制父块回填后的
    # 上下文膨胀。确定性规则、零 LLM 调用，压缩在注入清洗之后执行。
    CONTEXT_COMPRESSION_ENABLED: bool = True
    CONTEXT_MAX_CHARS_PER_SOURCE: int = 1500   # 单个 source 的字符预算
    CONTEXT_MAX_TOTAL_CHARS: int = 6000        # 全部 source 的总字符软预算（num_ctx 8k 时留足生成空间）
    CONTEXT_MIN_SOURCE_CHARS: int = 300        # 总预算耗尽后，末位 source 至少保住的字符数

    # ── Enhanced chunking (token-aware + 句子滑窗 + 表格/代码块保护 + small-to-big) ──
    # 旧 API 完全向后兼容：新增参数全部带 default；缺省值 = 关闭新特性，沿用旧行为。
    CHUNK_TOKEN_AWARE: bool = True            # 按 token 数切（chars_per_token 估算 token 上限）
    CHARS_PER_TOKEN: float = 1.6              # 中英文混合经验值
    CHUNK_SENTENCE_OVERLAP: bool = True       # 重叠区按句子边界对齐
    CHUNK_PROTECT_BLOCKS: bool = True         # ```代码块``` 和 Markdown 表格不得从中间拆开
    CHUNK_PARENT_SIZE_MULT: float = 2.5       # small-to-big：父块 = 子块目标大小 × 此倍数
    CHUNK_PARENT_OVERLAP: int = 200           # 父块之间重叠字符数
    HIERARCHICAL_RAG_ENABLED: bool = True     # retrieval 时优先回填父块文本

    # ── Parent-Child（结构感知三层：section → parent → child）────────────────────
    # 与旧 small-to-big 的区别（旧实现按**字符窗口**切父块，会把一个完整小节
    # 从中间劈开；且把父块正文重复写进每一个子块的 payload，1000 份文档时
    # 是纯存储/IO 放大）：
    #
    #     section  章节级（Markdown H1/H2 边界，整节）
    #       └ parent  小节级（H3/H4 或段落组，~2.5× child 目标）
    #           └ child  检索打分单元（~MAX_CHUNK_SIZE）
    #
    # child 只携带 parent_id / section_id / section_path，**不再携带父块正文**；
    # 命中后由检索层按 parent_id 批量回填（一次 IN 查询）。
    PARENT_CHILD_ENABLED: bool = True         # 总开关（关闭时退回旧字符窗口父块）
    PARENT_CHILD_SECTION_ENABLED: bool = True # 额外生成章节级父块（引用可到"3.2 节"）
    PARENT_MIN_CHARS: int = 1200              # 父块最小字符数（短于此的小节向上合并）
    PARENT_MAX_CHARS: int = 6000              # 父块最大字符数（超长小节强切）
    # 同一父块内第 2 条及以后的证据按此系数递减（decay^(k-1)），见
    # retrieval_service._apply_parent_score_decay。只在父块**回填成功**
    # （parent_text 有值）的 chunk 上生效。1.0 = 关闭。
    PARENT_SCORE_DECAY: float = 0.85          # 父块回填后同父冗余证据的打分衰减系数
                                              # （父块是"上下文"不是"证据"，不能让它
                                              #  因为更长就压过真正的精确命中）
    PARENT_CHILD_MAX_PER_PARENT: int = 2      # 同一父块最多保留几条子块证据
                                              # （1 = 每个出处只给一条；2 = 保留
                                              #  "结论+数据"这类互补组合；0 = 关闭去重）

    # ── Metadata 系统（自动元数据抽取 + 可过滤检索 + 溯源展示）──────────────────
    # 1000+ 文档时"检索范围"本身就是准确率的一部分：不做元数据预过滤时，
    # 大量同主题文档互相稀释候选池，精排会把跨年份/跨部门的近似段落排上来。
    METADATA_ENABLED: bool = True             # 总开关
    METADATA_FROM_FILENAME: bool = True       # 从文件名/目录派生 doc_type/年份/部门
    METADATA_MAX_KEYWORDS: int = 12           # 自动关键词上限
    METADATA_MIN_KEYWORD_LEN: int = 2         # 关键词最小长度
    METADATA_TITLE_MAX_CHARS: int = 200       # 标题截断
    METADATA_OUTLINE_MAX_NODES: int = 200     # 章节大纲节点上限（防失控目录）
    METADATA_FILTER_ENABLED: bool = True      # 允许检索时按元数据预过滤
    # 把个人库/部门库的可见性下推到向量层（ANN 之前）。关掉即回到"只按 tenant
    # 过滤 + PG 侧 ACL 兜底"的旧行为：正确性不变，但候选池会被别人的私库向量
    # 稀释。留这个开关是为了线上出问题时能一键回退。
    ACL_PREFILTER_ENABLED: bool = True

    # ── 文档结构解析（Marker PDF / MinerU / Docling 统一链）──────────────────────
    # 解析器按 chain 顺序逐个尝试**能力探测**，第一个可用的胜出；全不可用则回退
    # native（正则抽 Markdown heading）。任何一级失败都只是降级，绝不中断入库。
    #
    # ⚠️ 与原 Docling 路径的关键区别：这些 provider 都必须产出**逐页归属**，
    # 否则细粒度引用（第 3 页第 12 行）会全部变成"第 1 页"。做不到的就跳过。
    STRUCTURE_PARSER_ENABLED: bool = True
    STRUCTURE_PARSER_CHAIN: str = "mineru,marker,docling,native"
    STRUCTURE_PARSER_TIMEOUT_SECONDS: float = 600.0   # 单个 provider 子进程超时
    STRUCTURE_PARSER_MAX_OUTPUT_CHARS: int = 5_000_000 # 输出上限，防失控
    STRUCTURE_PROBE_CACHE_SECONDS: float = 300.0      # 能力探测结果缓存
    MINERU_COMMAND: str = "mineru"
    MINERU_BACKEND: str = "pipeline"          # pipeline | vlm
    MARKER_COMMAND: str = "marker_single"
    # 结构解析只对这些扩展名生效（其余格式走各自的原生解析器）
    STRUCTURE_PARSER_EXTENSIONS: str = "pdf,docx,pptx"

    # ── 关键词腿后端（规模关键）─────────────────────────────────────────────────
    # 历史实现把整个语料 scroll 进内存建 BM25，并**静默截断**到
    # HYBRID_MAX_CORPUS_POINTS —— 1000 份文档时关键词腿只看得到前 1 万条 chunk，
    # 剩下 95%+ 的语料永远召不回，而且没有任何报错（最危险的一类 bug）。
    #
    #   postgres — 词项落 PG（bigram + GIN + ts_rank），召回线性可控、无内存放大（推荐）
    #   memory   — 旧的内存 BM25（小语料可用，超限时显式告警而非静默截断）
    HYBRID_KEYWORD_BACKEND: str = "postgres"
    HYBRID_BM25_CACHE_MAX_ENTRIES: int = 8    # 内存 BM25 语料缓存上限（LRU）
    KEYWORD_BIGRAM_MAX_CHARS: int = 4000      # 单 chunk 词项化字符上限（防超长块）
    KEYWORD_SEARCH_MULTIPLIER: int = 4        # 关键词腿候选深度 = top_k × 此值

    # ── Query Rewrite 升级（HyDE / 查询分解 / 防漂移闸门）────────────────────────
    # 历史实现：空历史短问题也会**每次都调一次 LLM**（MULTI_QUERY_ENABLED 默认开），
    # 白付一次 1-3s 延迟；且改写产物没有任何校验，改写器一旦"发挥"，检索就会
    # 沿着一个和原问题不一致的方向跑偏（改写引入的幻觉无法被后续引用校验拦住）。
    QUERY_REWRITE_FASTPATH_MAX_CHARS: int = 18  # 短问题 + 无历史 → 跳过 LLM（快路径）
    QUERY_REWRITE_MIN_OVERLAP: float = 0.30     # 改写产物与原问题的 token 重叠下限；
                                               # 低于此值判为"漂移"，丢弃改写回退原查询
    QUERY_HYDE_ENABLED: bool = True            # HyDE：生成假设性答案段落用于向量召回
    QUERY_HYDE_MAX_CHARS: int = 400
    # HyDE 段落**只进向量腿**，不进关键词腿。它是一段"假设答案"而非"提问"，
    # 其价值在于把查询向量拉到答案分布上；而 BM25 吃的是词项精确匹配，把一段
    # 400 字通顺但凭空生成的段落丢进去，只会得到一份词项宽泛杂乱的排名，在
    # RRF 里稀释真正的精确词命中（编号/型号类问题受害最明显）。
    QUERY_HYDE_VECTOR_ONLY: bool = True
    QUERY_DECOMPOSITION_ENABLED: bool = True   # 多跳/对比类问题拆成子问题
    QUERY_DECOMPOSITION_MAX: int = 3
    QUERY_REWRITE_CACHE_TTL_SECONDS: float = 600.0
    QUERY_REWRITE_CACHE_MAX_ENTRIES: int = 512

    # ── 证据可信度加权（反幻觉：让"不可信的解析产物"自己沉下去）───────────────────
    # 规模场景下最隐蔽的幻觉来源不是模型编造，而是**检索回来的证据本身就是错的**
    # （OCR 读错数字、VLM 转写出通顺但错误的表格、结构解析失败只留残片）。
    # 这些 chunk 在语义上看起来完全合理，精排分也高。所以把解析期已知的可信度
    # 信号（解析引擎、置信度、人工复核标记、质检结论）直接接进排序。
    EVIDENCE_TRUST_ENABLED: bool = True
    EVIDENCE_TRUST_MIN: float = 0.30          # 低于此可信度的证据直接丢弃
    EVIDENCE_TRUST_RERANK_WEIGHT: float = 0.15 # 可信度在最终排序里的权重（0 = 关闭）

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
    # 跨文档相同内容去重：同一文件被重复上传 N 次时，同一段正文只保留排名最高
    # 的一份拷贝，把 top_k 名额还给多样证据（否则引用列表挂满重复来源）。
    RETRIEVAL_CONTENT_DEDUP_ENABLED: bool = True

    # ── Document Summary（架构图 Document Summary 分支）─────────────────────────
    DOC_SUMMARY_ENABLED: bool = True
    DOC_SUMMARY_MAX_CHARS_PER_DOC: int = 12000  # 单文档摘要输入上限
    DOC_SUMMARY_TIMEOUT_SECONDS: float = 60.0
    # 整库总结的文档覆盖上限。总结是逐文档 map-reduce（每份一次 LLM 调用），
    # 上限取的是"用户愿意等多久"而不是上下文装不装得下；被截断时答案末尾
    # 会明确告知"只覆盖了最近 N 份"，绝不静默漏文档（问题：总结所有文档
    # 只总结了出现最多的那几份）。
    DOC_SUMMARY_MAX_DOCUMENTS: int = 20
    # 逐份文档摘要是否开启 qwen3 的思考（reasoning）。
    # 实测（本机 qwen3:8b、1696 字摘要输入、CPU、num_gpu=0）：
    #   True  → 每份约 92s（thinking 占 563 字），4 份 + 概览的整库总结极易超时挂死；
    #   False → 每份约 33s，输出内容完整（282 字，格式反而更规整）。
    # 逐份摘要是"抽取/结构化"性质任务（挑数字、保口径），按本机约定**应当关
    # 思考**；生成类节点（rag_graph / general_chat）才保留 reasoning=True。
    # 默认 False（省 2.8× 时延，且避免逐个调用逼近 DOC_SUMMARY_TIMEOUT_SECONDS）。
    # 注意"总体概览"那次调用**恒定**不思考 —— 它的输入只是已写好的各节摘要。
    DOC_SUMMARY_REASONING: bool = False

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
    def chat_num_ctx(self) -> int:
        """
        所有聊天类节点统一使用的上下文窗口（**必须全项目一致**）.

        为什么"每个节点各挑一个合适的 num_ctx"是个陷阱
        ────────────────────────────────────────────────
        Ollama 按 **(模型, 运行参数)** 缓存常驻实例，而 ``num_ctx`` 决定了 KV cache
        的分配大小，因此**它一变，整个模型必须卸载重载**。实测本机 qwen3:8b：

            num_ctx 8192 → 8192   138ms / 92ms      load_duration ≈ 5ms     （命中常驻实例）
            num_ctx 8192 → 4096   11,508ms          load_duration ≈ 11,000ms（重载）
            再 4096 → 8192        11,478ms          load_duration ≈ 11,000ms（重载）

        而一次问答的节点序列天然是"辅助节点 ↔ 生成节点"交替：路由(小) → 改写(小)
        → 评估(小) → 生成(大)。只要两侧取值不同，**每一步交替就白付 11 秒**，
        且这笔开销在监控面板上完全不可见（只记了 total 时延）—— 实测线上
        knowledge_qa 平均 87 秒里约有 22 秒是模型反复加载，不是推理。

        "辅助节点用 4096 省内存"是**假的节省**：同一时刻只会有一个实例常驻，
        而生成节点本来就需要 8192，峰值分配由后者决定 —— 用小窗口并不能降低峰值，
        只换来反复重载。所以正确做法是统一，而不是各取所需。

        改这里即可全局生效；任何节点都不应再自己写 ``min(4096, ...)`` 之类的表达式，
        否则会把"11 秒 × 每次交替"重新引回来。
        """
        return int(self.OLLAMA_NUM_CTX)

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

    # ── Embedded-image recognition (问题3 / 部分1+2) ────────────────────────────
    ENABLE_IMAGE_OCR: bool = True          # OCR images embedded inside PDF/DOCX/PPTX
    # 单文档图片上限。实测逐份 VLM 调用 ~50s/图，20 张 → 单文档入库可达 ~17 分钟
    # （且 Ollama 单实例排队拖慢所有功能）。压到 8 张把上限压到 ~7 分钟。
    MAX_IMAGES_PER_DOCUMENT: int = 8       # hard cap per document (guards runaway docs)
    MAX_IMAGES_PER_PAGE: int = 6           # per-page cap for PDFs
    MIN_IMAGE_DIMENSION: int = 60          # icon guard: reject when BOTH sides are smaller (px)
    # 任一边小于该值即视为退化条带（1-2px 的边框/分隔线），直接丢弃。
    # 与 MIN_IMAGE_DIMENSION 的区别：只有"两边都小"才算图标/装饰；宽而矮的
    # 窄条（如一行终端输出截图）是内容，不能误杀。
    MIN_IMAGE_THIN_SIDE: int = 16          # any side below this = degenerate strip (px)
    MAX_IMAGE_OCR_CHARS: int = 600         # truncate OCR text per image
    # Tesseract 识别语言。中文知识库必须带 chi_sim；镜像里若没装该语言包，
    # 代码会自动降级到 eng 并打印明确告警（否则中文会静默变成空串）。
    OCR_TESSERACT_LANG: str = "chi_sim+eng"

    # 部分2：内嵌图片落盘。抽出的图片写入 uploads/{document_id}/images/，
    # 检索命中后可回显原始图片（下载 / 引用卡片缩略图）。
    ENABLE_IMAGE_SAVE: bool = True         # 关闭后图片只识别不落盘（无法回显原图）
    IMAGE_STORAGE_DIR: str = "uploads"     # 相对后端起跑目录；容器内挂持久卷
    ARCHIVE_ORIGINAL_DOCUMENT: bool = True # 同时归档原始文件到 uploads/{document_id}/

    # 部分1：图片不再当作纯文本。开启后图片文本**不**拼进正文，而是抽成
    # 独立的 image chunk（content_type="image"）—— 图片成为独立检索对象。
    # 关闭则退回旧行为（图片 OCR 文本混入正文）。
    IMAGE_AS_INDEPENDENT_OBJECT: bool = True

    # ── Vision（部分4：将 Vision 打开，没有模型就先放弃）────────────────────────
    # 开启后：入库期为图片生成 caption；检索期对命中的图片做"看图问答"。
    # 若 Ollama 中没有对应多模态模型，启动探测会判定不可用并自动降级 ——
    # 图片仍可被检索与回显，仅"视觉理解/看图问答"被跳过。
    VISION_ENABLED: bool = True
    OLLAMA_VISION_MODEL: str = "qwen2.5vl:7b"   # 空字符串 = 显式放弃 vision
    # 入库期单图推理超时阈值。实测本机 qwen2.5vl:3b（CPU, num_gpu=0）单图
    # 43-54s，首次调用还含约 6.7s 模型加载 —— 60s 余量仅 6-17s，系统稍忙即触发
    # httpx 超时 → 静默降级为纯 OCR 却仍标 route="vision"（见 pipeline 修复）。
    # 提到 120s，给尾延迟留足余量。
    VISION_TIMEOUT_SECONDS: float = 120.0
    VISION_CAPTION_MAX_CHARS: int = 400         # 入库期 caption 截断长度

    # ── 无文字图片的处理（实施手册 3.3.1：图片不得在分块阶段被直接跳过）─────────
    # True  = 图内**一个字都没读出来**时（OCR / 结构化引擎 / Vision 转写全空），
    #         再花一次多模态推理，让模型输出「图注 + 关键要素 + 数值信息」三段式
    #         描述作为这张图自己的检索文本 —— 图片因此能生成独立 image chunk，
    #         而不是在分块阶段消失。
    # False = 保持旧行为（无文字图片不建块，只在磁盘留一份原图）。
    # 只在"完全没有文字"时才触发：有文字的图不会多花这次推理。
    IMAGE_SUMMARY_WHEN_TEXTLESS: bool = True

    # ── 图片类型判断（Image Classification）──────────────────────────────────────
    # 文档里的图片落到磁盘之后不是"一律 OCR"，而是先判类型再分流：
    #
    #     Image → Picture Classification ─┬─ Table    → Table Parser
    #                                     ├─ Chart    → Vision
    #                                     ├─ Diagram  → Vision
    #                                     └─ 其他      → OCR
    #
    # 分类器本身是**纯规则**实现（横竖线投影 + 色彩统计 + OCR 版面），
    # 不需要任何模型，因此 Vision 缺失时分类与降级链路依然完整可用。
    IMAGE_CLASSIFICATION_ENABLED: bool = True
    IMAGE_CLASSIFIER_ENGINE: str = "rules"      # rules | rules+vision（后者用多模态模型复核）
    # 判定为表格图片所需的最少横线 / 竖线数量（表格线检测）
    TABLE_MIN_H_LINES: int = 3
    TABLE_MIN_V_LINES: int = 2
    # 表格结构识别：识别后的最少行 / 列数，低于此值判为"结构化失败"并降级
    TABLE_IMAGE_MIN_ROWS: int = 2
    TABLE_IMAGE_MIN_COLS: int = 2
    TABLE_IMAGE_MAX_ROWS: int = 200             # 防失控大表
    TABLE_IMAGE_MAX_COLS: int = 30
    # 逐单元格 OCR 的最大格数（表格图片还原上限，防止一张巨型表格把入库拖死）
    TABLE_IMAGE_MAX_OCR_CELLS: int = 240
    # 表格结构识别失败时，是否让 Vision 直接转写 Markdown 表格救场
    # （不依赖框线，对无边框/花边框表格更稳；无模型时自动跳过）
    TABLE_IMAGE_VISION_RESCUE: bool = True
    # 走 Vision 的图片类型（逗号分隔）。设计稿里"普通图片 → OCR"，
    # 只有图表 / 流程图 / 代码截图这类才需要"看图理解"。
    VISION_ANALYZE_TYPES: str = "chart,diagram,screenshot"
    VISION_CHART_MAX_CHARS: int = 600           # 图表结构化描述截断长度
    VISION_DIAGRAM_MAX_CHARS: int = 600         # 流程图 / 结构图描述截断长度

    # ── 图片理解引擎（多引擎 + 置信度门控）─────────────────────────────────────
    # 设计稿的引擎清单：
    #   PDF/DOCX 基础解析 → Docling
    #   Layout Detection  → PP-StructureV3
    #   普通 OCR          → PaddleOCR
    #   表格结构          → Table Transformer
    #   代码截图          → OCR + Code Parser
    #   公式              → PaddleOCR Formula
    #   流程图/架构图/描述 → Vision
    # 每个引擎都做**能力探测**：环境里装不上就自动退回下一条路径，不阻塞入库。

    # Docling 负责 PDF/DOCX 的正文与结构解析（不可用时回退 PyMuPDF/python-docx）
    DOCLING_ENABLED: bool = True
    # Docling 解析 PDF 的开关，**默认关**：走 Docling 会把正文全部归到第 1 页，
    # 丢失逐页归属（而逐页归属是细粒度引用溯源的基础）。复杂版面（多栏、
    # 图文混排）确实需要 Docling 时才显式打开，代价是牺牲页码准确性。
    # 注意：这个字段以前只写在 .env.example 里、Settings 中没有定义，
    # 于是 pydantic 静默忽略该环境变量 —— 用户设了也无效。现已补上。
    DOCLING_PDF_ENABLED: bool = False

    # PaddleOCR：语言 + 模型版本 + 模型缓存目录
    # ⚠️ PADDLE_OCR_VERSION 默认 PP-OCRv3：默认的 v4 中文识别模型会在部分
    # CPU 上触发 SelfAttentionFusePass 非法指令（SIGILL）。除非确认 CPU 支持，
    # 否则不要改成 PP-OCRv4。
    PADDLE_OCR_LANG: str = "ch"
    PADDLE_OCR_VERSION: str = "PP-OCRv3"
    PADDLE_CACHE_DIR: str = "/app/.cache"   # 容器内 appuser 的 HOME 不可写时的模型落盘目录

    # 版面检测（PP-Structure）。V3 需要 paddleocr ≥3.0；只有 2.9.x 时自动用 v2。
    LAYOUT_DETECTION_ENABLED: bool = True

    # 表格结构：Table Transformer（研究型方案）
    # 首次使用会从 HuggingFace 镜像下载模型（约 100+MB，之后走缓存卷）。
    # 关闭后表格只走"框线规则 + 逐格 OCR"，零模型依赖。
    TABLE_TRANSFORMER_ENABLED: bool = True

    # 预处理：对"文档感"强的图（线稿占比高）做纠偏
    IMAGE_PREPROCESS_GEOMETRIC: bool = True

    # ── 图片噪声处理（先量噪声，再决定动不动手）────────────────────────────────
    # 知识库里大多数图是"原生导出"（PDF/PPTX 里的矢量图转位图），几乎没有噪点；
    # 对它们去噪只会把笔画磨糊、让 OCR 更差。所以预处理先**估计噪声**：
    #   noise_sigma < NOISE_SIGMA_THRESHOLD → 一步都不做
    #   超过阈值                            → 按强度自适应去噪（h ∈ [3, 18]）
    #   四条边都有黑边                       → 判为扫描件（去边框 + 纠偏 + 二值化）
    # 噪声标准差阈值（0-255 灰度域）。经验值：
    #   原生导出 ≈ 0.5-1.5   截图 ≈ 2-4   扫描件 ≈ 5-12   翻拍 ≈ 10-25
    IMAGE_NOISE_SIGMA_THRESHOLD: float = 4.0
    # 强噪声阈值：超过即进 scan 档（去噪 + 去边框 + 摩尔纹抑制 + 纠偏 + 二值化）
    IMAGE_NOISE_SIGMA_STRONG: float = 18.0
    # 对比度跨度（P99.5-P0.5）/255 低于该值认为"发灰"，做对比度归一化（CLAHE）
    IMAGE_LOW_CONTRAST_SPAN: float = 0.55
    # 四条边至少这么多像素的黑边才判为扫描黑边（避免误判深色主题截图）
    IMAGE_BORDER_TRIGGER_PX: int = 6

    # 置信度门控阈值
    #   >= ACCEPT → 专用引擎结果直接采用（Accept）
    #   低于 ACCEPT → 走兜底（Vision / Second OCR）
    #   兜底后 >= PASS 且结构校验通过 → 采用（Pass），否则标记人工复核
    IMAGE_CONFIDENCE_ACCEPT: float = 0.75
    IMAGE_CONFIDENCE_PASS: float = 0.55

    # ── OCR / VLM 双通道（把"失败才兜底"升级为"并行互补"）─────────────────────
    # 原设计里 VLM 只在专用引擎置信度低时登场，于是"OCR 自信地读错"这类失败
    # 永远轮不到模型纠正（代码截图最典型：丢了缩进、把 != 读成 =，仍给 0.9）。
    # 打开后按类型**主动**跑一遍 VLM，与 OCR/结构化通道交叉校验后融合。
    # ⚠️ VLM 是整条链路最贵的一步（1-5 秒 + 显存），所以只对白名单类型开。
    IMAGE_DUAL_CHANNEL_ENABLED: bool = True
    IMAGE_DUAL_CHANNEL_TYPES: str = "code,table,formula,chart,diagram,screenshot"
    # 融合策略由图片类型决定（见 dual_channel.strategy_for）：
    #   complementary（table/code/formula）—— 结构通道给骨架，VLM 交叉校验
    #   vision-first （chart/diagram/screenshot）—— VLM 主力，OCR 作文字锚点
    #   ocr-first    （photo）—— OCR 够用就不花 VLM 的钱

    # ── 产出质量校验（VLM 解析后的事后质检）───────────────────────────────────
    # 结构形态对 ≠ 内容对。最常见的失败是"形态对但内容是错的"：
    #   · 代码截图 → VLM 转写出看起来合理的代码，但括号不配平、def 少了冒号
    #   · 扫描件   → OCR 把 1 读成 l、0 读成 O，文本通顺但数字全错
    # 所以能形式化验证的就真的去验证：Python 用 ast.parse、JSON 用 json.loads、
    # YAML 用 safe_load，其余语言用字符串感知的括号/引号配平。
    IMAGE_CODE_SYNTAX_CHECK: bool = True
    # OCR 行置信度门限：平均行置信度低于该值 → 本次读字不可信
    IMAGE_OCR_CONFIDENCE_MIN: float = 0.65
    # 低置信行（< 0.60）占比超过该值 → 本次读字不可信
    IMAGE_OCR_LOWCONF_RATIO_MAX: float = 0.35
    # 质检总分低于该值 → 判为"未通过"，产出会打回兜底/人工复核
    IMAGE_VLM_QUALITY_MIN: float = 0.5

    # ── Multimodal Context（部分5+6：检索结果的图文分流 → 看图 → LLM）──────────
    # 精排 Top5 之后拆成 text chunk 与 image chunk：
    #   text  → 直接进上下文；
    #   image → image_path 取原图 → Vision 看图 → 结论进上下文（并回显原图）。
    MULTIMODAL_CONTEXT_ENABLED: bool = True
    MAX_VISION_IMAGES_PER_QUERY: int = 3        # 单次查询最多对几张图跑 Vision（控延迟）
    MULTIMODAL_IMAGE_TEXT_BUDGET: int = 800     # 图片结论在上下文中占用的字符预算

    # ── Document Agent（最终效果：Word 写入 / Word 插入图片）─────────────────────
    DOCUMENT_AGENT_ENABLED: bool = True
    DOCUMENT_OUTPUT_DIR: str = "uploads/_generated"   # 生成的 Word 落盘目录
    DOCUMENT_DOWNLOAD_PREFIX: str = "/documents/generated"  # 下载路由前缀
    DOCUMENT_AGENT_MAX_SOURCES: int = 8          # 写入文档的检索片段上限
    DOCUMENT_AGENT_MAX_IMAGES: int = 4           # 插入文档的原始图片上限
    DOCUMENT_AGENT_MAX_TABLE_ROWS: int = 60      # 表格写入行数上限（防失控文档）

    # ── API gateway / edge protection ─────────────────────────────────────────
    # In-process gateway policy. For multi-instance production, pair this with
    # Kong/APISIX/Envoy and a Redis-backed distributed limiter.
    GATEWAY_MAX_REQUEST_BYTES: int = 55 * 1024 * 1024  # allows 50 MB upload + multipart overhead
    GATEWAY_AUTH_RATE_LIMIT: int = 10
    GATEWAY_UPLOAD_RATE_LIMIT: int = 12
    GATEWAY_ADMIN_RATE_LIMIT: int = 60
    GATEWAY_ENFORCE_ORIGIN: bool = False              # set true in production
    TRUSTED_PROXY_IPS: list[str] = []                 # only these may set X-Forwarded-For

    # ── 五维安全隔离（密级 / 项目）—— 见 docs/system_design_security_isolation.md ──
    # 严格模式：密级字段**缺失**时按最高档（3）处理，而不是按默认档（1）。
    # 默认关闭 —— 存量文档全部未标注，开严格模式会让它们一夜之间全部不可见
    # （那是不可退化基线的直接击穿）。只在回填脚本跑完、确认没有 NULL 之后再开。
    SECURITY_STRICT_MODE: bool = False
    # 存量/缺失密级的默认档位（已裁决 Q2 = 1 内部）。
    # ⚠️ 必须与 app.db.security_models.DEFAULT_SECURITY_LEVEL 一致 ——
    #    test_security_schema.py 有一条断言盯着这两处不漂移。
    DEFAULT_SECURITY_LEVEL: int = 1
    # 密级前置过滤形态：
    #   false（默认）→ deny-list（老向量缺字段不排除 → fail-open，交 PG 终判）
    #   true          → 白名单（must: lte=clearance）—— **只允许在回填完成后开启**
    ACL_SECURITY_PREFILTER_STRICT: bool = False
    # 项目维度总开关（关掉时 project_ids / visibility_mode 一律按 tier 处理）
    PROJECT_ENABLED: bool = True
    # ── ACL 双写一致性（PG 权威 → Qdrant 副本异步追平）─────────────────────────
    # stale 重试次数上限，超过则升级为 logger.error 并计入指标
    ACL_SYNC_STALE_MAX_RETRIES: int = 3
    # 单次异步推送的批大小
    ACL_SYNC_PUSH_BATCH: int = 256

    # ── CORS ───────────────────────────────────────────────────────────────────
    # Never use "*" in production. Configure the exact frontend origin(s).
    CORS_ORIGINS: list[str] = ["*"]


# ── 相对分带的硬上界（护栏常数）──────────────────────────────────────────────
# 刻意写成**模块级常量**而不是 Settings 字段。
#
# 为什么不能是字段：Settings 的每个字段都可被环境变量覆盖（env > default，
# 这是 pydantic-settings 的既定行为）。把护栏做成可覆盖的字段，等于没护栏 ——
# 一个 `.env` 里残留的旧值就能把护栏连同被护栏保护的对象一起改掉。
# 这正是本次踩到的坑：`.env` 里 `RERANK_MIN_SCORE=0.25` 会静默覆盖代码里的
# 0.05（`docker compose config` 已证实解析结果为 "0.25"），重构镜像后
# 「金标被阈值误杀」的旧 bug 会原样复活，而代码看起来已经修好了。
#
# 取值依据（实测，见 backend/eval/golden_v1.json 的 band_headroom_measured）：
#   多证据用例的 min(次证分 / 头名分) = 0.0582 / 0.832 = 0.07
#   ⇒ ratio ≥ 0.07 时，至少一条**合法**次要证据会被带砍掉（答案只答一半）。
# 该上界随语料与精排模型漂移，换模型/换语料后必须重测；
# scripts/run_eval_baseline.py 每次都会重新测量并与金标集声明的值对账。
RERANK_MIN_SCORE_RATIO_CEILING: float = 0.07


# ─────────────────────────────────────────────────────────────────────────────
# 出网代理豁免：把 HTTP_NO_PROXY 物化成真实的 NO_PROXY / no_proxy 环境变量
# ─────────────────────────────────────────────────────────────────────────────
#
# 背景见 Settings.HTTP_NO_PROXY 的注释。这里只做一件小事：**只增不减**地合并。
#
# 为什么是"只增"：NO_PROXY 是运维会用的东西（CI 里可能已经列了内网镜像站、
# 数据库域名、堡垒机）。若应用启动时把它整体覆盖成自己那份清单，就会把运维的
# 配置**静默吃掉** —— 那是比原问题更难查的故障。反向也成立：运维已写好完整
# 清单时，应用再补一遍是无害的（去重后结果不变）。
#
# 两个变量都要写：``no_proxy`` 是 ``NO_PROXY`` 的小写别名，不同库读的不是同一个
# 名字（requests / urllib 读 ``no_proxy``，httpx 读 ``NO_PROXY``）—— 只写一个会
# 让"另一条链路仍然走代理"这种问题继续存在。
_NO_PROXY_ENV_KEYS: tuple[str, ...] = ("NO_PROXY", "no_proxy")


def _split_no_proxy(raw: str | None) -> list[str]:
    """把 NO_PROXY 串切成条目列表：去空白、去空项、按大小写不敏感去重（保序）。

    分隔符同时接受逗号和分号 —— 注册表 ``ProxyOverride`` 惯用分号，运维从那儿
    复制过来时很容易带分号；不拆的话会得到一个永远匹配不上的怪条目。
    """
    items: list[str] = []
    seen: set[str] = set()
    for chunk in (raw or "").replace(";", ",").split(","):
        host = chunk.strip()
        if not host:
            continue
        key = host.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(host)
    return items


def merge_no_proxy(existing: str | None, required: str | None) -> str:
    """把 ``required`` 并入 ``existing``，返回合并后的逗号串。

    **只增不减**：``existing`` 的条目全部保留且排在前，重复项（大小写不敏感）
    只保留先出现的那份。两边都为空时返回 ``""``。
    """
    merged = _split_no_proxy(existing)
    lowered = {host.lower() for host in merged}
    for host in _split_no_proxy(required):
        if host.lower() not in lowered:
            merged.append(host)
            lowered.add(host.lower())
    return ",".join(merged)


def apply_no_proxy_env(settings: Settings) -> str:
    """把 ``settings.HTTP_NO_PROXY`` 写进 ``NO_PROXY`` / ``no_proxy`` 环境变量。

    幂等：重复调用结果不变。**绝不删除**已存在的条目。
    返回最终生效值（便于启动日志打印与测试断言）；无内容可写时返回 ``""``。
    """
    required = getattr(settings, "HTTP_NO_PROXY", "") or ""
    final = ""
    for key in _NO_PROXY_ENV_KEYS:
        current = os.environ.get(key)
        merged = merge_no_proxy(current, required)
        if not merged:
            continue
        final = merged
        if merged != current:
            os.environ[key] = merged
    return final


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of Settings."""
    settings = Settings()  # type: ignore[call-arg]
    # 出网代理豁免：httpx 只认 NO_PROXY 环境变量，不读 Windows 注册表的
    # ProxyOverride（详见 HTTP_NO_PROXY 字段的注释）。放在这里是因为
    # app.config 一定早于任何 httpx 客户端被导入。
    apply_no_proxy_env(settings)
    return settings


# ─────────────────────────────────────────────────────────────────────────────
# FIX-B（T5 预发布）：JWT 弱密钥判定
# ─────────────────────────────────────────────────────────────────────────────
#
# 判定口径（fail-closed，仅在**真正使用弱/默认值**时阻断启动）：
#   1. 密钥为空 / 仅空白                          → 弱
#   2. 长度 < 32                                  → 弱（HS256 最低安全要求）
#   3. 命中已知弱值白名单（精确匹配，大小写不敏感）→ 弱
#      —— 含本项目开发默认值强前缀 ``dev-insecure-secret-change-me``
#      （用户未改默认值即命中，但显式提供一个 ≥32 字符的强密钥不会命中）
#
# ⚠️ 白名单用**精确匹配**，不用子串匹配：避免误伤恰好包含 "secret" 的强密钥
# （如运维生成的 ``prod-secret-<64hex>``）。需要拦的是"值本身就是弱口令"，
# 不是"值里出现了某词"。
_KNOWN_WEAK_JWT_SECRETS: frozenset[str] = frozenset(
    {
        "changeme",
        "change-me",
        "secret",
        "your-secret-key",
        "your_secret_key",
        "insecure",
        "password",
        "123456",
        "test",
        "default",
        "dev-insecure-secret-change-me-0123456789abcdef0123456789abcdef",
    }
)


def is_jwt_secret_weak(secret: str | None) -> bool:
    """返回 True 表示 JWT_SECRET 必须被拒绝（fail-closed 启动）。

    显式提供了一个 ≥32 字符、且不在已知弱值白名单里的密钥 ⇒ 返回 False（照常启动）。
    """
    if not secret or not str(secret).strip():
        return True
    s = str(secret).strip()
    if len(s) < 32:
        return True
    low = s.lower()
    if low in _KNOWN_WEAK_JWT_SECRETS:
        return True
    # 开发默认值强前缀：用户未改默认即命中（显式提供强密钥不会命中）
    if low.startswith("dev-insecure-secret-change-me"):
        return True
    return False


def jwt_secret_must_fail_startup(
    secret: str | None, *, allow_insecure_jwt: bool = False, debug: bool = False
) -> bool:
    """启动决策（fail-closed）：True ⇒ 必须终止启动。

    仅当密钥**确实弱**且未显式开启 escape 开关（``ALLOW_INSECURE_JWT`` /
    ``DEBUG``）时才返回 True。显式提供强密钥 ⇒ False（照常启动）。
    """
    if not is_jwt_secret_weak(secret):
        return False
    if allow_insecure_jwt or debug:
        return False
    return True