"""
Cross-encoder reranker（精排层，粗排→精排两阶段检索的第二阶段）.

粗排（retrieval_service 的 向量 ANN + BM25 RRF 融合）先把候选集从全库
缩到 ~20 条，本模块用更重的模型对 query-chunk 逐对精细打分，只保留真正
最相关的 top_k。这是企业级 RAG（LangGraph RAG / Dify / Databricks 等）
的标准「两阶段检索」做法，直接决定最终上下文的质量 —— 精排越准，喂给
LLM 的证据越干净，幻觉与答非所问越少。

三级后端，按可用性自动降级：

    Tier 1  FlagReranker (BAAI/bge-reranker-base)      — FlagEmbedding 自带
    Tier 2  sentence_transformers.CrossEncoder          — 通用兜底
    Tier 3  纯 Python 启发式精排                          — 零依赖保底
            （查询词覆盖率 × 向量相似度加权融合）

所有后端的输出统一归一化到 [0, 1]：
    Tier 1/2 用 sigmoid 把 cross-encoder logit 压到概率；
    Tier 3 本身就是 [0, 1] 的加权覆盖率分数。
分数量纲一致，下游 RERANK_MIN_SCORE 阈值与前端"相关度 %"展示都不用
区分后端。
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
from typing import TYPE_CHECKING

from app.config import get_settings
from app.utils.logging import get_logger

if TYPE_CHECKING:
    from app.services.retrieval_service import RetrievedChunk

logger = get_logger(__name__)


def _sigmoid(x: float) -> float:
    """Numerically-safe sigmoid: maps cross-encoder logits to [0, 1]."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


# ── Tier 3: heuristic fallback ───────────────────────────────────────────────

def _heuristic_scores(query: str, texts: list[str], vector_scores: list[float]) -> list[float]:
    """
    Zero-dependency reranking signal: 查询词覆盖率 + 向量相似度先验.

    覆盖率 = |query 词集合 ∩ chunk 词集合| / |query 词集合|，
    直接衡量"这个 chunk 覆盖了问题的多少个关键概念"，与
    cross-encoder 的相关性判断高度相关，可作为无模型环境下的保底精排。
    """
    from app.services.hybrid_search import tokenize

    q_terms = set(tokenize(query))
    if not q_terms:
        # 查询没有可提取的词（纯符号等）——退化为向量先验
        return [max(0.0, min(1.0, s)) for s in vector_scores]

    scores: list[float] = []
    for text, vscore in zip(texts, vector_scores):
        d_terms = set(tokenize(text))
        coverage = len(q_terms & d_terms) / len(q_terms)
        prior = max(0.0, min(1.0, vscore))
        # 覆盖率为主信号（0.7），向量分作先验（0.3）
        scores.append(0.7 * coverage + 0.3 * prior)
    return scores


# ── Reranker singleton ───────────────────────────────────────────────────────

def dedup_candidates(chunks: "list[RetrievedChunk]") -> "list[RetrievedChunk]":
    """
    精排前去重：文本完全相同的候选只保留排名最高的一个.

    多查询扩展 + small-to-big 场景下，同一个 chunk 经常从向量腿和 BM25 腿
    各进一次候选池（或不同查询召回同一父块下的相同子块）。cross-encoder
    对重复候选是纯粹的浪费推理，还会把 top_k 名额挤占成重复引用。
    *chunks* 需已按粗排名次排序（保留下标最小的那个）。
    """
    seen: set[str] = set()
    unique: "list[RetrievedChunk]" = []
    for c in chunks:
        # 只哈希文本前 1000 字：足以识别重复块，又避免对超长文本全量哈希
        fingerprint = hashlib.md5(c.text[:1000].encode("utf-8", "ignore")).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(c)
    if len(unique) != len(chunks):
        logger.info("rerank dedup: %d candidates → %d unique", len(chunks), len(unique))
    return unique


def filter_by_min_score(
    ranked: "list[RetrievedChunk]",
    min_score: float,
    ratio: float = 0.0,
) -> "list[RetrievedChunk]":
    """
    Relevance Threshold（框架图精排 → Top 3~5 → 相关性阈值）：
    丢弃精排置信度低于阈值的候选，不让"擦边"证据进入上下文诱导幻觉.

    接受条件（**必须同时**满足）：

        c.score >= min_score          绝对下限（针对已校准模型的绝对水位）
        c.score >= best * ratio       相对分带（在同一批内砍掉与头名差距过大的尾巴）

    ⚠️ 两条之间必须是 **AND**，不能是 OR —— 这是一个已实测过的语义坑：
    ``band = best × ratio`` 且 ``ratio ≤ 1``，所以 ``band`` 恒 ≤ ``best``,
    ``score >= band`` 永远比 ``score >= min_score`` 更容易满足。写成 OR 之后
    ``score >= min_score or score >= band`` 等价于把生效下限降成
    ``min(min_score, best × ratio)``：**分带只能放宽、永远不能收紧**，
    它宣称的"砍尾巴"从未生效（实测 7 档 ratio∈[0, 0.5] 指标逐位相同），
    同时把校准过的绝对下限**静默削弱** —— ``best=0.22`` 时真实下限从 0.05
    掉到 0.022，"距噪声上界约 100 倍"的校准保证被悄悄作废，而这正是
    ``test_rerank_threshold_below_weakest_true_positive`` 那套校准门禁
    所依赖的前提（config.py 对同一旋钮的描述也是"低于下限**且**低于相对
    分带"，与本函数旧实现的 OR 相互矛盾）。

    为什么两条并存
    ──────────────
    主判据是校准过的**绝对下限**（*min_score*）：实测（bge-reranker-base +
    本仓库中文语料）金标分片 0.22~0.9999、"库里根本没有"的问题 ≤0.0005，
    两者可分。
    **相对分带**（*ratio*）是纵深防御：绝对分水位会随模型换版、语料语言与
    体裁整体漂移；以"本次检索的最好结果"为基准即随分布自适应。它只在
    **已过绝对下限**的候选里做二次收紧（"同档证据"= 与头名相差不超过
    1/ratio 倍）。
    检索层的职责是召回与排序，不该因为一个常数把整批证据清空 —— 因此
    本批全被收紧时**保留头名**，够不够格进上下文仍由下游
    evidence_gate / 路由守卫判定（那里有独立的分数+覆盖率判据）。

    ⚠️ 本函数的前提是 ``c.score`` **确实是精排分**。若上游跳过了精排
    （历史实现里"候选 ≤1 条就跳过"），这里拿到的是**粗排分**（向量余弦 /
    关键词占位 0.30），与 *min_score* 不同量纲 —— 那会让一条毫不相关的
    候选看起来"分数尚可"从而绕过拒答。该泄漏点已在
    ``retrieval_service`` 修掉，并有门禁测试看住。

    ``ratio=0`` 时退化为纯绝对阈值 —— 与升级前行为一致，便于回滚与单测。

    返回空：仅当 *ranked* 本身为空，或 ``ratio=0`` 时整批都低于绝对下限。
    后者是校准阈值的正常产物（"库里确实没有"），交给上层 grader 走
    retry/refuse 路径，比硬塞噪声证据给 LLM 更诚实；``ratio>0`` 时头名兜底
    保证非空。
    """
    if not ranked:
        return []

    kept = ranked
    if min_score > 0.0 or ratio > 0.0:
        head = ranked[0]                      # 调用方已按分数降序排列
        # ① 绝对下限（硬下限）。ratio<=0 时这就是全部判据 ——
        #    与升级前逐字一致，便于回滚与既有单测。
        kept = [c for c in ranked if c.score >= min_score]
        # ② 相对分带：只在"已过绝对下限"的集合里砍尾。
        #    不可写成 `or`（见 docstring：OR 下分带只放宽、不收紧，还会把
        #    生效下限降成 min(min_score, best*ratio)，静默废掉校准锁）。
        if ratio > 0.0:
            # ratio>1 属配置错误（band 会超过头名 → 清空证据），按 1.0 夹紧。
            band = head.score * min(ratio, 1.0)
            kept = [c for c in kept if c.score >= band]
            # ③ 恒非空：整批都没过绝对下限时保留头名。
            #    只要 head 过了 min_score 就必然也在 ② 中存活
            #    （head.score >= head.score*ratio），所以这条兜底最多只交出一条，
            #    不会把噪声成批带进来。
            if not kept:
                kept = [head]

    if len(kept) != len(ranked):
        logger.info(
            "relevance threshold: %d → %d chunks "
            "(min_score=%.3f ratio=%.2f best=%.4f)",
            len(ranked), len(kept), min_score, ratio, ranked[0].score,
        )
    return kept


class ChunkReranker:
    """
    线程安全的精排器单例.

    cross-encoder 模型加载要数秒并常驻内存，因此整个进程只加载一次，
    所有查询复用。``score_pairs`` 本身是 CPU/GPU 推理，通过
    ``asyncio.to_thread`` 调用以免阻塞事件循环（与 BGE embedding 的
    调用方式保持一致）。
    """

    _instance: "ChunkReranker | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ChunkReranker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        settings = get_settings()
        self.model_name: str = settings.RERANKER_MODEL_NAME
        self._model = None
        self._backend: str | None = None
        self._load_model()
        self._initialized = True

    # ── Model loading (three-tier) ──────────────────────────────────────────

    def _load_model(self) -> None:
        # Tier 1: FlagReranker —— BGE 官方 reranker，与项目现有 embedding
        # 依赖（FlagEmbedding）同源，无新增依赖。
        settings = get_settings()
        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self.model_name,
                use_fp16=settings.RERANKER_USE_FP16,  # CUDA 环境开 True 省一半显存
            )
            self._backend = "flagreranker"
            logger.info("Reranker loaded via FlagReranker: %s", self.model_name)
            return
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed — trying sentence-transformers "
                "CrossEncoder for reranking"
            )
        except Exception:
            # 模型文件缺失 / 下载失败等 —— 降级而不是让服务起不来
            logger.exception(
                "FlagReranker failed to load %s — falling back", self.model_name
            )

        # Tier 2: sentence-transformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self._backend = "cross_encoder"
            logger.info(
                "Reranker loaded via sentence-transformers: %s", self.model_name
            )
            return
        except ImportError:
            logger.warning(
                "sentence-transformers not installed either — "
                "reranking falls back to heuristic term-coverage scoring"
            )
        except Exception:
            logger.exception(
                "CrossEncoder failed to load %s — heuristic fallback", self.model_name
            )

        # Tier 3: 无模型启发式（在 score_pairs 里处理）
        self._backend = "heuristic"
        logger.warning("Reranker running in HEURISTIC mode (no ML model)")

    # ── Scoring ─────────────────────────────────────────────────────────────

    def _score_pairs_sync(self, query: str, texts: list[str], vector_scores: list[float]) -> list[float]:
        """Synchronous pairwise scoring — run via asyncio.to_thread."""
        if not texts:
            return []

        # bge-reranker 系列的窗口只有 512 token，超长部分本来就截断；
        # 先按字符截到 RERANKER_MAX_TEXT_CHARS，省掉无效 tokenization
        # 与推理开销（粗排候选是 1000~2000 字的块，父块回填后更长）。
        limit = get_settings().RERANKER_MAX_TEXT_CHARS
        if limit > 0:
            texts = [t[:limit] for t in texts]

        if self._backend == "flagreranker":
            raw = self._model.compute_score(
                [[query, t] for t in texts], normalize=False
            )
            # compute_score 对单条输入返回标量而非列表
            if not isinstance(raw, list):
                raw = [raw]
            return [_sigmoid(float(r)) for r in raw]

        if self._backend == "cross_encoder":
            raw = self._model.predict(
                [(query, t) for t in texts], show_progress_bar=False
            )
            if not isinstance(raw, list):
                raw = [raw]
            return [_sigmoid(float(r)) for r in raw]

        return _heuristic_scores(query, texts, vector_scores)

    async def score_pairs(
        self, query: str, texts: list[str], vector_scores: list[float]
    ) -> list[float]:
        """Async wrapper: normalize=False + sigmoid keeps the [0,1] scale stable."""
        return await asyncio.to_thread(self._score_pairs_sync, query, texts, vector_scores)

    async def score_pairs_multi(
        self,
        queries: list[str],
        texts: list[str],
        vector_scores: list[float],
    ) -> list[float]:
        """
        多查询精排：对每路查询分别打分，每个候选取各路中的最高分.

        动机：粗排用多查询扩展提升了召回，但精排只用主查询打分会漏掉
        "变体措辞比主查询贴得更近"的候选（查询改写本来就是近似）。
        取 max 是标准的多查询精排融合做法；代价是推理次数 × 查询数，
        由 RERANKER_USE_QUERY_VARIANTS 开关控制（默认关）。
        """
        if len(queries) <= 1:
            return await self.score_pairs(queries[0], texts, vector_scores)
        score_lists = await asyncio.gather(*(
            self.score_pairs(q, texts, vector_scores) for q in queries
        ))
        return [
            max(per_candidate) for per_candidate in zip(*score_lists)
        ]

    def backend_name(self) -> str:
        return self._backend or "heuristic"


def get_reranker() -> ChunkReranker:
    return ChunkReranker()


# ── Public rerank API ─────────────────────────────────────────────────────────

async def rerank_chunks(
    query: str,
    chunks: "list[RetrievedChunk]",
    top_k: int,
    queries: "list[str] | None" = None,
    min_score: float | None = None,
) -> "list[RetrievedChunk]":
    """
    精排：对粗排候选逐对打分并重排序，返回 top_k.

    流程（对应框架图 Candidate Top 20~50 → Reranker → Top 3~5 → Threshold）：
      1. 去重      —— 相同文本候选只保留一个，省推理也不挤占 top_k 名额
      2. 逐对打分  —— cross-encoder 打分；传入 *queries*（含检索变体）且
                      RERANKER_USE_QUERY_VARIANTS 开启时按多查询取最大分
      3. 阈值过滤  —— *min_score* 以下丢弃（None = 不过滤）
      4. 截断      —— 取前 top_k

    - chunk.score 被替换为归一化精排分（[0,1]，可直接用于
      RERANK_MIN_SCORE 幻觉守卫和前端"相关度 %"展示）。
    - **只有一个候选时也要打分**：直接返回会把粗排/关键词分留在 score 上，
      造成量纲泄漏（详见函数内注释）。只有"精排不可用"（降级）才允许保留原分，
      此时 except 分支会显式记录。
    - 任一异常都降级为"保持粗排顺序"——精排失败不应让查询失败。
    """
    settings = get_settings()
    if not chunks:
        return []
    if top_k <= 0:
        return []

    # ⚠️ 单候选**不能**直接返回（除非精排不可用）。
    #
    # 历史实现是 `if len(chunks) == 1: return chunks[:top_k]` —— 看着是省一次
    # 推理，实际是一个**量纲泄漏**：返回时 `c.score` 仍是**粗排分**
    # （向量余弦，基线普遍 0.1~0.4）或关键词腿的占位分（0.30），
    # 而下游所有消费方（rag_graph 的幻觉守卫、evidence_gate、
    # retrieval_grader、前端"相关度 %"）都按**精排 sigmoid 分**口径解读它。
    #
    # 后果不是"少了点精度"，而是**决策错向**：关键词腿把一条语义上毫不相关的
    # chunk（向量分 ≈0.19）提为唯一候选时，它带着 0.19 的余弦分被当作"精排判它
    # 相关"送进上下文 —— 实测两条"库里根本没有"的问题就是这样绕过拒答的
    # （见 tests/test_reliability_guards.py 的门禁）。而交叉编码器真打分通常
    # 接近 0，本可以正确拒答。
    #
    # 代价只是一次 forward pass（单条），换来 `chunk.score` **始终**是精排分 ——
    # 这条不变式是下面那个绝对阈值能有意义的前提。
    try:
        reranker = get_reranker()

        # 1) 去重（保序：粗排名次靠前的优先保留）
        unique_chunks = dedup_candidates(chunks)

        vector_scores = [c.score for c in unique_chunks]
        score_queries: list[str] = [query]
        if settings.RERANKER_USE_QUERY_VARIANTS and queries:
            score_queries = [q for q in dict.fromkeys(queries) if q]
            if query not in score_queries:
                score_queries.insert(0, query)

        # 2) 打分（单查询或多查询取 max）
        if len(score_queries) > 1:
            scores = await reranker.score_pairs_multi(
                score_queries, [c.text for c in unique_chunks], vector_scores
            )
        else:
            scores = await reranker.score_pairs(
                query, [c.text for c in unique_chunks], vector_scores
            )

        ranked = list(zip(unique_chunks, scores))
        ranked.sort(key=lambda cs: cs[1], reverse=True)

        reranked: "list[RetrievedChunk]" = []
        for chunk, score in ranked:
            chunk.score = round(float(score), 4)
            reranked.append(chunk)

        # 3) 相关性阈值（绝对下限 **AND** 相对分带；仅当输入为空才会返回空，见
        #    filter_by_min_score 的说明）
        #
        #    ⚠️ 这里曾写着"绝对下限 OR 相对分带" —— 与实现（AND）矛盾。注释不是
        #    装饰：滤器写错方向时，读代码的人会依据注释推断成"分带只能放宽"，
        #    从而放过一次真实的收紧失效（实测 7 档 ratio 指标逐位相同）。
        if min_score is not None:
            reranked = filter_by_min_score(
                reranked,
                min_score,
                float(getattr(settings, "RERANK_MIN_SCORE_RATIO", 0.0) or 0.0),
            )

        # 4) 截断到 top_k
        reranked = reranked[:top_k]

        logger.info(
            "rerank: %d candidates → top %d (backend=%s queries=%d best=%.4f)",
            len(chunks), len(reranked), reranker.backend_name(),
            len(score_queries), ranked[0][1] if ranked else 0.0,
        )
        return reranked
    except Exception:
        logger.exception("Rerank failed — keeping coarse ranking order")
        return chunks[:top_k]
