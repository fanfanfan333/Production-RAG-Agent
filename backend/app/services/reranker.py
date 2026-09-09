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

    def backend_name(self) -> str:
        return self._backend or "heuristic"


def get_reranker() -> ChunkReranker:
    return ChunkReranker()


# ── Public rerank API ─────────────────────────────────────────────────────────

async def rerank_chunks(
    query: str,
    chunks: "list[RetrievedChunk]",
    top_k: int,
) -> "list[RetrievedChunk]":
    """
    精排：对粗排候选逐对打分并重排序，返回 top_k.

    - chunk.score 被替换为归一化精排分（[0,1]，可直接用于
      RERANK_MIN_SCORE 幻觉守卫和前端"相关度 %"展示）。
    - 任一异常都降级为"保持粗排顺序"——精排失败不应让查询失败。
    - 候选数为 0/1 时直接返回，省掉模型调用。
    """
    settings = get_settings()
    if not chunks:
        return []
    if len(chunks) == 1 or top_k <= 0:
        return chunks[:top_k]

    try:
        reranker = get_reranker()
        vector_scores = [c.score for c in chunks]
        scores = await reranker.score_pairs(query, [c.text for c in chunks], vector_scores)

        ranked = list(zip(chunks, scores))
        ranked.sort(key=lambda cs: cs[1], reverse=True)

        reranked: list[RetrievedChunk] = []
        for chunk, score in ranked[:top_k]:
            chunk.score = round(float(score), 4)
            reranked.append(chunk)

        logger.info(
            "rerank: %d candidates → top %d (backend=%s best=%.4f)",
            len(chunks), len(reranked), reranker.backend_name(),
            ranked[0][1] if ranked else 0.0,
        )
        return reranked
    except Exception:
        logger.exception("Rerank failed — keeping coarse ranking order")
        return chunks[:top_k]
