"""
Local BGE embedding provider (replaces the Gemini-based provider).

Implements the exact interface that embedding_service.embed_batch_with_retry
expects from get_embedding_provider():

    provider.embed_batch(texts: list[str], task_type: str) -> list[list[float]]

Runs fully offline — no network calls, no API keys, no rate limits.
The retry/backoff wrapper in embedding_service.py becomes a no-op safety
net for this provider (kept for interface parity with cloud providers,
e.g. transient OOM or model-swap scenarios), but will never see 429s.

Model options (set via settings.BGE_MODEL_NAME):
    BAAI/bge-small-zh-v1.5   -> 512-dim,  fastest, CPU-friendly
    BAAI/bge-base-zh-v1.5    -> 768-dim,  balanced
    BAAI/bge-large-zh-v1.5   -> 1024-dim, best quality, slower  (default)
    BAAI/bge-m3              -> 1024-dim, multilingual, heaviest

IMPORTANT: settings.EMBEDDING_DIMENSION must match the chosen model's
output dimension, and the Qdrant collection must be (re)created with
that exact vector size — a mismatch fails at write time, not import time.
"""

from __future__ import annotations

import threading
from typing import List

import numpy as np

from app.config import get_settings
from app.services.embeddings.base import BaseEmbeddingProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)

_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# Known output dimensions for the common BGE checkpoints. Used only as a
# sanity-check against settings.EMBEDDING_DIMENSION at load time.
_KNOWN_DIMS = {
    "bge-small-zh-v1.5": 512,
    "bge-base-zh-v1.5": 768,
    "bge-large-zh-v1.5": 1024,
    "bge-m3": 1024,
}


class BGEEmbeddingProvider(BaseEmbeddingProvider):
    """
    Thread-safe singleton wrapping a local BGE embedding model.

    Model loading is expensive (seconds, holds GPU/CPU memory for the
    process lifetime), so this class loads once and is reused across
    every embed_batch call via the module-level factory below.
    """

    _instance: "BGEEmbeddingProvider | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "BGEEmbeddingProvider":
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
        self.model_name: str = settings.BGE_MODEL_NAME
        self.batch_size: int = settings.EMBEDDING_BATCH_SIZE
        self._model = None
        self._backend: str | None = None

        self._load_model()
        self._sanity_check_dimension(settings.EMBEDDING_DIMENSION)

        self._initialized = True

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        logger.info("Loading BGE embedding model: %s", self.model_name)

        try:
            from FlagEmbedding import FlagModel

            self._model = FlagModel(
                self.model_name,
                query_instruction_for_retrieval=_QUERY_INSTRUCTION,
                use_fp16=False,  # keep False for CPU; set True only on CUDA GPUs
            )
            self._backend = "flagembedding"
            logger.info("BGE model loaded via FlagEmbedding: %s", self.model_name)
            return

        except ImportError:
            logger.warning(
                "FlagEmbedding not installed, falling back to sentence-transformers "
                "(pip install FlagEmbedding for the officially recommended backend)"
            )

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._backend = "sentence_transformers"
            logger.info("BGE model loaded via sentence-transformers: %s", self.model_name)

        except ImportError as exc:
            raise RuntimeError(
                "No embedding backend available. Install one of:\n"
                "  pip install FlagEmbedding\n"
                "  pip install sentence-transformers"
            ) from exc
        except Exception:
            logger.exception("Failed to load BGE model: %s", self.model_name)
            raise

    def _sanity_check_dimension(self, configured_dim: int) -> None:
        """Warn loudly if EMBEDDING_DIMENSION doesn't match the loaded model."""
        actual_dim = self.embedding_dim
        if actual_dim != configured_dim:
            logger.error(
                "EMBEDDING_DIMENSION mismatch: config says %d, model '%s' "
                "actually outputs %d. Qdrant writes WILL fail until "
                "settings.EMBEDDING_DIMENSION is corrected and the "
                "collection is recreated with the right vector size.",
                configured_dim,
                self.model_name,
                actual_dim,
            )

    # ── Public interface (matches embedding_service.py's expectations) ─────

    def embed_batch(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        """
        Synchronous batch embedding call.

        Called from embedding_service.embed_batch_with_retry via
        asyncio.to_thread, so blocking here is fine and expected.

        task_type:
            "RETRIEVAL_DOCUMENT" -> document encoding (no query prefix)
            "RETRIEVAL_QUERY"    -> query encoding (BGE query prefix applied)
        """
        if not texts:
            return []

        is_query = task_type == "RETRIEVAL_QUERY"

        if self._backend == "flagembedding":
            vectors = (
                self._model.encode_queries(texts)
                if is_query
                else self._model.encode(texts, batch_size=self.batch_size)
            )
        elif self._backend == "sentence_transformers":
            prefixed = (
                [f"{_QUERY_INSTRUCTION}{t}" for t in texts] if is_query else texts
            )
            vectors = self._model.encode(
                prefixed,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            raise RuntimeError(f"Unknown BGE backend: {self._backend}")

        return np.asarray(vectors).tolist()

    @property
    def embedding_dim(self) -> int:
        name = self.model_name.lower()
        for key, dim in _KNOWN_DIMS.items():
            if key in name:
                return dim
        # Unknown checkpoint — ask the loaded model directly rather than guess.
        if self._backend == "sentence_transformers" and self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        logger.warning(
            "Could not determine embedding_dim for unrecognised model '%s'; "
            "defaulting to 1024. Verify against settings.EMBEDDING_DIMENSION.",
            self.model_name,
        )
        return 1024

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "backend": self._backend,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton — used only in tests."""
        cls._instance = None


def get_bge_provider() -> BGEEmbeddingProvider:
    """Module-level accessor, for wiring into app.services.embeddings' factory."""
    return BGEEmbeddingProvider()