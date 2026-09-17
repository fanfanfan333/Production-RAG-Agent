"""
Embedding service orchestrator.

Provides a robust, queue-ready retry wrapper around the configured embedding provider.
Implements exponential backoff and jitter, and tracks metrics.

NOTE: The embedding provider is fully local (bge_provider.BGEEmbeddingProvider),
so there is no HTTP layer, no cloud quota and no 429 responses. The retry logic
below therefore only needs to absorb generic transient errors (e.g. brief
resource contention or a cold model load) rather than provider-specific
rate-limit responses.
"""

import asyncio
import random

from app.config import get_settings
from app.services.embeddings import get_embedding_provider
from app.services.metrics import metrics
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def embed_batch_with_retry(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Embed a single batch of texts with robust transient error recovery.

    This function handles exponential backoff and limits.
    It does NOT handle chunking or large-document batching; that is the
    caller's responsibility.
    """
    if not texts:
        return []

    settings = get_settings()
    provider = get_embedding_provider()

    retries = 0

    while True:
        metrics.record_attempt()
        try:
            return await asyncio.to_thread(provider.embed_batch, texts, task_type)

        except Exception as exc:
            # Generic transient errors — e.g. brief filesystem/resource
            # contention when the model is loading, or OS-level interrupts.
            # A local BGE model has no HTTP layer, so there are no 429s or
            # provider-specific rate-limit responses to special-case here.
            is_retryable = isinstance(exc, (ConnectionError, TimeoutError, OSError))

            if not is_retryable or retries >= settings.MAX_EMBED_RETRIES:
                metrics.record_failure(retries)
                if is_retryable:
                    logger.error("Embedding permanently failed after exhausting %d retries.", retries)
                raise RuntimeError(f"Embedding failed: {exc}") from exc

            # Calculate backoff for next attempt
            retries += 1
            sleep_time = min(settings.INITIAL_BACKOFF * (2 ** (retries - 1)), settings.MAX_BACKOFF)
            if settings.ENABLE_JITTER:
                sleep_time = sleep_time * random.uniform(0.8, 1.2)

            logger.warning(
                "Retry #%d | Waiting %.1fs (Backoff) | Reason: %s",
                retries, sleep_time, exc.__class__.__name__
            )

            await asyncio.sleep(sleep_time)