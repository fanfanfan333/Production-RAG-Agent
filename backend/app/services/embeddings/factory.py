from app.config import get_settings
from app.services.embeddings.base import BaseEmbeddingProvider
from app.services.embeddings.bge_provider import BGEEmbeddingProvider


def get_embedding_provider() -> BaseEmbeddingProvider:
    """
    Factory function to retrieve the configured embedding provider.
    Future extensions (e.g., OpenAI, Voyage) can be added here
    without modifying the caller's logic.
    """
    settings = get_settings()

    # In the future, this could inspect an `EMBEDDING_PROVIDER` setting.
    # BGE (local, offline) is currently the only implemented provider —
    # replaces the previous GeminiEmbeddingProvider.
    return BGEEmbeddingProvider()