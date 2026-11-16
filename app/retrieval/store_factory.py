
"""Factory for selecting the configured vector store."""

from app.core.config import get_settings
from app.retrieval.vector_store import VectorStore
from app.retrieval.faiss_store import FAISSVectorStore


def create_vector_store() -> VectorStore | FAISSVectorStore:
    """Create ChromaDB or FAISS based on application settings."""

    settings = get_settings()

    if settings.vector_store_provider == "faiss":
        return FAISSVectorStore()

    if settings.vector_store_provider == "chroma":
        return VectorStore()

    raise ValueError(
        f"Unsupported vector store: {settings.vector_store_provider}"
    )
