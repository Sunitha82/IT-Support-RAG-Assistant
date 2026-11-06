
"""Embedding service for converting text into semantic vectors."""

import logging
from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings using a reusable SentenceTransformer model."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()

        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._load_lock = Lock()

    def _get_model(self) -> SentenceTransformer:
        """
        Lazily load the embedding model.

        The model is loaded only when embeddings are first requested.
        A lock prevents concurrent requests from loading it twice.
        """
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    logger.info(
                        "Loading embedding model: %s",
                        self.model_name,
                    )
                    self._model = SentenceTransformer(self.model_name)

        return self._model

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate normalized embeddings for document chunks.

        Args:
            texts: Non-empty list of document text strings.
            batch_size: Number of texts processed per model batch.

        Returns:
            NumPy array of shape (number_of_texts, embedding_dimension).
        """
        if not texts:
            raise ValueError("Cannot embed an empty document list")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("All documents must contain non-empty text")

        model = self._get_model()

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        logger.info(
            "Generated %d embeddings with %d dimensions",
            embeddings.shape[0],
            embeddings.shape[1],
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a normalized embedding for a user question."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        model = self._get_model()

        embedding = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding
