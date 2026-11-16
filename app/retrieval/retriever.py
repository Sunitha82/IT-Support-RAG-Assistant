
"""Semantic retrieval service for IT support knowledge."""

import logging
from dataclasses import dataclass


from app.core.config import get_settings
from app.core.exceptions import (
    EmbeddingServiceError,
    VectorStoreError,
)
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.faiss_store import FAISSVectorStore
from app.retrieval.store_factory import create_vector_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document chunk with its ranking metadata."""

    chunk_id: str
    source: str
    chunk_index: int
    content: str
    distance: float
    rank: int
    child_id: str | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None


class SemanticRetriever:
    """Retrieve relevant knowledge chunks using vector similarity."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | FAISSVectorStore | None = None,
    ) -> None:

        self.embedding_service = (
            embedding_service
            if embedding_service is not None
            else EmbeddingService()
        )

        self.vector_store = (
            vector_store
            if vector_store is not None
            else create_vector_store()
        )

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant knowledge chunks using vector similarity."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")

        settings = get_settings()

        limit = (
            top_k
            if top_k is not None
            else settings.retrieval_top_k
        )

        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit <= 0
        ):
            raise ValueError("top_k must be a positive integer")

        # Step 1: Generate the question embedding.
        try:
            query_embedding = self.embedding_service.embed_query(
                question.strip()
            )

        except Exception as exc:
            logger.exception("Question embedding generation failed")
            raise EmbeddingServiceError(
                "Unable to generate the question embedding"
            ) from exc

        # Step 2: Search the configured vector store.
        try:
            matches = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=limit,
            )

        except Exception as exc:
            logger.exception("Vector database search failed")
            raise VectorStoreError(
                "Unable to search the knowledge base"
            ) from exc

        # Step 3: Filter insufficiently similar chunks.
        matches = [
            match
            for match in matches
            if match["distance"] <= settings.retrieval_max_distance
        ]

        # Step 4: Convert matches into ranked results.
        results = [
            RetrievalResult(
                chunk_id=match["chunk_id"],
                source=match["source"],
                chunk_index=match["chunk_index"],
                content=match["content"],
                distance=match["distance"],
                rank=index,
            )
            for index, match in enumerate(matches, start=1)
        ]

        logger.info(
            "Retrieved %d chunks for the question",
            len(results),
        )

        return results
