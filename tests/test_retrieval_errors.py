
"""Tests for retrieval error handling."""

from unittest.mock import Mock

import pytest

from app.core.exceptions import (
    EmbeddingServiceError,
    VectorStoreError,
)
from app.retrieval.retriever import SemanticRetriever


@pytest.fixture
def retriever():
    """Create a retriever with mocked dependencies."""
    embedding_service = Mock()
    vector_store = Mock()

    embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]
    vector_store.search.return_value = []

    return SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def test_embedding_failure(retriever):
    """Embedding failures should raise EmbeddingServiceError."""

    retriever.embedding_service.embed_query.side_effect = RuntimeError(
        "Embedding model failed"
    )

    with pytest.raises(EmbeddingServiceError):
        retriever.retrieve("How do I fix my VPN?")


def test_vector_store_failure(retriever):
    """Database failures should raise VectorStoreError."""

    retriever.vector_store.search.side_effect = RuntimeError(
        "ChromaDB connection failed"
    )

    with pytest.raises(VectorStoreError):
        retriever.retrieve("How do I fix my VPN?")


def test_no_matching_documents_is_not_an_error(retriever):
    """An empty search result is a valid retrieval outcome."""

    retriever.vector_store.search.return_value = []

    results = retriever.retrieve("Unknown IT policy question")

    assert results == []
