
"""Unit tests for semantic retrieval."""

from unittest.mock import Mock

import numpy as np
import pytest

from app.retrieval.retriever import SemanticRetriever


@pytest.fixture
def retriever():
    """Create a retriever with mocked dependencies."""
    embedding_service = Mock()
    vector_store = Mock()

    embedding_service.embed_query.return_value = np.array(
        [0.1, 0.2, 0.3]
    )

    vector_store.search.return_value = [
        {
            "chunk_id": "vpn.txt::0",
            "source": "vpn.txt",
            "chunk_index": 0,
            "content": "Reconnect to the company VPN.",
            "distance": 0.2,
        }
    ]

    return SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def test_retrieves_relevant_document(retriever):
    """Verify retrieved results preserve source metadata."""
    results = retriever.retrieve("How do I reconnect VPN?")

    assert len(results) == 1
    assert results[0].source == "vpn.txt"
    assert results[0].rank == 1
    assert results[0].content == "Reconnect to the company VPN."


def test_empty_question_rejected(retriever):
    """Reject empty user questions."""
    with pytest.raises(ValueError):
        retriever.retrieve("")


def test_whitespace_question_rejected(retriever):
    """Reject whitespace-only questions."""
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_invalid_top_k_rejected(retriever):
    """Reject invalid retrieval limits."""
    with pytest.raises(ValueError):
        retriever.retrieve("VPN problem", top_k=0)


def test_no_documents_found(retriever):
    """Return an empty list when the vector store has no matches."""
    retriever.vector_store.search.return_value = []

    results = retriever.retrieve("How do I reset my password?")

    assert results == []


def test_custom_top_k(retriever):
    """Verify the requested retrieval limit reaches the vector store."""
    retriever.retrieve("VPN connection issue", top_k=5)

    assert retriever.vector_store.search.call_args.kwargs["top_k"] == 5


def test_query_is_embedded(retriever):
    """Verify the question is passed to the embedding service."""
    retriever.retrieve("  VPN connection issue  ")

    retriever.embedding_service.embed_query.assert_called_once_with(
        "VPN connection issue"
    )


def test_irrelevant_chunks_are_filtered(retriever):
    """Weak matches must not be returned as evidence."""

    retriever.vector_store.search.return_value = [
        {
            "chunk_id": "password.txt::0",
            "source": "password.txt",
            "chunk_index": 0,
            "content": "Employees change passwords every 90 days.",
            "distance": 0.8219,
        }
    ]

    results = retriever.retrieve(
        "How many days of maternity leave do employees receive?"
    )

    assert results == []


def test_relevant_chunks_are_preserved(retriever):
    """Strong matches should remain available to the LLM."""

    results = retriever.retrieve(
        "How do I reconnect my VPN?"
    )

    assert len(results) == 1
    assert results[0].distance == 0.2
