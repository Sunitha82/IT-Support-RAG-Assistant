
"""Tests for the FAISS vector store."""

import numpy as np
import pytest

from app.ingestion.chunking import DocumentChunk
from app.retrieval.faiss_store import FAISSVectorStore


@pytest.fixture
def sample_chunks():
    """Create three sample IT support chunks."""
    return [
        DocumentChunk(
            chunk_id="IT-001",
            content="Reset your password using the password portal.",
            source="password_policy.txt",
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="IT-002",
            content="Reconnect to the VPN when the connection fails.",
            source="vpn_troubleshooting.txt",
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="IT-003",
            content="Contact the helpdesk for software installation.",
            source="helpdesk_support.txt",
            chunk_index=0,
        ),
    ]


@pytest.fixture
def sample_embeddings():
    """Use simple vectors so expected rankings are predictable."""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_add_and_count(tmp_path, sample_chunks, sample_embeddings):
    store = FAISSVectorStore(tmp_path)

    store.add_chunks(sample_chunks, sample_embeddings)

    assert store.count() == 3


def test_search_returns_correct_chunk(
    tmp_path, sample_chunks, sample_embeddings
):
    store = FAISSVectorStore(tmp_path)
    store.add_chunks(sample_chunks, sample_embeddings)

    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    results = store.search(query, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "IT-002"
    assert results[0]["source"] == "vpn_troubleshooting.txt"
    assert results[0]["distance"] == pytest.approx(0.0)


def test_persistence(tmp_path, sample_chunks, sample_embeddings):
    store = FAISSVectorStore(tmp_path)
    store.add_chunks(sample_chunks, sample_embeddings)

    # Simulate restarting the application.
    reloaded_store = FAISSVectorStore(tmp_path)

    assert reloaded_store.count() == 3

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = reloaded_store.search(query, top_k=1)

    assert results[0]["chunk_id"] == "IT-001"


def test_upsert_does_not_duplicate(
    tmp_path, sample_chunks, sample_embeddings
):
    store = FAISSVectorStore(tmp_path)

    store.add_chunks(sample_chunks, sample_embeddings)
    store.add_chunks(sample_chunks, sample_embeddings)

    assert store.count() == 3


def test_invalid_top_k(tmp_path):
    store = FAISSVectorStore(tmp_path)

    with pytest.raises(ValueError, match="top_k"):
        store.search(np.array([1.0, 0.0]), top_k=0)


def test_dimension_mismatch(
    tmp_path, sample_chunks, sample_embeddings
):
    store = FAISSVectorStore(tmp_path)
    store.add_chunks(sample_chunks, sample_embeddings)

    with pytest.raises(ValueError, match="dimension"):
        store.search(np.array([1.0, 0.0]), top_k=1)
