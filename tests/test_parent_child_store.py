
"""Tests for parent-child FAISS storage and retrieval."""

import numpy as np

from app.ingestion.parent_child_chunking import ParentChunk, ChildChunk
from app.retrieval.parent_child_store import ParentChildStore


class FakeEmbeddingService:
    """Deterministic embeddings without downloading a model."""

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            if "VPN" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "password" in text.lower():
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query):
        return self.embed_documents([query])[0]


def test_child_retrieval_expands_to_parent(tmp_path):
    store = ParentChildStore(
        persist_directory=tmp_path / "parent_child",
        embedding_service=FakeEmbeddingService(),
    )

    parents = [
        ParentChunk(
            parent_id="IT-001::parent-1",
            source="vpn.txt",
            title="VPN Guide",
            content="Complete VPN troubleshooting procedure.",
        ),
        ParentChunk(
            parent_id="IT-002::parent-1",
            source="password.txt",
            title="Password Policy",
            content="Complete password policy.",
        ),
    ]

    children = [
        ChildChunk(
            child_id="IT-001::child-1",
            parent_id="IT-001::parent-1",
            source="vpn.txt",
            content="Check VPN connection.",
        ),
        ChildChunk(
            child_id="IT-002::child-1",
            parent_id="IT-002::parent-1",
            source="password.txt",
            content="Change your password every 90 days.",
        ),
    ]

    store.add_chunks(parents, children)

    results = store.search("VPN not connecting", top_k=1)

    assert len(results) == 1
    assert results[0]["child_id"] == "IT-001::child-1"
    assert results[0]["parent_id"] == "IT-001::parent-1"
    assert results[0]["content"] == "Complete VPN troubleshooting procedure."


def test_parent_mapping_survives_restart(tmp_path):
    directory = tmp_path / "parent_child"

    store = ParentChildStore(
        persist_directory=directory,
        embedding_service=FakeEmbeddingService(),
    )

    store.add_chunks(
        parents=[
            ParentChunk(
                parent_id="IT-001::parent-1",
                source="vpn.txt",
                title="VPN Guide",
                content="Complete VPN troubleshooting procedure.",
            )
        ],
        children=[
            ChildChunk(
                child_id="IT-001::child-1",
                parent_id="IT-001::parent-1",
                source="vpn.txt",
                content="Check VPN connection.",
            )
        ],
    )

    # Simulate restarting the application.
    restarted_store = ParentChildStore(
        persist_directory=directory,
        embedding_service=FakeEmbeddingService(),
    )

    results = restarted_store.search("VPN problem", top_k=1)

    assert len(results) == 1
    assert results[0]["parent_id"] == "IT-001::parent-1"
    assert results[0]["content"] == "Complete VPN troubleshooting procedure."
