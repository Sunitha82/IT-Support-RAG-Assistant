
"""Persistent FAISS vector store for IT support documents."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from app.core.config import get_settings
from app.ingestion.chunking import DocumentChunk

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Manage document embeddings using a persistent FAISS index."""

    def __init__(self, persist_directory: Path | None = None) -> None:
        settings = get_settings()

        self.persist_directory = Path(
            persist_directory
            if persist_directory is not None
            else settings.faiss_dir
        )

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.index_path = self.persist_directory / "index.faiss"
        self.metadata_path = self.persist_directory / "metadata.json"
        self.embedding_model = settings.embedding_model

        self.index = None
        self.documents = []

        # Load previously stored vectors if they exist.
        if self.index_path.exists() and self.metadata_path.exists():
            self._load()
        elif self.index_path.exists() or self.metadata_path.exists():
            raise ValueError(
                "Incomplete FAISS store. Both index and metadata are required."
            )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Insert or update document chunks and their embeddings."""

        if not chunks:
            raise ValueError("No chunks provided")

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count must match")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid values")

        ids = [chunk.chunk_id for chunk in chunks]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs detected")

        dimension = embeddings.shape[1]

        if dimension == 0:
            raise ValueError("Embedding dimension cannot be zero")

        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)
        elif self.index.d != dimension:
            raise ValueError("Embedding dimension mismatch")

        # Normalize embeddings for cosine similarity.
        embeddings = np.ascontiguousarray(embeddings)
        if np.any(np.linalg.norm(embeddings, axis=1) == 0):
            raise ValueError("Zero-length embeddings are not supported")

        faiss.normalize_L2(embeddings)

        # IndexFlatIP does not support direct upsert.
        # Reconstruct existing vectors, replace matching IDs,
        # and rebuild the index to preserve ChromaDB-like behaviour.
        existing_vectors = (
            np.vstack([
                self.index.reconstruct(i)
                for i in range(self.index.ntotal)
            ]).astype(np.float32)
            if self.index.ntotal
            else np.empty((0, dimension), dtype=np.float32)
        )

        existing = {
            document["chunk_id"]: (document, vector)
            for document, vector in zip(self.documents, existing_vectors)
        }

        for chunk, vector in zip(chunks, embeddings):
            existing[chunk.chunk_id] = (
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                },
                vector,
            )

        self.documents = [item[0] for item in existing.values()]
        all_vectors = np.ascontiguousarray(
            np.vstack([item[1] for item in existing.values()]),
            dtype=np.float32,
        )

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(all_vectors)
        self._save()

        logger.info("Stored %d chunks in FAISS", len(self.documents))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict]:
        """Return nearest chunks using cosine distance."""

        query_embedding = np.asarray(
            query_embedding, dtype=np.float32
        )

        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be 1D")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains invalid values")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if self.index is None or self.index.ntotal == 0:
            return []

        if query_embedding.shape[0] != self.index.d:
            raise ValueError("Query embedding dimension mismatch")

        if np.linalg.norm(query_embedding) == 0:
            raise ValueError("Zero-length query embedding")

        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1)
        )
        faiss.normalize_L2(query)

        scores, indices = self.index.search(
            query, min(top_k, self.index.ntotal)
        )

        matches = []

        for score, index_id in zip(scores[0], indices[0]):
            if index_id < 0:
                continue

            document = self.documents[int(index_id)]

            matches.append({
                **document,
                "distance": float(
                    max(0.0, min(2.0, 1.0 - float(score)))
                ),
            })

        return matches

    def count(self) -> int:
        """Return number of stored chunks."""
        return self.index.ntotal if self.index is not None else 0

    def _save(self) -> None:
        """Persist the FAISS index and document metadata."""

        faiss.write_index(self.index, str(self.index_path))

        payload = {
            "embedding_model": self.embedding_model,
            "documents": self.documents,
        }

        self.metadata_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        """Load the FAISS index and verify its metadata."""

        payload = json.loads(
            self.metadata_path.read_text(encoding="utf-8")
        )

        if payload["embedding_model"] != self.embedding_model:
            raise ValueError(
                "Embedding model mismatch. Rebuild the FAISS index."
            )

        self.index = faiss.read_index(str(self.index_path))
        self.documents = payload["documents"]

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                "FAISS index and metadata count mismatch"
            )

        logger.info(
            "Loaded %d chunks from FAISS", self.index.ntotal
        )
