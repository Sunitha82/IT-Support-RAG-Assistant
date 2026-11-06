
"""Persistent ChromaDB vector store for IT support documents."""

import logging
from pathlib import Path

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection

from app.core.config import get_settings
from app.ingestion.chunking import DocumentChunk

logger = logging.getLogger(__name__)

# Versioned collection for the refactored RAG implementation.
COLLECTION_NAME = "it_support_knowledge_v2"


class VectorStore:
    """Manage persistent document embeddings in ChromaDB."""

    def __init__(self, persist_directory: Path | None = None) -> None:
        settings = get_settings()

        self.persist_directory = (
            persist_directory
            if persist_directory is not None
            else settings.chroma_dir
        )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Store vectors on disk rather than in memory only.
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection: Collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={
                    "embedding_model": settings.embedding_model,
                },
            )
        )

        # Prevent mixing embeddings from different models.
        stored_model = self.collection.metadata.get(
            "embedding_model"
        )

        if stored_model != settings.embedding_model:
            raise ValueError(
                "Embedding model mismatch. Rebuild the collection "
                "before using a different embedding model."
            )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Insert or update chunks and their embeddings."""

        if not chunks:
            raise ValueError("No chunks provided")

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunk count and embedding count must match"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid values")

        ids = [chunk.chunk_id for chunk in chunks]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs detected")

        documents = [chunk.content for chunk in chunks]

        metadatas = [
            {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        logger.info(
            "Stored %d chunks in ChromaDB",
            len(chunks),
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict]:
        """Return the nearest document chunks."""

        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be 1D")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains invalid values")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        count = self.collection.count()

        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, count),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        matches = []

        for chunk_id, content, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append({
                "chunk_id": chunk_id,
                "content": content,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": float(distance),
            })

        return matches

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()
