"""Parent-child FAISS retrieval for the IT Support RAG application."""

import json
from pathlib import Path
import numpy as np

from app.ingestion.chunking import DocumentChunk
from app.ingestion.parent_child_chunking import ParentChunk, ChildChunk
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.faiss_store import FAISSVectorStore
from app.retrieval.reranker import CrossEncoderReranker

class ParentChildStore:
    """Index child chunks and retrieve their complete parent sections."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.vector_store = FAISSVectorStore(
            persist_directory=self.persist_directory / "children"
        )

        self.embedding_service = embedding_service or EmbeddingService()

        self.parents_path = self.persist_directory / "parents.json"
        self.parents = {}
        self.child_to_parent = {}

        if self.parents_path.exists():
            self._load_parents()

    def add_chunks(
        self,
        parents: list[ParentChunk],
        children: list[ChildChunk],
    ) -> None:
        """Index child embeddings and save their parent relationships."""

        parent_lookup = {p.parent_id: p for p in parents}

        if len(parent_lookup) != len(parents):
            raise ValueError("Duplicate parent IDs detected")

        if len({c.child_id for c in children}) != len(children):
            raise ValueError("Duplicate child IDs detected")

        for child in children:
            if child.parent_id not in parent_lookup:
                raise ValueError(
                    f"Missing parent for child: {child.child_id}"
                )

        if not children:
            raise ValueError("No child chunks provided")

        # Adapt ChildChunk to the existing FAISS DocumentChunk format.
        faiss_chunks = [
            DocumentChunk(
                chunk_id=child.child_id,
                source=child.source,
                chunk_index=index,
                content=child.content,
            )
            for index, child in enumerate(children)
        ]

        embeddings = self.embedding_service.embed_documents(
            [child.content for child in children]
        )

        self.vector_store.add_chunks(faiss_chunks, embeddings)

        self.parents.update(parent_lookup)

        for child in children:
            self.child_to_parent[child.child_id] = child.parent_id

        self._save_parents()


    def search(
        self,
        question: str,
        top_k: int = 3,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[dict]:
        """Search eligible child vectors and expand matches to parents."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if metadata_filter is not None and (
            not isinstance(metadata_filter, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in metadata_filter.items()
            )
        ):
            raise ValueError("metadata_filter must contain string keys and values")

        query_embedding = np.asarray(
            self.embedding_service.embed_query(question.strip()),
            dtype=np.float32,
        ).reshape(-1)

        index = self.vector_store.index

        if index is None or index.ntotal == 0:
            return []

        if query_embedding.shape[0] != index.d:
            raise ValueError("Query embedding dimension mismatch")

        norm = np.linalg.norm(query_embedding)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Invalid query embedding")

        query_embedding = query_embedding / norm

        candidates = []

        for position, document in enumerate(self.vector_store.documents):
            child_id = document["chunk_id"]
            parent_id = self.child_to_parent.get(child_id)

            if parent_id is None or parent_id not in self.parents:
                raise ValueError(f"Missing parent mapping for {child_id}")

            parent = self.parents[parent_id]

            metadata = {
                **parent.metadata,
                "source": parent.source,
            }

            document_id = metadata.get("document_id", "")
            if document_id.startswith("IT-"):
                metadata["department"] = "IT"
            elif document_id.startswith("HR-"):
                metadata["department"] = "HR"

            if metadata_filter and any(
                metadata.get(key) != value
                for key, value in metadata_filter.items()
            ):
                continue

            # FAISS IndexFlatIP supports exact vector reconstruction.
            vector = index.reconstruct(position)
            score = float(np.dot(query_embedding, vector))

            candidates.append((score, document, parent_id))

        candidates.sort(key=lambda item: item[0], reverse=True)

        results = []


        for score, document, parent_id in candidates:
            parent = self.parents[parent_id]

            results.append({
                "child_id": document["chunk_id"],
                "parent_id": parent_id,
                "source": parent.source,
                "matched_child": document["content"],
                "content": parent.content,
                "distance": max(0.0, min(2.0, 1.0 - score)),
                "metadata": {
                    **parent.metadata,
                    "source": parent.source,
                },
            })

            if len(results) >= top_k:
                break

        return results


    def multi_query_search(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None,
        additional_queries: list[str] | None = None,
    ) -> list[dict]:
        """
        Retrieve child chunks using multiple related questions.

        Deduplicate by child ID and combine rankings using
        Reciprocal Rank Fusion (RRF).
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        queries = [question.strip()]

        if additional_queries is not None:
            if not isinstance(additional_queries, list):
                raise ValueError("additional_queries must be a list")

            for query in additional_queries:
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("Additional queries must be non-empty strings")

                cleaned = query.strip()

                if cleaned not in queries:
                    queries.append(cleaned)

        fused_results = {}
        rrf_scores = {}

        for query in queries:
            matches = self.search(
                question=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )

            for rank, match in enumerate(matches, start=1):
                child_id = match["child_id"]

                # Reciprocal Rank Fusion
                rrf_scores[child_id] = (
                    rrf_scores.get(child_id, 0.0)
                    + 1.0 / (60 + rank)
                )

                if child_id not in fused_results:
                    fused_results[child_id] = match.copy()

        ranked_ids = sorted(
            rrf_scores,
            key=lambda child_id: rrf_scores[child_id],
            reverse=True,
        )

        results = []

        for child_id in ranked_ids[:top_k]:
            result = fused_results[child_id].copy()
            result["rrf_score"] = rrf_scores[child_id]
            results.append(result)

        return results

    def enhanced_search(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None,
        additional_queries: list[str] | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> list[dict]:
        """Multi-query retrieval followed by cross-encoder re-ranking."""

        candidates = self.multi_query_search(
            question=question,
            top_k=max(top_k * 3, 10),
            metadata_filter=metadata_filter,
            additional_queries=additional_queries,
        )

        if not candidates:
            return []

        ranker = reranker or CrossEncoderReranker()

        reranked = ranker.rerank(
            question=question,
            results=candidates,
            top_k=top_k,
        )

        # Keep individual child matches through re-ranking.
        # Expand and deduplicate parents only when building LLM context.
        return reranked


    def _save_parents(self) -> None:
        """Persist parent content and child-to-parent mappings."""

        payload = {
            "parents": {
                parent_id: {
                    "parent_id": parent.parent_id,
                    "source": parent.source,
                    "title": parent.title,
                    "content": parent.content,
                    "metadata": parent.metadata,
                }
                for parent_id, parent in self.parents.items()
            },
            "child_to_parent": self.child_to_parent,
        }

        self.parents_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _load_parents(self) -> None:
        """Restore parent records and child relationships."""

        payload = json.loads(
            self.parents_path.read_text(encoding="utf-8")
        )

        self.parents = {
            parent_id: ParentChunk(**data)
            for parent_id, data in payload["parents"].items()
        }

        self.child_to_parent = payload["child_to_parent"]
