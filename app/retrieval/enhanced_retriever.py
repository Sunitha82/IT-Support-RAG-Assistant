"""Multi-query semantic retrieval, RRF, reranking and parent expansion."""
import re
from pathlib import Path
from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.observability.langfuse_tracing import observation
from app.retrieval.parent_child_store import ParentChildStore
from app.retrieval.retriever import RetrievalResult
from app.retrieval.reranker import CrossEncoderReranker

def expand_it_terms(question: str) -> str:
    """Explicit vocabulary expansion, not an LLM-generated rewrite."""
    replacements = (
        (r"\b(?:secure work connection|secure remote connection|virtual private network)\b", "VPN"),
        (r"\bcompany websites\b", "internal applications"),
    )
    for pattern, replacement in replacements:
        question = re.sub(pattern, replacement, question, flags=re.I)
    return question

class EnhancedRetriever:
    def __init__(self, store=None, reranker=None):
        settings = get_settings()
        self.store = store if store is not None else ParentChildStore(Path(settings.parent_child_faiss_dir))
        self.reranker = reranker if reranker is not None else CrossEncoderReranker()

    def retrieve(self, question: str, top_k: int | None = None, metadata_filter=None):
        settings = get_settings()
        limit = top_k if top_k is not None else settings.retrieval_top_k
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("top_k must be a positive integer")
        search_question = expand_it_terms(question)
        queries = list(dict.fromkeys([
            question, search_question,
            f"{search_question} troubleshooting steps and requirements",
            f"{search_question} restrictions, exceptions and escalation",
        ]))
        try:
            with observation("multi-query-search", as_type="retriever",
                             input={"query_count": len(queries), "candidate_limit": max(limit * 3, 18)}) as span:
                candidates = self.store.multi_query_search(
                    question=queries[0], additional_queries=queries[1:],
                    top_k=max(limit * 3, 18), metadata_filter=metadata_filter)
                span.update(output={"candidate_count": len(candidates)})
            with observation("cross-encoder-reranking", input={"candidate_count": len(candidates)}) as span:
                ranked = self.reranker.rerank(search_question, candidates, top_k=len(candidates)) if candidates else []
                span.update(output={"ranked_count": len(ranked)})
        except Exception as exc:
            raise VectorStoreError("Enhanced retrieval failed") from exc
        results = []
        seen_parents = set()
        with observation("parent-expansion-and-filtering") as span:
            for match in ranked:
                if match["distance"] > settings.retrieval_max_distance or match["rerank_score"] < 0.0:
                    continue
                if match["parent_id"] in seen_parents:
                    continue
                seen_parents.add(match["parent_id"])
                results.append(RetrievalResult(
                    chunk_id=match["parent_id"], source=match["source"],
                    chunk_index=len(results), content=match["content"],
                    distance=float(match["distance"]), rank=len(results) + 1,
                    child_id=match["child_id"], rerank_score=float(match["rerank_score"]),
                    rrf_score=float(match["rrf_score"])))
                if len(results) >= limit:
                    break
            span.update(output={"parent_count": len(results)},
                        metadata={"distance_limit": settings.retrieval_max_distance})
        return results
