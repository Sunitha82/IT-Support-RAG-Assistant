
"""Cross-encoder re-ranking for retrieved RAG child chunks."""

import logging
from threading import Lock

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-rank retrieved child chunks against the original question."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self._model = None
        self._load_lock = Lock()

    def _get_model(self) -> CrossEncoder:
        """Load the model only when first needed."""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    logger.info("Loading cross-encoder: %s", self.model_name)
                    self._model = CrossEncoder(self.model_name)

        return self._model

    def rerank(
        self,
        question: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Score and rank child chunks by relevance to the question."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be non-empty")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if not results:
            return []

        pairs = [
            (question.strip(), result["matched_child"])
            for result in results
        ]

        scores = self._get_model().predict(pairs)

        ranked = []

        for result, score in zip(results, scores, strict=True):
            item = result.copy()
            item["rerank_score"] = float(score)
            ranked.append(item)

        ranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return ranked[:top_k]
