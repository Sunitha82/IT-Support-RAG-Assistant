
"""Orchestrate retrieval-augmented IT support answers."""

from dataclasses import dataclass, field

from app.llm.azure_client import AzureLLMClient
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.retriever import SemanticRetriever


@dataclass(frozen=True)
class RetrievalDetail:
    """Information about one retrieved document chunk."""

    source: str
    chunk_id: str
    rank: int
    distance: float
    similarity: float


@dataclass(frozen=True)
class RAGResponse:
    """An answer, document sources and retrieval information."""

    question: str
    answer: str
    sources: list[str]
    retrieval_details: list[RetrievalDetail] = field(
        default_factory=list
    )


class RAGService:
    """Combine semantic retrieval with Azure OpenAI generation."""

    NO_EVIDENCE_MESSAGE = (
        "I don't have enough information in the IT knowledge base."
    )

    def __init__(
        self,
        retriever: SemanticRetriever | None = None,
        llm_client: AzureLLMClient | None = None,
    ) -> None:
        self.retriever = (
            retriever if retriever is not None
            else SemanticRetriever()
        )

        self.llm_client = (
            llm_client if llm_client is not None
            else AzureLLMClient()
        )

    def ask(self, question: str) -> RAGResponse:
        """Retrieve evidence and generate a grounded answer."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must not be empty")

        question = question.strip()

        # Step 1: Retrieve relevant document chunks.
        results = self.retriever.retrieve(question)

        # Step 2: Avoid calling the LLM if no evidence is found.
        if not results:
            return RAGResponse(
                question=question,
                answer=self.NO_EVIDENCE_MESSAGE,
                sources=[],
                retrieval_details=[],
            )

        # Step 3: Prepare retrieval information for the UI.
        retrieval_details = [
            RetrievalDetail(
                source=result.source,
                chunk_id=result.chunk_id,
                rank=result.rank,
                distance=round(float(result.distance), 4),
                similarity=round(
                    1.0 - float(result.distance), 4
                ),
            )
            for result in results
        ]

        # Step 4: Build the evidence-grounded prompt.
        user_prompt = build_user_prompt(question, results)

        # Step 5: Generate an answer using Azure OpenAI.
        answer = self.llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Step 6: Return the answer and retrieved sources.
        sources = list(dict.fromkeys(
            result.source for result in results
        ))

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            retrieval_details=retrieval_details,
        )
