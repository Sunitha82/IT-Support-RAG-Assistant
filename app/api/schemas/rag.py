
"""Pydantic schemas for IT Support RAG API requests and responses."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of document chunks to retrieve"
    )


class RetrievalDetailResponse(BaseModel):
    """Details of a document chunk retrieved from ChromaDB."""

    source: str
    chunk_id: str
    rank: int
    distance: float
    similarity: float
    content: str
    child_id: str | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None


class AskResponse(BaseModel):
    """RAG-generated answer, sources and retrieval information."""

    question: str
    answer: str
    sources: list[str]
    retrieval_details: list[RetrievalDetailResponse] = Field(
        default_factory=list
    )
    privacy: dict = Field(default_factory=dict)
    trace_id: str | None = None
    tracing_status: str = "disabled_or_unavailable"


class HealthResponse(BaseModel):
    """Application health response."""

    status: str


class ReadinessResponse(BaseModel):
    """Knowledge base readiness response."""

    status: str
    indexed_chunks: int
