
"""Pydantic schemas for IT Support RAG API requests and responses."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request containing an employee's IT support question."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The IT support question to answer",
    )


class RetrievalDetailResponse(BaseModel):
    """Details of a document chunk retrieved from ChromaDB."""

    source: str
    chunk_id: str
    rank: int
    distance: float
    similarity: float


class AskResponse(BaseModel):
    """RAG-generated answer, sources and retrieval information."""

    question: str
    answer: str
    sources: list[str]
    retrieval_details: list[RetrievalDetailResponse] = Field(
        default_factory=list
    )


class HealthResponse(BaseModel):
    """Application health response."""

    status: str


class ReadinessResponse(BaseModel):
    """Knowledge base readiness response."""

    status: str
    indexed_chunks: int
