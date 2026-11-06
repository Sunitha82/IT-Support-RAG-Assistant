
"""FastAPI routes for IT support question answering."""

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from app.api.security import verify_api_key

from app.api.schemas.rag import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ReadinessResponse,
)
from app.core.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServiceError,
    LLMTimeoutError,
    LLMUnavailableError,
    EmbeddingServiceError,
    VectorStoreError,
)
from app.retrieval.vector_store import VectorStore
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()


# -------------------------------------------------
# Reusable RAG service
# -------------------------------------------------

@lru_cache
def get_rag_service() -> RAGService:
    """Create the RAG service once and reuse it."""
    return RAGService()


# -------------------------------------------------
# Health endpoint
# -------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check whether the API process is running."""
    return HealthResponse(status="healthy")


# -------------------------------------------------
# Readiness endpoint
# -------------------------------------------------

@router.get("/ready", response_model=ReadinessResponse)
def ready() -> ReadinessResponse:
    """Check whether indexed documents are available."""

    try:
        store = VectorStore()
        count = store.count()

        if count == 0:
            raise HTTPException(
                status_code=503,
                detail="Knowledge base has no indexed documents",
            )

        return ReadinessResponse(
            status="ready",
            indexed_chunks=count,
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Readiness check failed")
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is unavailable",
        ) from None


# -------------------------------------------------
# Question-answering endpoint
# -------------------------------------------------

@router.post(
    "/ask",
    response_model=AskResponse,
    dependencies=[Depends(verify_api_key)],
)
def ask_question(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    """Answer an IT support question using RAG."""

    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be blank",
        )

    try:
        result = service.ask(request.question)

        return AskResponse(
            question=result.question,
            answer=result.answer,
            sources=result.sources,
            retrieval_details=[
                {
                    "source": detail.source,
                    "chunk_id": detail.chunk_id,
                    "rank": detail.rank,
                    "distance": detail.distance,
                    "similarity": detail.similarity,
                }
                for detail in result.retrieval_details
            ],
        )

    except LLMTimeoutError:
        logger.warning("LLM request timed out")
        raise HTTPException(
            status_code=504,
            detail="The AI service timed out. Please try again.",
        ) from None

    except LLMRateLimitError:
        logger.warning("LLM rate limit exceeded")
        raise HTTPException(
            status_code=429,
            detail="The AI service is busy. Please try again later.",
        ) from None

    except LLMAuthenticationError:
        logger.error("LLM authentication failed")
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable.",
        ) from None

    except LLMUnavailableError:
        logger.error("LLM connection failed")
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable.",
        ) from None

    except LLMResponseError:
        logger.error("LLM returned an invalid response")
        raise HTTPException(
            status_code=502,
            detail="The AI service returned an invalid response.",
        ) from None

    except LLMServiceError:
        logger.error("LLM service request failed")
        raise HTTPException(
            status_code=502,
            detail="The AI service encountered an error.",
        ) from None

    except VectorStoreError:
        logger.error("Knowledge base retrieval failed")
        raise HTTPException(
            status_code=503,
            detail="The knowledge base is temporarily unavailable.",
        ) from None

    except EmbeddingServiceError:
        logger.error("Question embedding generation failed")
        raise HTTPException(
            status_code=503,
            detail="The search service is temporarily unavailable.",
        ) from None

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected RAG question-answering failure")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question",
        ) from None
