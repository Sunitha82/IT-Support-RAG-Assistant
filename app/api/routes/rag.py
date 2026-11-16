
"""FastAPI routes for IT support question answering."""

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from app.api.security import verify_api_key
from app.guardrails.input_guard import InputGuardrailError

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
from app.retrieval.store_factory import create_vector_store
from app.core.config import get_settings
from app.retrieval.parent_child_store import ParentChildStore
from app.services.rag_service import RAGService
from app.ui.topic_catalog import build_topic_catalog

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/topics", dependencies=[Depends(verify_api_key)])
def available_topics():
    """Expose only topic metadata for documents in the selected active index."""
    try:
        settings = get_settings()
        if settings.retrieval_mode == "enhanced":
            store = ParentChildStore(settings.parent_child_faiss_dir)
            sources = {parent.source: parent.title for parent in store.parents.values()}
        else:
            store = create_vector_store()
            if hasattr(store, "documents"):
                metadata = store.documents
            else:
                metadata = store.collection.get(include=["metadatas"])["metadatas"]
            sources = {item["source"]: "" for item in metadata if item and item.get("source")}
        return {"topics": build_topic_catalog(sources), "source_count": len(sources)}
    except Exception:
        logger.warning("Topic metadata is unavailable")
        raise HTTPException(status_code=503, detail="Available topics could not be loaded") from None


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
        settings = get_settings()
        if settings.retrieval_mode == "enhanced":
            store = ParentChildStore(settings.parent_child_faiss_dir)
            count = store.vector_store.count()
            # Child vectors need their parent text and relationships to answer.
            child_ids = {doc["chunk_id"] for doc in store.vector_store.documents}
            if any(
                store.child_to_parent.get(child_id) not in store.parents
                for child_id in child_ids
            ):
                raise ValueError("Parent-child index has missing parent records")
        else:
            store = create_vector_store()
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
        result = service.ask_with_tracing(request.question, top_k=request.top_k)

        return AskResponse(
            question=result.question,
            answer=result.answer,
            sources=result.sources,
            privacy=result.privacy,
            trace_id=result.trace_id,
            tracing_status=result.tracing_status,
            retrieval_details=[
                {
                    "source": detail.source,
                    "chunk_id": detail.chunk_id,
                    "rank": detail.rank,
                    "distance": detail.distance,
                    "similarity": detail.similarity,
                    "content": detail.content,
                    "child_id": detail.child_id,
                    "rerank_score": detail.rerank_score,
                    "rrf_score": detail.rrf_score,
                }
                for detail in result.retrieval_details
            ],
        )
    except InputGuardrailError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": str(exc)},
        ) from None

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
