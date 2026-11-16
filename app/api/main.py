
"""FastAPI application entry point."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from app.observability.langfuse_tracing import flush_traces

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.rag import router
from app.core.config import get_settings
from app.core.logging_config import configure_logging


configure_logging()

logger = logging.getLogger("app.api")

settings = get_settings()


@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        flush_traces()

app = FastAPI(
    lifespan=lifespan,
    title=settings.app_name,
    description="Internal IT Support Assistant powered by RAG",
    version="1.0.0",
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log each request with an ID and execution time."""

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    request.state.request_id = request_id

    try:
        response = await call_next(request)

    except Exception:
        logger.exception(
            "Unhandled request exception",
            extra={"request_id": request_id},
        )

        response = JSONResponse(
            status_code=500,
            content={
                "detail": "An internal server error occurred.",
                "request_id": request_id,
            },
        )

    duration_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    return response


app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "IT Support RAG API is running",
        "documentation": "/docs",
    }
