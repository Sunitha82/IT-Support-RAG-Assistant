
"""API key authentication for protected FastAPI endpoints."""


from app.core.config import get_settings
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def verify_api_key(
    provided_key: str | None = Security(api_key_header),
) -> None:
    """Reject requests without a valid API key."""

    expected_key = get_settings().rag_api_key

    # Fail closed if the server key is not configured.
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if (
        not provided_key
        or not secrets.compare_digest(
            provided_key,
            expected_key,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "APIKey"},
        )
