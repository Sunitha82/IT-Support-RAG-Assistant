
"""Tests for API error handling."""

from unittest.mock import Mock

import pytest
TEST_API_KEY = "test-only-api-key-do-not-use-in-production"
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes.rag import get_rag_service
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



@pytest.fixture
def client(monkeypatch):
    """Create an authenticated test client with a mocked RAG service."""

    monkeypatch.setenv("RAG_API_KEY", TEST_API_KEY)

    mock_service = Mock()
    app.dependency_overrides[get_rag_service] = lambda: mock_service

    try:
        with TestClient(
            app,
            headers={"X-API-Key": TEST_API_KEY},
        ) as test_client:
            yield test_client, mock_service
    finally:
        app.dependency_overrides.clear()

@pytest.mark.parametrize(
    "exception, expected_status",
    [
        (LLMTimeoutError("Timeout"), 504),
        (LLMRateLimitError("Rate limit"), 429),
        (LLMAuthenticationError("Invalid credentials"), 503),
        (LLMUnavailableError("Connection failed"), 503),
        (LLMResponseError("Empty response"), 502),
        (LLMServiceError("Service error"), 502),
    ],
)
def test_ask_handles_llm_errors(
    client,
    exception,
    expected_status,
):
    """Verify that Azure failures map to safe HTTP responses."""

    test_client, mock_service = client
    mock_service.ask_with_tracing.side_effect = exception

    response = test_client.post(
        "/ask",
        json={"question": "How do I fix my VPN?"},
    )

    assert response.status_code == expected_status
    assert "detail" in response.json()

    # Ensure internal exception messages are not exposed.
    assert str(exception) not in response.json()["detail"]

    # Every response should have a request ID.
    assert response.headers.get("X-Request-ID")


def test_unexpected_error_returns_500(client):
    """Unexpected failures should return a generic error."""

    test_client, mock_service = client
    mock_service.ask_with_tracing.side_effect = RuntimeError(
        "Sensitive internal failure details"
    )

    response = test_client.post(
        "/ask",
        json={"question": "How do I fix my VPN?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Unable to process the question"
    )
    assert "Sensitive internal failure details" not in response.text


def test_vector_store_failure_returns_503(client):
    """ChromaDB failures should return HTTP 503."""

    test_client, mock_service = client

    mock_service.ask_with_tracing.side_effect = VectorStoreError(
        "Sensitive ChromaDB internal error"
    )

    response = test_client.post(
        "/ask",
        json={"question": "How do I fix my VPN?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The knowledge base is temporarily unavailable."
    )
    assert "Sensitive ChromaDB internal error" not in response.text
    assert response.headers.get("X-Request-ID")


def test_embedding_failure_returns_503(client):
    """Embedding failures should return HTTP 503."""

    test_client, mock_service = client

    mock_service.ask_with_tracing.side_effect = EmbeddingServiceError(
        "Sensitive embedding model error"
    )

    response = test_client.post(
        "/ask",
        json={"question": "How do I reset my password?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The search service is temporarily unavailable."
    )
    assert "Sensitive embedding model error" not in response.text
    assert response.headers.get("X-Request-ID")
