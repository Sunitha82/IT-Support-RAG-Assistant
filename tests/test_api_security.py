
"""Tests for API key authentication."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes.rag import get_rag_service
from app.core.config import get_settings


TEST_API_KEY = "test-only-api-key-do-not-use-in-production"


@pytest.fixture
def client(monkeypatch):
    """Create a test client with a temporary API key."""

    monkeypatch.setenv("RAG_API_KEY", TEST_API_KEY)

    # Prevent the tests from calling the real RAG service.
    class FakeRAGService:
        def ask(self, question):
            raise AssertionError(
                "RAG service should not be called "
                "during authentication tests."
            )

    app.dependency_overrides[get_rag_service] = (
        lambda: FakeRAGService()
    )

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_missing_api_key(client):
    """Requests without an API key must be rejected."""

    response = client.post(
        "/ask",
        json={"question": "How do I reset my password?"},
    )

    assert response.status_code == 401


def test_invalid_api_key(client):
    """Requests with an incorrect API key must be rejected."""

    response = client.post(
        "/ask",
        json={"question": "How do I reset my password?"},
        headers={"X-API-Key": "incorrect-key"},
    )

    assert response.status_code == 401



def test_missing_server_configuration(client, monkeypatch):
    """Authentication must fail closed if the server key is missing."""

    monkeypatch.setenv("RAG_API_KEY", "")
    get_settings.cache_clear()

    response = client.post(
        "/ask",
        json={"question": "How do I reset my password?"},
        headers={"X-API-Key": TEST_API_KEY},
    )

    assert response.status_code == 503



def test_valid_api_key_passes_authentication(client):
    """A valid key must pass authentication and reach the RAG service."""

    response = client.post(
        "/ask",
        json={"question": "How do I reset my password?"},
        headers={"X-API-Key": TEST_API_KEY},
    )

    # Our fake RAG service raises an AssertionError,
    # which the existing route converts into HTTP 500.
    # This confirms authentication succeeded.
    assert response.status_code == 500
