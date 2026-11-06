
"""Automated tests for the FastAPI endpoints."""

from unittest.mock import Mock

import pytest
TEST_API_KEY = "test-only-api-key-do-not-use-in-production"
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes.rag import get_rag_service
from app.services.rag_service import RAGResponse


@pytest.fixture
def client(monkeypatch):
    """Create an API client with a mocked RAG service."""
    monkeypatch.setenv("RAG_API_KEY", TEST_API_KEY)
    mock_service = Mock()

    mock_service.ask.return_value = RAGResponse(
        question="How do I fix my VPN?",
        answer="Reconnect to the VPN.",
        sources=["vpn_troubleshooting.txt"],
    )

    app.dependency_overrides[get_rag_service] = (
        lambda: mock_service
    )

    with TestClient(
    app,
    headers={"X-API-Key": TEST_API_KEY},
    ) as test_client:
        yield test_client, mock_service

    app.dependency_overrides.clear()


def test_health_endpoint(client):
    test_client, _ = client

    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask_endpoint_success(client):
    test_client, mock_service = client

    response = test_client.post(
        "/ask",
        json={"question": "How do I fix my VPN?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Reconnect to the VPN."
    assert response.json()["sources"] == [
        "vpn_troubleshooting.txt"
    ]

    mock_service.ask.assert_called_once_with(
        "How do I fix my VPN?"
    )


def test_ask_rejects_empty_question(client):
    test_client, mock_service = client

    response = test_client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422
    mock_service.ask.assert_not_called()


def test_ask_rejects_whitespace_question(client):
    test_client, mock_service = client

    response = test_client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422
    mock_service.ask.assert_not_called()


def test_ask_rejects_missing_question(client):
    test_client, _ = client

    response = test_client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


def test_ask_handles_missing_evidence(client):
    test_client, mock_service = client

    mock_service.ask.return_value = RAGResponse(
        question="What is the maternity leave policy?",
        answer="I don't have enough information in the IT knowledge base.",
        sources=[],
    )

    response = test_client.post(
        "/ask",
        json={
            "question": "What is the maternity leave policy?"
        },
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert "don't have enough information" in (
        response.json()["answer"]
    )


def test_ask_handles_service_failure(client):
    test_client, mock_service = client

    mock_service.ask.side_effect = RuntimeError(
        "Simulated Azure service failure"
    )

    response = test_client.post(
        "/ask",
        json={"question": "How do I fix my VPN?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Unable to process the question"
    )
