
"""Unit tests for the RAG orchestration service."""

from unittest.mock import Mock

import pytest

from app.retrieval.retriever import RetrievalResult
from app.services.rag_service import RAGService
from app.services.rag_service import NO_EVIDENCE_MESSAGE


@pytest.fixture
def rag_service():
    """Create a RAG service with mocked dependencies."""

    retriever = Mock()
    llm_client = Mock()

    retriever.retrieve.return_value = [
        RetrievalResult(
            chunk_id="vpn_troubleshooting.txt::0",
            source="vpn_troubleshooting.txt",
            chunk_index=0,
            content="Disconnect and reconnect to the VPN.",
            distance=0.21,
            rank=1,
        )
    ]

    llm_client.generate.return_value = (
        "Disconnect and reconnect to the VPN. "
        "Source: vpn_troubleshooting.txt"
    )

    return RAGService(
        retriever=retriever,
        llm_client=llm_client,
    )


def test_generates_answer_from_retrieved_context(rag_service):
    """An answer should be generated when evidence exists."""

    response = rag_service.ask("How do I fix my VPN?")

    assert "Disconnect and reconnect" in response.answer
    assert response.question == "How do I fix my VPN?"


def test_returns_document_sources(rag_service):
    """The response should include retrieved source filenames."""

    response = rag_service.ask("How do I fix my VPN?")

    assert response.sources == ["vpn_troubleshooting.txt"]


def test_passes_evidence_to_llm(rag_service):
    """The LLM prompt must contain the retrieved evidence."""

    rag_service.ask("How do I fix my VPN?")

    call_args = rag_service.llm_client.generate.call_args.kwargs

    assert "Disconnect and reconnect to the VPN" in (
        call_args["user_prompt"]
    )

    assert "How do I fix my VPN?" in call_args["user_prompt"]


def test_no_evidence_returns_fallback(rag_service):
    """An empty retrieval result must produce a fallback answer."""

    rag_service.retriever.retrieve.return_value = []

    response = rag_service.ask(
        "How many days of maternity leave do employees receive?"
    )

    assert response.answer == NO_EVIDENCE_MESSAGE
    assert response.sources == []


def test_no_evidence_skips_azure_call(rag_service):
    """Azure must not be called when no evidence is available."""

    rag_service.retriever.retrieve.return_value = []

    rag_service.ask("What is the maternity leave policy?")

    rag_service.llm_client.generate.assert_not_called()


def test_empty_question_is_rejected(rag_service):
    """Empty questions must fail before retrieval."""

    with pytest.raises(ValueError):
        rag_service.ask("")


def test_whitespace_question_is_rejected(rag_service):
    """Whitespace-only questions must fail validation."""

    with pytest.raises(ValueError):
        rag_service.ask("   ")


def test_duplicate_sources_are_removed(rag_service):
    """Multiple chunks from one document should produce one source."""

    rag_service.retriever.retrieve.return_value = [
        RetrievalResult(
            chunk_id="vpn_troubleshooting.txt::0",
            source="vpn_troubleshooting.txt",
            chunk_index=0,
            content="Reconnect to the VPN.",
            distance=0.21,
            rank=1,
        ),
        RetrievalResult(
            chunk_id="vpn_troubleshooting.txt::1",
            source="vpn_troubleshooting.txt",
            chunk_index=1,
            content="Contact the network support team.",
            distance=0.48,
            rank=2,
        ),
    ]

    response = rag_service.ask("VPN troubleshooting")

    assert response.sources == ["vpn_troubleshooting.txt"]


def test_question_is_trimmed(rag_service):
    """Leading and trailing spaces should be removed."""

    response = rag_service.ask("  How do I fix my VPN?  ")

    assert response.question == "How do I fix my VPN?"

    rag_service.retriever.retrieve.assert_called_once_with(
    "How do I fix my VPN?",
    top_k=5
    )
