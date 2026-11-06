
"""Tests for Azure OpenAI client error handling."""

from unittest.mock import Mock
from app.core.exceptions import LLMResponseError

import httpx
import pytest

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.core.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.azure_client import AzureLLMClient


@pytest.fixture
def azure_client(monkeypatch):
    """Create an Azure client without making network calls."""

    monkeypatch.setenv(
        "AZURE_OPENAI_DEPLOYMENT",
        "test-deployment",
    )

    # Clear cached settings if configuration uses lru_cache.
    from app.core.config import get_settings
    get_settings.cache_clear()

    mock_sdk_client = Mock()

    client = AzureLLMClient(client=mock_sdk_client)

    yield client, mock_sdk_client

    get_settings.cache_clear()


def make_http_error(error_class, status_code):
    """Create an Azure SDK HTTP error."""

    request = httpx.Request(
        "POST",
        "https://example.com/openai/test",
    )

    response = httpx.Response(
        status_code=status_code,
        request=request,
    )

    return error_class(
        "Simulated Azure failure",
        response=response,
        body=None,
    )


def test_azure_timeout(azure_client):
    client, mock_sdk = azure_client

    request = httpx.Request(
        "POST",
        "https://example.com/openai/test",
    )

    mock_sdk.chat.completions.create.side_effect = (
        APITimeoutError(request=request)
    )

    with pytest.raises(LLMTimeoutError):
        client.generate("System prompt", "User prompt")


def test_azure_rate_limit(azure_client):
    client, mock_sdk = azure_client

    mock_sdk.chat.completions.create.side_effect = (
        make_http_error(RateLimitError, 429)
    )

    with pytest.raises(LLMRateLimitError):
        client.generate("System prompt", "User prompt")


def test_azure_authentication_failure(azure_client):
    client, mock_sdk = azure_client

    mock_sdk.chat.completions.create.side_effect = (
        make_http_error(AuthenticationError, 401)
    )

    with pytest.raises(LLMAuthenticationError):
        client.generate("System prompt", "User prompt")


def test_azure_connection_failure(azure_client):
    client, mock_sdk = azure_client

    request = httpx.Request(
        "POST",
        "https://example.com/openai/test",
    )

    mock_sdk.chat.completions.create.side_effect = (
        APIConnectionError(request=request)
    )

    with pytest.raises(LLMUnavailableError):
        client.generate("System prompt", "User prompt")


def test_azure_returns_no_choices(azure_client):
    """Azure responses without choices must be rejected."""

    client, mock_sdk = azure_client

    mock_sdk.chat.completions.create.return_value = Mock(
        choices=[]
    )

    with pytest.raises(LLMResponseError):
        client.generate(
            "System prompt",
            "User prompt",
        )


def test_azure_returns_empty_answer(azure_client):
    """Empty Azure-generated answers must be rejected."""

    client, mock_sdk = azure_client

    mock_sdk.chat.completions.create.return_value = Mock(
        choices=[
            Mock(
                message=Mock(content="   ")
            )
        ]
    )

    with pytest.raises(LLMResponseError):
        client.generate(
            "System prompt",
            "User prompt",
        )

