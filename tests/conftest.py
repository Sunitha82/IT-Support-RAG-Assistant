
"""Shared pytest fixtures for the IT Support RAG application."""

import pytest

from app.core.config import get_settings

"""Shared pytest configuration."""

import os

# Unit tests use local test credentials, not live Azure Key Vault.
os.environ["USE_AZURE_KEY_VAULT"] = "false"
os.environ["LANGFUSE_ENABLED"] = "false"



@pytest.fixture(autouse=True)
def clear_settings_cache():
    """
    Ensure each test reads its own environment configuration.

    Prevent a cached API key from leaking between tests.
    """
    get_settings.cache_clear()

    yield

    # Clear again so settings from this test are not reused.
    get_settings.cache_clear()
