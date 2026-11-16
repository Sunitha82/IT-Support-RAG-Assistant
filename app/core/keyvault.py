
"""Read application secrets securely from Azure Key Vault."""

import os
from functools import lru_cache

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


@lru_cache(maxsize=1)
def get_secret_client():
    """Create and reuse the Azure Key Vault client."""
    vault_url = os.getenv("AZURE_KEY_VAULT_URL")

    if not vault_url:
        raise ValueError("AZURE_KEY_VAULT_URL is not configured")

    credential = DefaultAzureCredential(
        exclude_interactive_browser_credential=True
    )

    return SecretClient(
        vault_url=vault_url,
        credential=credential,
    )


def get_secret(secret_name: str) -> str:
    """Retrieve a secret without logging its value."""
    secret = get_secret_client().get_secret(secret_name)

    if not secret.value:
        raise ValueError(f"Secret '{secret_name}' is empty")

    return secret.value
