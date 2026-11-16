
import os

from azure.identity import AzureCliCredential
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv

load_dotenv()

vault_url = os.getenv("AZURE_KEY_VAULT_URL")

if not vault_url:
    raise ValueError("AZURE_KEY_VAULT_URL is missing from .env")

# Authenticate using the Azure CLI login.
credential = AzureCliCredential()

# Connect to Azure Key Vault.
client = SecretClient(
    vault_url=vault_url,
    credential=credential,
)

# Verify access without displaying secret values.
secret_names = [
    "AZURE-OPENAI-KEY",
    "RAG-API-KEY",
    "langfuse-secret-key",
]

for name in secret_names:
    secret = client.get_secret(name)
    print(f"{name}: {'OK' if secret.value else 'EMPTY'}")

print("Azure Key Vault connection successful!")
