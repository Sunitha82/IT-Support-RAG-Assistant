
from dotenv import load_dotenv

load_dotenv()

from app.core.keyvault import get_secret

secret_names = [
    "AZURE-OPENAI-KEY",
    "RAG-API-KEY",
    "langfuse-secret-key",
]

for name in secret_names:
    value = get_secret(name)
    print(f"{name}: {'OK' if value else 'EMPTY'}")

print("Reusable Key Vault module works!")
