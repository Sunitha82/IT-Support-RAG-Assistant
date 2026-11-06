
"""Smoke test for the Azure OpenAI connection."""

from app.llm.azure_client import AzureLLMClient


def main() -> None:
    """Send a simple request to verify the Azure deployment."""

    llm = AzureLLMClient()

    response = llm.generate(
        system_prompt=(
            "You are a helpful IT support assistant. "
            "Answer clearly and briefly."
        ),
        user_prompt=(
            "In one sentence, explain what a VPN is."
        ),
    )

    print("\n=== AZURE OPENAI RESPONSE ===")
    print(response)


if __name__ == "__main__":
    main()
