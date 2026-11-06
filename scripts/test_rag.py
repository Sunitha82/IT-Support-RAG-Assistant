
"""End-to-end smoke test for the IT Support RAG pipeline."""

from app.services.rag_service import RAGService


def main() -> None:
    """Ask an IT question and display the grounded answer."""

    rag = RAGService()

    question = input("\nAsk an IT support question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    response = rag.ask(question)

    print("\n=== IT SUPPORT ANSWER ===")
    print(response.answer)

    print("\n=== RETRIEVED SOURCES ===")

    if response.sources:
        for source in response.sources:
            print(f"- {source}")
    else:
        print("No supporting documents retrieved.")


if __name__ == "__main__":
    main()
