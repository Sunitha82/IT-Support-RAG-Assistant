
"""Manual smoke test for semantic document retrieval."""

import logging

from app.retrieval.retriever import SemanticRetriever

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(message)s",
)


def main() -> None:
    """Run a question against the indexed IT knowledge base."""

    retriever = SemanticRetriever()

    question = input(
        "\nAsk an IT support question: "
    ).strip()

    if not question:
        print("Please enter a valid question.")
        return

    results = retriever.retrieve(
        question=question,
        top_k=3,
    )

    if not results:
        print("No indexed documents were found.")
        return

    print("\n=== RETRIEVAL RESULTS ===")

    for result in results:
        print("\n" + "-" * 55)
        print(f"Rank: {result.rank}")
        print(f"Source: {result.source}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Cosine distance: {result.distance:.4f}")
        print(f"Content:\n{result.content}")

    print("\n" + "=" * 55)
    print(f"Retrieved {len(results)} chunks.")


if __name__ == "__main__":
    main()
