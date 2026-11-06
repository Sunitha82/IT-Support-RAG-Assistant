
"""Manual smoke test for the embedding service."""

from app.retrieval.embeddings import EmbeddingService


def main() -> None:
    """Generate sample document and query embeddings."""
    service = EmbeddingService()

    documents = [
        "Employees must connect using the company VPN.",
        "Passwords must be changed every 90 days.",
        "The IT helpdesk operates from 9 AM to 6 PM.",
    ]

    document_embeddings = service.embed_documents(documents)

    print("Document embeddings:", document_embeddings.shape)

    query = "How do I connect to the company network?"
    query_embedding = service.embed_query(query)

    print("Query embedding:", query_embedding.shape)

    # Normalized vectors allow cosine similarity via dot product.
    similarities = document_embeddings @ query_embedding

    for text, score in zip(documents, similarities):
        print(f"\nSimilarity: {score:.4f}")
        print(f"Document: {text}")


if __name__ == "__main__":
    main()
