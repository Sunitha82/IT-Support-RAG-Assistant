
from vector_store import get_collection
from embeddings import generate_embeddings


def search_documents(question, top_k=3):
    collection = get_collection()

    if collection.count() == 0:
        print("Vector database is empty. Run vector_store.py first.")
        return []

    # Convert the question into an embedding
    query_embedding = generate_embeddings([question])[0]

    # Search for nearest vectors
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    matches = []

    for doc_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        matches.append({
            "id": doc_id,
            "content": document,
            "source": metadata["source"],
            "distance": distance
        })

    return matches


if __name__ == "__main__":
    question = input("Ask an IT support question: ").strip()

    if not question:
        print("Please enter a question.")
    else:
        results = search_documents(question, top_k=3)

        print(f"\nQuestion: {question}")

        for index, result in enumerate(results, start=1):
            print("\n------------------------")
            print(f"Rank: {index}")
            print("Source:", result["source"])
            print("Distance:", round(result["distance"], 4))
            print("Content:", result["content"])
