
from pathlib import Path

import chromadb

from document_loader import load_documents
from chunking import chunk_documents
from embeddings import generate_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "chroma_db"
DATA_PATH = PROJECT_ROOT / "data"
COLLECTION_NAME = "it_support_knowledge_v3"


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def build_vector_store():
    # 1. Load the documents
    documents = load_documents(DATA_PATH)
    print(f"Documents loaded: {len(documents)}")

    if not documents:
        raise ValueError("No documents found in data folder")

    # 2. Split into chunks
    chunks = chunk_documents(documents)
    print(f"Chunks created: {len(chunks)}")

    # 3. Extract text
    texts = [chunk["content"] for chunk in chunks]

    # 4. Generate embeddings
    embeddings = generate_embeddings(texts)
    print("Embedding shape:", embeddings.shape)

    # 5. Prepare IDs and metadata
    ids = [chunk["chunk_id"] for chunk in chunks]

    metadatas = [
        {
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"]
        }
        for chunk in chunks
    ]

    # 6. Store in ChromaDB
    collection = get_collection()

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )

    print(f"Total vectors stored: {collection.count()}")

    return collection


if __name__ == "__main__":
    build_vector_store()
