
"""Index IT support documents into ChromaDB."""

import logging

from app.ingestion.document_loader import load_documents
from app.ingestion.chunking import chunk_documents
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Load, chunk, embed and index documents."""

    # Step 1: Load the source documents.
    documents = load_documents()

    if not documents:
        raise ValueError("No documents found for indexing")

    logger.info("Documents loaded: %d", len(documents))

    # Step 2: Split documents into chunks.
    chunks = chunk_documents(documents)

    logger.info("Chunks created: %d", len(chunks))

    # Step 3: Generate semantic embeddings.
    embedding_service = EmbeddingService()

    texts = [chunk.content for chunk in chunks]

    embeddings = embedding_service.embed_documents(texts)

    logger.info("Embedding shape: %s", embeddings.shape)

    # Step 4: Persist chunks and vectors in ChromaDB.
    vector_store = VectorStore()

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    logger.info(
        "Total stored vectors: %d",
        vector_store.count(),
    )


if __name__ == "__main__":
    main()
