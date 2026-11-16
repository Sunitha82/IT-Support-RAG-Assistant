
"""Build the separate parent-child FAISS index."""
import os
from app.core.config import get_settings
from app.ingestion.document_loader import load_documents
from app.ingestion.parent_child_chunking import chunk_documents_parent_child
from app.retrieval.parent_child_store import ParentChildStore
from app.ingestion.azure_blob_loader import load_documents_from_azure_blob


def main() -> None:
    settings = get_settings()

    # Load existing IT Support knowledge documents.
    
    document_source = os.getenv("DOCUMENT_SOURCE", "local").lower()

    if document_source == "azure_blob":
        documents = load_documents_from_azure_blob(
            storage_account="stitragdocs2026",
            container_name="it-support-documents",
        )
    else:
        documents = load_documents()

    # Generate parent and child chunks.
    parents, children = chunk_documents_parent_child(documents)

    print(f"Documents: {len(documents)}")
    print(f"Parent chunks: {len(parents)}")
    print(f"Child chunks: {len(children)}")

    # Store child embeddings and parent relationships separately.
    store = ParentChildStore(
        persist_directory=settings.parent_child_faiss_dir
    )

    store.add_chunks(parents, children)

    print(f"Indexed children: {store.vector_store.count()}")
    print(f"Parent records: {len(store.parents)}")
    print("Parent-child indexing completed.")


if __name__ == "__main__":
    main()
