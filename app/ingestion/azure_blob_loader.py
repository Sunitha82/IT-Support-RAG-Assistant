"""Load IT Support documents directly from Azure Blob Storage."""

import logging

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from app.ingestion.document_loader import Document

logger = logging.getLogger(__name__)


def load_documents_from_azure_blob(
    storage_account: str,
    container_name: str,
) -> list[Document]:
    """Download non-empty UTF-8 TXT documents from Azure Blob Storage."""

    credential = DefaultAzureCredential()

    service = BlobServiceClient(
        account_url=f"https://{storage_account}.blob.core.windows.net",
        credential=credential,
    )

    container = service.get_container_client(container_name)
    documents = []

    for blob in sorted(container.list_blobs(), key=lambda item: item.name):
        if not blob.name.lower().endswith(".txt"):
            continue

        content = (
            container.download_blob(blob.name)
            .readall()
            .decode("utf-8")
            .strip()
        )

        if not content:
            logger.warning("Skipping empty Azure Blob: %s", blob.name)
            continue

        documents.append(
            Document(
                source=blob.name,
                content=content,
            )
        )

        logger.info("Loaded document from Azure Blob: %s", blob.name)

    if not documents:
        raise RuntimeError("No TXT documents found in Azure Blob Storage.")

    logger.info("Total Azure documents loaded: %d", len(documents))
    return documents