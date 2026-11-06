
"""Document loading utilities for the IT Support RAG application."""

import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """Represents a source document and its metadata."""

    source: str
    content: str


def load_documents(data_dir: Path | None = None) -> list[Document]:
    """
    Load all non-empty UTF-8 TXT files from the data directory.

    Args:
        data_dir: Optional directory containing source documents.

    Returns:
        A list of Document objects.

    Raises:
        FileNotFoundError: If the data directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    directory = data_dir or get_settings().data_dir

    if not directory.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {directory}"
        )

    documents = []

    for file_path in sorted(directory.glob("*.txt")):
        try:
            content = file_path.read_text(encoding="utf-8").strip()

        except (OSError, UnicodeError):
            logger.exception(
                "Failed to read document: %s", file_path.name
            )
            raise

        if not content:
            logger.warning(
                "Skipping empty document: %s", file_path.name
            )
            continue

        documents.append(
            Document(
                source=file_path.name,
                content=content,
            )
        )

        logger.info("Loaded document: %s", file_path.name)

    logger.info("Total documents loaded: %d", len(documents))

    return documents
