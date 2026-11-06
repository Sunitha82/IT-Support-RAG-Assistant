
"""Document chunking utilities for the IT Support RAG application."""

import logging
from dataclasses import dataclass

from app.core.config import get_settings
from app.ingestion.document_loader import Document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a chunk and its source metadata."""

    chunk_id: str
    source: str
    chunk_index: int
    content: str




def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split documents while preserving numbered instructions.

    Uses paragraph and line boundaries, with word-based
    splitting only for oversized units.
    """
    import re

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    if not text or not text.strip():
        return []

    # Normalise whitespace while keeping paragraph boundaries.
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    units = []

    for line in lines:
        # Merge a standalone list number with its following text.
        if units and re.fullmatch(r"\d+[.)]", units[-1]):
            units[-1] = f"{units[-1]} {line}"
        else:
            units.append(line)

    # Split oversized units at word boundaries where possible.
    bounded_units = []

    for unit in units:
        while len(unit) > chunk_size:
            split_at = unit.rfind(" ", 0, chunk_size + 1)

            if split_at <= 0:
                split_at = chunk_size

            bounded_units.append(unit[:split_at].strip())
            unit = unit[split_at:].strip()

        if unit:
            bounded_units.append(unit)

    chunks = []
    current_units = []

    for unit in bounded_units:
        candidate = "\n".join(current_units + [unit])

        if len(candidate) <= chunk_size:
            current_units.append(unit)
            continue

        if current_units:
            chunks.append("\n".join(current_units))

        # Reuse complete previous lines, subject to overlap.
        carried = []

        for previous in reversed(current_units):
            proposed = [previous] + carried

            if len("\n".join(proposed)) > overlap:
                break

            if len("\n".join(proposed + [unit])) > chunk_size:
                break

            carried.insert(0, previous)

        current_units = carried + [unit]

    if current_units:
        chunks.append("\n".join(current_units))

    return chunks




def chunk_documents(
    documents: list[Document],
) -> list[DocumentChunk]:
    """
    Split loaded documents into chunks with source metadata.

    Chunk size and overlap are controlled by application settings.
    """
    settings = get_settings()
    all_chunks = []

    for document in documents:
        text_chunks = chunk_text(
            text=document.content,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        for index, content in enumerate(text_chunks):
            all_chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.source}::{index}",
                    source=document.source,
                    chunk_index=index,
                    content=content,
                )
            )

        logger.info(
            "Document %s produced %d chunks",
            document.source,
            len(text_chunks),
        )

    logger.info("Total chunks created: %d", len(all_chunks))

    return all_chunks
