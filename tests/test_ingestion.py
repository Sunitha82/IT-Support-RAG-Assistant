
"""Automated tests for document loading and chunking."""

import pytest

from app.ingestion.document_loader import (
    Document,
    load_documents,
)
from app.ingestion.chunking import (
    chunk_text,
    chunk_documents,
)


def test_load_documents(tmp_path):
    """Verify that valid documents are loaded correctly."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "This is an IT support document.",
        encoding="utf-8",
    )

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].source == "sample.txt"
    assert documents[0].content == "This is an IT support document."


def test_empty_document_is_skipped(tmp_path):
    """Verify that empty documents are ignored."""
    (tmp_path / "empty.txt").write_text(
        "   ",
        encoding="utf-8",
    )

    documents = load_documents(tmp_path)

    assert documents == []


def test_missing_directory(tmp_path):
    """Verify that missing directories raise an error."""
    missing_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_documents(missing_path)


def test_chunk_text_preserves_sentences():
    text = (
        "Step one: Check the VPN connection.\n"
        "Step two: Restart the VPN client.\n"
        "Step three: Contact network support."
    )

    chunks = chunk_text(
        text=text,
        chunk_size=75,
        overlap=20,
    )

    assert chunks
    assert all(len(chunk) <= 75 for chunk in chunks)

    for instruction in [
        "Step one: Check the VPN connection.",
        "Step two: Restart the VPN client.",
        "Step three: Contact network support.",
    ]:
        assert any(instruction in chunk for chunk in chunks)


def test_long_text_respects_chunk_size():
    text = "network " * 100

    chunks = chunk_text(
        text=text,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_invalid_overlap():
    """Verify that invalid overlap values are rejected."""
    with pytest.raises(ValueError):
        chunk_text(
            text="Sample text",
            chunk_size=10,
            overlap=10,
        )


def test_empty_text():
    """Verify that empty text produces no chunks."""
    assert chunk_text("", chunk_size=10, overlap=2) == []


def test_chunk_documents():
    """Verify that chunks retain source metadata."""
    documents = [
        Document(
            source="test_policy.txt",
            content="This is a sample IT support policy.",
        )
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "test_policy.txt::0"
    assert chunks[0].source == "test_policy.txt"
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == documents[0].content


def test_numbered_instructions_remain_together():
    text = (
        "VPN Troubleshooting Guide\n"
        "1.\n"
        "Verify the VPN connection.\n"
        "2.\n"
        "Restart the VPN client.\n"
        "3.\n"
        "Contact the network support team."
    )

    chunks = chunk_text(
        text=text,
        chunk_size=75,
        overlap=15,
    )

    combined = "\n".join(chunks)

    assert "1. Verify the VPN connection." in combined
    assert "2. Restart the VPN client." in combined
    assert "3. Contact the network support team." in combined
    assert all(len(chunk) <= 75 for chunk in chunks)
