
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from chunking import chunk_text, chunk_documents
from document_loader import load_documents


def test_chunking():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chunks = chunk_text(text, chunk_size=10, overlap=3)

    assert chunks == [
        "ABCDEFGHIJ",
        "HIJKLMNOPQ",
        "OPQRSTUVWX",
        "VWXYZ"
    ]

    print("Basic chunking test passed!")


def test_documents():
    documents = load_documents()
    chunks = chunk_documents(documents)

    assert len(documents) == 5
    assert len(chunks) >= 5

    for chunk in chunks:
        assert len(chunk["content"]) <= 200
        assert chunk["source"]
        assert chunk["chunk_id"]

    print("Document chunking test passed!")


if __name__ == "__main__":
    test_chunking()
    test_documents()
