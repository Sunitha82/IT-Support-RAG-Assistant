
"""Tests for parent-child chunking."""

from app.ingestion.document_loader import Document
from app.ingestion.parent_child_chunking import (
    chunk_documents_parent_child,
)


def test_parent_child_relationships():
    documents = [
        Document(
            source="password_policy.txt",
            content=(
                "Document ID: IT-002\n"
                "Title: Password Policy\n\n"
                "Passwords must contain at least 12 characters. "
                "Employees must not share passwords."
            ),
        )
    ]

    parents, children = chunk_documents_parent_child(documents)

    assert len(parents) == 1
    assert len(children) == 2

    parent = parents[0]

    assert parent.parent_id == "IT-002::parent-1"
    assert all(child.parent_id == parent.parent_id for child in children)
    assert all(child.source == parent.source for child in children)

    assert "12 characters" in children[0].content
    assert "not share passwords" in children[1].content


def test_numbered_steps_preserve_context():
    documents = [
        Document(
            source="vpn.txt",
            content=(
                "Document ID: IT-001\n"
                "Title: VPN Guide\n\n"
                "If the VPN is connected but an application is inaccessible:\n"
                "1. Check the VPN connection.\n"
                "2. Contact network support if all applications fail."
            ),
        )
    ]

    parents, children = chunk_documents_parent_child(documents)

    assert len(parents) == 1
    assert len(children) == 2

    assert all(
        "If the VPN is connected" in child.content
        for child in children
    )

    assert "network support" in children[1].content


def test_empty_documents():
    parents, children = chunk_documents_parent_child([])

    assert parents == []
    assert children == []
