
"""Parent-child chunk models for the IT Support RAG application."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParentChunk:
    """Complete document section supplied to the LLM."""

    parent_id: str
    source: str
    title: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ChildChunk:
    """Small searchable chunk linked to its parent section."""

    child_id: str
    parent_id: str
    source: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


import re

from app.ingestion.document_loader import Document


def chunk_documents_parent_child(
    documents: list[Document],
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Create parent sections and smaller searchable child chunks."""

    parents = []
    children = []

    for document in documents:
        lines = [
            line.strip()
            for line in document.content.splitlines()
            if line.strip()
        ]

        document_id = next(
            (
                line.split(":", 1)[1].strip()
                for line in lines
                if line.startswith("Document ID:")
            ),
            document.source,
        )

        title = next(
            (
                line.split(":", 1)[1].strip()
                for line in lines
                if line.startswith("Title:")
            ),
            document.source,
        )

        body_lines = [
            line for line in lines
            if not line.startswith(("Document ID:", "Title:"))
        ]

        parent_id = f"{document_id}::parent-1"
        parent_content = "\n".join(body_lines)

        parents.append(
            ParentChunk(
                parent_id=parent_id,
                source=document.source,
                title=title,
                content=parent_content,
                metadata={"document_id": document_id},
            )
        )

        # Numbered procedures: preserve their introductory condition.
        numbered = [
            line for line in body_lines
            if re.match(r"^\d+[.)]\s+", line)
        ]

        if numbered:
            introduction = "\n".join(
                line for line in body_lines
                if not re.match(r"^\d+[.)]\s+", line)
            )

            child_texts = [
                f"{introduction}\n{step}".strip()
                for step in numbered
            ]
        else:
            # Simple policy documents: split into individual sentences.
            child_texts = [
                sentence.strip()
                for sentence in re.split(
                    r"(?<=[.!?])\s+",
                    parent_content,
                )
                if sentence.strip()
            ]

        for index, content in enumerate(child_texts, start=1):
            children.append(
                ChildChunk(
                    child_id=f"{document_id}::child-{index}",
                    parent_id=parent_id,
                    source=document.source,
                    # Titles keep short policy sentences meaningful in isolation.
                    content=f"{title}\n{content}",
                    metadata={"document_id": document_id},
                )
            )

    return parents, children
