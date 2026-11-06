
from document_loader import load_documents


def chunk_text(text, chunk_size=200, overlap=40):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size - 1"
        )

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end]

        chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_documents(documents):
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(
            doc["content"],
            chunk_size=200,
            overlap=40
        )

        for index, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['source']}::{index}",
                "source": doc["source"],
                "chunk_index": index,
                "content": chunk
            })

    return all_chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

    for chunk in chunks:
        print("\n------------------------")
        print("Chunk ID:", chunk["chunk_id"])
        print("Source:", chunk["source"])
        print("Chunk index:", chunk["chunk_index"])
        print("Characters:", len(chunk["content"]))
        print("Content:", repr(chunk["content"]))
