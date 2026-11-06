
from pathlib import Path


def load_documents(data_dir="data"):
    documents = []

    for file_path in sorted(Path(data_dir).glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Skipping empty file: {file_path.name}")
            continue

        documents.append({
            "source": file_path.name,
            "content": text
        })

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Total documents loaded: {len(documents)}")

    for doc in documents:
        print("\n------------------------")
        print("Source:", doc["source"])
        print("Characters:", len(doc["content"]))
        print("Content preview:", doc["content"][:150])
