
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_embedding_model():
    global _model

    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def generate_embeddings(texts):
    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":
    sentences = [
        "My VPN is not working.",
        "I cannot connect to the company network.",
        "How many annual leave days do I get?"
    ]

    embeddings = generate_embeddings(sentences)

    print("Embedding shape:", embeddings.shape)
    print("First 5 values:", embeddings[0][:5])
