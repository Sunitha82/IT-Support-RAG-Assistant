
from embeddings import generate_embeddings


sentences = [
    "My VPN is not working.",
    "I cannot connect to the company network.",
    "How many annual leave days do I get?"
]

embeddings = generate_embeddings(sentences)

# Vectors are normalized, so dot product = cosine similarity
similarity_matrix = embeddings @ embeddings.T

for i, sentence in enumerate(sentences):
    print(f"\nSentence {i + 1}: {sentence}")

    for j, score in enumerate(similarity_matrix[i]):
        print(f"  Similarity with sentence {j + 1}: {score:.3f}")
