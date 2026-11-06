
import json
from pathlib import Path

from app.retrieval.retriever import SemanticRetriever


EVALUATION_FILE = Path(__file__).parent / "test_questions.json"
TOP_K = 3


def load_test_questions():
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate():
    retriever = SemanticRetriever()
    questions = load_test_questions()

    answerable_count = 0
    recall_hits = 0
    reciprocal_rank_total = 0.0

    unanswerable_count = 0
    correct_rejections = 0

    print("\n========== RAG RETRIEVAL EVALUATION ==========\n")

    for test in questions:
        results = retriever.retrieve(
            question=test["question"],
            top_k=TOP_K,
        )

        retrieved_sources = [result.source for result in results]
        expected_source = test["expected_source"]

        print(f"Test: {test['id']}")
        print(f"Question: {test['question']}")
        print(f"Expected source: {expected_source}")
        print(f"Retrieved sources: {retrieved_sources}")

        if test["answerable"]:
            answerable_count += 1

            if expected_source in retrieved_sources:
                rank = retrieved_sources.index(expected_source) + 1
                recall_hits += 1
                reciprocal_rank_total += 1 / rank
                print(f"PASS - Correct source found at rank {rank}")
            else:
                print("FAIL - Expected source not retrieved")

        else:
            unanswerable_count += 1

            if not results:
                correct_rejections += 1
                print("PASS - No relevant evidence returned")
            else:
                print("FAIL - Unrelated evidence returned")

        print("-" * 55)

    recall_at_k = (
        recall_hits / answerable_count
        if answerable_count else 0.0
    )

    mrr = (
        reciprocal_rank_total / answerable_count
        if answerable_count else 0.0
    )

    rejection_rate = (
        correct_rejections / unanswerable_count
        if unanswerable_count else 0.0
    )

    print("\n========== FINAL METRICS ==========")
    print(f"Answerable questions: {answerable_count}")
    print(f"Unanswerable questions: {unanswerable_count}")
    print(f"Recall@{TOP_K}: {recall_at_k:.2%}")
    print(f"MRR: {mrr:.4f}")
    print(f"No-evidence rejection rate: {rejection_rate:.2%}")


if __name__ == "__main__":
    evaluate()
