
import json
import re
from pathlib import Path

from app.services.rag_service import RAGService


EVALUATION_FILE = Path(__file__).parent / "test_questions.json"
OUTPUT_FILE = Path(__file__).parent / "answer_results.json"

FALLBACK = "I don't have enough information in the IT knowledge base."


def normalize(text: str) -> str:
    """Normalize text for basic phrase matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def evaluate():
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    service = RAGService()
    results = []

    print("\n========== RAG ANSWER EVALUATION ==========\n")

    for test in questions:
        # Initially evaluate only cases with labelled expected facts.
        if "expected_facts" not in test:
            continue

        print(f"Evaluating {test['id']}: {test['question']}")

        try:
            response = service.ask(test["question"])
            answer = response.answer

            expected_facts = test["expected_facts"]

            matched_facts = [
                fact
                for fact in expected_facts
                if normalize(fact) in normalize(answer)
            ]

            missing_facts = [
                fact
                for fact in expected_facts
                if fact not in matched_facts
            ]

            completeness = (
                len(matched_facts) / len(expected_facts)
                if expected_facts else 0.0
            )

            result = {
                "id": test["id"],
                "question": test["question"],
                "answer": answer,
                "sources": response.sources,
                "expected_facts": expected_facts,
                "matched_facts": matched_facts,
                "missing_facts": missing_facts,
                "completeness": round(completeness, 4),
            }

            results.append(result)

            print(f"Completeness: {completeness:.2%}")
            print(f"Matched facts: {len(matched_facts)}")
            print(f"Missing facts: {len(missing_facts)}")

            if missing_facts:
                print("\nMissing facts:")
                for fact in missing_facts:
                    print(f" - {fact}")

            print("\nGenerated answer:")
            print(answer)
            print("-" * 60)

        except Exception as exc:
            print(f"ERROR: {test['id']} failed: {type(exc).__name__}: {exc}")

            results.append({
                "id": test["id"],
                "error": f"{type(exc).__name__}: {exc}",
            })

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nEvaluation results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()
