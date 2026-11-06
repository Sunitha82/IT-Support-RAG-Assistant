
import json
from pathlib import Path

from app.retrieval.retriever import SemanticRetriever
from app.services.rag_service import RAGService
from evaluation.llm_judge import LLMJudge


EVALUATION_DIR = Path(__file__).parent
INPUT_FILE = EVALUATION_DIR / "test_questions.json"
OUTPUT_FILE = EVALUATION_DIR / "judge_results.json"


def main():
    tests = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    retriever = SemanticRetriever()
    rag_service = RAGService(retriever=retriever)
    judge = LLMJudge()

    reports = []

    for test in tests:
        expected_facts = test.get("expected_facts")

        # Only evaluate labelled answerable questions.
        if not test["answerable"] or not expected_facts:
            continue

        print(f"\nEvaluating {test['id']}")

        try:
            # Retrieve the same evidence used by the RAG service.
            retrieved = retriever.retrieve(test["question"])

            context = "\n\n".join(
                f"Source: {item.source}\n"
                f"Chunk ID: {item.chunk_id}\n"
                f"Content: {item.content}"
                for item in retrieved
            )

            if not context:
                raise ValueError("No relevant evidence retrieved")

            response = rag_service.ask(test["question"])

            judgment = judge.evaluate(
                question=test["question"],
                answer=response.answer,
                expected_facts=expected_facts,
                retrieved_context=context,
            )

            report = {
                "id": test["id"],
                "question": test["question"],
                "answer": response.answer,
                "sources": response.sources,
                "judgment": judgment.model_dump(),
            }

            reports.append(report)

            print(f"Completeness: {judgment.completeness_score:.2%}")
            print(f"Faithfulness: {judgment.faithfulness_score:.2%}")
            #print(f"Unsupported claims: {judgment.unsupported_claims}")
            unsupported_claims = [
                item.claim
                for item in judgment.claim_assessments
                if not item.supported
            ]

            print(f"Unsupported claims: {unsupported_claims}")
            print(f"Verdict: {judgment.verdict.upper()}")
            print(f"Explanation: {judgment.explanation}")

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
            reports.append({
                "id": test["id"],
                "error": f"{type(exc).__name__}: {exc}",
            })

    OUTPUT_FILE.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
