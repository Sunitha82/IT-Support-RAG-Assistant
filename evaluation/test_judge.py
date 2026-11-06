
from app.retrieval.retriever import SemanticRetriever
from evaluation.llm_judge import LLMJudge


QUESTION = (
    "My VPN is connected but I cannot access an internal "
    "company application. What should I do?"
)

EXPECTED_FACTS = [
    "Verify that the VPN client reports a successful connection",
    "Disconnect and reconnect to the VPN",
    "Check whether other internal applications are accessible",
    "Verify that the correct internal application URL is being used",
    "If only one application is inaccessible, contact the application support team",
    "If all internal applications are inaccessible, contact the network support team",
]

CORRECT_ANSWER = """
1. Verify that the VPN client reports a successful connection.
2. Disconnect and reconnect to the VPN.
3. Check whether other internal applications are accessible.
4. Verify that you are using the correct internal application URL.
5. If only one application is inaccessible, contact application support.
6. If all internal applications are inaccessible, contact network support.
"""

INCOMPLETE_ANSWER = """
1. Verify that the VPN client reports a successful connection.
2. Disconnect and reconnect to the VPN.
3. Check whether other internal applications are accessible.
"""

HALLUCINATED_ANSWER = CORRECT_ANSWER + """
7. Disable the company firewall permanently to restore access.
"""


def main():
    retriever = SemanticRetriever()
    judge = LLMJudge()

    retrieved = retriever.retrieve(QUESTION)

    context = "\n\n".join(
        f"Source: {item.source}\nContent: {item.content}"
        for item in retrieved
    )

    if not context:
        raise RuntimeError("No VPN evidence retrieved.")

    cases = [
        ("Correct answer", CORRECT_ANSWER, "pass"),
        ("Incomplete answer", INCOMPLETE_ANSWER, "fail"),
        ("Hallucinated answer", HALLUCINATED_ANSWER, "fail"),
    ]

    print("\n========== JUDGE VALIDATION ==========\n")

    passed = 0

    for name, answer, expected_verdict in cases:
        print(f"\nTesting: {name}")

        try:
            result = judge.evaluate(
                question=QUESTION,
                answer=answer,
                expected_facts=EXPECTED_FACTS,
                retrieved_context=context,
            )

            print(f"Completeness: {result.completeness_score:.2%}")
            print(f"Faithfulness: {result.faithfulness_score:.2%}")
            print(f"Verdict: {result.verdict.upper()}")
            print(f"Expected verdict: {expected_verdict.upper()}")

            if result.verdict == expected_verdict:
                print("TEST PASSED")
                passed += 1
            else:
                print("TEST FAILED")

        except Exception as exc:
            print(f"TEST ERROR: {type(exc).__name__}: {exc}")

    print(f"\nValidation: {passed}/{len(cases)} tests passed")


if __name__ == "__main__":
    main()
