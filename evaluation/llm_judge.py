
import json

from pydantic import ValidationError

from app.llm.azure_client import AzureLLMClient
from evaluation.judge_schema import JudgeResult


JUDGE_SYSTEM_PROMPT = """
You are an impartial evaluator of a RAG application.

You will receive:
1. A user question.
2. A generated answer.
3. Expected facts.
4. Retrieved document context.

Evaluate completeness and faithfulness separately.

COMPLETENESS:
- Assess every expected fact.
- Accept accurate paraphrases.
- Mark covered=true when the answer communicates the fact.

FAITHFULNESS:
- Examine every factual claim in the generated answer.
- For each claim, decide whether retrieved context supports it.
- Additional information is allowed if supported by context.
- A claim is NOT unsupported merely because it is absent
  from the expected facts.
- Do not use outside knowledge to justify claims.
- Treat claims contradicting the context as unsupported.
- Include all substantive factual claims in claim_assessments.

SCORING:
- completeness_score = covered facts / expected facts.
- faithfulness_score = supported factual claims / total factual claims.
- verdict = pass only if all expected facts are covered
  AND every factual claim is supported.

Return valid JSON only, without Markdown formatting.

Required JSON structure:

{
  "fact_assessments": [
    {
      "expected_fact": "exact expected fact",
      "covered": true,
      "explanation": "reason"
    }
  ],
  "completeness_score": 1.0,
  "faithfulness_score": 1.0,
  "claim_assessments": [
    {
      "claim": "factual claim from generated answer",
      "supported": true,
      "explanation": "evidence from retrieved context"
    }
  ],
  "verdict": "pass",
  "explanation": "overall assessment"
}
"""


class LLMJudge:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or AzureLLMClient()

    def evaluate(
        self,
        question: str,
        answer: str,
        expected_facts: list[str],
        retrieved_context: str,
    ) -> JudgeResult:

        if not retrieved_context.strip():
            raise ValueError(
                "Retrieved context is required for evaluation."
            )

        payload = {
            "question": question,
            "generated_answer": answer,
            "expected_facts": expected_facts,
            "retrieved_context": retrieved_context,
        }

        raw_response = self.llm_client.generate(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )

        try:
            result = JudgeResult.model_validate_json(raw_response)
        except ValidationError as exc:
            raise ValueError(
                f"Judge returned invalid JSON or schema: {exc}"
            ) from exc

        assessed_facts = [
            item.expected_fact
            for item in result.fact_assessments
        ]

        if assessed_facts != expected_facts:
            raise ValueError(
                "Judge did not assess every expected fact in order."
            )

        # Calculate completeness in Python.
        covered_count = sum(
            item.covered for item in result.fact_assessments
        )

        completeness = (
            covered_count / len(expected_facts)
            if expected_facts else 1.0
        )

        result.completeness_score = round(completeness, 4)

        # Calculate faithfulness from individual claim assessments.
        claim_count = len(result.claim_assessments)

        if claim_count == 0:
            raise ValueError(
                "Judge returned no claim assessments."
            )

        supported_count = sum(
            item.supported for item in result.claim_assessments
        )

        faithfulness = supported_count / claim_count
        result.faithfulness_score = round(faithfulness, 4)

        # Derive unsupported claims from structured assessments.
        unsupported_claims = [
            item.claim
            for item in result.claim_assessments
            if not item.supported
        ]

        result.verdict = (
            "pass"
            if completeness == 1.0 and not unsupported_claims
            else "fail"
        )

        return result
