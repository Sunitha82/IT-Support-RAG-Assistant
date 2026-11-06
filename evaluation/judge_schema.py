
from typing import Literal

from pydantic import BaseModel, Field


class FactAssessment(BaseModel):
    expected_fact: str
    covered: bool
    explanation: str


class ClaimAssessment(BaseModel):
    claim: str
    supported: bool
    explanation: str


class JudgeResult(BaseModel):
    fact_assessments: list[FactAssessment]

    completeness_score: float = Field(ge=0.0, le=1.0)
    faithfulness_score: float = Field(ge=0.0, le=1.0)

    claim_assessments: list[ClaimAssessment]

    verdict: Literal["pass", "fail"]
    explanation: str
