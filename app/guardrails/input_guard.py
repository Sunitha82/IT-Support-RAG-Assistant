"""Basic, deterministic input checks. Not a complete prompt-injection defence."""
import re

MAX_QUESTION_LENGTH = 2000

class InputGuardrailError(ValueError):
    """Raised when a question fails the input guardrail."""

    def __init__(self, message: str, code: str = "invalid_question"):
        super().__init__(message)
        self.code = code

_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above|system|developer)\s+(?:instructions?|prompts?|rules?)\b", re.I),
    re.compile(r"\b(?:reveal|print|show|disclose)\s+(?:me\s+)?(?:your\s+)?(?:hidden\s+|secret\s+|internal\s+)?(?:system|developer)\s+(?:prompt|instructions?)\b", re.I),
    re.compile(r"\b(?:you\s+are\s+now|act\s+as)\s+(?:an?\s+)?(?:unrestricted|unfiltered|jailbroken)\b", re.I),
)

def validate_question(question: str) -> str:
    """Return stripped question or raise a safe, user-facing validation error."""
    if not isinstance(question, str) or not question.strip():
        raise InputGuardrailError("Question cannot be blank")
    cleaned = question.strip()
    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise InputGuardrailError(f"Question must be {MAX_QUESTION_LENGTH} characters or fewer")
    if any(pattern.search(cleaned) for pattern in _INJECTION_PATTERNS):
        raise InputGuardrailError(
            "I can help with IT support questions, but I can't reveal internal "
            "instructions or follow requests to override them. Please ask about "
            "VPN, passwords, or software access.",
            code="input_guardrail_rejected",
        )
    return cleaned
