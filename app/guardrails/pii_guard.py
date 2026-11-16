
"""PII detection and anonymization using Microsoft Presidio."""

from functools import lru_cache

PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
]


@lru_cache(maxsize=1)
def get_pii_engines():
    """Initialize Presidio engines once, when first needed."""
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    analyzer = AnalyzerEngine()

    # Recognize UK mobile numbers.
    uk_phone_pattern = Pattern(
        name="uk_mobile_pattern",
        regex=r"(?<!\w)(?:\+44\s?7\d{3}\s?\d{6}|07\d{3}\s?\d{6})(?!\w)",
        score=0.85,
    )

    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[uk_phone_pattern],
        )
    )

    return analyzer, AnonymizerEngine()


def mask_pii(text: str) -> tuple[str, int]:
    """Return anonymized text and the number of PII findings."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    analyzer, anonymizer = get_pii_engines()

    findings = analyzer.analyze(
        text=text,
        language="en",
        entities=PII_ENTITIES,
    )

    protected_text = anonymizer.anonymize(
        text=text,
        analyzer_results=findings,
    ).text

    return protected_text, len(findings)
