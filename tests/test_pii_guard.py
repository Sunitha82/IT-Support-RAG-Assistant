
"""Tests for Microsoft Presidio PII protection."""

from app.guardrails.pii_guard import mask_pii


def test_email_masking():
    text = "Contact john.smith@example.com for help."
    protected, count = mask_pii(text)

    assert "john.smith@example.com" not in protected
    assert "<EMAIL_ADDRESS>" in protected
    assert count >= 1


def test_uk_phone_masking():
    text = "Call +44 7700 900123 for assistance."
    protected, count = mask_pii(text)

    assert "+44 7700 900123" not in protected
    assert "<PHONE_NUMBER>" in protected
    assert count >= 1


def test_normal_it_question():
    text = "How do I reset my VPN password?"
    protected, count = mask_pii(text)

    assert protected == text
    assert count == 0
