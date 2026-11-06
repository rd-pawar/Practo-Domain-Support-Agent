"""
test_autogen.py – Acceptance tests for AutoGen two-agent review.
"""

import pytest
from practo_agent.agents.autogen_review import run_autogen_review
from practo_agent.agents.schemas import AutoGenVerdict

POLICY_CONTEXT = (
    "Patients may book appointments up to 30 days in advance. "
    "For specialists, booking 3-5 days ahead is recommended."
)


def test_autogen_returns_autogen_verdict():
    response = {
        "answer": "You can book appointments up to 30 days in advance via the clinic portal.",
        "intent": "policy",
        "source_ids": ["KB-001"],
        "confidence": 0.85,
        "needs_escalation": False,
        "refusal_reason": None,
    }
    verdict = run_autogen_review(response, POLICY_CONTEXT, verbose=False)
    assert isinstance(verdict, AutoGenVerdict)
    assert isinstance(verdict.approved, bool)
    assert isinstance(verdict.final_answer, str)
    assert isinstance(verdict.reason, str)


def test_grounded_answer_approved():
    """A grounded policy answer should be approved."""
    response = {
        "answer": "You can book appointments up to 30 days in advance via the clinic portal. "
                  "Specialists should be booked 3-5 business days ahead.",
        "intent": "policy",
        "source_ids": ["KB-001"],
        "confidence": 0.90,
        "needs_escalation": False,
        "refusal_reason": None,
    }
    verdict = run_autogen_review(response, POLICY_CONTEXT, verbose=False)
    assert verdict.approved is True
    assert len(verdict.final_answer) > 0


def test_ungrounded_answer_revised():
    """An ungrounded answer should be revised (not approved)."""
    response = {
        "answer": "The stock market rallied today and crypto is trending.",
        "intent": "policy",
        "source_ids": [],
        "confidence": 0.10,
        "needs_escalation": False,
        "refusal_reason": None,
    }
    verdict = run_autogen_review(response, POLICY_CONTEXT, verbose=False)
    assert verdict.approved is False
    # Final answer should be the safe fallback, not the original
    assert "stock market" not in verdict.final_answer.lower()


def test_refusal_answer_always_approved():
    """A proper refusal should always be approved by the reviewer."""
    response = {
        "answer": "I'm unable to diagnose medical conditions. Please consult a doctor.",
        "intent": "out_of_scope",
        "source_ids": [],
        "confidence": 1.0,
        "needs_escalation": True,
        "refusal_reason": "Diagnosis outside scope.",
    }
    verdict = run_autogen_review(response, "", verbose=False)
    assert verdict.approved is True


def test_verdict_has_all_required_fields():
    response = {
        "answer": "Book 30 days ahead.",
        "intent": "policy",
        "source_ids": ["KB-001"],
        "confidence": 0.85,
        "needs_escalation": False,
        "refusal_reason": None,
    }
    verdict = run_autogen_review(response, POLICY_CONTEXT, verbose=False)
    # Pydantic model means these fields must be present
    verdict_dict = verdict.model_dump()
    assert "approved" in verdict_dict
    assert "final_answer" in verdict_dict
    assert "reason" in verdict_dict
