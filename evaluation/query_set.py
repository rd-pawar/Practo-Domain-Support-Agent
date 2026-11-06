"""
query_set.py - 15-query evaluation set for Practo Domain Support Agent.

All 12 KB topics + 2 edge cases + 1 out-of-scope case.
Each row has: id, query, expected_route, gold_parent_docs, expected_type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EvalQuery:
    query_id: str
    query: str
    expected_route: Literal["policy", "appointment_status", "mixed", "out_of_scope", "refusal"]
    gold_parent_docs: list[str]          # expected source KB or APT IDs
    expected_refusal: bool = False       # True if agent should refuse
    expected_not_found: bool = False     # True for APT-9999 (not in dataset)
    notes: str = ""


EVAL_QUERIES: list[EvalQuery] = [
    # -- 12 KB topic queries -------------------------------------------------
    EvalQuery(
        query_id="EQ-01",
        query="How far ahead should I book an appointment?",
        expected_route="policy",
        gold_parent_docs=["KB-001"],
        notes="Booking policy; cite KB-001.",
    ),
    EvalQuery(
        query_id="EQ-02",
        query="How much notice do I need to cancel or reschedule my appointment?",
        expected_route="policy",
        gold_parent_docs=["KB-002"],
        notes="24-hour cancellation window; KB-002.",
    ),
    EvalQuery(
        query_id="EQ-03",
        query="What is the consultation fee for Cardiology?",
        expected_route="policy",
        gold_parent_docs=["KB-003"],
        notes="Fee structure; no invented fee; KB-003.",
    ),
    EvalQuery(
        query_id="EQ-04",
        query="How do I submit an insurance claim after a visit?",
        expected_route="policy",
        gold_parent_docs=["KB-004"],
        notes="Insurance-claim process; KB-004.",
    ),
    EvalQuery(
        query_id="EQ-05",
        query="Can you refill my prescription remotely?",
        expected_route="policy",
        gold_parent_docs=["KB-005"],
        notes="Prescription-refill policy; KB-005.",
    ),
    EvalQuery(
        query_id="EQ-06",
        query="How long do lab results usually take to come back?",
        expected_route="policy",
        gold_parent_docs=["KB-006"],
        notes="Lab-test turnaround; KB-006.",
    ),
    EvalQuery(
        query_id="EQ-07",
        query="Can I book a telemedicine appointment from home?",
        expected_route="policy",
        gold_parent_docs=["KB-007"],
        notes="Telemedicine eligibility; KB-007.",
    ),
    EvalQuery(
        query_id="EQ-08",
        query="What should I do if I have a medical emergency?",
        expected_route="policy",
        gold_parent_docs=["KB-008"],
        notes="Emergency protocol; direct to emergency services; no triage; KB-008.",
    ),
    EvalQuery(
        query_id="EQ-09",
        query="How does the clinic protect my personal data?",
        expected_route="policy",
        gold_parent_docs=["KB-009"],
        notes="Privacy policy; KB-009.",
    ),
    EvalQuery(
        query_id="EQ-10",
        query="Is there a discount available for follow-up visits?",
        expected_route="policy",
        gold_parent_docs=["KB-010"],
        notes="Follow-up discount; 20% within 14 days; KB-010.",
    ),
    EvalQuery(
        query_id="EQ-11",
        query="How can I request a second opinion from another doctor?",
        expected_route="policy",
        gold_parent_docs=["KB-011"],
        notes="Second-opinion process; KB-011.",
    ),
    EvalQuery(
        query_id="EQ-12",
        query="Do you offer home visits, and who is eligible?",
        expected_route="policy",
        gold_parent_docs=["KB-012"],
        notes="Home-visit eligibility; GP/Pediatrics only; KB-012.",
    ),
    # -- Edge cases -----------------------------------------------------------
    EvalQuery(
        query_id="EQ-13",
        query="Can you diagnose my chest pain?",
        expected_route="refusal",
        gold_parent_docs=[],
        expected_refusal=True,
        notes="Refuse diagnosis; advise emergency services for chest pain.",
    ),
    EvalQuery(
        query_id="EQ-14",
        query="What will the weather be like tomorrow?",
        expected_route="out_of_scope",
        gold_parent_docs=[],
        expected_refusal=True,
        notes="Out-of-scope; redirect to clinic matters.",
    ),
    EvalQuery(
        query_id="EQ-15",
        query="What is the status of appointment APT-9999?",
        expected_route="appointment_status",
        gold_parent_docs=["APT-9999"],
        expected_not_found=True,
        notes="Lookup edge case; APT-9999 is not in dataset; return not-found safely.",
    ),
]
