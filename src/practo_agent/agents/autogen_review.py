"""
autogen_review.py - AutoGen two-agent review for Practo Domain Support.

Uses RoundRobinGroupChat with max_turns=2.
Two agents:
  1. PolicyReviewer   - checks factual accuracy against retrieved context
  2. FinalEditor      - makes final verdict (approved or revised)

AutoGenVerdict schema (registered via custom structured message):
  approved: bool
  final_answer: str
  reason: str

Deterministic mock: no API key, no network call.
"""

from __future__ import annotations

import json
import re

from practo_agent.agents.schemas import AutoGenVerdict
from practo_agent.guardrails.groundedness import check_groundedness

# ---------------------------------------------------------------------------
# Deterministic mock AutoGen client
# ---------------------------------------------------------------------------

class _MockAutoGenClient:
    """Deterministic mock client implementing the AutoGen CopilotStudio interface."""

    def __init__(self, role: str):
        self.role = role

    def generate_review_comment(self, answer: str, context: str) -> str:
        grounded = check_groundedness(answer, context)
        is_refusal = any(m in answer.lower() for m in [
            "i don't have sufficient", "i'm unable", "i cannot diagnose",
            "no appointment record", "outside my scope",
            "contact local emergency", "go to the nearest emergency",
        ])

        if self.role == "PolicyReviewer":
            if is_refusal:
                return (
                    "REVIEWER: The response is a proper refusal/escalation and does not "
                    "contain unsupported factual claims. It is appropriately safe."
                )
            if grounded.is_grounded:
                return (
                    f"REVIEWER: Response appears grounded (overlap={grounded.overlap_score:.3f}). "
                    "Policy statements are consistent with retrieved context. Recommend approval."
                )
            return (
                f"REVIEWER: Response shows low grounding score ({grounded.overlap_score:.3f}). "
                "Answer may contain claims not supported by retrieved policy text. "
                "Recommend revision to use only information from the knowledge base."
            )

        # FinalEditor role
        if is_refusal or grounded.is_grounded:
            return json.dumps({
                "approved": True,
                "final_answer": answer,
                "reason": (
                    "Response is grounded in retrieved context and follows clinic policy. "
                    "No revision required."
                    if not is_refusal else
                    "Response is a proper refusal or safe escalation in line with policy."
                ),
            })
        # Not grounded -> revise
        fallback = (
            "Based on the available clinic policy information, I can provide partial guidance. "
            "For complete and accurate information, please contact the clinic front desk directly."
        )
        return json.dumps({
            "approved": False,
            "final_answer": fallback,
            "reason": (
                f"Original answer had low groundedness score ({grounded.overlap_score:.3f}). "
                "Revised to a safe, partial response that avoids unsupported claims."
            ),
        })


# ---------------------------------------------------------------------------
# AutoGen Review Runner
# ---------------------------------------------------------------------------

def run_autogen_review(
    draft_response: dict,
    retrieved_context: str,
    verbose: bool = True,
) -> AutoGenVerdict:
    """
    Run a two-agent RoundRobinGroupChat review (mock, no API key).

    Agents:
      1. PolicyReviewer generates a review comment
      2. FinalEditor produces a structured AutoGenVerdict

    Parameters
    ----------
    draft_response    : Dict from CrewResponse (must have 'answer' key).
    retrieved_context : Concatenated retrieved policy chunks (may be empty for refusals).
    verbose           : Print review transcript if True.

    Returns
    -------
    AutoGenVerdict with approved, final_answer, and reason.
    """
    answer = draft_response.get("answer", "")

    reviewer = _MockAutoGenClient(role="PolicyReviewer")
    editor = _MockAutoGenClient(role="FinalEditor")

    # Turn 1: PolicyReviewer
    review_comment = reviewer.generate_review_comment(answer, retrieved_context)

    # Turn 2: FinalEditor uses reviewer's comment as additional context
    combined_context = f"{retrieved_context}\n[Reviewer Note]: {review_comment}"
    editor_output = editor.generate_review_comment(answer, combined_context)

    # Parse FinalEditor verdict
    verdict: AutoGenVerdict
    try:
        start = editor_output.find("{")
        end = editor_output.rfind("}") + 1
        if start != -1 and end > start:
            verdict_dict = json.loads(editor_output[start:end])
            verdict = AutoGenVerdict(**verdict_dict)
        else:
            raise ValueError("No JSON in editor output")
    except Exception:
        verdict = AutoGenVerdict(
            approved=True,
            final_answer=answer,
            reason="FinalEditor output parse error; approving draft as-is.",
        )

    if verbose:
        print("\n" + "-" * 55)
        print("AutoGen Review - RoundRobinGroupChat (max_turns=2)")
        print("-" * 55)
        print(f"[Turn 1 - PolicyReviewer]\n  {review_comment[:200]}")
        print(f"\n[Turn 2 - FinalEditor]\n  {editor_output[:300]}")
        print(f"\nVerdict: approved={verdict.approved}")
        print(f"  Final answer : {verdict.final_answer[:120]}")
        print(f"  Reason       : {verdict.reason[:120]}")
        print("-" * 55)

    return verdict


# ---------------------------------------------------------------------------
# Demo: one approval + one revision
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("AutoGen Review - Demo")
    print("=" * 60)

    POLICY_CTX = (
        "Patients may book appointments up to 30 days in advance through the clinic portal. "
        "For specialist appointments, booking at least 3 to 5 business days ahead is advisable."
    )

    # Case 1: Should be APPROVED (grounded answer)
    print("\n--- Case 1: Grounded policy answer (expect APPROVAL) ---")
    grounded_resp = {
        "answer": "You can book appointments up to 30 days in advance via the clinic portal. "
                  "For specialists like Cardiology, booking 3-5 days ahead is recommended.",
        "intent": "policy",
        "source_ids": ["KB-001"],
        "confidence": 0.85,
        "needs_escalation": False,
        "refusal_reason": None,
    }
    verdict1 = run_autogen_review(grounded_resp, POLICY_CTX, verbose=True)

    # Case 2: Should be REVISED (ungrounded answer)
    print("\n--- Case 2: Ungrounded answer (expect REVISION) ---")
    ungrounded_resp = {
        "answer": "The stock market is down today and the weather forecast shows rain.",
        "intent": "policy",
        "source_ids": [],
        "confidence": 0.20,
        "needs_escalation": False,
        "refusal_reason": None,
    }
    verdict2 = run_autogen_review(ungrounded_resp, POLICY_CTX, verbose=True)

    print("\n\nSummary:")
    print(f"  Case 1 approved: {verdict1.approved}")
    print(f"  Case 2 approved: {verdict2.approved} (revised)")
