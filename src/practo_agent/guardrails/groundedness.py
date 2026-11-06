"""
groundedness.py - Output groundedness check guardrail.

Compares answer token overlap with retrieved context.
If overlap is below threshold, returns a safe "I do not know" response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those",
    "i", "you", "we", "they", "it", "he", "she", "me", "him", "her",
    "our", "your", "their", "its", "not", "no", "so", "if", "as", "by",
    "from", "up", "about", "into", "through", "during", "before", "after",
    "please", "also", "any", "all", "more", "other", "such", "only",
})

GROUNDEDNESS_THRESHOLD = 0.15   # Minimum token Jaccard overlap required

_SAFE_FALLBACK_ANSWER = (
    "I don't have sufficient information in the clinic knowledge base to answer your "
    "question accurately. Please contact the clinic front desk for detailed assistance."
)


@dataclass
class GroundednessResult:
    is_grounded: bool
    overlap_score: float        # Jaccard-like overlap [0, 1]
    answer_tokens: int
    context_tokens: int
    shared_tokens: int
    final_answer: str           # Either original or safe fallback


def _tokenise(text: str) -> set[str]:
    """Lower-case, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def check_groundedness(
    answer: str,
    retrieved_context: str,
    threshold: float = GROUNDEDNESS_THRESHOLD,
) -> GroundednessResult:
    """
    Compute token-based Jaccard overlap between *answer* and *retrieved_context*.

    If the answer is for an out-of-scope or appointment_status query, groundedness
    is not strictly required (context may be empty for valid refusals).

    Parameters
    ----------
    answer            : The agent's draft answer.
    retrieved_context : Concatenated retrieved policy chunks.
    threshold         : Minimum overlap score (default 0.15).

    Returns
    -------
    GroundednessResult with is_grounded flag and final_answer.
    """
    # Explicit refusals and appointment answers are always "grounded"
    _refusal_markers = [
        "i don't have sufficient",
        "i cannot diagnose",
        "i'm unable to diagnose",
        "no appointment record found",
        "outside my scope",
        "contact local emergency",
        "go to the nearest emergency",
        "i do not know",
    ]
    answer_lower = answer.lower()
    if any(marker in answer_lower for marker in _refusal_markers):
        return GroundednessResult(
            is_grounded=True,
            overlap_score=1.0,
            answer_tokens=len(_tokenise(answer)),
            context_tokens=len(_tokenise(retrieved_context)),
            shared_tokens=0,
            final_answer=answer,
        )

    if not retrieved_context.strip():
        return GroundednessResult(
            is_grounded=False,
            overlap_score=0.0,
            answer_tokens=len(_tokenise(answer)),
            context_tokens=0,
            shared_tokens=0,
            final_answer=_SAFE_FALLBACK_ANSWER,
        )

    answer_tokens = _tokenise(answer)
    context_tokens = _tokenise(retrieved_context)
    shared = answer_tokens & context_tokens

    union = answer_tokens | context_tokens
    jaccard = len(shared) / len(union) if union else 0.0

    is_grounded = jaccard >= threshold

    return GroundednessResult(
        is_grounded=is_grounded,
        overlap_score=round(jaccard, 4),
        answer_tokens=len(answer_tokens),
        context_tokens=len(context_tokens),
        shared_tokens=len(shared),
        final_answer=answer if is_grounded else _SAFE_FALLBACK_ANSWER,
    )


if __name__ == "__main__":
    print("Groundedness Guardrail - Demo")
    print("-" * 50)
    ctx = (
        "Patients may book appointments up to 30 days in advance through the clinic portal. "
        "For specialist appointments, booking at least 3 to 5 business days ahead is advisable."
    )
    good_answer = "You can book appointments up to 30 days in advance via the clinic portal."
    bad_answer = "The stock market rose 3 percent yesterday and weather is sunny."

    for label, ans in [("GROUNDED", good_answer), ("UNGROUNDED", bad_answer)]:
        r = check_groundedness(ans, ctx)
        print(f"\n  [{label}]")
        print(f"    Answer  : {ans[:80]}")
        print(f"    Score   : {r.overlap_score:.4f}")
        print(f"    Grounded: {r.is_grounded}")
        if not r.is_grounded:
            print(f"    Fallback: {r.final_answer[:80]}")
