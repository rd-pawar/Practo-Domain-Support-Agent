"""
budget.py - Token estimation and request budget enforcement guardrail.

Caps:
  - MAX_ESTIMATED_TOKENS = 2000 per request
  - MAX_TURNS            = 6 model/tool calls per request

Rejects oversized requests before any model or tool call.
Token count is an estimate, not billed tokens; labeled accordingly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Budget caps (frozen demo values - adjust and document in README)
# ---------------------------------------------------------------------------
MAX_ESTIMATED_TOKENS: int = 2000
MAX_TURNS: int = 6

_CHARS_PER_TOKEN_ESTIMATE: float = 4.0    # rough approximation


@dataclass
class BudgetCheckResult:
    within_budget: bool
    estimated_tokens: int
    max_tokens: int
    max_turns: int
    rejection_code: Optional[str]          # e.g. "BUDGET_EXCEEDED_TOKENS"
    rejection_message: Optional[str]


def estimate_tokens(
    query: str,
    memory_context: str = "",
    retrieved_context: str = "",
    prompt_overhead: int = 200,
) -> int:
    """
    Estimate total token count for a request before dispatch.

    Breakdown:
      - query tokens
      - memory context tokens
      - retrieved context tokens
      - prompt overhead (system prompts, role descriptions, task templates)

    NOTE: This is an estimate using approx 4 chars/token.
          It is NOT billed token count.  Label outputs accordingly.
    """
    total_chars = (
        len(query)
        + len(memory_context)
        + len(retrieved_context)
        + prompt_overhead * _CHARS_PER_TOKEN_ESTIMATE
    )
    return int(total_chars / _CHARS_PER_TOKEN_ESTIMATE)


def check_budget(
    query: str,
    memory_context: str = "",
    retrieved_context: str = "",
    prompt_overhead: int = 200,
    max_tokens: int = MAX_ESTIMATED_TOKENS,
    max_turns: int = MAX_TURNS,
    turns_used: int = 0,
) -> BudgetCheckResult:
    """
    Check if a request is within token and turn budget.

    Parameters
    ----------
    query             : User's query text.
    memory_context    : Serialised session history.
    retrieved_context : Concatenated retrieved KB chunks.
    prompt_overhead   : Estimated overhead for prompts/templates (in tokens).
    max_tokens        : Maximum estimated tokens per request.
    max_turns         : Maximum model/tool calls per request.
    turns_used        : Number of turns already consumed this session.

    Returns
    -------
    BudgetCheckResult - if within_budget is False, reject before any model call.
    """
    estimated = estimate_tokens(query, memory_context, retrieved_context, prompt_overhead)

    if estimated > max_tokens:
        return BudgetCheckResult(
            within_budget=False,
            estimated_tokens=estimated,
            max_tokens=max_tokens,
            max_turns=max_turns,
            rejection_code="BUDGET_EXCEEDED_TOKENS",
            rejection_message=(
                f"Request rejected: estimated {estimated} tokens exceeds cap of {max_tokens}. "
                "Please shorten your query or reduce context. "
                "[MOCK_LLM estimate - not billed tokens]"
            ),
        )

    if turns_used >= max_turns:
        return BudgetCheckResult(
            within_budget=False,
            estimated_tokens=estimated,
            max_tokens=max_tokens,
            max_turns=max_turns,
            rejection_code="BUDGET_EXCEEDED_TURNS",
            rejection_message=(
                f"Request rejected: turn limit of {max_turns} reached "
                f"(currently at {turns_used}). "
                "Please start a new session."
            ),
        )

    return BudgetCheckResult(
        within_budget=True,
        estimated_tokens=estimated,
        max_tokens=max_tokens,
        max_turns=max_turns,
        rejection_code=None,
        rejection_message=None,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Budget Guardrail - Demo")
    print("-" * 55)

    # Normal request
    normal = check_budget("How far ahead should I book an appointment?")
    print(f"\nNormal request:")
    print(f"  Estimated tokens: {normal.estimated_tokens}  (cap: {normal.max_tokens})")
    print(f"  Within budget   : {normal.within_budget}")

    # Oversized request
    huge_query = "X" * 9000
    oversized = check_budget(huge_query)
    print(f"\nOversized request ({len(huge_query)} chars):")
    print(f"  Estimated tokens: {oversized.estimated_tokens}  (cap: {oversized.max_tokens})")
    print(f"  Within budget   : {oversized.within_budget}")
    print(f"  Rejection code  : {oversized.rejection_code}")
    print(f"  Message         : {oversized.rejection_message}")
