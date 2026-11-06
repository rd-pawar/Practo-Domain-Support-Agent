"""
pii_masker.py - PII masking guardrail for Practo Domain Support Agent.

Masks fixed-format Indian mobile contact numbers before:
  - Model input
  - Memory/session context
  - Log fields
  - Exception messages

Supported format: +91-XXXXXXXXXX (10 digits after +91-)
Also handles 10-digit standalone numbers and with spaces.

IMPORTANT: Never write the raw masked contact value to logs.
The masked placeholder is [CONTACT-MASKED].
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Contact number patterns
# ---------------------------------------------------------------------------
# Pattern 1: +91-XXXXXXXXXX (canonical format used in synthetic dataset)
_CANONICAL_CONTACT = re.compile(r"\+91-\d{10}")

# Pattern 2: Standalone 10-digit number (must start with 6-9 for Indian mobile)
_STANDALONE_10 = re.compile(r"\b[6-9]\d{9}\b")

# Pattern 3: +91 XXXXXXXXXX (with space instead of hyphen)
_SPACE_FORMAT = re.compile(r"\+91\s\d{10}")

_MASK_PLACEHOLDER = "[CONTACT-MASKED]"

# Demonstration pair (for test/transcript output only)
DEMO_SYNTHETIC_NUMBER = "+91-9876543210"
DEMO_MASKED_OUTPUT = _MASK_PLACEHOLDER


def mask_contact_numbers(text: str) -> str:
    """
    Replace all recognised contact number formats with [CONTACT-MASKED].

    Applies masks in order of specificity (most specific first) to avoid
    double-masking or partial matches.

    Parameters
    ----------
    text : Any string that may contain contact numbers.

    Returns
    -------
    String with all contact numbers replaced by [CONTACT-MASKED].
    """
    if not text:
        return text

    # Order: most specific formats first
    result = _CANONICAL_CONTACT.sub(_MASK_PLACEHOLDER, text)
    result = _SPACE_FORMAT.sub(_MASK_PLACEHOLDER, result)
    result = _STANDALONE_10.sub(_MASK_PLACEHOLDER, result)

    return result


def is_contact_present(text: str) -> bool:
    """Return True if any unmasked contact number pattern is found."""
    return bool(
        _CANONICAL_CONTACT.search(text)
        or _SPACE_FORMAT.search(text)
        or _STANDALONE_10.search(text)
    )


def mask_dict_values(data: dict) -> dict:
    """
    Recursively mask contact numbers in all string values of a dict.
    Safe for log event dictionaries.
    """
    masked: dict = {}
    for key, value in data.items():
        if isinstance(value, str):
            masked[key] = mask_contact_numbers(value)
        elif isinstance(value, dict):
            masked[key] = mask_dict_values(value)
        elif isinstance(value, list):
            masked[key] = [
                mask_contact_numbers(item) if isinstance(item, str)
                else mask_dict_values(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            masked[key] = value
    return masked


# ---------------------------------------------------------------------------
# Demo (for transcript evidence)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("PII Masker - Demo")
    print("=" * 60)
    examples = [
        f"Patient contact: {DEMO_SYNTHETIC_NUMBER}",
        "Call us at +91 8765432109 for support.",
        "Mobile number 9123456789 registered.",
        "No phone number here.",
        "+91-9000000001 and 9000000002 are two different numbers.",
    ]
    for ex in examples:
        masked = mask_contact_numbers(ex)
        print(f"  Input : {ex}")
        print(f"  Output: {masked}")
        print()
