"""
logger.py - Structured JSONL event logger for Practo Domain Support API.

Emits one JSON object per line to logs/api_events.jsonl.
PII (contact numbers) is masked BEFORE log field construction.
Fields: trace_id, timestamp, route, session_id, query_masked,
        intent, retrieval_ids, cache_state, guardrail_outcomes,
        result_code, duration_ms, budget_tokens.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from practo_agent.guardrails.pii_masker import mask_contact_numbers, mask_dict_values

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "api_events.jsonl"

_json_logger = logging.getLogger("practo.api.events")
_json_logger.setLevel(logging.DEBUG)

_handler: Optional[logging.FileHandler] = None


def _ensure_handler() -> None:
    global _handler
    if _handler is None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        _handler.setFormatter(logging.Formatter("%(message)s"))
        _json_logger.addHandler(_handler)
        _json_logger.propagate = False


def emit_event(
    trace_id: str,
    route: str,
    session_id: str,
    query: str,                    # raw query - will be masked before logging
    result_code: str = "ok",
    intent: Optional[str] = None,
    retrieval_ids: Optional[list[str]] = None,
    cache_state: str = "miss",
    guardrail_outcomes: Optional[dict] = None,
    duration_ms: float = 0.0,
    budget_tokens: int = 0,
) -> None:
    """
    Emit one masked JSONL log event.

    NOTE: query is masked immediately here; the raw value is NEVER written
    to the log file, trace, or exception message.
    """
    _ensure_handler()

    masked_query = mask_contact_numbers(query)
    safe_guardrails = mask_dict_values(guardrail_outcomes or {})

    event = {
        "trace_id": trace_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "route": route,
        "session_id": session_id,
        "query_masked": masked_query,
        "intent": intent,
        "retrieval_ids": retrieval_ids or [],
        "cache_state": cache_state,
        "guardrail_outcomes": safe_guardrails,
        "result_code": result_code,
        "duration_ms": duration_ms,
        "budget_tokens": budget_tokens,
    }

    _json_logger.info(json.dumps(event, ensure_ascii=False))


def get_recent_events(n: int = 10) -> list[dict]:
    """Read the last *n* events from the JSONL log file (for inspection)."""
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    recent = lines[-n:] if len(lines) >= n else lines
    events = []
    for line in recent:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events
