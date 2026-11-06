"""
mock_llm.py - Deterministic Mock LLM for CrewAI-compatible capstone.

Subclasses BaseLLM from crewai_compat (Python 3.9-compatible).
No API key, no network call.

Routing logic:
  - "APT-" / appointment keywords -> appointment_status
  - Diagnosis keywords  -> refusal (out_of_scope)
  - Out-of-domain keywords -> refusal (out_of_scope)
  - Everything else  -> policy intent
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from practo_agent.agents.crewai_compat import BaseLLM


# ---------------------------------------------------------------------------
# Intent patterns
# ---------------------------------------------------------------------------
_APT_PATTERN = re.compile(r"\bAPT-\d{4}\b", re.IGNORECASE)
_STATUS_KEYWORDS = re.compile(
    r"\b(status|appointment status|status of appointment|look up appointment)\b",
    re.IGNORECASE,
)
_DIAGNOSIS_KEYWORDS = re.compile(
    r"\b(diagnose|diagnosis|what disease|do i have|treat my|cure my|"
    r"is it cancer|chest pain diagnosis|diagnose me|can you diagnose)\b",
    re.IGNORECASE,
)
_OUT_OF_SCOPE_KEYWORDS = re.compile(
    r"\b(weather|restaurant|recipe|stock price|capital of|football|cricket score|news today)\b",
    re.IGNORECASE,
)


def _detect_intent(text: str) -> str:
    if _APT_PATTERN.search(text) or _STATUS_KEYWORDS.search(text):
        return "appointment_status"
    if _DIAGNOSIS_KEYWORDS.search(text):
        return "diagnosis_refusal"
    if _OUT_OF_SCOPE_KEYWORDS.search(text):
        return "out_of_scope"
    return "policy"


# ---------------------------------------------------------------------------
# Deterministic response templates
# ---------------------------------------------------------------------------
def _build_policy_response(context: str, source_ids: List[str]) -> dict:
    snippet = (context[:250] + "...") if len(context) > 250 else context
    return {
        "answer": (
            f"Based on clinic policy: {snippet} "
            "For complete details, please refer to the official clinic documentation "
            "or contact the front desk."
        ),
        "intent": "policy",
        "source_ids": source_ids,
        "confidence": 0.85,
        "needs_escalation": False,
        "refusal_reason": None,
    }


def _build_appointment_response(apt_id: str, record: Optional[dict]) -> dict:
    if record and record.get("found"):
        return {
            "answer": (
                f"Appointment {apt_id}: Status is '{record['status']}'. "
                f"Category: {record['category']}, "
                f"Days since created: {record['days_since_created']}. "
                f"Follow-up required: {record['follow_up_required']}."
            ),
            "intent": "appointment_status",
            "source_ids": [apt_id],
            "confidence": 0.95,
            "needs_escalation": record.get("escalation_score", 0) >= 0.65,
            "refusal_reason": None,
        }
    return {
        "answer": (
            f"No appointment record found for ID '{apt_id}'. "
            "Please verify the appointment ID and contact the clinic if you need assistance."
        ),
        "intent": "appointment_status",
        "source_ids": [],
        "confidence": 0.90,
        "needs_escalation": False,
        "refusal_reason": "Appointment ID not found in system.",
    }


def _build_diagnosis_refusal() -> dict:
    return {
        "answer": (
            "I'm unable to diagnose medical conditions. "
            "If you are experiencing symptoms that concern you, please consult a qualified "
            "healthcare professional immediately. "
            "For emergencies, contact local emergency services or go to the nearest emergency department."
        ),
        "intent": "out_of_scope",
        "source_ids": [],
        "confidence": 1.0,
        "needs_escalation": True,
        "refusal_reason": "Diagnosis requests are outside the scope of this administrative support agent.",
    }


def _build_out_of_scope_refusal(query: str) -> dict:
    return {
        "answer": (
            "I can only assist with clinic-related questions such as appointment booking, "
            "cancellations, fees, insurance, telemedicine, and related healthcare administrative "
            "matters. Your question appears to be outside my scope. "
            "Please contact the clinic directly for further assistance."
        ),
        "intent": "out_of_scope",
        "source_ids": [],
        "confidence": 0.95,
        "needs_escalation": False,
        "refusal_reason": f"Query is outside clinic administrative domain: '{query[:80]}'",
    }


def _build_fallback_response() -> dict:
    return {
        "answer": (
            "I don't have sufficient information in the knowledge base to answer your question "
            "accurately. Please contact the clinic front desk for detailed assistance."
        ),
        "intent": "policy",
        "source_ids": [],
        "confidence": 0.30,
        "needs_escalation": True,
        "refusal_reason": "No sufficiently relevant policy document found (below retrieval threshold).",
    }


# ---------------------------------------------------------------------------
# Mock LLM class (subclasses compat BaseLLM)
# ---------------------------------------------------------------------------
class MockCrewLLM(BaseLLM):
    """
    Deterministic mock LLM. Subclasses BaseLLM (CrewAI's documented custom LLM
    extension point, implemented via crewai_compat).

    No network, no API key. Parses only the last user message to avoid
    reading the ReAct system template which contains placeholder text.
    """
    model: str = "mock-practo-v1"

    def call(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[list] = None,
        **kwargs: Any,
    ) -> str:
        user_text = self._extract_user_text(messages)
        return self._generate(user_text)

    def supports_function_calling(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 4096

    @staticmethod
    def _extract_user_text(messages: List[Dict[str, str]]) -> str:
        """Return only the last human/user turn content."""
        for msg in reversed(messages):
            if msg.get("role") in ("user", "human"):
                return msg.get("content", "")
        return messages[-1].get("content", "") if messages else ""

    def _generate(self, user_text: str) -> str:
        """Build deterministic JSON response based on intent."""
        # Check for tool results embedded in the context
        tool_context, source_ids = self._extract_tool_context(user_text)

        # Extract the actual query from the text
        # For composer tasks: 'for: "<actual query>"' pattern
        import re as _re
        query_match = _re.search(r'for:\s*"([^"]+)"', user_text)
        query_for_intent = query_match.group(1) if query_match else user_text[:200]

        intent = _detect_intent(query_for_intent)

        if intent == "appointment_status":
            apt_match = _APT_PATTERN.search(query_for_intent)
            apt_id = apt_match.group(0).upper() if apt_match else "UNKNOWN"
            try:
                from practo_agent.dataset import check_appointment_status
                record = check_appointment_status(apt_id)
            except Exception:
                record = None
            response = _build_appointment_response(apt_id, record)
        elif intent == "diagnosis_refusal":
            response = _build_diagnosis_refusal()
        elif intent == "out_of_scope":
            response = _build_out_of_scope_refusal(query_for_intent)
        else:
            # Policy: use extracted tool context if available
            context = tool_context if tool_context else user_text[:300]
            response = _build_policy_response(context, source_ids)

        return json.dumps(response)

    @staticmethod
    def _extract_tool_context(text: str) -> tuple:
        """Extract retrieved text and source IDs from embedded tool results."""
        source_ids = []
        context_parts = []
        try:
            # Look for JSON tool output embedded in context
            import re as _re
            json_blocks = _re.findall(r'\{[^{}]{20,}\}', text)
            for block in json_blocks:
                try:
                    data = json.loads(block)
                    if "results" in data and isinstance(data["results"], list):
                        for r in data["results"]:
                            if "text" in r:
                                context_parts.append(r["text"])
                            if "parent_doc_id" in r and r["parent_doc_id"] not in source_ids:
                                source_ids.append(r["parent_doc_id"])
                except Exception:
                    pass
        except Exception:
            pass
        return " ".join(context_parts), source_ids


# ---------------------------------------------------------------------------
# Context-aware response builder (used by crew_runner)
# ---------------------------------------------------------------------------
def build_response_from_context(
    query: str,
    retrieved_context: str,
    source_ids: List[str],
    appointment_record: Optional[dict] = None,
    fallback_triggered: bool = False,
) -> dict:
    if fallback_triggered:
        return _build_fallback_response()

    intent = _detect_intent(query)
    if intent == "appointment_status" and appointment_record is not None:
        apt_match = _APT_PATTERN.search(query)
        apt_id = apt_match.group(0).upper() if apt_match else "UNKNOWN"
        return _build_appointment_response(apt_id, appointment_record)
    elif intent == "diagnosis_refusal":
        return _build_diagnosis_refusal()
    elif intent == "out_of_scope":
        return _build_out_of_scope_refusal(query)
    else:
        return _build_policy_response(retrieved_context, source_ids)
