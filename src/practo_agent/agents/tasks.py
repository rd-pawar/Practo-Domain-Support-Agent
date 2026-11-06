"""
tasks.py - CrewAI-compatible Task definitions for Practo Domain Support.
Uses crewai_compat.Task.
"""
from __future__ import annotations

from practo_agent.agents.crewai_compat import Task


def make_retrieval_task(agent, query: str, strategy: str = "fixed") -> Task:
    return Task(
        description=(
            f'Search the clinic policy knowledge base for: "{query}"\n'
            f"Use the rag_lookup tool with strategy='{strategy}' and top_k=3.\n"
            "Return JSON with retrieved chunks, source IDs, and similarity scores."
        ),
        expected_output=(
            "JSON with keys: fallback_triggered (bool), collection (str), "
            "results (list of chunk_id, parent_doc_id, text, similarity)."
        ),
        agent=agent,
    )


def make_lookup_task(agent, appointment_id: str) -> Task:
    return Task(
        description=(
            f"Look up appointment record for ID: '{appointment_id}'. "
            "Use the check_appointment_status tool. "
            "If not found, return a clear not-found response. Do NOT invent details."
        ),
        expected_output=(
            "JSON with keys: found (bool), appointment_id, status, category, "
            "days_since_created, follow_up_required, escalation_score. "
            "If not found: found=False, message explaining situation."
        ),
        agent=agent,
    )


def make_composer_task(agent, query: str, context: str = "") -> Task:
    return Task(
        description=(
            f'Compose a final structured response for: "{query}"\n'
            f"Context from retrieval/lookup:\n{context}\n\n"
            "Rules:\n"
            "1. Answer only from the provided context. Do not fabricate.\n"
            "2. If context is empty or fallback was triggered, say 'I do not know'.\n"
            "3. Never diagnose or recommend treatment.\n"
            "4. For emergencies, direct to local emergency services.\n"
            "5. Output must be valid JSON matching the CrewResponse schema."
        ),
        expected_output=(
            "JSON: answer (str), intent (policy|appointment_status|mixed|out_of_scope), "
            "source_ids (list), confidence (0-1), needs_escalation (bool), refusal_reason (str|null)."
        ),
        agent=agent,
    )
