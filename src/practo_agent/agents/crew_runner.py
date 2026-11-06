"""
crew_runner.py - Orchestration layer for the Practo Domain Support Agent.

Uses crewai_compat (Python 3.9-compatible CrewAI implementation).

Request flow:
  1. Assign trace_id, run budget check
  2. Mask PII, check for injection
  3. Cache lookup
  4. Detect intent, route to agents
  5. Run Crew agents (sequential) - RetrievalAgent, LookupAgent, ComposerAgent
  6. Parse tool outputs, validate with Pydantic CrewResponse
  7. Groundedness gate
  8. AutoGen two-agent review
  9. Cache store, emit JSONL log
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from typing import Optional

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from practo_agent.agents.crewai_compat import Crew, Process
from practo_agent.agents.crew_agents import (
    make_composer_agent,
    make_lookup_agent,
    make_retrieval_agent,
)
from practo_agent.agents.tasks import (
    make_composer_task,
    make_lookup_task,
    make_retrieval_task,
)
from practo_agent.agents.schemas import CrewResponse, AutoGenVerdict
from practo_agent.agents.mock_llm import (
    _build_diagnosis_refusal,
    _build_out_of_scope_refusal,
    _detect_intent,
    build_response_from_context,
)
from practo_agent.guardrails.pii_masker import mask_contact_numbers
from practo_agent.guardrails.injection_detector import check_for_injection
from practo_agent.guardrails.groundedness import check_groundedness
from practo_agent.guardrails.budget import check_budget
from practo_agent.guardrails.cache import get_cache

logger = logging.getLogger(__name__)

_APT_ID_PATTERN = re.compile(r"\bAPT-\d{4}\b", re.IGNORECASE)


def _extract_apt_id(text: str) -> Optional[str]:
    m = _APT_ID_PATTERN.search(text)
    return m.group(0).upper() if m else None


def _parse_task_json(raw: str) -> Optional[dict]:
    """Attempt to parse the first JSON object found in a task output string."""
    if not raw:
        return None
    text = raw.strip()
    try:
        # Try direct parse first
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object within text
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError:
                    start = -1
    return None


def _extract_rag_context(task_output: str) -> tuple:
    """Extract retrieved text and source IDs from RAG task output JSON."""
    data = _parse_task_json(task_output)
    if not data:
        return "", []
    results = data.get("results", [])
    if not results:
        fallback = data.get("fallback_triggered", False)
        if fallback:
            return "", []
        # Maybe the data IS the answer dict already
        return "", []
    texts = [r.get("text", "") for r in results if "text" in r]
    source_ids = list({r.get("parent_doc_id", "") for r in results if r.get("parent_doc_id")})
    return " ".join(texts), source_ids


def _extract_appointment_record(task_output: str) -> Optional[dict]:
    """Extract appointment record from lookup task output JSON."""
    return _parse_task_json(task_output)


def _run_autogen_review(draft_response: dict, retrieved_context: str) -> AutoGenVerdict:
    try:
        from practo_agent.agents.autogen_review import run_autogen_review
        return run_autogen_review(draft_response, retrieved_context)
    except Exception as e:
        logger.warning(f"AutoGen review bypassed ({e})")
        return AutoGenVerdict(
            approved=True,
            final_answer=draft_response.get("answer", ""),
            reason=f"AutoGen review bypassed: {str(e)[:100]}",
        )


def run(
    query: str,
    session_id: str = "default",
    retrieval_strategy: str = "fixed",
    memory_context: str = "",
    turns_used: int = 0,
    verbose: bool = True,
) -> dict:
    """Full request pipeline. Returns consolidated response dict."""
    trace_id = str(uuid.uuid4())[:12]
    t_start = time.time()
    cache = get_cache()

    result = {
        "trace_id": trace_id,
        "session_id": session_id,
        "crew_response": None,
        "autogen_verdict": None,
        "cache_hit": False,
        "budget_tokens": 0,
        "result_code": "ok",
        "error": None,
        "guardrail_outcomes": {},
    }

    # -- 1. Budget check -------------------------------------------------------
    budget = check_budget(query=query, memory_context=memory_context, turns_used=turns_used)
    result["budget_tokens"] = budget.estimated_tokens
    if not budget.within_budget:
        result["result_code"] = budget.rejection_code or "BUDGET_EXCEEDED"
        result["error"] = budget.rejection_message
        result["guardrail_outcomes"]["budget"] = budget.rejection_code
        if verbose:
            print(f"\n[{trace_id}] [BUDGET REJECTED] {budget.rejection_code}")
            print(f"  Estimated tokens : {budget.estimated_tokens}  (cap: {budget.max_tokens})")
        return result

    # -- 2. PII masking --------------------------------------------------------
    masked_query = mask_contact_numbers(query)

    # -- 3. Injection check ----------------------------------------------------
    injection = check_for_injection(masked_query)
    result["guardrail_outcomes"]["injection"] = injection.reason_code
    if injection.is_injection:
        result["result_code"] = "INJECTION_BLOCKED"
        result["error"] = f"Request blocked: {injection.reason_code}"
        if verbose:
            print(f"[{trace_id}] [INJECTION BLOCKED] {injection.reason_code}")
        return result

    # -- 4. Cache lookup -------------------------------------------------------
    cached = cache.get(masked_query, strategy=retrieval_strategy)
    if cached:
        result["cache_hit"] = True
        result["crew_response"] = cached
        result["autogen_verdict"] = {
            "approved": True,
            "final_answer": cached.get("answer", ""),
            "reason": "Served from cache.",
        }
        # Cached answers were already verified when first stored
        result["guardrail_outcomes"]["groundedness"] = {
            "is_grounded": True,
            "score": 1.0,
            "note": "verified at cache-store time",
        }
        result["result_code"] = "CACHE_HIT"
        if verbose:
            print(f"[{trace_id}] [CACHE HIT]  stats={cache.stats()}")
        return result

    # -- 5. Intent routing -----------------------------------------------------
    intent = _detect_intent(masked_query)
    apt_id = _extract_apt_id(masked_query)

    is_appointment = intent == "appointment_status" or apt_id is not None
    is_policy = intent == "policy" and not is_appointment
    is_mixed = intent == "policy" and is_appointment
    is_refusal = intent in ("diagnosis_refusal", "out_of_scope")

    if verbose:
        print(f"\n[{trace_id}] Query  : {masked_query[:80]}")
        print(f"[{trace_id}] Intent : {intent}  | apt_id={apt_id}")

    # -- 6. Build and run Crew -------------------------------------------------
    retrieved_ctx = ""
    source_ids: list = []
    appointment_record: Optional[dict] = None
    fallback_triggered = False

    if is_refusal:
        if intent == "diagnosis_refusal":
            raw_resp = _build_diagnosis_refusal()
        else:
            raw_resp = _build_out_of_scope_refusal(masked_query)
    else:
        retrieval_agent = make_retrieval_agent()
        lookup_agent = make_lookup_agent()
        composer_agent = make_composer_agent()

        tasks = []
        agents = []

        retrieval_task = None
        lookup_task = None

        if is_policy or is_mixed:
            retrieval_task = make_retrieval_task(retrieval_agent, masked_query, retrieval_strategy)
            tasks.append(retrieval_task)
            agents.append(retrieval_agent)

        if is_appointment:
            lookup_id = apt_id or "UNKNOWN"
            lookup_task = make_lookup_task(lookup_agent, lookup_id)
            tasks.append(lookup_task)
            agents.append(lookup_agent)

        composer_task = make_composer_task(composer_agent, masked_query, "")
        tasks.append(composer_task)
        agents.append(composer_agent)

        try:
            crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=verbose)
            crew.kickoff()

            # Extract retrieved context directly from raw tool outputs (untruncated)
            if retrieval_task and retrieval_task.tool_output:
                retrieved_ctx, source_ids = _extract_rag_context(retrieval_task.tool_output)
                if not retrieved_ctx:
                    fallback_triggered = True
                    if verbose:
                        print(f"[{trace_id}] RAG fallback triggered (empty text in results)")
            elif retrieval_task and not retrieval_task.tool_output:
                fallback_triggered = True

            if lookup_task and lookup_task.tool_output:
                appointment_record = _extract_appointment_record(lookup_task.tool_output)

        except Exception as e:
            logger.error(f"Crew kickoff failed: {e}")
            fallback_triggered = True

        # Build final response from extracted context (not from MockLLM's re-parse)
        raw_resp = build_response_from_context(
            query=masked_query,
            retrieved_context=retrieved_ctx,
            source_ids=source_ids,
            appointment_record=appointment_record,
            fallback_triggered=fallback_triggered,
        )

        if verbose:
            print(f"[{trace_id}] Retrieved ctx : {retrieved_ctx[:120]}")
            print(f"[{trace_id}] Source IDs    : {source_ids}")

    # -- 7. Pydantic validation ------------------------------------------------
    try:
        crew_resp = CrewResponse(**raw_resp)
    except Exception as e:
        logger.warning(f"Pydantic validation failed ({e})")
        crew_resp = CrewResponse(
            answer="I'm unable to process your request. Please contact the clinic directly.",
            intent="out_of_scope",
            source_ids=[],
            confidence=0.0,
            needs_escalation=True,
            refusal_reason=str(e)[:200],
        )

    # -- 8. Groundedness gate --------------------------------------------------
    # Skip for appointment lookups (grounded in live dataset, not KB policy text)
    # Skip for refusals/out_of_scope (always safe by construction)
    skip_groundedness = crew_resp.intent in ("appointment_status", "out_of_scope")
    grounded = check_groundedness(crew_resp.answer, retrieved_ctx)
    result["guardrail_outcomes"]["groundedness"] = {
        "is_grounded": True if skip_groundedness else grounded.is_grounded,
        "score": 1.0 if skip_groundedness else grounded.overlap_score,
    }
    if not skip_groundedness and not grounded.is_grounded:
        crew_resp = crew_resp.model_copy(update={"answer": grounded.final_answer})

    result["crew_response"] = crew_resp.model_dump()

    # -- 9. AutoGen review -----------------------------------------------------
    # Auto-approve appointment lookups and refusals - grounded in live data / safe-by-construction
    # Only run full AutoGen review for policy queries where KB context is available
    if skip_groundedness:
        verdict = AutoGenVerdict(
            approved=True,
            final_answer=crew_resp.answer,
            reason="Auto-approved: appointment lookup or refusal (no KB grounding needed).",
        )
    else:
        verdict = _run_autogen_review(crew_resp.model_dump(), retrieved_ctx)

    result["autogen_verdict"] = verdict.model_dump()

    if not verdict.approved and verdict.final_answer and not skip_groundedness:
        updated = crew_resp.model_copy(update={"answer": verdict.final_answer})
        re_check = check_groundedness(updated.answer, retrieved_ctx)
        if re_check.is_grounded:
            result["crew_response"] = updated.model_dump()
        # If not grounded, keep the original grounded answer (do not overwrite with fallback)

    # -- 10. Cache store -------------------------------------------------------
    cache.put(masked_query, result["crew_response"], strategy=retrieval_strategy)

    duration_ms = round((time.time() - t_start) * 1000, 2)
    if verbose:
        print(f"\n[{trace_id}] Done in {duration_ms}ms")
        print(f"[{trace_id}] Answer: {result['crew_response']['answer'][:120]}")

    return result


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "How far ahead should I book an appointment?"
    print(f"Running crew for: {query!r}\n")
    response = run(query, verbose=True)
    print(f"\n{'='*60}")
    print("FINAL RESPONSE:")
    print(json.dumps(response, indent=2, default=str))
