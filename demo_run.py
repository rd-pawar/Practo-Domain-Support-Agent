import json
import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import sys
sys.path.insert(0, r"d:\Healthcare_AgenticSystem\src")
from practo_agent.agents.crew_runner import run

queries = [
    "How far ahead should I book an appointment?",
    "What is the cancellation policy?",
    "What is the status of appointment APT-0001?",
    "Can you diagnose my chest pain?",
    "What will the weather be like tomorrow?",
    "APT-9999 status please",
]

for q in queries:
    print("\n" + "=" * 65)
    r = run(q, session_id="demo", verbose=False)
    cr = r.get("crew_response") or {}
    intent = cr.get("intent", "?")
    src = cr.get("source_ids", [])
    ans = (cr.get("answer") or "")[:160]
    cache_hit = r["cache_hit"]
    grounded_info = r["guardrail_outcomes"].get("groundedness", {})
    grounded = grounded_info.get("is_grounded", "?")
    score = grounded_info.get("score", 0)
    ag = r.get("autogen_verdict") or {}
    approved = ag.get("approved", "?")
    print(f"Q:       {q}")
    print(f"Intent:  {intent} | Sources: {src}")
    print(f"Answer:  {ans}")
    print(f"Grnd: {grounded} ({score:.2f}) | Cache: {cache_hit} | AutoGen approved: {approved}")
