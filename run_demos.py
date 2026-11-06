"""
run_demos.py - CLI runner for all capstone demonstrations.

Usage:
  python run_demos.py [demo_name]

Available demos:
  dataset     - Generate and validate the 50-record dataset
  index       - Build ChromaDB indexes (both chunking strategies)
  calibrate   - Run retrieval calibration and print threshold
  metrics     - Run P@k / R@k comparison for both strategies
  rag         - Run a sample policy query through the full crew
  lookup      - Run a sample appointment lookup
  guardrails  - Demo PII masking, injection detection, groundedness
  cache       - Demo cache hit/miss counters
  budget      - Demo budget rejection
  autogen     - Demo AutoGen approval and revision
  memory      - Demo same-session and fresh-session memory
  evaluate    - Run 15-query evaluation and print report
  api         - Start the FastAPI server (Ctrl+C to stop)
  all         - Run all demos except 'api'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be before any CrewAI import
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

sys.path.insert(0, str(Path(__file__).parent / "src"))


def demo_dataset():
    print("\n" + "="*60)
    print("DEMO: Dataset Generator")
    print("="*60)
    import subprocess
    subprocess.run([sys.executable, "-m", "practo_agent.dataset"], check=True)


def demo_index():
    print("\n" + "="*60)
    print("DEMO: ChromaDB Indexing")
    print("="*60)
    from practo_agent.rag.chroma_index import build_all_indexes
    build_all_indexes(verbose=True)


def demo_calibrate():
    print("\n" + "="*60)
    print("DEMO: Retrieval Calibration")
    print("="*60)
    from practo_agent.rag.calibration import run_calibration
    run_calibration(verbose=True)


def demo_metrics():
    print("\n" + "="*60)
    print("DEMO: Retrieval Metrics (P@k / R@k)")
    print("="*60)
    from practo_agent.rag.metrics import run_retrieval_comparison
    run_retrieval_comparison(top_k=3, verbose=True)


def demo_rag():
    print("\n" + "="*60)
    print("DEMO: RAG Policy Query")
    print("="*60)
    from practo_agent.agents.crew_runner import run
    result = run("How far ahead should I book an appointment?", session_id="demo-rag", verbose=True)
    print("\nFINAL ANSWER:", result.get("crew_response", {}).get("answer", ""))


def demo_lookup():
    print("\n" + "="*60)
    print("DEMO: Appointment Lookup")
    print("="*60)
    from practo_agent.agents.crew_runner import run
    result = run("What is the status of appointment APT-0001?", session_id="demo-lookup", verbose=True)
    print("\nFINAL ANSWER:", result.get("crew_response", {}).get("answer", ""))


def demo_guardrails():
    print("\n" + "="*60)
    print("DEMO: Guardrails")
    print("="*60)
    import subprocess
    subprocess.run([sys.executable, "-m", "practo_agent.guardrails.pii_masker"], check=True)
    subprocess.run([sys.executable, "-m", "practo_agent.guardrails.injection_detector"], check=True)
    subprocess.run([sys.executable, "-m", "practo_agent.guardrails.groundedness"], check=True)


def demo_cache():
    print("\n" + "="*60)
    print("DEMO: Cache Hit/Miss")
    print("="*60)
    import subprocess
    subprocess.run([sys.executable, "-m", "practo_agent.guardrails.cache"], check=True)


def demo_budget():
    print("\n" + "="*60)
    print("DEMO: Budget Rejection")
    print("="*60)
    import subprocess
    subprocess.run([sys.executable, "-m", "practo_agent.guardrails.budget"], check=True)


def demo_autogen():
    print("\n" + "="*60)
    print("DEMO: AutoGen Review")
    print("="*60)
    import subprocess
    subprocess.run([sys.executable, "-m", "practo_agent.agents.autogen_review"], check=True)


def demo_memory():
    print("\n" + "="*60)
    print("DEMO: Session Memory")
    print("="*60)
    import subprocess
    subprocess.run([sys.executable, "-m", "practo_agent.agents.memory"], check=True)


def demo_evaluate():
    print("\n" + "="*60)
    print("DEMO: 15-Query Evaluation")
    print("="*60)
    import subprocess
    subprocess.run([sys.executable, "evaluation/report.py"], check=True)


def demo_api():
    print("\n" + "="*60)
    print("DEMO: FastAPI Server (Ctrl+C to stop)")
    print("="*60)
    import uvicorn
    uvicorn.run("practo_agent.api.app:app", host="0.0.0.0", port=8000, reload=False)


DEMOS = {
    "dataset":   demo_dataset,
    "index":     demo_index,
    "calibrate": demo_calibrate,
    "metrics":   demo_metrics,
    "rag":       demo_rag,
    "lookup":    demo_lookup,
    "guardrails":demo_guardrails,
    "cache":     demo_cache,
    "budget":    demo_budget,
    "autogen":   demo_autogen,
    "memory":    demo_memory,
    "evaluate":  demo_evaluate,
    "api":       demo_api,
}


if __name__ == "__main__":
    demo_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    if demo_name == "all":
        for name, fn in DEMOS.items():
            if name == "api":
                continue
            try:
                fn()
            except Exception as e:
                print(f"[DEMO {name}] ERROR: {e}")
    elif demo_name in DEMOS:
        DEMOS[demo_name]()
    else:
        print(f"Unknown demo: {demo_name!r}")
        print(f"Available: {', '.join(DEMOS.keys())}")
        sys.exit(1)
