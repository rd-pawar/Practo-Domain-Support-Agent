"""
report.py - Evaluation report generator for Practo Domain Support Agent.

Runs all 15-query evaluation, prints per-query table, and writes results.json.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from evaluator import QueryScore, run_evaluation


def print_report(scores: list[QueryScore]) -> dict:
    print("\n" + "=" * 100)
    print("PRACTO DOMAIN SUPPORT AGENT - 15-QUERY EVALUATION REPORT")
    print("NOTE: These are MOCK_LLM scores. They verify plumbing, NOT real-model quality.")
    print("=" * 100)

    header = (
        f"{'QID':<7} {'Query':<55} {'Route':<20} "
        f"{'A':>2} {'G':>2} {'C':>2} {'S':>2} {'Avg':>5}"
    )
    print(header)
    print("-" * 100)

    all_avg: list[float] = []
    rows: list[dict] = []

    for s in scores:
        avg = round((s.accuracy + s.grounding + s.completeness + s.safety) / 4, 2)
        all_avg.append(avg)
        print(
            f"{s.query_id:<7} {s.query[:53]:<55} {s.actual_intent or s.expected_route:<20} "
            f"{s.accuracy:>2} {s.grounding:>2} {s.completeness:>2} {s.safety:>2} {avg:>5.2f}"
        )
        rows.append({
            "query_id": s.query_id,
            "query": s.query,
            "expected_route": s.expected_route,
            "actual_intent": s.actual_intent,
            "answer_snippet": s.answer[:200],
            "accuracy": s.accuracy,
            "grounding": s.grounding,
            "completeness": s.completeness,
            "safety": s.safety,
            "average": avg,
        })

    print("-" * 100)
    avg_acc = round(sum(s.accuracy for s in scores) / len(scores), 2)
    avg_grd = round(sum(s.grounding for s in scores) / len(scores), 2)
    avg_cmp = round(sum(s.completeness for s in scores) / len(scores), 2)
    avg_sft = round(sum(s.safety for s in scores) / len(scores), 2)
    overall = round(sum(all_avg) / len(all_avg), 2)

    print(
        f"{'AVERAGE':<7} {'(all 15 queries)':<55} {'':<20} "
        f"{avg_acc:>2.1f} {avg_grd:>2.1f} {avg_cmp:>2.1f} {avg_sft:>2.1f} {overall:>5.2f}"
    )
    print("=" * 100)
    print(f"\nAverage Accuracy    : {avg_acc}/5")
    print(f"Average Grounding   : {avg_grd}/5")
    print(f"Average Completeness: {avg_cmp}/5")
    print(f"Average Safety      : {avg_sft}/5")
    print(f"Overall Average     : {overall}/5")

    return {
        "queries": rows,
        "averages": {
            "accuracy": avg_acc,
            "grounding": avg_grd,
            "completeness": avg_cmp,
            "safety": avg_sft,
            "overall": overall,
        },
        "note": "MOCK_LLM evaluation scores - verifies plumbing, not real model quality.",
    }


if __name__ == "__main__":
    print("Running 15-query evaluation...\n")
    scores = run_evaluation(verbose=True)
    report = print_report(scores)

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
