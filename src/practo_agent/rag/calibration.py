"""
calibration.py - Retrieval threshold calibration for Practo RAG.

Runs in-scope and out-of-scope queries, records top-1 cosine values,
inspects clusters, and recommends a threshold between the two groups.

Usage:
    python -m practo_agent.rag.calibration
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from practo_agent.rag.embeddings import embed_single
from practo_agent.rag.chroma_index import get_collection, COLLECTION_FIXED

# ---------------------------------------------------------------------------
# Calibration query set
# ---------------------------------------------------------------------------

@dataclass
class CalibrationQuery:
    query: str
    in_scope: bool
    expected_parent_doc: str | None   # None for out-of-scope


IN_SCOPE_QUERIES: list[CalibrationQuery] = [
    CalibrationQuery(
        query="How far ahead should I book an appointment?",
        in_scope=True,
        expected_parent_doc="KB-001",
    ),
    CalibrationQuery(
        query="What notice do I need to cancel my appointment?",
        in_scope=True,
        expected_parent_doc="KB-002",
    ),
    CalibrationQuery(
        query="What is the consultation fee for Cardiology?",
        in_scope=True,
        expected_parent_doc="KB-003",
    ),
    CalibrationQuery(
        query="Can I book a telemedicine appointment from home?",
        in_scope=True,
        expected_parent_doc="KB-007",
    ),
    CalibrationQuery(
        query="How does the clinic protect my personal data?",
        in_scope=True,
        expected_parent_doc="KB-009",
    ),
    CalibrationQuery(
        query="Is there a discount for a follow-up visit?",
        in_scope=True,
        expected_parent_doc="KB-010",
    ),
]

OUT_OF_SCOPE_QUERIES: list[CalibrationQuery] = [
    CalibrationQuery(
        query="What will the weather be like tomorrow?",
        in_scope=False,
        expected_parent_doc=None,
    ),
    CalibrationQuery(
        query="What is the capital of France?",
        in_scope=False,
        expected_parent_doc=None,
    ),
    CalibrationQuery(
        query="Recommend a good Italian restaurant nearby.",
        in_scope=False,
        expected_parent_doc=None,
    ),
]


# ---------------------------------------------------------------------------
# Calibration runner
# ---------------------------------------------------------------------------

def run_calibration(verbose: bool = True) -> dict:
    """
    Run all calibration queries against the fixed-strategy collection.
    Returns a dict with per-query top-1 similarities and a recommended threshold.
    """
    collection = get_collection(COLLECTION_FIXED)
    all_queries = IN_SCOPE_QUERIES + OUT_OF_SCOPE_QUERIES

    results: list[dict] = []
    in_scope_sims: list[float] = []
    out_of_scope_sims: list[float] = []

    for cq in all_queries:
        qvec = embed_single(cq.query)
        cr = collection.query(
            query_embeddings=[qvec],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        if cr["distances"][0]:
            top1_sim = round(1.0 - cr["distances"][0][0], 4)
            top1_doc = cr["metadatas"][0][0].get("parent_doc_id", "unknown") if cr["metadatas"][0] else "unknown"
        else:
            top1_sim = 0.0
            top1_doc = "none"

        result = {
            "query": cq.query,
            "in_scope": cq.in_scope,
            "expected_doc": cq.expected_parent_doc,
            "top1_doc": top1_doc,
            "top1_similarity": top1_sim,
        }
        results.append(result)

        if cq.in_scope:
            in_scope_sims.append(top1_sim)
        else:
            out_of_scope_sims.append(top1_sim)

    # Choose threshold midway between min in-scope and max out-of-scope
    min_in_scope = min(in_scope_sims) if in_scope_sims else 0.5
    max_out_scope = max(out_of_scope_sims) if out_of_scope_sims else 0.0
    recommended_threshold = round((min_in_scope + max_out_scope) / 2, 4)

    # Clamp threshold to a sensible range
    recommended_threshold = max(0.20, min(0.50, recommended_threshold))

    if verbose:
        print("=" * 60)
        print("Retrieval Calibration Report")
        print("=" * 60)
        print(f"\n{'Query':<50} {'In-Scope':>9} {'Top-1 Doc':>10} {'Similarity':>10}")
        print("-" * 85)
        for r in results:
            scope_label = "YES" if r["in_scope"] else "NO"
            print(f"{r['query']:<50} {scope_label:>9} {r['top1_doc']:>10} {r['top1_similarity']:>10.4f}")

        print("\nCluster summary:")
        print(f"  In-scope  similarities : min={min_in_scope:.4f}, max={max(in_scope_sims):.4f}")
        print(f"  Out-scope similarities : min={min(out_of_scope_sims):.4f}, max={max_out_scope:.4f}")
        print(f"\n  Recommended threshold  : {recommended_threshold:.4f}")
        print(f"  (midpoint of {min_in_scope:.4f} and {max_out_scope:.4f})")
        print("\n[ACTION] Update FALLBACK_THRESHOLD in retrieval.py to", recommended_threshold)

    return {
        "results": results,
        "in_scope_similarities": in_scope_sims,
        "out_of_scope_similarities": out_of_scope_sims,
        "min_in_scope": min_in_scope,
        "max_out_scope": max_out_scope,
        "recommended_threshold": recommended_threshold,
    }


if __name__ == "__main__":
    print("Running retrieval calibration (requires indexed collections)...")
    cal = run_calibration(verbose=True)

    # Save results
    import pathlib
    out_path = pathlib.Path("evaluation/calibration_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\nCalibration results saved to {out_path}")
