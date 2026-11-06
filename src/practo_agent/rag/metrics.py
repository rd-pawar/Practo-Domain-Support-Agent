"""
metrics.py - Retrieval precision@k and recall@k for both chunking strategies.

Metrics are computed at the parent-document level (unique parent IDs),
not the chunk level. This avoids inflating scores when a document
produces multiple retrieved chunks.

For each query:
  Precision@k = |relevant_retrieved_parent_docs| / |unique_retrieved_parent_docs|
  Recall@k    = |relevant_retrieved_parent_docs| / |all_gold_relevant_parent_docs|
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from practo_agent.rag.retrieval import retrieve

# ---------------------------------------------------------------------------
# Gold standard: 5 in-scope queries with gold parent document IDs
# ---------------------------------------------------------------------------

@dataclass
class MetricQuery:
    query_id: str
    query: str
    gold_parent_docs: list[str]   # Ground-truth relevant parent doc IDs


METRIC_QUERIES: list[MetricQuery] = [
    MetricQuery(
        query_id="MQ-001",
        query="How far ahead should I book an appointment?",
        gold_parent_docs=["KB-001"],
    ),
    MetricQuery(
        query_id="MQ-002",
        query="What notice period is required to cancel an appointment without a fee?",
        gold_parent_docs=["KB-002"],
    ),
    MetricQuery(
        query_id="MQ-003",
        query="What is the consultation fee for Cardiology specialist visits?",
        gold_parent_docs=["KB-003"],
    ),
    MetricQuery(
        query_id="MQ-004",
        query="How do I submit an insurance claim after a visit?",
        gold_parent_docs=["KB-004"],
    ),
    MetricQuery(
        query_id="MQ-005",
        query="Can I book a telemedicine video consultation from home?",
        gold_parent_docs=["KB-007"],
    ),
]

TOP_K = 3


@dataclass
class QueryMetrics:
    query_id: str
    query: str
    strategy: str
    retrieved_parent_docs: list[str]          # deduplicated
    gold_parent_docs: list[str]
    relevant_retrieved: list[str]             # intersection
    precision_at_k: float
    recall_at_k: float
    fallback_triggered: bool


def compute_metrics_for_query(
    mq: MetricQuery,
    strategy: Literal["fixed", "sentence"],
    top_k: int = TOP_K,
) -> QueryMetrics:
    """Compute precision@k and recall@k for a single query."""
    resp = retrieve(mq.query, strategy=strategy, top_k=top_k)

    # Deduplicate retrieved parent doc IDs
    seen: set[str] = set()
    retrieved_parents: list[str] = []
    for r in resp.results:
        if r.parent_doc_id not in seen:
            seen.add(r.parent_doc_id)
            retrieved_parents.append(r.parent_doc_id)

    gold_set = set(mq.gold_parent_docs)
    retrieved_set = set(retrieved_parents)
    relevant_retrieved = list(gold_set & retrieved_set)

    precision = len(relevant_retrieved) / len(retrieved_parents) if retrieved_parents else 0.0
    recall = len(relevant_retrieved) / len(gold_set) if gold_set else 0.0

    return QueryMetrics(
        query_id=mq.query_id,
        query=mq.query,
        strategy=strategy,
        retrieved_parent_docs=retrieved_parents,
        gold_parent_docs=mq.gold_parent_docs,
        relevant_retrieved=relevant_retrieved,
        precision_at_k=round(precision, 4),
        recall_at_k=round(recall, 4),
        fallback_triggered=resp.fallback_triggered,
    )


def run_retrieval_comparison(top_k: int = TOP_K, verbose: bool = True) -> dict:
    """
    Evaluate both strategies on all 5 metric queries.
    Returns a dict with per-query results and aggregate averages.
    """
    fixed_results: list[QueryMetrics] = []
    sentence_results: list[QueryMetrics] = []

    for mq in METRIC_QUERIES:
        fixed_results.append(compute_metrics_for_query(mq, "fixed", top_k))
        sentence_results.append(compute_metrics_for_query(mq, "sentence", top_k))

    if verbose:
        _print_comparison_table(fixed_results, sentence_results, top_k)

    avg_p_fixed = sum(r.precision_at_k for r in fixed_results) / len(fixed_results)
    avg_r_fixed = sum(r.recall_at_k for r in fixed_results) / len(fixed_results)
    avg_p_sent = sum(r.precision_at_k for r in sentence_results) / len(sentence_results)
    avg_r_sent = sum(r.recall_at_k for r in sentence_results) / len(sentence_results)

    recommendation = _make_recommendation(avg_p_fixed, avg_r_fixed, avg_p_sent, avg_r_sent)

    if verbose:
        print(f"\nAverages (fixed   ): P@{top_k}={avg_p_fixed:.4f}  R@{top_k}={avg_r_fixed:.4f}")
        print(f"Averages (sentence): P@{top_k}={avg_p_sent:.4f}  R@{top_k}={avg_r_sent:.4f}")
        print(f"\nRecommendation: {recommendation}")

    return {
        "top_k": top_k,
        "fixed": [_qm_to_dict(r) for r in fixed_results],
        "sentence": [_qm_to_dict(r) for r in sentence_results],
        "averages": {
            "fixed": {"precision": round(avg_p_fixed, 4), "recall": round(avg_r_fixed, 4)},
            "sentence": {"precision": round(avg_p_sent, 4), "recall": round(avg_r_sent, 4)},
        },
        "recommendation": recommendation,
    }


def _make_recommendation(p_fix: float, r_fix: float, p_sent: float, r_sent: float) -> str:
    f1_fix = 2 * p_fix * r_fix / (p_fix + r_fix) if (p_fix + r_fix) > 0 else 0.0
    f1_sent = 2 * p_sent * r_sent / (p_sent + r_sent) if (p_sent + r_sent) > 0 else 0.0
    if f1_fix >= f1_sent:
        return (
            f"Fixed-size chunking (F1={f1_fix:.4f}) performs at least as well as "
            f"sentence chunking (F1={f1_sent:.4f}). "
            "Fixed strategy recommended for production due to predictable chunk boundaries."
        )
    return (
        f"Sentence-boundary chunking (F1={f1_sent:.4f}) outperforms "
        f"fixed-size chunking (F1={f1_fix:.4f}). "
        "Sentence strategy recommended for higher recall of complete policy clauses."
    )


def _print_comparison_table(
    fixed: list[QueryMetrics],
    sentence: list[QueryMetrics],
    top_k: int,
) -> None:
    print("=" * 100)
    print(f"Retrieval Comparison - Precision@{top_k} and Recall@{top_k} (parent-document level)")
    print("=" * 100)
    header = (
        f"{'QID':<8} {'Query':<52} "
        f"{'P@k(fixed)':>10} {'R@k(fixed)':>10} "
        f"{'P@k(sent)':>10}  {'R@k(sent)':>10}"
    )
    print(header)
    print("-" * 100)
    for f, s in zip(fixed, sentence):
        # Show numerator/denominator for fixed precision
        p_num = len(f.relevant_retrieved)
        p_den = len(f.retrieved_parent_docs)
        r_num = len(f.relevant_retrieved)
        r_den = len(f.gold_parent_docs)
        s_p_num = len(s.relevant_retrieved)
        s_p_den = len(s.retrieved_parent_docs)
        s_r_num = len(s.relevant_retrieved)
        s_r_den = len(s.gold_parent_docs)

        print(
            f"{f.query_id:<8} {f.query[:50]:<52} "
            f"{p_num}/{p_den}={f.precision_at_k:.2f}".rjust(10) + " "
            f"{r_num}/{r_den}={f.recall_at_k:.2f}".rjust(10) + " "
            f"{s_p_num}/{s_p_den}={s.precision_at_k:.2f}".rjust(10) + "  "
            f"{s_r_num}/{s_r_den}={s.recall_at_k:.2f}".rjust(10)
        )


def _qm_to_dict(qm: QueryMetrics) -> dict:
    return {
        "query_id": qm.query_id,
        "query": qm.query,
        "strategy": qm.strategy,
        "retrieved_parent_docs": qm.retrieved_parent_docs,
        "gold_parent_docs": qm.gold_parent_docs,
        "relevant_retrieved": qm.relevant_retrieved,
        "precision_at_k": qm.precision_at_k,
        "recall_at_k": qm.recall_at_k,
        "fallback_triggered": qm.fallback_triggered,
    }


if __name__ == "__main__":
    print("Running retrieval comparison metrics...")
    comparison = run_retrieval_comparison(top_k=TOP_K, verbose=True)

    out_path = Path("evaluation/retrieval_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nMetrics saved to {out_path}")
