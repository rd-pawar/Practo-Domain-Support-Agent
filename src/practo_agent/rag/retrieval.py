"""
retrieval.py - Top-k retrieval from ChromaDB with fallback threshold.

Returns chunk IDs, parent doc IDs, and similarity scores.
Falls back to refusal response if best score is below threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from practo_agent.rag.chroma_index import (
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    get_collection,
)
from practo_agent.rag.embeddings import embed_single

# Empirically calibrated fallback threshold (set after calibration run)
# This value is determined from calibration.py and frozen here.
FALLBACK_THRESHOLD = 0.3978   # Empirical midpoint: in-scope min=0.6084, out-scope max=0.1871

TOP_K_DEFAULT = 3


@dataclass
class RetrievalResult:
    """Container for a single retrieved chunk."""
    chunk_id: str
    parent_doc_id: str
    text: str
    similarity: float
    chunking_strategy: str


@dataclass
class RetrievalResponse:
    """Full retrieval response, including fallback flag."""
    results: list[RetrievalResult]
    query: str
    collection_name: str
    fallback_triggered: bool
    fallback_reason: Optional[str] = None


def retrieve(
    query: str,
    strategy: Literal["fixed", "sentence"] = "fixed",
    top_k: int = TOP_K_DEFAULT,
    threshold: float = FALLBACK_THRESHOLD,
) -> RetrievalResponse:
    """
    Embed *query* and retrieve top-*k* chunks from the selected collection.

    If the best cosine similarity is below *threshold*, sets
    fallback_triggered=True and returns an empty results list (triggers
    a "I do not know" or safe escalation response upstream).

    Parameters
    ----------
    query    : Natural-language question from the user.
    strategy : "fixed" or "sentence" (selects Chroma collection).
    top_k    : Number of chunks to retrieve.
    threshold: Minimum acceptable cosine similarity; below this -> fallback.
    """
    collection_name = COLLECTION_FIXED if strategy == "fixed" else COLLECTION_SENTENCE
    collection = get_collection(collection_name)

    query_vec = embed_single(query)

    # Chroma uses cosine distance: similarity = 1 - distance
    chroma_results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    ids = chroma_results["ids"][0]
    documents = chroma_results["documents"][0]
    metadatas = chroma_results["metadatas"][0]
    distances = chroma_results["distances"][0]

    # Convert cosine distance -> similarity
    results: list[RetrievalResult] = []
    for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
        sim = 1.0 - dist
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                parent_doc_id=meta.get("parent_doc_id", "unknown"),
                text=text,
                similarity=round(sim, 4),
                chunking_strategy=meta.get("chunking_strategy", strategy),
            )
        )

    # Check threshold
    if not results or results[0].similarity < threshold:
        return RetrievalResponse(
            results=[],
            query=query,
            collection_name=collection_name,
            fallback_triggered=True,
            fallback_reason=(
                f"Best similarity {results[0].similarity:.4f} is below "
                f"threshold {threshold:.4f}" if results else "No chunks retrieved"
            ),
        )

    # Filter out below-threshold results (keep only above-threshold top-k)
    results = [r for r in results if r.similarity >= threshold]

    return RetrievalResponse(
        results=results,
        query=query,
        collection_name=collection_name,
        fallback_triggered=False,
    )


def retrieve_with_ids(
    query: str,
    strategy: Literal["fixed", "sentence"] = "fixed",
    top_k: int = TOP_K_DEFAULT,
    threshold: float = FALLBACK_THRESHOLD,
) -> tuple[list[str], list[str], list[float], bool]:
    """
    Convenience wrapper returning (chunk_ids, parent_doc_ids, similarities, fallback_triggered).
    Used by CrewAI tools and evaluation harness.
    """
    resp = retrieve(query, strategy=strategy, top_k=top_k, threshold=threshold)
    chunk_ids = [r.chunk_id for r in resp.results]
    parent_ids = [r.parent_doc_id for r in resp.results]
    sims = [r.similarity for r in resp.results]
    return chunk_ids, parent_ids, sims, resp.fallback_triggered
