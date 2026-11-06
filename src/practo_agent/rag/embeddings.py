"""
embeddings.py - Sentence-Transformer embedding utilities.

Uses all-MiniLM-L6-v2 (local, no API key required).
Cosine similarity is used for all retrieval comparisons.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np

# Cache model files locally (set env var before import if needed)
_CACHE_DIR = Path(os.getenv("SENTENCE_TRANSFORMERS_HOME", "sentence_transformers_cache"))
_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the SentenceTransformer model (loaded once per process)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        _MODEL_NAME,
        cache_folder=str(_CACHE_DIR),
    )
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings and return a list of float vectors.
    Uses all-MiniLM-L6-v2 (384-dimensional output).
    """
    model = get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.tolist()


def embed_single(text: str) -> list[float]:
    """Embed a single string."""
    return embed_texts([text])[0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    Vectors are assumed to be L2-normalised (as produced by embed_texts with
    normalize_embeddings=True), so dot product == cosine similarity.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    dot = float(np.dot(a, b))
    # Clamp to [-1, 1] for numerical safety
    return max(-1.0, min(1.0, dot))


def batch_cosine_similarities(
    query_vec: list[float], candidate_vecs: list[list[float]]
) -> list[float]:
    """Return similarity scores for a single query against many candidates."""
    q = np.array(query_vec, dtype=np.float32)
    C = np.array(candidate_vecs, dtype=np.float32)
    scores = (C @ q).tolist()
    return scores
