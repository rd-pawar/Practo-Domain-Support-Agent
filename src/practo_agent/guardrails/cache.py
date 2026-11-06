"""
cache.py - In-memory result cache for Practo Domain Support Agent.

Caches ONLY grounded generation results (policy answers).
Does NOT cache appointment lookup results (mutable data).

Cache key = normalize(query) + kb_version + retrieval_strategy
Counter tracks call savings (pre-cache vs post-cache).
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CacheEntry:
    key: str
    response: dict
    hit_count: int = 0


class ResultCache:
    """
    Thread-safe in-memory cache for grounded generation results.

    Key construction:
      SHA-256(normalize(query) + "|" + kb_version + "|" + strategy)

    Rules:
      - Only cache policy answers (intent != "appointment_status")
      - Namespace by KB version + strategy so edits invalidate stale answers
      - Count total calls and cache hits for evidence reporting
    """

    def __init__(self, kb_version: str = "1.0"):
        self._store: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._total_calls: int = 0
        self._cache_hits: int = 0
        self.kb_version = kb_version

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #
    def get(
        self,
        query: str,
        strategy: str = "fixed",
    ) -> Optional[dict]:
        """Return cached response or None on miss."""
        key = self._make_key(query, strategy)
        with self._lock:
            self._total_calls += 1
            entry = self._store.get(key)
            if entry is not None:
                entry.hit_count += 1
                self._cache_hits += 1
                return dict(entry.response)
        return None

    def put(
        self,
        query: str,
        response: dict,
        strategy: str = "fixed",
    ) -> None:
        """Store a grounded generation result. Skip appointment results."""
        intent = response.get("intent", "policy")
        if intent == "appointment_status":
            return   # Never cache mutable appointment data

        key = self._make_key(query, strategy)
        with self._lock:
            self._store[key] = CacheEntry(key=key, response=dict(response))

    def stats(self) -> dict:
        """Return call and hit counters for evidence reporting."""
        with self._lock:
            return {
                "total_calls": self._total_calls,
                "cache_hits": self._cache_hits,
                "cache_misses": self._total_calls - self._cache_hits,
                "hit_rate": (
                    round(self._cache_hits / self._total_calls, 4)
                    if self._total_calls > 0 else 0.0
                ),
                "entries": len(self._store),
                "kb_version": self.kb_version,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._total_calls = 0
            self._cache_hits = 0

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                          #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _normalise(text: str) -> str:
        """Lowercase, collapse whitespace."""
        return " ".join(text.lower().split())

    def _make_key(self, query: str, strategy: str) -> str:
        raw = f"{self._normalise(query)}|{self.kb_version}|{strategy}"
        return hashlib.sha256(raw.encode()).hexdigest()


# Process-level singleton
_default_cache = ResultCache(kb_version="1.0")


def get_cache() -> ResultCache:
    """Return the process-level cache singleton."""
    return _default_cache


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Cache - Demo")
    print("-" * 55)
    cache = ResultCache(kb_version="1.0")

    # Simulate two queries - second is a cache hit
    policy_response = {
        "answer": "You can book up to 30 days in advance.",
        "intent": "policy",
        "source_ids": ["KB-001"],
        "confidence": 0.85,
        "needs_escalation": False,
        "refusal_reason": None,
    }
    query = "How far ahead should I book an appointment?"

    print(f"\nCall 1: get('{query[:40]}...')")
    result1 = cache.get(query)
    print(f"  Result: {'HIT' if result1 else 'MISS'}")

    cache.put(query, policy_response)

    print(f"\nCall 2: get('{query[:40]}...') [should be HIT]")
    result2 = cache.get(query)
    print(f"  Result: {'HIT' if result2 else 'MISS'}")
    if result2:
        print(f"  Cached answer: {result2['answer']}")

    # Appointment results are NOT cached
    apt_response = {
        "answer": "Status: Scheduled",
        "intent": "appointment_status",
        "source_ids": ["APT-0001"],
        "confidence": 0.95,
        "needs_escalation": False,
        "refusal_reason": None,
    }
    cache.put("Status of APT-0001", apt_response)   # should be skipped

    print(f"\nStats after demo:")
    stats = cache.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
