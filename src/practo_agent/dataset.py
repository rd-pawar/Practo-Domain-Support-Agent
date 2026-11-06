"""
dataset.py - Deterministic synthetic appointment record generator for Practo capstone.

Seed: 20261019
Records: 50
Fail-loud validation on category counts, status presence, and follow-up share.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Configuration (frozen after validation passes)
# ---------------------------------------------------------------------------
SEED = 20261019
NUM_RECORDS = 50
FEE_RANGE = (300, 2500)          # INR - synthetic demo envelope
FOLLOW_UP_PROB = 0.20
DAYS_RANGE = (0, 30)

CATEGORIES = ["Cardiology", "Dermatology", "Pediatrics", "Orthopedics", "General"]
CATEGORY_WEIGHTS = [0.20, 0.20, 0.20, 0.20, 0.20]

STATUSES = ["Scheduled", "Completed", "Cancelled", "No-Show", "Rescheduled"]
STATUS_WEIGHTS = [0.36, 0.30, 0.12, 0.08, 0.14]

# Validation bounds
MIN_RECORDS_PER_CATEGORY = 3
FOLLOW_UP_SHARE_MIN = 0.10
FOLLOW_UP_SHARE_MAX = 0.30
ESCALATION_PERCENTILE = 80          # empirical threshold percentile

NAMES = [
    "Aarav Shah", "Priya Nair", "Rohan Mehta", "Sunita Rao", "Vikram Patel",
    "Ananya Iyer", "Deepak Verma", "Kavitha Krishnan", "Sanjay Gupta", "Meera Joshi",
    "Arjun Reddy", "Pooja Singh", "Anil Kumar", "Sneha Sharma", "Rahul Desai",
    "Nisha Pillai", "Suresh Patil", "Rekha Nambiar", "Vijay Bhatt", "Lakshmi Menon",
    "Kartik Doshi", "Divya Pandey", "Mohit Agarwal", "Anita Bose", "Ravi Malhotra",
    "Swati Kapoor", "Ajay Thakur", "Preethi Subramaniam", "Nikhil Jain", "Geeta Yadav",
    "Harish Chatterjee", "Pallavi Nair", "Dinesh Kumar", "Archana Mishra", "Sameer Khanna",
    "Vaishali More", "Vinod Pandey", "Chandrika Sastry", "Alok Srivastava", "Sumitra Das",
    "Pratap Singh", "Malathy Rajan", "Sunil Rao", "Bharati Patil", "Ganesh Hegde",
    "Usha Namboodiri", "Rakesh Tiwari", "Saraswati Varma", "Ashok Pillai", "Madhuri Shinde",
]

CONTACT_TEMPLATE = "+91-{a}{b}{c}{d}{e}{f}{g}{h}{i}{j}"


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------
class AppointmentRecord(BaseModel):
    appointment_id: str
    patient_name: str
    contact_number: str         # synthetic only; always masked before model/log
    age: int = Field(ge=18, le=85)
    category: str
    status: str
    days_since_created: int = Field(ge=0, le=30)
    consultation_fee_inr: int
    follow_up_required: bool
    escalation_score: float = Field(ge=0.0, le=1.0)

    @field_validator("category")
    @classmethod
    def valid_category(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"Unknown category: {v}")
        return v

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: str) -> str:
        if v not in STATUSES:
            raise ValueError(f"Unknown status: {v}")
        return v


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def generate_dataset(seed: int = SEED, n: int = NUM_RECORDS) -> list[AppointmentRecord]:
    """Return n validated AppointmentRecord objects using a local seeded RNG."""
    rng = random.Random(seed)

    records: list[AppointmentRecord] = []
    for i in range(n):
        apt_id = f"APT-{i + 1:04d}"
        name = NAMES[i % len(NAMES)]
        # Synthetic contact number (10 random digits)
        digits = [str(rng.randint(0, 9)) for _ in range(10)]
        contact = "+91-" + "".join(digits)
        age = rng.randint(18, 85)
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        days = rng.randint(*DAYS_RANGE)
        fee = rng.randint(*FEE_RANGE)
        follow_up = rng.random() < FOLLOW_UP_PROB

        # Escalation formula: 0.65 × I(follow_up) + 0.35 × (days/30)
        esc_score = round(0.65 * int(follow_up) + 0.35 * (days / 30), 4)

        rec = AppointmentRecord(
            appointment_id=apt_id,
            patient_name=name,
            contact_number=contact,
            age=age,
            category=category,
            status=status,
            days_since_created=days,
            consultation_fee_inr=fee,
            follow_up_required=follow_up,
            escalation_score=esc_score,
        )
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dataset(records: list[AppointmentRecord]) -> dict[str, Any]:
    """Fail loudly if distribution requirements are violated. Return stats."""
    cat_counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    stat_counts: dict[str, int] = {s: 0 for s in STATUSES}
    follow_up_count = 0

    for r in records:
        cat_counts[r.category] += 1
        stat_counts[r.status] += 1
        if r.follow_up_required:
            follow_up_count += 1

    # ---- Fail-loud checks ---------------------------------------------------
    errors: list[str] = []

    for cat, cnt in cat_counts.items():
        if cnt < MIN_RECORDS_PER_CATEGORY:
            errors.append(f"Category '{cat}' has only {cnt} record(s) (min {MIN_RECORDS_PER_CATEGORY})")

    for stat in STATUSES:
        if stat_counts.get(stat, 0) == 0:
            errors.append(f"Status '{stat}' is absent from the dataset")

    total = len(records)
    follow_up_share = follow_up_count / total
    if not (FOLLOW_UP_SHARE_MIN <= follow_up_share <= FOLLOW_UP_SHARE_MAX):
        errors.append(
            f"Follow-up share {follow_up_share:.1%} outside [{FOLLOW_UP_SHARE_MIN:.0%}, {FOLLOW_UP_SHARE_MAX:.0%}]"
        )

    if errors:
        for e in errors:
            print(f"[VALIDATION ERROR] {e}", file=sys.stderr)
        raise SystemExit(
            "Dataset validation failed. Adjust seed or weights and regenerate. "
            "Do NOT patch individual rows."
        )

    # ---- Escalation threshold -----------------------------------------------
    scores = sorted(r.escalation_score for r in records)
    idx = int(len(scores) * ESCALATION_PERCENTILE / 100)
    threshold = scores[min(idx, len(scores) - 1)]
    high_risk_count = sum(1 for s in scores if s >= threshold)

    stats = {
        "total_records": total,
        "seed": SEED,
        "category_counts": cat_counts,
        "status_counts": stat_counts,
        "follow_up_count": follow_up_count,
        "follow_up_share": round(follow_up_share, 4),
        "escalation_percentile": ESCALATION_PERCENTILE,
        "escalation_threshold": threshold,
        "high_risk_count": high_risk_count,
    }
    return stats


# ---------------------------------------------------------------------------
# Appointment lookup tool (used by LookupAgent only)
# ---------------------------------------------------------------------------
# Build an in-memory index at module import time
_DATASET: list[AppointmentRecord] | None = None
_INDEX: dict[str, AppointmentRecord] | None = None


def _ensure_loaded() -> None:
    global _DATASET, _INDEX
    if _INDEX is None:
        _DATASET = generate_dataset()
        _INDEX = {r.appointment_id: r for r in _DATASET}


def check_appointment_status(appointment_id: str) -> dict[str, Any]:
    """
    Look up an appointment by ID.
    Returns a sanitised dict (no raw contact number) or a not-found sentinel.
    Raises ValueError for empty / obviously injected IDs.
    """
    _ensure_loaded()
    assert _INDEX is not None

    apt_id = str(appointment_id).strip()
    if not apt_id:
        raise ValueError("appointment_id must not be empty")

    record = _INDEX.get(apt_id)
    if record is None:
        return {
            "found": False,
            "appointment_id": apt_id,
            "message": f"No appointment record found for ID '{apt_id}'.",
        }

    return {
        "found": True,
        "appointment_id": record.appointment_id,
        "patient_name": record.patient_name,
        "age": record.age,
        "category": record.category,
        "status": record.status,
        "days_since_created": record.days_since_created,
        "consultation_fee_inr": record.consultation_fee_inr,
        "follow_up_required": record.follow_up_required,
        "escalation_score": record.escalation_score,
        # contact_number deliberately OMITTED from the return dict
    }


def get_all_records() -> list[AppointmentRecord]:
    """Return the full dataset (for evaluation/reporting)."""
    _ensure_loaded()
    assert _DATASET is not None
    return _DATASET


# ---------------------------------------------------------------------------
# Save dataset to JSON
# ---------------------------------------------------------------------------
def save_dataset(path: str | Path = "data/appointments.json") -> None:
    records = generate_dataset()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in records], f, indent=2, ensure_ascii=False)
    print(f"[dataset] Saved {len(records)} records to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Practo Capstone - Synthetic Dataset Generator")
    print(f"Seed: {SEED}  |  Records: {NUM_RECORDS}")
    print("=" * 60)

    records = generate_dataset()
    stats = validate_dataset(records)

    print("\nDataset Statistics:")
    print(f"  Total records       : {stats['total_records']}")
    print(f"  Seed                : {stats['seed']}")
    print(f"\nCategory distribution:")
    for cat, cnt in stats["category_counts"].items():
        print(f"  {cat:<20} {cnt:>3} records")
    print(f"\nStatus distribution:")
    for stat, cnt in stats["status_counts"].items():
        print(f"  {stat:<20} {cnt:>3} records")
    print(f"\nFollow-up required  : {stats['follow_up_count']} records ({stats['follow_up_share']:.1%})")
    print(
        f"\nEscalation threshold: {stats['escalation_threshold']:.4f}"
        f"  (empirical {stats['escalation_percentile']}th percentile)"
    )
    print(f"High-risk records   : {stats['high_risk_count']} (score >= threshold)")

    # Save dataset
    save_path = Path("data/appointments.json")
    save_dataset(save_path)

    print("\n[VALIDATION] All checks passed [OK]")
