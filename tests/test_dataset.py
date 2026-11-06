"""
test_dataset.py – Acceptance tests for the dataset generator.
"""

import pytest
from practo_agent.dataset import (
    CATEGORIES,
    FOLLOW_UP_SHARE_MAX,
    FOLLOW_UP_SHARE_MIN,
    MIN_RECORDS_PER_CATEGORY,
    STATUSES,
    check_appointment_status,
    generate_dataset,
    validate_dataset,
)


def test_dataset_record_count():
    records = generate_dataset()
    assert len(records) == 50, f"Expected 50 records, got {len(records)}"


def test_all_categories_present():
    records = generate_dataset()
    found = {r.category for r in records}
    assert found == set(CATEGORIES)


def test_all_statuses_present():
    records = generate_dataset()
    found = {r.status for r in records}
    assert found == set(STATUSES)


def test_category_minimum_records():
    records = generate_dataset()
    from collections import Counter
    counts = Counter(r.category for r in records)
    for cat in CATEGORIES:
        assert counts[cat] >= MIN_RECORDS_PER_CATEGORY, f"Category '{cat}' has {counts[cat]} records"


def test_follow_up_share_in_band():
    records = generate_dataset()
    share = sum(1 for r in records if r.follow_up_required) / len(records)
    assert FOLLOW_UP_SHARE_MIN <= share <= FOLLOW_UP_SHARE_MAX, f"Follow-up share {share:.2%} out of band"


def test_escalation_score_range():
    records = generate_dataset()
    for r in records:
        assert 0.0 <= r.escalation_score <= 1.0, f"{r.appointment_id} score={r.escalation_score}"


def test_fee_range():
    records = generate_dataset()
    for r in records:
        assert 300 <= r.consultation_fee_inr <= 2500


def test_age_range():
    records = generate_dataset()
    for r in records:
        assert 18 <= r.age <= 85


def test_validate_dataset_passes():
    records = generate_dataset()
    stats = validate_dataset(records)
    assert stats["total_records"] == 50
    assert "escalation_threshold" in stats
    assert 0.0 <= stats["escalation_threshold"] <= 1.0


def test_appointment_lookup_known():
    result = check_appointment_status("APT-0001")
    assert result["found"] is True
    assert result["appointment_id"] == "APT-0001"
    assert "contact_number" not in result, "Contact number must NOT appear in lookup result"


def test_appointment_lookup_unknown():
    result = check_appointment_status("APT-9999")
    assert result["found"] is False
    assert "APT-9999" in result["message"]


def test_appointment_lookup_empty_id():
    with pytest.raises(ValueError):
        check_appointment_status("")


def test_deterministic_seed():
    r1 = generate_dataset()
    r2 = generate_dataset()
    assert [r.appointment_id for r in r1] == [r.appointment_id for r in r2]
    assert [r.status for r in r1] == [r.status for r in r2]
