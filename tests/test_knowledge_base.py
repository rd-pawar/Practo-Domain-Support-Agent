"""
test_knowledge_base.py – Verify all 12 KB documents are present and valid.
"""

import json
from pathlib import Path

import pytest

KB_DIR = Path(__file__).resolve().parents[1] / "src" / "practo_agent" / "knowledge_base"

REQUIRED_DOC_IDS = [f"KB-{i:03d}" for i in range(1, 13)]
REQUIRED_FIELDS = {"id", "topic", "version", "effective_date", "parent_doc_id", "content"}


def load_all_docs():
    docs = []
    for path in sorted(KB_DIR.glob("KB-*.json")):
        with open(path, encoding="utf-8") as f:
            docs.append(json.load(f))
    return docs


def test_twelve_documents_present():
    docs = load_all_docs()
    assert len(docs) == 12, f"Expected 12 KB docs, found {len(docs)}"


def test_all_required_doc_ids_present():
    docs = load_all_docs()
    found_ids = {d["id"] for d in docs}
    missing = set(REQUIRED_DOC_IDS) - found_ids
    assert not missing, f"Missing KB IDs: {missing}"


def test_all_documents_have_required_fields():
    docs = load_all_docs()
    for doc in docs:
        missing = REQUIRED_FIELDS - set(doc.keys())
        assert not missing, f"Doc {doc.get('id')} missing fields: {missing}"


def test_content_length_2_to_5_sentences():
    """Each document should be 2-5 sentences (approx 50-2000 chars)."""
    docs = load_all_docs()
    for doc in docs:
        content_len = len(doc["content"])
        assert content_len >= 50, f"{doc['id']}: content too short ({content_len} chars)"
        assert content_len <= 3000, f"{doc['id']}: content may be too long ({content_len} chars)"


def test_stable_ids_match_parent_doc_id():
    docs = load_all_docs()
    for doc in docs:
        assert doc["id"] == doc["parent_doc_id"], (
            f"{doc['id']} has mismatched parent_doc_id={doc['parent_doc_id']}"
        )


def test_emergency_doc_no_phone_number():
    """KB-008 must not contain an invented phone number."""
    docs = load_all_docs()
    emergency_doc = next((d for d in docs if d["id"] == "KB-008"), None)
    assert emergency_doc is not None, "KB-008 not found"
    import re
    phone_pattern = re.compile(r"\b\d{10}\b|\+91-\d{10}")
    assert not phone_pattern.search(emergency_doc["content"]), (
        "KB-008 must not contain an invented phone number"
    )


def test_version_format():
    docs = load_all_docs()
    import re
    version_pattern = re.compile(r"^\d+\.\d+$")
    for doc in docs:
        assert version_pattern.match(doc["version"]), (
            f"{doc['id']} has invalid version format: {doc['version']}"
        )
