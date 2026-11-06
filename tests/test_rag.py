"""
test_rag.py – Acceptance tests for the RAG layer (chunkers, embeddings, retrieval).
"""

import pytest
from practo_agent.rag.chunkers import fixed_size_chunks, sentence_chunks, chunk_document


SAMPLE_TEXT = (
    "Patients may book appointments up to 30 days in advance through the clinic portal. "
    "Walk-in consultations are accepted subject to doctor availability. "
    "For specialist appointments, booking at least 3 to 5 business days ahead is advisable. "
    "Same-day bookings may be available for General practice slots."
)

SAMPLE_DOC = {
    "id": "KB-001",
    "content": SAMPLE_TEXT,
}


# ── Chunkers ─────────────────────────────────────────────────────────────────

def test_fixed_chunks_produced():
    chunks = fixed_size_chunks(SAMPLE_TEXT, doc_id="KB-001")
    assert len(chunks) > 0


def test_fixed_chunks_have_parent_doc_id():
    chunks = fixed_size_chunks(SAMPLE_TEXT, doc_id="KB-001")
    for c in chunks:
        assert c.parent_doc_id == "KB-001"


def test_fixed_chunks_strategy_label():
    chunks = fixed_size_chunks(SAMPLE_TEXT, doc_id="KB-001")
    for c in chunks:
        assert c.chunking_strategy == "fixed"


def test_sentence_chunks_produced():
    chunks = sentence_chunks(SAMPLE_TEXT, doc_id="KB-001")
    assert len(chunks) > 0


def test_sentence_chunks_have_parent_doc_id():
    chunks = sentence_chunks(SAMPLE_TEXT, doc_id="KB-001")
    for c in chunks:
        assert c.parent_doc_id == "KB-001"


def test_sentence_chunks_strategy_label():
    chunks = sentence_chunks(SAMPLE_TEXT, doc_id="KB-001")
    for c in chunks:
        assert c.chunking_strategy == "sentence"


def test_chunk_document_fixed():
    chunks = chunk_document(SAMPLE_DOC, strategy="fixed")
    assert all(c.chunking_strategy == "fixed" for c in chunks)


def test_chunk_document_sentence():
    chunks = chunk_document(SAMPLE_DOC, strategy="sentence")
    assert all(c.chunking_strategy == "sentence" for c in chunks)


def test_fixed_chunk_ids_unique():
    chunks = fixed_size_chunks(SAMPLE_TEXT, doc_id="KB-001")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_sentence_chunk_ids_unique():
    chunks = sentence_chunks(SAMPLE_TEXT, doc_id="KB-001")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_text_returns_no_chunks():
    assert fixed_size_chunks("", doc_id="KB-001") == []
    assert sentence_chunks("", doc_id="KB-001") == []


# ── Embeddings ────────────────────────────────────────────────────────────────

def test_embed_single_returns_vector():
    from practo_agent.rag.embeddings import embed_single
    vec = embed_single("book appointment")
    assert isinstance(vec, list)
    assert len(vec) == 384   # all-MiniLM-L6-v2 output dim


def test_cosine_similarity_self():
    from practo_agent.rag.embeddings import embed_single, cosine_similarity
    vec = embed_single("test sentence")
    sim = cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 0.01   # self-similarity ≈ 1


def test_cosine_similarity_different():
    from practo_agent.rag.embeddings import embed_single, cosine_similarity
    v1 = embed_single("book clinic appointment")
    v2 = embed_single("stock market crashed today")
    sim = cosine_similarity(v1, v2)
    assert sim < 0.9   # different topics should not be near-identical
