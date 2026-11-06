"""
chroma_index.py - ChromaDB indexing for Practo Knowledge Base.

Two named collections:
  - practo_fixed    (chunks from fixed-size strategy)
  - practo_sentence (chunks from sentence-boundary strategy)

Embeddings computed by SentenceTransformer all-MiniLM-L6-v2.
All chunks carry parent_doc_id and chunking_strategy in metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings

from practo_agent.rag.chunkers import Chunk, chunk_document
from practo_agent.rag.embeddings import embed_texts

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KB_DIR = Path(__file__).resolve().parents[2] / "practo_agent" / "knowledge_base"
CHROMA_PATH = Path(__file__).resolve().parents[4] / "chroma_db"

COLLECTION_FIXED = "practo_fixed"
COLLECTION_SENTENCE = "practo_sentence"

KB_VERSION = "1.0"


# ---------------------------------------------------------------------------
# ChromaDB client (persistent, single instance per process)
# ---------------------------------------------------------------------------
def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(name: str) -> chromadb.Collection:
    """Get or create a named Chroma collection with cosine distance."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Load all 12 KB documents
# ---------------------------------------------------------------------------
def load_kb_documents(kb_dir: Path = KB_DIR) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(kb_dir.glob("KB-*.json")):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        docs.append(doc)
    if not docs:
        raise FileNotFoundError(f"No KB documents found in {kb_dir}")
    return docs


# ---------------------------------------------------------------------------
# Index documents into a collection
# ---------------------------------------------------------------------------
def index_documents(
    documents: list[dict],
    collection_name: str,
    strategy: str,
    verbose: bool = True,
) -> chromadb.Collection:
    """
    Chunk all documents using *strategy*, embed them, and upsert into the
    named Chroma collection.  Safe to re-run (idempotent via upsert).
    """
    collection = get_collection(collection_name)

    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunk_document(doc, strategy=strategy)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError(f"No chunks produced for strategy '{strategy}'")

    texts = [c.text for c in all_chunks]
    embeddings = embed_texts(texts)

    ids = [c.chunk_id for c in all_chunks]
    metadatas = [
        {
            "parent_doc_id": c.parent_doc_id,
            "chunking_strategy": c.chunking_strategy,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "kb_version": KB_VERSION,
        }
        for c in all_chunks
    ]

    # Upsert in batches of 100 to be safe
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    if verbose:
        print(f"[chroma_index] '{collection_name}': upserted {len(all_chunks)} chunks from {len(documents)} docs")

    return collection


# ---------------------------------------------------------------------------
# Build both collections
# ---------------------------------------------------------------------------
def build_all_indexes(verbose: bool = True) -> None:
    documents = load_kb_documents()
    index_documents(documents, COLLECTION_FIXED, strategy="fixed", verbose=verbose)
    index_documents(documents, COLLECTION_SENTENCE, strategy="sentence", verbose=verbose)
    if verbose:
        print("[chroma_index] Both collections indexed successfully.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Practo Capstone - ChromaDB Indexer")
    print("=" * 60)
    build_all_indexes(verbose=True)
