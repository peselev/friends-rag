"""
Bootstrap script for HF Spaces deployment.

Builds the Chroma collection the demo needs, if it doesn't exist.
Runs on Space startup before app.py serves.

The demo modes (hybrid_window_bm25, hybrid_window_bm25_reranked_bge) use
only ONE Chroma collection: friends_window_noheader. We build just that
one, not the other 5 collections used during evaluation.
"""
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_DIR, WINDOW_NOHEADER_COLLECTION
from src.indexer import index_chunks

WINDOW_NOHEADER_CHUNKS = Path("data/processed/chunks_window_noheader.jsonl")


def collection_exists_and_populated(client, name: str, expected_min: int = 1000) -> bool:
    """Return True if the named collection exists and has at least expected_min items."""
    try:
        coll = client.get_collection(name=name)
        return coll.count() >= expected_min
    except Exception:
        return False


def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    if collection_exists_and_populated(client, WINDOW_NOHEADER_COLLECTION):
        print(f"[bootstrap] {WINDOW_NOHEADER_COLLECTION} already populated, skipping.")
        return

    print(f"[bootstrap] Building {WINDOW_NOHEADER_COLLECTION} from {WINDOW_NOHEADER_CHUNKS}...")
    print("[bootstrap] This takes ~5 minutes on first Space boot.")
    index_chunks(
        chunks_path=WINDOW_NOHEADER_CHUNKS,
        collection_name=WINDOW_NOHEADER_COLLECTION,
    )
    print("[bootstrap] Done.")


if __name__ == "__main__":
    main()