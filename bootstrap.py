"""
Bootstrap script for HF Spaces deployment.

Builds the Chroma collection the demo needs, if it doesn't exist.
Runs on Space startup before app.py serves.

The demo modes (hybrid_window_bm25, hybrid_window_bm25_reranked_bge) use
only ONE Chroma collection: friends_window_noheader. We build just that
one, not the other 5 collections used during evaluation.
"""
from pathlib import Path
import shutil

import chromadb

from src.config import CHROMA_DIR, CHROMA_SETTINGS, WINDOW_NOHEADER_COLLECTION
from src.indexer import index_chunks

WINDOW_NOHEADER_CHUNKS = Path("data/processed/chunks_window_noheader.jsonl")


def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=CHROMA_SETTINGS)


def collection_exists_and_populated(client, name: str, expected_min: int = 1000) -> bool:
    """Return True if the named collection exists and has at least expected_min items."""
    try:
        coll = client.get_collection(name=name)
        return coll.count() >= expected_min
    except Exception:
        return False


def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        client = _client()
        populated = collection_exists_and_populated(client, WINDOW_NOHEADER_COLLECTION)
    except Exception as e:
        # The persistent dir exists but can't be opened — e.g. a leftover or
        # version-incompatible store ("no such table: tenants"). Wipe it and
        # start clean; the rebuild below repopulates from the jsonl.
        print(f"[bootstrap] Could not open Chroma at {CHROMA_DIR} "
              f"({type(e).__name__}: {e}). Resetting and rebuilding.")
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = _client()
        populated = False

    if populated:
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
