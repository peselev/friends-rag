"""
Indexer: read chunks JSONL, embed in batches, store in Chroma.

Examples:
    # Scene chunks (default - same as Weekend 1)
    python -m src.indexer

    # Naive chunks into a separate collection
    python -m src.indexer --chunks data/processed/chunks_naive.jsonl --collection friends_naive

    # Smoke test with first 100
    python -m src.indexer --limit 100
"""
import argparse
import json
from pathlib import Path

import chromadb

from src.config import DATA_PROCESSED, CHROMA_DIR
from src.embedder import embed_texts

DEFAULT_COLLECTION = "friends_scenes"
DEFAULT_CHUNKS_FILE = DATA_PROCESSED / "chunks.jsonl"
BATCH_SIZE = 100


def load_chunks(path: Path, limit: int | None = None) -> list[dict]:
    """Read a chunks JSONL file into memory."""
    chunks = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            chunks.append(json.loads(line))
    return chunks


def get_or_reset_collection(client, name: str):
    """Delete the collection if it exists, then recreate. Fresh index every run."""
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        print(f"  (Deleting existing '{name}' collection)")
        client.delete_collection(name)
    return client.create_collection(name=name)


def index_chunks(chunks_path: Path, collection_name: str, limit: int | None = None):
    """Main entry point."""
    print(f"Loading chunks from {chunks_path.name} (limit={limit})...")
    chunks = load_chunks(chunks_path, limit=limit)
    print(f"  Loaded {len(chunks):,} chunks")

    print(f"\nConnecting to Chroma at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = get_or_reset_collection(client, collection_name)

    print(f"\nEmbedding and indexing into '{collection_name}' in batches of {BATCH_SIZE}...")
    n_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, start in enumerate(range(0, len(chunks), BATCH_SIZE), start=1):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        vectors = embed_texts(texts)

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  Batch {batch_num}/{n_batches}: indexed {len(batch)} chunks")

    final_count = collection.count()
    print(f"\nDone. Collection '{collection_name}' now contains {final_count:,} items.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_FILE,
        help=f"Path to chunks JSONL file (default: {DEFAULT_CHUNKS_FILE})",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Index only the first N chunks (for testing)",
    )
    args = parser.parse_args()
    index_chunks(chunks_path=args.chunks, collection_name=args.collection, limit=args.limit)
