"""
Indexer: read chunks.jsonl, embed them in batches, store in Chroma.

Run with:
    python -m src.indexer              # full corpus
    python -m src.indexer --limit 100  # smoke test with first 100 chunks
"""
import argparse
import json

import chromadb

from src.config import DATA_PROCESSED, CHROMA_DIR
from src.embedder import embed_texts

COLLECTION_NAME = "friends_scenes"
BATCH_SIZE = 100


def load_chunks(limit: int | None = None) -> list[dict]:
    """Read chunks.jsonl into memory."""
    chunks_path = DATA_PROCESSED / "chunks.jsonl"
    chunks = []
    with open(chunks_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            chunks.append(json.loads(line))
    return chunks


def get_or_reset_collection(client, name: str):
    """
    Delete the collection if it exists, then recreate.
    Re-running the indexer always produces a fresh index from scratch.
    Simpler and safer than partial updates while we're still iterating.
    """
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        print(f"  (Deleting existing '{name}' collection)")
        client.delete_collection(name)
    return client.create_collection(name=name)


def index_chunks(limit: int | None = None):
    """Main entry point."""
    print(f"Loading chunks (limit={limit})...")
    chunks = load_chunks(limit=limit)
    print(f"  Loaded {len(chunks):,} chunks")

    print(f"\nConnecting to Chroma at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = get_or_reset_collection(client, COLLECTION_NAME)

    print(f"\nEmbedding and indexing in batches of {BATCH_SIZE}...")
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
    print(f"\nDone. Collection '{COLLECTION_NAME}' now contains {final_count:,} items.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Index only the first N chunks (for testing)",
    )
    args = parser.parse_args()
    index_chunks(limit=args.limit)
