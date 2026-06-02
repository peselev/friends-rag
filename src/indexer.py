"""
Indexer: read chunks JSONL, embed in batches, store in Chroma.

Examples:
    # Scene chunks (default, small model)
    python -m src.indexer

    # Naive chunks into a separate collection
    python -m src.indexer --chunks data/processed/chunks_naive.jsonl --collection friends_naive

    # Build a LARGE-model collection alongside the small one (different name!)
    python -m src.indexer \
        --chunks data/processed/chunks.jsonl \
        --collection friends_scenes_large \
        --embedding-model text-embedding-3-large

    # Smoke test with first 100
    python -m src.indexer --limit 100
"""
import argparse
import json
from pathlib import Path

import chromadb

from src.config import DATA_PROCESSED, CHROMA_DIR, EMBEDDING_MODEL
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


def index_chunks(
    chunks_path: Path,
    collection_name: str,
    limit: int | None = None,
    embedding_model: str = EMBEDDING_MODEL,
):
    """Main entry point."""
    print(f"Loading chunks from {chunks_path.name} (limit={limit})...")
    chunks = load_chunks(chunks_path, limit=limit)
    print(f"  Loaded {len(chunks):,} chunks")

    print(f"\nConnecting to Chroma at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = get_or_reset_collection(client, collection_name)

    print(f"\nEmbedding with '{embedding_model}' and indexing into "
          f"'{collection_name}' in batches of {BATCH_SIZE}...")
    n_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, start in enumerate(range(0, len(chunks), BATCH_SIZE), start=1):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        vectors = embed_texts(texts, model=embedding_model)

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
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"OpenAI embedding model (default: {EMBEDDING_MODEL}). "
             f"Use a DIFFERENT --collection name when changing this, since "
             f"vector dimensionality differs between models.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Index only the first N chunks (for testing)",
    )
    args = parser.parse_args()
    index_chunks(
        chunks_path=args.chunks,
        collection_name=args.collection,
        limit=args.limit,
        embedding_model=args.embedding_model,
    )
