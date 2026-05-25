"""
BM25 retriever: keyword search over scene chunks.

Build index:  python -c "from src.bm25_retriever import build_index; build_index()"
Query (CLI):  python -m src.bm25_retriever "your query here"
"""
import json
import pickle
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_PATH, CHUNKS_JSONL, TOP_K
from src.retriever import RetrievalResult

_tokenize = lambda text: re.findall(r"[a-z0-9]+", text.lower())

_cache: tuple[BM25Okapi, list[dict]] | None = None


def _load_chunks(path: Path) -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_index() -> tuple[BM25Okapi, list[dict]]:
    """Build BM25 index from chunks JSONL, save to disk, update cache. Always rebuilds."""
    global _cache

    print(f"Loading chunks from {CHUNKS_JSONL.name}...")
    chunks = _load_chunks(CHUNKS_JSONL)
    print(f"  Loaded {len(chunks):,} chunks")

    print("Tokenizing and building BM25 index...")
    tokenized = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    index = (bm25, chunks)

    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"Saved index to {BM25_INDEX_PATH}")

    _cache = index
    return index


def _get_index() -> tuple[BM25Okapi, list[dict]]:
    """Load index from cache or disk; auto-build if missing."""
    global _cache
    if _cache is not None:
        return _cache

    if not BM25_INDEX_PATH.exists():
        _cache = build_index()
    else:
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        _cache = (data["bm25"], data["chunks"])

    return _cache


def retrieve_bm25(query: str, top_k: int = TOP_K) -> list[RetrievalResult]:
    """Return top-k chunks by BM25 score. distance = -score (lower is better)."""
    bm25, chunks = _get_index()
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    results = []
    for i in top_indices:
        chunk = chunks[i]
        score = float(scores[i])
        results.append(RetrievalResult(
            scene_id=chunk["id"],
            text=chunk["text"],
            metadata=chunk["metadata"],
            distance=-score,
        ))
    return results


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Ross and Rachel break up"
    print(f"Query: {query!r}\n")
    for i, r in enumerate(retrieve_bm25(query), start=1):
        score = -r.distance
        print(f"{i}. {r.citation}  (score: {score:.3f})")
        print(f"   {r.text[:150]}...")
        print()
