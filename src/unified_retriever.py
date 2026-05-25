"""
Unified retriever: dispatch to vector, vector_naive, BM25, or hybrid by mode.

CLI: python -m src.unified_retriever <mode> "<query>"
"""
import sys

from src.bm25_retriever import retrieve_bm25
from src.config import NAIVE_COLLECTION, SCENE_COLLECTION, TOP_K
from src.hybrid_retriever import retrieve_hybrid
from src.retriever import RetrievalResult, retrieve, retrieve_from_collection

AVAILABLE_MODES = ("vector", "vector_naive", "bm25", "hybrid")

_USAGE = f"""Usage: python -m src.unified_retriever <mode> "<query>"

Modes: {", ".join(AVAILABLE_MODES)}
"""


def retrieve_unified(
    query: str,
    mode: str,
    top_k: int = TOP_K,
) -> list[RetrievalResult]:
    """Retrieve using the given mode. Raises ValueError for unknown mode."""
    if mode not in AVAILABLE_MODES:
        raise ValueError(
            f"Unknown mode: {mode!r}. Must be one of: {AVAILABLE_MODES}"
        )

    if mode == "vector":
        return retrieve(query, top_k)
    if mode == "vector_naive":
        return retrieve_from_collection(query, top_k, NAIVE_COLLECTION)
    if mode == "bm25":
        return retrieve_bm25(query, top_k)
    return retrieve_hybrid(query, top_k)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(_USAGE, file=sys.stderr)
        sys.exit(1)

    mode, query = sys.argv[1], sys.argv[2]

    try:
        results = retrieve_unified(query, mode)
    except ValueError as e:
        print(e, file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        sys.exit(1)

    print(f"Mode: {mode}  Query: {query!r}\n")
    for i, r in enumerate(results, start=1):
        print(f"{i}. {r.citation}  (distance: {r.distance:.3f})")
        print(f"   {r.text[:150]}...")
        print()
