"""
Unified retriever: dispatch to one of ten retrieval modes by name.
"""
import sys

from src.bm25_retriever import retrieve_bm25
from src.config import (
    NAIVE_COLLECTION, SCENE_COLLECTION, TOP_K,
    UTTERANCE_COLLECTION, WINDOW_COLLECTION,
    UTTERANCE_NOHEADER_COLLECTION, WINDOW_NOHEADER_COLLECTION,
)
from src.hybrid_retriever import retrieve_hybrid
from src.retriever import RetrievalResult, retrieve, retrieve_from_collection

AVAILABLE_MODES = (
    "vector", "vector_naive",
    "vector_utterance", "vector_window",
    "vector_utterance_noheader", "vector_window_noheader",
    "bm25", "hybrid",
    "parent_scene", "parent_scene_reranked",
)

_USAGE = f"""Usage: python -m src.unified_retriever <mode> "<query>"

Modes: {", ".join(AVAILABLE_MODES)}
"""

_MODE_TO_COLLECTION = {
    "vector": SCENE_COLLECTION,
    "vector_naive": NAIVE_COLLECTION,
    "vector_utterance": UTTERANCE_COLLECTION,
    "vector_window": WINDOW_COLLECTION,
    "vector_utterance_noheader": UTTERANCE_NOHEADER_COLLECTION,
    "vector_window_noheader": WINDOW_NOHEADER_COLLECTION,
}


def retrieve_unified(
    query: str,
    mode: str,
    top_k: int = TOP_K,
) -> list[RetrievalResult]:
    if mode not in AVAILABLE_MODES:
        raise ValueError(
            f"Unknown mode: {mode!r}. Must be one of: {AVAILABLE_MODES}"
        )

    if mode == "bm25":
        return retrieve_bm25(query, top_k)
    if mode == "hybrid":
        return retrieve_hybrid(query, top_k)

    if mode == "smart_hybrid":
        from src.smart_hybrid import retrieve_smart_hybrid
        return retrieve_smart_hybrid(query, top_k=top_k)

    # Parent-child modes: import here to avoid circular import at module load
    if mode == "parent_scene":
        from src.parent_child import retrieve_parent_scene
        return retrieve_parent_scene(query, top_k=top_k)
    if mode == "parent_scene_reranked":
        from src.parent_child import retrieve_parent_scene
        from src.reranker import rerank
        # Pull more candidates so reranker has options
        candidates = retrieve_parent_scene(query, top_k=20)
        return rerank(query, candidates, top_k=top_k)

    collection_name = _MODE_TO_COLLECTION[mode]
    if mode == "vector":
        return retrieve(query, top_k)
    return retrieve_from_collection(query, top_k, collection_name)


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