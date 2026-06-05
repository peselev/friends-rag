"""
Unified retriever: dispatch to one of several retrieval modes by name.
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

# ⚠️ The two header modes ("vector_utterance", "vector_window") are DISABLED.
# They embed a "Friends S01E01, Scene 1" prefix and measured worse than the
# *_noheader variants for short chunks. They are commented out below so they
# can never run by accident (retrieve_unified raises on any mode not listed).
# Use "vector_utterance_noheader" / "vector_window_noheader" instead.
AVAILABLE_MODES = (
    "vector", "vector_naive",
    # "vector_utterance",   # DISABLED — header variant, underperforms. Use _noheader.
    # "vector_window",      # DISABLED — header variant, underperforms. Use _noheader.
    "vector_utterance_noheader", "vector_window_noheader",
    "bm25", "hybrid",
    "hybrid_window_bm25",
    "hybrid_window_bm25_smart",
    "hybrid_window_bm25_reranked",
    "hybrid_window_bm25_reranked_bge",
    "hybrid_window_bm25_reranked_cohere",
)

_USAGE = f"""Usage: python -m src.unified_retriever <mode> "<query>"

Modes: {", ".join(AVAILABLE_MODES)}
"""

_MODE_TO_COLLECTION = {
    "vector": SCENE_COLLECTION,
    "vector_naive": NAIVE_COLLECTION,
    # "vector_utterance": UTTERANCE_COLLECTION,   # DISABLED — header variant
    # "vector_window": WINDOW_COLLECTION,         # DISABLED — header variant
    "vector_utterance_noheader": UTTERANCE_NOHEADER_COLLECTION,
    "vector_window_noheader": WINDOW_NOHEADER_COLLECTION,
}

BGE_RERANKER = "BAAI/bge-reranker-base"

def retrieve_unified(
    query: str,
    mode: str,
    top_k: int = TOP_K,
) -> list[RetrievalResult]:
    if mode not in AVAILABLE_MODES:
        raise ValueError(
            f"Unknown mode: {mode!r}. Must be one of: {AVAILABLE_MODES}"
        )

    # Keyword and simple-hybrid modes
    if mode == "bm25":
        return retrieve_bm25(query, top_k)
    if mode == "hybrid":
        return retrieve_hybrid(query, top_k)

    # Window + BM25 hybrids (lazy imports avoid loading these unless used)
    if mode == "hybrid_window_bm25":
        from src.hybrid_window import retrieve_hybrid_window
        return retrieve_hybrid_window(query, top_k=top_k)
    if mode == "hybrid_window_bm25_smart":
        from src.smart_hybrid import retrieve_hybrid_window_smart
        return retrieve_hybrid_window_smart(query, top_k=top_k)

    if mode == "hybrid_window_bm25_reranked":
        from src.hybrid_window import retrieve_hybrid_window
        from src.reranker import rerank
        candidates = retrieve_hybrid_window(query, top_k=20)
        return rerank(query, candidates, top_k=top_k)

    if mode == "hybrid_window_bm25_reranked_bge":
        from src.hybrid_window import retrieve_hybrid_window
        from src.reranker import rerank
        candidates = retrieve_hybrid_window(query, top_k=20)
        return rerank(query, candidates, top_k=top_k, model_name=BGE_RERANKER)

    if mode == "hybrid_window_bm25_reranked_cohere":
        # The shipped "higher accuracy" path. Fuse the equal-weight top-N, hand
        # them to Cohere rerank-v3.5, keep the best. cohere_rerank raises
        # CohereUnavailable on any failure; the app catches it and falls back to
        # the fast hybrid (we deliberately do NOT swallow it here).
        from src.config import RERANK_CANDIDATE_DEPTH
        from src.cohere_reranker import cohere_rerank
        from src.hybrid_window import retrieve_hybrid_window
        candidates = retrieve_hybrid_window(query, top_k=RERANK_CANDIDATE_DEPTH)
        return cohere_rerank(query, candidates, top_k=top_k)

    # Pure vector modes (dispatch by collection)
    collection_name = _MODE_TO_COLLECTION[mode]
    if mode == "vector":
        return retrieve(query, top_k)  # original convenience entry point
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