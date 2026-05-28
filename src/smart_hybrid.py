"""
Smart hybrid retrieval: IDF-weighted RRF fusion of BM25 and
vector_window_noheader, returning full parent scenes.

Identical to hybrid_window.retrieve_hybrid_window EXCEPT the fusion weight
adapts to the query:
  - High avg IDF of query terms (rare, dialogue-like words) -> trust BM25 more
  - Low avg IDF (common or out-of-vocabulary words)         -> trust vector more

This directly targets the failure we measured: BM25 dominates on rare-term
queries but collapses on abstract paraphrases. A fixed equal weight can't
serve both; an adaptive weight can.
"""
import numpy as np

from src.bm25_retriever import _get_index, _tokenize, retrieve_bm25
from src.config import TOP_K, WINDOW_NOHEADER_COLLECTION
from src.retriever import RetrievalResult, retrieve_from_collection

# Reuse the shared machinery from the naive hybrid so the two modes differ
# ONLY in their weighting logic.
from src.hybrid_window import (
    RRF_K,
    OVERSAMPLE_MULTIPLIER,
    _dedupe_windows_to_scenes,
    _make_scene_result,
    _SCENES,
)

# Weight clamping: even at extremes, neither retriever is fully disabled,
# because consensus across both is itself a useful signal.
WEIGHT_BOUNDS = (0.2, 0.8)  # (min BM25 weight, max BM25 weight)


def query_avg_idf(query: str) -> float:
    """
    Average IDF of query terms that exist in the BM25 corpus.
    Out-of-corpus terms are excluded (they contribute nothing to BM25 anyway).
    Higher = query has rarer terms = BM25 has more signal.
    """
    bm25, _chunks = _get_index()
    tokens = _tokenize(query)
    idfs = [bm25.idf[t] for t in tokens if t in bm25.idf]
    if not idfs:
        return 0.0
    return float(np.mean(idfs))

# Empirically-calibrated IDF thresholds based on observed query avg-IDF range.
# Queries dominated by common words (~1.5) -> trust vector.
# Queries with rare proper nouns (~6+)    -> trust BM25.
IDF_LOW = 2.0    # at or below -> min BM25 weight
IDF_HIGH = 5.0   # at or above -> max BM25 weight


def compute_bm25_weight(query: str) -> float:
    """
    Map the query's average term-IDF to a BM25 weight in WEIGHT_BOUNDS.

    Calibrated against the observed range of real-query avg IDF (~1.5 to ~6.5),
    NOT against the corpus-wide token IDF distribution (which is dominated by
    ultra-rare tokens that never appear in queries).
    """
    q_idf = query_avg_idf(query)

    if q_idf <= 0:
        return WEIGHT_BOUNDS[0]  # no in-corpus terms -> all vector
    if q_idf <= IDF_LOW:
        return WEIGHT_BOUNDS[0]
    if q_idf >= IDF_HIGH:
        return WEIGHT_BOUNDS[1]

    progress = (q_idf - IDF_LOW) / (IDF_HIGH - IDF_LOW)
    return WEIGHT_BOUNDS[0] + progress * (WEIGHT_BOUNDS[1] - WEIGHT_BOUNDS[0])


def retrieve_hybrid_window_smart(query: str, top_k: int = TOP_K) -> list[RetrievalResult]:
    """IDF-weighted RRF fusion of BM25 and window_noheader, returning scenes."""
    n = top_k * OVERSAMPLE_MULTIPLIER

    bm25_results = retrieve_bm25(query, top_k=n)
    window_results = retrieve_from_collection(query, n, WINDOW_NOHEADER_COLLECTION)

    bm25_scenes = [r.scene_id for r in bm25_results]
    window_scenes = _dedupe_windows_to_scenes(window_results)

    bm25_w = compute_bm25_weight(query)
    vector_w = 1.0 - bm25_w

    rrf = {}
    for rank, sid in enumerate(bm25_scenes, start=1):
        rrf[sid] = rrf.get(sid, 0.0) + bm25_w / (RRF_K + rank)
    for rank, sid in enumerate(window_scenes, start=1):
        rrf[sid] = rrf.get(sid, 0.0) + vector_w / (RRF_K + rank)

    top = sorted(rrf.keys(), key=lambda s: rrf[s], reverse=True)[:top_k]
    return [_make_scene_result(sid, -rrf[sid]) for sid in top if sid in _SCENES]


if __name__ == "__main__":
    import sys
    queries = sys.argv[1:] if len(sys.argv) > 1 else [
        "Nina Bookbinder",
        "Who was Joey in love with?",
        "What does Ross yell when moving a couch?",
        "What program is the group viewing on the television set?",
    ]
    for query in queries:
        w = compute_bm25_weight(query)
        print(f"\nQuery: {query!r}")
        print(f"  avg IDF: {query_avg_idf(query):.3f}  ->  BM25 weight: {w:.3f} (vector: {1-w:.3f})")
        for i, r in enumerate(retrieve_hybrid_window_smart(query), start=1):
            print(f"    {i}. {r.citation}  (score: {-r.distance:.4f})")