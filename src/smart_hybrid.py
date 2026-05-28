"""
Smart hybrid retrieval: weighted RRF over BM25 and parent-child results,
where the weight adapts to query characteristics.

For each query:
1. Compute average IDF of query terms in the corpus
2. High avg IDF (rare query terms) -> trust BM25 more
3. Low avg IDF (common or absent query terms) -> trust vector more
4. Fuse rankings with the computed weight

This addresses the Weekend 3 finding that uniform RRF is dominated by
whichever retriever happens to be stronger on the corpus, and the
parent-child finding that BM25 collapses on out-of-vocab paraphrases.
"""
import math
import re
from collections import OrderedDict

import numpy as np

from src.bm25_retriever import _get_index, _tokenize
from src.config import TOP_K
from src.parent_child import retrieve_parent_scene
from src.retriever import RetrievalResult


# RRF parameters
RRF_K = 60                    # Standard RRF smoothing constant
OVERSAMPLE_MULTIPLIER = 4     # Pull 4x more candidates than top_k before fusing

# Weight clamping: even at extremes, neither retriever fully disabled.
# Format: (min_bm25_weight, max_bm25_weight)
WEIGHT_BOUNDS = (0.2, 0.8)


def query_avg_idf(query: str) -> float:
    """
    Compute average IDF of query terms that exist in the BM25 corpus.
    Terms not in the corpus are excluded from the average (they contribute
    nothing to BM25's score anyway).

    Higher value = query contains rarer terms = BM25 has more signal.
    """
    bm25, _chunks = _get_index()
    query_tokens = _tokenize(query)

    # rank_bm25's BM25Okapi stores per-token IDF in self.idf
    idfs_present = [bm25.idf[t] for t in query_tokens if t in bm25.idf]
    if not idfs_present:
        # No query terms appear in the corpus at all - BM25 useless
        return 0.0
    return float(np.mean(idfs_present))


def compute_bm25_weight(query: str) -> float:
    """
    Map query characteristics to a BM25 weight in [0.2, 0.8].

    The mapping: we use the corpus's overall IDF distribution as the
    reference scale. If the query's avg IDF is at or above the corpus's
    median IDF, trust BM25 heavily. Below median, trust vector more.
    """
    bm25, _chunks = _get_index()
    query_idf = query_avg_idf(query)

    # Reference scale: median IDF in the corpus
    all_idfs = list(bm25.idf.values())
    median_idf = float(np.median(all_idfs))
    p90_idf = float(np.percentile(all_idfs, 90))

    if query_idf <= 0:
        # No query terms in corpus -> all-vector
        return WEIGHT_BOUNDS[0]

    # Smooth ramp from min weight at median to max weight at p90
    if query_idf <= median_idf:
        return WEIGHT_BOUNDS[0]
    if query_idf >= p90_idf:
        return WEIGHT_BOUNDS[1]

    # Linear interpolation between bounds
    span = p90_idf - median_idf
    progress = (query_idf - median_idf) / span
    return WEIGHT_BOUNDS[0] + progress * (WEIGHT_BOUNDS[1] - WEIGHT_BOUNDS[0])


def retrieve_smart_hybrid(query: str, top_k: int = TOP_K) -> list[RetrievalResult]:
    """
    Weighted hybrid: parent-child (vector) + BM25, weighted by query analysis.
    """
    from src.bm25_retriever import retrieve_bm25

    n_candidates = top_k * OVERSAMPLE_MULTIPLIER
    bm25_results = retrieve_bm25(query, top_k=n_candidates)
    vector_results = retrieve_parent_scene(query, top_k=n_candidates)

    bm25_weight = compute_bm25_weight(query)
    vector_weight = 1.0 - bm25_weight

    rrf_scores: dict[str, float] = {}
    result_lookup: dict[str, RetrievalResult] = {}

    for rank, result in enumerate(bm25_results, start=1):
        contrib = bm25_weight / (RRF_K + rank)
        rrf_scores[result.scene_id] = rrf_scores.get(result.scene_id, 0.0) + contrib
        result_lookup.setdefault(result.scene_id, result)

    for rank, result in enumerate(vector_results, start=1):
        contrib = vector_weight / (RRF_K + rank)
        rrf_scores[result.scene_id] = rrf_scores.get(result.scene_id, 0.0) + contrib
        # Prefer the parent-child result since it has full scene text
        result_lookup[result.scene_id] = result

    sorted_ids = sorted(rrf_scores.keys(), key=lambda sid: rrf_scores[sid], reverse=True)
    top_ids = sorted_ids[:top_k]

    results = []
    for sid in top_ids:
        original = result_lookup[sid]
        results.append(RetrievalResult(
            scene_id=original.scene_id,
            text=original.text,
            metadata=original.metadata,
            distance=-rrf_scores[sid],
        ))
    return results


if __name__ == "__main__":
    import sys

    queries = sys.argv[1:] if len(sys.argv) > 1 else [
        "Who was Joey in love with?",
        "What does Ross yell when moving a couch?",
        "Nina Bookbinder",
        "the gang celebrates Thanksgiving",
        "What program is the group viewing on the television set?",
    ]

    for query in queries:
        avg_idf = query_avg_idf(query)
        weight = compute_bm25_weight(query)
        print(f"\nQuery: {query!r}")
        print(f"  Avg IDF of query terms: {avg_idf:.3f}")
        print(f"  Computed BM25 weight:   {weight:.3f}  (vector weight: {1-weight:.3f})")
        print(f"  Results:")
        for i, r in enumerate(retrieve_smart_hybrid(query), start=1):
            print(f"    {i}. {r.citation}  (RRF score: {-r.distance:.4f})")