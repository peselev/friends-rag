"""
Hybrid retriever: combines vector and BM25 results using Reciprocal Rank Fusion.

For each document seen by either retriever, compute its RRF score:
    rrf_score(doc) = sum( 1 / (k + rank) for each retriever that ranked it )

Documents ranked highly by either retriever get boosted; documents ranked highly
by both get boosted twice. No score normalization needed.
"""
from src.config import TOP_K
from src.retriever import retrieve as retrieve_vector, RetrievalResult
from src.bm25_retriever import retrieve_bm25

RRF_K = 60                    # Standard RRF constant from the original paper
OVERSAMPLE_MULTIPLIER = 3     # Pull 3x more from each retriever before fusing


def retrieve_hybrid(query: str, top_k: int = TOP_K) -> list[RetrievalResult]:
    """
    Retrieve using both vector and BM25, fuse rankings via RRF, return top-k.
    """
    # Pull more candidates than we need, since the right doc might be ranked
    # low in one retriever and high in the other.
    n_candidates = top_k * OVERSAMPLE_MULTIPLIER

    vector_results = retrieve_vector(query, top_k=n_candidates)
    bm25_results = retrieve_bm25(query, top_k=n_candidates)

    # Build a map: scene_id -> RRF score (and keep the result object for return)
    rrf_scores: dict[str, float] = {}
    result_lookup: dict[str, RetrievalResult] = {}

    for rank, result in enumerate(vector_results, start=1):
        rrf_scores[result.scene_id] = rrf_scores.get(result.scene_id, 0.0) + 1.0 / (RRF_K + rank)
        result_lookup[result.scene_id] = result

    for rank, result in enumerate(bm25_results, start=1):
        rrf_scores[result.scene_id] = rrf_scores.get(result.scene_id, 0.0) + 1.0 / (RRF_K + rank)
        # Only set if not already there - prefer the vector RetrievalResult since
        # it has a "real" vector distance, which the UI can display
        result_lookup.setdefault(result.scene_id, result)

    # Sort by RRF score descending, take top_k
    sorted_ids = sorted(rrf_scores.keys(), key=lambda sid: rrf_scores[sid], reverse=True)
    top_ids = sorted_ids[:top_k]

    # Build final result list. Overwrite the distance field with the negated
    # RRF score so the UI's "lower = better" convention still holds.
    results = []
    for sid in top_ids:
        original = result_lookup[sid]
        results.append(RetrievalResult(
            scene_id=original.scene_id,
            text=original.text,
            metadata=original.metadata,
            distance=-rrf_scores[sid],  # negated to match other modes
        ))
    return results


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Joey is in love"
    print(f"Query: {query!r}\n")
    for i, r in enumerate(retrieve_hybrid(query), start=1):
        rrf_score = -r.distance
        print(f"{i}. {r.citation}  (RRF score: {rrf_score:.4f})")
        print(f"   {r.text[:150]}...")
        print()
