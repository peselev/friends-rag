"""
Cross-encoder re-ranker: second-stage refinement of retrieved candidates.

A cross-encoder reads (query, candidate) together and scores their relevance
directly - more accurate than bi-encoder similarity but much slower. We use it
to re-rank a small set of candidates retrieved by a faster first-stage method.

Public function:
    rerank(query, candidates, top_k=5) -> list[RetrievalResult]
"""
from sentence_transformers import CrossEncoder

from src.retriever import RetrievalResult

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level singleton. Model loads on first use (~5-10s), then is fast.
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Lazy-load the cross-encoder model on first use."""
    global _model
    if _model is None:
        print(f"Loading cross-encoder model {MODEL_NAME}...")
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """
    Re-rank candidates using a cross-encoder, return top-k.

    The cross-encoder scores each (query, candidate.text) pair directly.
    Returned results have their .distance replaced with -score so the
    "lower = better" convention is preserved.
    """
    if not candidates:
        return []

    model = _get_model()

    # Build pairs for the model
    pairs = [(query, c.text) for c in candidates]

    # predict() returns a numpy array of scores (higher = more relevant)
    scores = model.predict(pairs)

    # Pair each candidate with its score, sort by score descending, take top_k
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    # Return new RetrievalResults with the cross-encoder score as -distance
    results = []
    for candidate, score in top:
        results.append(RetrievalResult(
            scene_id=candidate.scene_id,
            text=candidate.text,
            metadata=candidate.metadata,
            distance=-float(score),
        ))
    return results


if __name__ == "__main__":
    # Smoke test: pull candidates via parent_child, re-rank them
    import sys
    from src.parent_child import retrieve_parent_scene

    query = sys.argv[1] if len(sys.argv) > 1 else "Who was Joey in love with?"

    print(f"Query: {query!r}\n")
    print("Stage 1: parent-child retrieval (top 20)...")
    candidates = retrieve_parent_scene(query, top_k=20)
    print(f"  Got {len(candidates)} candidates")
    print("  Before reranking, top 5:")
    for i, c in enumerate(candidates[:5], start=1):
        print(f"    {i}. {c.citation}  (distance: {c.distance:.3f})")

    print("\nStage 2: cross-encoder reranking...")
    reranked = rerank(query, candidates, top_k=5)
    print("  After reranking, top 5:")
    for i, r in enumerate(reranked, start=1):
        print(f"    {i}. {r.citation}  (rerank score: {-r.distance:.3f})")