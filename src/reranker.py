"""
Cross-encoder re-ranker: second-stage refinement of retrieved candidates.

Supports multiple cross-encoder models, cached per model name so different
rerankers can be compared without reloading.

Public function:
    rerank(query, candidates, top_k=5, model_name=DEFAULT_MODEL) -> list[RetrievalResult]
"""
from sentence_transformers import CrossEncoder

from src.retriever import RetrievalResult

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cache one loaded model per model name.
_models: dict[str, CrossEncoder] = {}


def _get_model(model_name: str) -> CrossEncoder:
    """Lazy-load and cache a cross-encoder model by name."""
    if model_name not in _models:
        print(f"Loading cross-encoder model {model_name}...")
        _models[model_name] = CrossEncoder(model_name)
    return _models[model_name]


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> list[RetrievalResult]:
    """
    Re-rank candidates using the named cross-encoder, return top-k.
    Returned results have .distance replaced with -score (lower = better).
    """
    if not candidates:
        return []

    model = _get_model(model_name)
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

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
    import sys
    from src.hybrid_window import retrieve_hybrid_window

    query = sys.argv[1] if len(sys.argv) > 1 else "Who was Joey in love with?"
    model_name = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    print(f"Query: {query!r}")
    print(f"Model: {model_name}\n")
    candidates = retrieve_hybrid_window(query, top_k=20)
    reranked = rerank(query, candidates, top_k=5, model_name=model_name)
    for i, r in enumerate(reranked, start=1):
        print(f"{i}. {r.citation}  (score: {-r.distance:.3f})")
        print(f"   {r.text[:120].replace(chr(10), ' | ')}...")