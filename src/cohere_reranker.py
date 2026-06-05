"""
Cohere reranker: second-stage refinement backed by Cohere's rerank-v3.5 API.

This mirrors the local cross-encoder reranker in src/reranker.py (same
RetrievalResult in / RetrievalResult out contract, .distance set to -score so
lower is better), but the scoring is a metered API call instead of a local
model. It is the reranker the published writeup commits to and the only one
that earned its place in scripts/reranker_experiment.py.

Three-level graceful degradation (matches the writeup's promise):
    1. Try COHERE_API_KEY        (free trial key)
    2. On rate-limit / monthly-cap / auth failure -> try COHERE_API_KEY_PROD
    3. If both keys fail, no key is configured, or the `cohere` package is not
       installed -> raise CohereUnavailable.

The app catches CohereUnavailable and silently serves the fast hybrid result
instead, so the "higher accuracy" toggle deactivates itself on failure rather
than erroring out the demo.

Public surface:
    CohereUnavailable                         - exception the app catches
    cohere_rerank(query, candidates, top_k)   -> list[RetrievalResult]
"""
from src.config import (
    COHERE_API_KEY,
    COHERE_API_KEY_PROD,
    COHERE_RERANK_MODEL,
)
from src.retriever import RetrievalResult


class CohereUnavailable(Exception):
    """Raised when reranking via Cohere cannot be completed for any reason.

    The app treats this as the signal to fall back to the fast hybrid result.
    """


# Cache one client per API key so we don't rebuild it on every query.
_clients: dict = {}

# A small in-process retry for transient (non-quota) errors before we give up
# on a key. Quota/auth errors are NOT retried here — they jump straight to the
# next key in the chain.
_TRANSIENT_RETRIES = 2


def _get_client(api_key: str):
    """Lazy-build and cache a Cohere ClientV2 for a key. Raises CohereUnavailable
    if the cohere package isn't installed."""
    if api_key not in _clients:
        try:
            import cohere  # imported lazily so the package is optional
        except ImportError as e:
            raise CohereUnavailable("cohere package not installed") from e
        _clients[api_key] = cohere.ClientV2(api_key)
    return _clients[api_key]


def _is_quota_or_auth_error(exc: Exception) -> bool:
    """True for errors that mean 'this key is spent or invalid' — the trigger to
    move on to the next key in the chain (rate limit, monthly cap, bad/forbidden
    key). Everything else is treated as a transient/service error."""
    try:
        import cohere
    except ImportError:
        return False
    return isinstance(
        exc,
        (
            cohere.TooManyRequestsError,   # 429: rate limit or monthly trial cap
            cohere.ForbiddenError,         # 403: often quota/billing exhausted
            cohere.UnauthorizedError,      # 401: missing/invalid key
        ),
    )


def _rerank_with_key(api_key: str, query: str, documents: list[str]) -> list:
    """Call Cohere with one key. Returns resp.results (index + relevance_score).

    Raises the original cohere exception so the caller can decide whether to
    switch keys (quota/auth) or bail (transient/service)."""
    client = _get_client(api_key)
    last_exc = None
    for attempt in range(_TRANSIENT_RETRIES + 1):
        try:
            resp = client.rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=len(documents),   # rank all candidates; we cut to top_k ourselves
            )
            return resp.results
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_quota_or_auth_error(exc):
                # No point retrying the same spent/invalid key — surface it so
                # the chain can switch keys.
                raise
            # Transient/service error: brief retry, then give up on this key.
            if attempt < _TRANSIENT_RETRIES:
                import time
                time.sleep(2 ** attempt)
    # Exhausted transient retries on a service error.
    raise last_exc if last_exc else RuntimeError("Cohere rerank failed")


def cohere_rerank(
    query: str,
    candidates: list[RetrievalResult],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Re-rank candidates with Cohere rerank-v3.5 and return the top_k.

    Returned results have .distance set to -relevance_score (lower = better),
    matching the local reranker's contract so downstream code is identical.

    Raises CohereUnavailable if no key works, no key is configured, or the
    package is missing.
    """
    if not candidates:
        return []

    keys = [k for k in (COHERE_API_KEY, COHERE_API_KEY_PROD) if k]
    if not keys:
        raise CohereUnavailable("no Cohere API key configured")

    documents = [c.text for c in candidates]

    results = None
    for i, key in enumerate(keys):
        try:
            results = _rerank_with_key(key, query, documents)
            break
        except Exception as exc:  # noqa: BLE001
            if _is_quota_or_auth_error(exc) and i < len(keys) - 1:
                # Trial key spent/invalid — fall through to the next key.
                continue
            # Last key, or a service/transient failure we already retried:
            # nothing left to try.
            raise CohereUnavailable(
                f"Cohere rerank failed ({type(exc).__name__})"
            ) from exc

    if results is None:
        raise CohereUnavailable("Cohere rerank produced no result")

    # resp.results carry .index (into `documents`) and .relevance_score.
    ranked = sorted(results, key=lambda r: r.relevance_score, reverse=True)[:top_k]

    out = []
    for r in ranked:
        cand = candidates[r.index]
        out.append(RetrievalResult(
            scene_id=cand.scene_id,
            text=cand.text,
            metadata=cand.metadata,
            distance=-float(r.relevance_score),
        ))
    return out
