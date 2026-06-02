"""
Embedder: turn text strings into vectors using OpenAI's API.

This module is intentionally tiny - just the function that takes text in and
returns vectors out. We'll use it from both the indexer (to embed chunks)
and the retriever (to embed the user's query).

The embedding model is selectable per call (defaulting to the project default
in config). This lets us build parallel collections with a different model
(e.g. text-embedding-3-large) for comparison without disturbing the default.
"""
from openai import OpenAI

from src.config import OPENAI_API_KEY, EMBEDDING_MODEL

# Single client, reused across calls. OpenAI's SDK handles connection pooling.
_client = OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """
    Embed a batch of texts. Returns a list of vectors (one per input).

    OpenAI's API supports up to ~2000 inputs per call, but we'll batch in
    smaller groups upstream for safer retries.

    Dimensions depend on the model: text-embedding-3-small -> 1536,
    text-embedding-3-large -> 3072. A Chroma collection is fixed to one
    dimensionality, so different models must go into different collections.

    NOTE: no caching here on purpose - this is the bulk/chunk path used by the
    indexer. The query cache lives in embed_one().
    """
    if not texts:
        return []

    response = _client.embeddings.create(
        model=model,
        input=texts,
    )
    # The API returns results in the same order as inputs - critical assumption.
    return [item.embedding for item in response.data]


def embed_one(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    Embed a single string (the query path).

    Checks the on-disk query cache first; only calls OpenAI on a cache miss,
    then stores the result. The cache key includes the model, so small and
    large embeddings of the same question never collide.
    """
    from src.embed_cache import _cache   # local import avoids an import cycle
    cached = _cache.get(text, model=model)
    if cached is not None:
        return cached
    vector = embed_texts([text], model=model)[0]
    _cache.set(text, vector, model=model)
    return vector
