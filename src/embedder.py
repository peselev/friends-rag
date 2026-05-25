"""
Embedder: turn text strings into vectors using OpenAI's API.

This module is intentionally tiny - just the function that takes text in and
returns vectors out. We'll use it from both the indexer (to embed chunks)
and the retriever (to embed the user's query).
"""
from openai import OpenAI

from src.config import OPENAI_API_KEY, EMBEDDING_MODEL

# Single client, reused across calls. OpenAI's SDK handles connection pooling.
_client = OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts. Returns a list of vectors (one per input).

    OpenAI's API supports up to ~2000 inputs per call, but we'll batch in
    smaller groups upstream for safer retries.

    Each output vector is 1536 floats for text-embedding-3-small.
    """
    if not texts:
        return []

    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # The API returns results in the same order as inputs - critical assumption.
    return [item.embedding for item in response.data]


def embed_one(text: str) -> list[float]:
    """Convenience for embedding a single string (used at query time)."""
    return embed_texts([text])[0]
