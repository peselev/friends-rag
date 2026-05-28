"""
Retriever: given a query string, return the top-k most relevant scene chunks.

This is the "retrieval" part of RAG. The "augmented generation" part lives
in generator.py and uses the output of this module as context.
"""
from dataclasses import dataclass

import chromadb

from src.config import CHROMA_DIR, SCENE_COLLECTION, TOP_K
from src.embedder import embed_one


@dataclass
class RetrievalResult:
    """One retrieved chunk with its metadata and similarity score."""
    scene_id: str
    text: str
    metadata: dict
    distance: float  # Lower = more similar (Chroma uses L2 by default)

    @property
    def citation(self) -> str:
        """Human-readable scene reference. Falls back gracefully for naive chunks."""
        m = self.metadata
        if "season" in m:
            return f"S{m['season']:02d}E{m['episode']:02d}, Scene {m['scene_num']}"
        if "started_in_scene" in m:
            return f"naive chunk (started in {m['started_in_scene']})"
        return "(unknown source)"   
    

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collections: dict[str, chromadb.Collection] = {}


def _get_collection(name: str) -> chromadb.Collection:
    if name not in _collections:
        _collections[name] = _client.get_collection(name)
    return _collections[name]


def retrieve_from_collection(
    query: str,
    top_k: int,
    collection_name: str,
) -> list[RetrievalResult]:
    """Retrieve top-k from a named Chroma collection."""
    collection = _get_collection(collection_name)
    query_vec = embed_one(query)
    raw = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
    )

    # Chroma returns lists-of-lists (one inner list per query). We only sent
    # one query, so we always want index [0].
    results = []
    for scene_id, doc, meta, dist in zip(
        raw["ids"][0],
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append(RetrievalResult(
            scene_id=scene_id,
            text=doc,
            metadata=meta,
            distance=dist,
        ))
    return results


def retrieve(query: str, top_k: int = TOP_K) -> list[RetrievalResult]:
    """
    Retrieve the top-k most semantically similar scenes for a query.
    """
    return retrieve_from_collection(query, top_k, SCENE_COLLECTION)


def format_for_prompt(results: list[RetrievalResult]) -> str:
    """
    Format retrieved chunks for inclusion in an LLM prompt.

    The format matters: clear separators and source labels help the LLM
    cite correctly and reduce confusion between sources.
    """
    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(f"[Source {i} — {r.citation}]\n{r.text}")
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    # Quick CLI test: python -m src.retriever "your query here"
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Ross and Rachel break up"
    print(f"Query: {query!r}\n")
    for i, r in enumerate(retrieve(query), start=1):
        print(f"{i}. {r.citation}  (distance: {r.distance:.3f})")
        print(f"   {r.text[:150]}...")
        print()
