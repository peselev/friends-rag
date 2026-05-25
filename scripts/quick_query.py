"""
Quick sanity-check query against the indexed Chroma collection.
Not part of the main pipeline - just a tool to poke at the database manually.

Run with: python -m scripts.quick_query "your query here"
                 python -m scripts.quick_query             (uses default)
"""
import sys
import chromadb

from src.embedder import embed_one
from src.config import CHROMA_DIR

COLLECTION_NAME = "friends_scenes"
DEFAULT_QUERY = "Ross paleontologist"
TOP_K = 3


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    print(f"Query: {query!r}\n")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    query_vec = embed_one(query)
    results = collection.query(query_embeddings=[query_vec], n_results=TOP_K)

    # Chroma returns lists-of-lists (one list per query). We sent one query,
    # so we want index [0] for ids, documents, distances, metadatas.
    ids = results["ids"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]
    metas = results["metadatas"][0]

    for i, (id_, doc, dist, meta) in enumerate(zip(ids, docs, distances, metas), start=1):
        # Lower distance = closer match. Chroma uses squared L2 by default.
        print(f"{i}. {id_}  (distance: {dist:.3f})")
        print(f"   speakers: {meta['speakers'][:80]}...")
        print(f"   text: {doc[:200].replace(chr(10), ' | ')}...")
        print()


if __name__ == "__main__":
    main()
