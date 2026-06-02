"""
scripts/set_search_ef.py

Bake a fixed hnsw:search_ef into existing Chroma collections WITHOUT re-embedding.

Chroma 0.5.x fixes search_ef at collection creation -- modify() updates the
metadata field but does NOT change query behaviour (verified). So for each
target collection we:
  1. pull its vectors / metadatas / documents (no embedding calls)
  2. build a temp collection created WITH the desired search_ef
  3. verify the row count matches the original
  4. delete the original, then rename the temp into its place
     (rename preserves search_ef and is near-instant, so the swap is safe;
      if any step before the delete fails, the original is left untouched)

Run once after choosing the value with scripts.search_ef_sweep.

USAGE
-----
    python -m scripts.set_search_ef                 # ef=400 on the default set
    python -m scripts.set_search_ef --search-ef 400
    python -m scripts.set_search_ef --collections friends_window_noheader
"""
import argparse

import chromadb

from src.config import (
    CHROMA_DIR,
    SCENE_COLLECTION,
    NAIVE_COLLECTION,
    WINDOW_NOHEADER_COLLECTION,
    UTTERANCE_NOHEADER_COLLECTION,
)

ADD_BATCH = 5000

# The small-model collections the shipped pipeline and eval actually use.
# (Disabled header collections and the tabled *_large collections are excluded.)
DEFAULT_COLLECTIONS = [
    SCENE_COLLECTION,
    NAIVE_COLLECTION,
    WINDOW_NOHEADER_COLLECTION,
    UTTERANCE_NOHEADER_COLLECTION,
]


def migrate_one(client, name: str, search_ef: int) -> None:
    existing = [c.name for c in client.list_collections()]
    if name not in existing:
        print(f"  SKIP '{name}' (not found)")
        return

    col = client.get_collection(name)
    if (col.metadata or {}).get("hnsw:search_ef") == search_ef:
        print(f"  SKIP '{name}' (already search_ef={search_ef})")
        return

    data = col.get(include=["embeddings", "metadatas", "documents"])
    ids = data["ids"]
    n = len(ids)
    embs, metas, docs = data["embeddings"], data["metadatas"], data["documents"]

    tmp_name = f"{name}__migrate_tmp"
    try:
        client.delete_collection(tmp_name)
    except Exception:
        pass

    tmp = client.create_collection(tmp_name, metadata={"hnsw:search_ef": search_ef})
    for s in range(0, n, ADD_BATCH):
        tmp.add(
            ids=ids[s:s + ADD_BATCH],
            embeddings=embs[s:s + ADD_BATCH],
            metadatas=metas[s:s + ADD_BATCH],
            documents=docs[s:s + ADD_BATCH],
        )

    if tmp.count() != n:
        raise RuntimeError(
            f"count mismatch for '{name}': temp has {tmp.count()} vs original {n}. "
            f"ABORTING this collection; original left intact. "
            f"(Temp '{tmp_name}' kept for inspection.)"
        )

    # Safe swap: original still intact until this point.
    client.delete_collection(name)
    tmp.modify(name=name)
    print(f"  OK   '{name}': {n:,} vectors, search_ef={search_ef}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-ef", type=int, default=400)
    ap.add_argument("--collections", default=None,
                    help="Comma-separated collection names. Default: the small-model set.")
    args = ap.parse_args()

    cols = (args.collections.split(",") if args.collections else DEFAULT_COLLECTIONS)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    print(f"Setting hnsw:search_ef={args.search_ef} (no re-embedding) on:")
    for c in cols:
        print(f"  - {c}")
    print()

    for name in cols:
        migrate_one(client, name, args.search_ef)

    print("\nDone. Re-run the eval/coverage scripts to pick up the new accuracy.")


if __name__ == "__main__":
    main()
