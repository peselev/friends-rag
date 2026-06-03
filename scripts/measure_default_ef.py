"""
scripts/measure_default_ef.py

Produce the two "pre-optimization" tables at TRUE default ef:
  - First results:     BM25, scene-vector, scene+BM25 hybrid
  - Different chunks:   scene-vector, single-utterance, 5-utterance window

Why a dedicated script: the production collections are pinned at search_ef=400,
and you can't query below that (effective ef = max(search_ef, fetch_depth)). So
to measure at default ef we rebuild the relevant collections at Chroma's default
search_ef (no metadata) under TEMP names, measure against them, then delete them.
Production collections are never modified. No re-embedding (vectors are pulled
from the existing collections); query embeddings come from the warm cache.

Everything is scene-level (chunks deduped to parent scenes, dedup-fixed harness)
and uses the same fetch depth as the search_ef sweep, so the window number here
should reproduce the sweep's "default" row (~0.435 direct) as a consistency check.

USAGE
-----
    python -m scripts.measure_default_ef
    python -m scripts.measure_default_ef --keep   # don't delete temp collections
"""
import argparse
import json
from pathlib import Path

import chromadb

from src.config import (
    CHROMA_DIR, DATA_PROCESSED, EMBEDDING_MODEL,
    SCENE_COLLECTION, WINDOW_NOHEADER_COLLECTION, UTTERANCE_NOHEADER_COLLECTION,
)
from src.bm25_retriever import retrieve_bm25
from src.embedder import embed_one

FETCH = 80                 # matches the search_ef sweep (effective ef ~ default)
K_VALUES = [5, 10]
RRF_K = 60                 # matches production hybrid_retriever
ADD_BATCH = 5000


def build_default_collection(client, source_name: str):
    """Pull vectors from the (ef=400) source collection and rebuild a copy at
    Chroma's DEFAULT search_ef (no metadata), under a temp name."""
    src = client.get_collection(source_name)
    data = src.get(include=["embeddings", "metadatas", "documents"])
    ids, embs = data["ids"], data["embeddings"]
    metas, docs = data["metadatas"], data["documents"]
    tmp_name = f"{source_name}__defaultef_tmp"
    try:
        client.delete_collection(tmp_name)
    except Exception:
        pass
    col = client.create_collection(tmp_name)            # NO metadata -> default search_ef
    for s in range(0, len(ids), ADD_BATCH):
        col.add(ids=ids[s:s + ADD_BATCH], embeddings=embs[s:s + ADD_BATCH],
                metadatas=metas[s:s + ADD_BATCH], documents=docs[s:s + ADD_BATCH])
    return col, tmp_name


def vector_scenes(col, qvec, n=FETCH) -> list[str]:
    """Query a collection, dedupe chunks to ordered unique parent scenes."""
    r = col.query(query_embeddings=[qvec], n_results=n)
    ids, metas = r["ids"][0], r["metadatas"][0]
    seen = []
    for cid, m in zip(ids, metas):
        sid = (m or {}).get("scene_id", cid)
        if sid not in seen:
            seen.append(sid)
    return seen


def bm25_scenes(query, n=FETCH) -> list[str]:
    res = retrieve_bm25(query, top_k=n)
    seen = []
    for x in res:
        sid = x.metadata.get("scene_id", x.scene_id)
        if sid not in seen:
            seen.append(sid)
    return seen


def rrf_fuse(*lists) -> list[str]:
    scores = {}
    for lst in lists:
        for rank, sid in enumerate(lst, start=1):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=lambda s: scores[s], reverse=True)


def rank_of(scenes, gold):
    for i, s in enumerate(scenes, start=1):
        if s == gold:
            return i
    return None


def recall_table(per_q_scenes_by_mode, golds, strata, modes):
    """per_q_scenes_by_mode[mode] = list of scene-lists (one per question, aligned to golds/strata)."""
    out = {}
    for mode in modes:
        out[mode] = {}
        for strat in ["direct", "reworded"]:
            idxs = [i for i, s in enumerate(strata) if s == strat]
            n = len(idxs)
            row = {}
            for k in K_VALUES:
                hits = sum(1 for i in idxs
                           if (r := rank_of(per_q_scenes_by_mode[mode][i], golds[i])) and r <= k)
                row[k] = (hits / n) if n else 0.0
            out[mode][strat] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=DATA_PROCESSED / "eval_sample.jsonl")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    print("Building default-ef temp collections (no re-embedding)...")
    scene_col, scene_tmp = build_default_collection(client, SCENE_COLLECTION)
    window_col, window_tmp = build_default_collection(client, WINDOW_NOHEADER_COLLECTION)
    utt_col, utt_tmp = build_default_collection(client, UTTERANCE_NOHEADER_COLLECTION)
    print(f"  built: {scene_tmp}, {window_tmp}, {utt_tmp}\n")

    qs = [json.loads(l) for l in open(args.questions)]
    qs = [q for q in qs if not q.get("is_paraphrased")] + [q for q in qs if q.get("is_paraphrased")]
    golds = [q["ground_truth_scene_id"] for q in qs]
    strata = ["reworded" if q.get("is_paraphrased") else "direct" for q in qs]
    print(f"Measuring {len(qs)} questions (direct + reworded)...")

    modes = ["bm25", "scene", "hybrid", "utterance", "window"]
    scenes_by_mode = {m: [] for m in modes}
    for q in qs:
        qvec = embed_one(q["question"], model=EMBEDDING_MODEL)
        sc = vector_scenes(scene_col, qvec)
        bm = bm25_scenes(q["question"])
        scenes_by_mode["scene"].append(sc)
        scenes_by_mode["bm25"].append(bm)
        scenes_by_mode["hybrid"].append(rrf_fuse(sc, bm))
        scenes_by_mode["window"].append(vector_scenes(window_col, qvec))
        scenes_by_mode["utterance"].append(vector_scenes(utt_col, qvec))

    res = recall_table(scenes_by_mode, golds, strata, modes)

    def line(label, mode):
        d, r = res[mode]["direct"], res[mode]["reworded"]
        return f"| {label} | {d[5]:.3f} | {r[5]:.3f} |   (@10: {d[10]:.3f} / {r[10]:.3f})"

    print("\n=== FIRST RESULTS (default ef) ===")
    print("| Retrieval mode | Direct (recall@5) | Reworded (recall@5) |")
    print(line("BM25 (keyword)", "bm25"))
    print(line("Vector (scene-level)", "scene"))
    print(line("Hybrid (scene-level vector + BM25)", "hybrid"))

    print("\n=== DIFFERENT CHUNKS (default ef) ===")
    print("| Retrieval mode | Direct (recall@5) | Reworded (recall@5) |")
    print(line("Vector (scene-level)", "scene"))
    print(line("Vector (single utterance)", "utterance"))
    print(line("Vector (5-utterance window)", "window"))
    print(f"\nCross-check: window direct@5 should be ~0.435 (the sweep's 'default' row). "
          f"Got {res['window']['direct'][5]:.3f}.")

    if not args.keep:
        for name in [scene_tmp, window_tmp, utt_tmp]:
            try:
                client.delete_collection(name)
            except Exception:
                pass
        print("\nDeleted temp collections.")
    else:
        print(f"\nKept: {scene_tmp}, {window_tmp}, {utt_tmp}")


if __name__ == "__main__":
    main()
