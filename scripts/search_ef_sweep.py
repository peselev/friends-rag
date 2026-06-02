"""
scripts/search_ef_sweep.py

Find the accuracy/latency knee for Chroma's HNSW `search_ef`.

WHY
---
`search_ef` controls how thoroughly the approximate (HNSW) index is searched.
Higher = closer to the true nearest neighbours = slower. In Chroma 0.5.x it is
fixed at COLLECTION CREATION (`modify()` updates metadata but does NOT change
query behaviour), so to compare values we must rebuild the collection at each
setting. We do that WITHOUT re-embedding: the vectors already live in the
existing collection, so we pull them out with `get(include=["embeddings"])` and
re-add them under each `search_ef`. No OpenAI cost.

WHAT IT MEASURES
----------------
At a fixed fetch depth (`--n-results`), for each `search_ef`:
  - scene-level recall@5 / @10 (deduped to parent scenes, like the bot)
  - per-query search latency: median and p95 (milliseconds)
Plus the current collection as the baseline ("default" ef).

The effective ef at query time is max(search_ef, n_results), so we test ef
values >= n_results to isolate the setting cleanly.

DECISION
--------
Pick the smallest ef where recall@5 has plateaued; read its latency cost there.
If the cost is negligible, make it the default; if it's large but the lift is
worth it, expose it as a "high accuracy" toggle.

USAGE
-----
    python -m scripts.search_ef_sweep
    python -m scripts.search_ef_sweep --ef-values 100,200,400,800,1600
    python -m scripts.search_ef_sweep --source friends_window_noheader --n-results 80
    python -m scripts.search_ef_sweep --keep    # don't delete the temp collections
"""
import argparse
import statistics
import time
from pathlib import Path
import json

import chromadb

from src.config import CHROMA_DIR, DATA_PROCESSED, EMBEDDING_MODEL, WINDOW_NOHEADER_COLLECTION
from src.embedder import embed_one

ADD_BATCH = 5000


def parent_scene(chunk_id: str, meta: dict | None) -> str:
    return (meta or {}).get("scene_id", chunk_id)


def pull_collection(client, name: str):
    """Pull all ids/embeddings/metadatas/documents out of an existing collection."""
    col = client.get_collection(name)
    data = col.get(include=["embeddings", "metadatas", "documents"])
    n = len(data["ids"])
    print(f"  Pulled {n:,} vectors from '{name}'.")
    return data


def build_variant(client, base_name: str, ef: int, data: dict):
    """Create (or replace) a collection built WITH hnsw:search_ef=ef and the
    same vectors. Returns the collection."""
    name = f"{base_name}_efsweep_{ef}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:search_ef": ef})
    ids, embs = data["ids"], data["embeddings"]
    metas, docs = data["metadatas"], data["documents"]
    for s in range(0, len(ids), ADD_BATCH):
        col.add(
            ids=ids[s:s + ADD_BATCH],
            embeddings=embs[s:s + ADD_BATCH],
            metadatas=metas[s:s + ADD_BATCH],
            documents=docs[s:s + ADD_BATCH],
        )
    return col, name


def load_questions(path: Path):
    rows = [json.loads(l) for l in open(path)]
    originals = [q for q in rows if not q.get("is_paraphrased")]
    paraphrased = [q for q in rows if q.get("is_paraphrased")]
    return originals, paraphrased


def measure(col, qvecs, n_results: int):
    """Return (recall@5, recall@10, latencies_ms) for a list of (question, vector)."""
    lat = []
    hits5 = hits10 = 0
    for q, v in qvecs:
        t0 = time.perf_counter()
        r = col.query(query_embeddings=[v], n_results=n_results)
        lat.append((time.perf_counter() - t0) * 1000.0)
        rids = r["ids"][0]
        rmetas = r["metadatas"][0]
        seen = []
        for cid, m in zip(rids, rmetas):
            s = parent_scene(cid, m)
            if s not in seen:
                seen.append(s)
        gold = q["ground_truth_scene_id"]
        hits5 += gold in seen[:5]
        hits10 += gold in seen[:10]
    n = len(qvecs)
    return hits5 / n, hits10 / n, lat


def pctl(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(len(xs) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=WINDOW_NOHEADER_COLLECTION,
                    help="Collection to sweep (default: the shipped window collection).")
    ap.add_argument("--ef-values", default="100,200,400,800",
                    help="Comma-separated search_ef values to test.")
    ap.add_argument("--n-results", type=int, default=80,
                    help="Fetch depth per query (kept constant across ef).")
    ap.add_argument("--questions", type=Path, default=DATA_PROCESSED / "eval_sample.jsonl")
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    ap.add_argument("--max-questions", type=int, default=None,
                    help="Cap questions per split (for a quick pass).")
    ap.add_argument("--keep", action="store_true",
                    help="Keep the temp _efsweep_ collections instead of deleting.")
    args = ap.parse_args()

    ef_values = [int(x) for x in args.ef_values.split(",")]
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    print(f"Source collection: {args.source}")
    print(f"search_ef values:  {ef_values}")
    print(f"Fetch depth:       n_results={args.n_results}  "
          f"(effective ef = max(search_ef, n_results))\n")

    print("Pulling vectors (no re-embedding)...")
    data = pull_collection(client, args.source)

    print("Loading + embedding questions (uses query cache; no API if warm)...")
    originals, paraphrased = load_questions(args.questions)
    if args.max_questions:
        originals = originals[:args.max_questions]
        paraphrased = paraphrased[:args.max_questions]

    def embed_split(split):
        return [(q, embed_one(q["question"], model=args.embedding_model)) for q in split]

    qv_orig = embed_split(originals)
    qv_par = embed_split(paraphrased)
    print(f"  Originals: {len(qv_orig)}   Paraphrased: {len(qv_par)}\n")

    rows = []  # (label, ef_effective, r5_o, r10_o, r5_p, r10_p, med, p95)

    # Baseline: the existing collection as-is (current default ef).
    print("Measuring baseline (current collection, default search_ef)...")
    base = client.get_collection(args.source)
    r5o, r10o, lat_o = measure(base, qv_orig, args.n_results)
    r5p, r10p, lat_p = measure(base, qv_par, args.n_results)
    lat = lat_o + lat_p
    rows.append(("default", None, r5o, r10o, r5p, r10p,
                 statistics.median(lat), pctl(lat, 0.95)))

    # Each search_ef variant.
    created = []
    for ef in ef_values:
        print(f"Building + measuring search_ef={ef}...")
        col, name = build_variant(client, args.source, ef, data)
        created.append(name)
        r5o, r10o, lat_o = measure(col, qv_orig, args.n_results)
        r5p, r10p, lat_p = measure(col, qv_par, args.n_results)
        lat = lat_o + lat_p
        rows.append((f"ef={ef}", ef, r5o, r10o, r5p, r10p,
                     statistics.median(lat), pctl(lat, 0.95)))

    # Report
    print("\n\n=== search_ef sweep ===")
    print(f"source={args.source}  n_results={args.n_results}  "
          f"model={args.embedding_model}\n")
    hdr = f"{'setting':<10} {'R@5 orig':>9} {'R@10 orig':>10} {'R@5 par':>9} {'R@10 par':>10} {'med ms':>8} {'p95 ms':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, ef, r5o, r10o, r5p, r10p, med, p95 in rows:
        print(f"{label:<10} {r5o:>9.3f} {r10o:>10.3f} {r5p:>9.3f} {r10p:>10.3f} {med:>8.2f} {p95:>8.2f}")

    # Simple knee hint: smallest ef within 0.005 R@5 (orig) of the best ef.
    ef_rows = [r for r in rows if r[1] is not None]
    if ef_rows:
        best = max(r[2] for r in ef_rows)
        knee = min((r for r in ef_rows if r[2] >= best - 0.005), key=lambda r: r[1])
        base_med = rows[0][6]
        print(f"\nBest R@5 (orig) among ef variants: {best:.3f}")
        print(f"Knee (smallest ef within 0.005 of best): search_ef={knee[1]}  "
              f"-> R@5 orig {knee[2]:.3f}, median {knee[6]:.2f} ms "
              f"(baseline median {base_med:.2f} ms, {knee[6] - base_med:+.2f} ms)")

    # Cleanup
    if args.keep:
        print(f"\nKept temp collections: {created}")
    else:
        for name in created:
            try:
                client.delete_collection(name)
            except Exception:
                pass
        print(f"\nDeleted {len(created)} temp collections.")


if __name__ == "__main__":
    main()
