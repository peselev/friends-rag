"""
scripts/compare_embeddings_default_ef.py

Small vs large embedding model, BOTH measured at DEFAULT ef, in one run.

This is the comparison for the "Use a larger embedding model" section, which in
the narrative sits BEFORE the Fine-tune Chroma step, i.e. at default ef. So we do
NOT use the coverage script (it always fetches 600 -> effective ef ~600, the
wrong regime). We reuse measure_default_ef's harness: rebuild each collection at
Chroma's default search_ef under a temp name, query at the same shallow fetch,
delete the temp. Identical method for both models -> the only thing that differs
is the embedding model.

Built-in check: the SMALL numbers here must reproduce the earlier default-ef run
(scene 0.300/0.357, window 0.440/0.472, utterance 0.268/0.265). If they do, the
regime is right and the large column is trustworthy.

Requires the *_large collections to still exist (built during the original large
experiment). Query embeddings for each model come from the warm cache; the large
queries may cost ~800 embedding calls if not cached. Corpus vectors are reused.

USAGE
-----
    python -m scripts.compare_embeddings_default_ef
"""
import json
from pathlib import Path

import chromadb

from src.config import (
    CHROMA_DIR, DATA_PROCESSED, EMBEDDING_MODEL,
    SCENE_COLLECTION, WINDOW_NOHEADER_COLLECTION, UTTERANCE_NOHEADER_COLLECTION,
)
from src.embedder import embed_one
from scripts.measure_default_ef import (
    build_default_collection, vector_scenes, bm25_scenes, rank_of,
)

LARGE_MODEL = "text-embedding-3-large"
VEC_MODES = ["scene", "window", "utterance"]
SOURCES = {
    "scene": SCENE_COLLECTION,
    "window": WINDOW_NOHEADER_COLLECTION,
    "utterance": UTTERANCE_NOHEADER_COLLECTION,
}


def measure_model(client, model: str, suffix: str, qs, golds, strata) -> dict:
    """Build default-ef temp collections for the 3 vector modes, measure recall@5."""
    cols, tmps = {}, []
    for mode in VEC_MODES:
        try:
            col, tmp = build_default_collection(client, SOURCES[mode] + suffix)
        except Exception as e:
            raise SystemExit(
                f"\nCould not open collection '{SOURCES[mode] + suffix}': {e}\n"
                f"If the *_large collections were deleted, they must be rebuilt "
                f"(re-embedding with {LARGE_MODEL}) before this can run."
            )
        cols[mode], _ = col, tmps.append(tmp)

    per_q = {m: [] for m in VEC_MODES}
    for q in qs:
        qvec = embed_one(q["question"], model=model)
        for mode in VEC_MODES:
            per_q[mode].append(vector_scenes(cols[mode], qvec))

    for tmp in tmps:
        try:
            client.delete_collection(tmp)
        except Exception:
            pass

    out = {}
    for mode in VEC_MODES:
        out[mode] = {}
        for strat in ["direct", "reworded"]:
            idxs = [i for i, s in enumerate(strata) if s == strat]
            n = len(idxs)
            hits = sum(1 for i in idxs if (r := rank_of(per_q[mode][i], golds[i])) and r <= 5)
            out[mode][strat] = (hits / n) if n else 0.0
    return out


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    qs = [json.loads(l) for l in open(DATA_PROCESSED / "eval_sample.jsonl")]
    golds = [q["ground_truth_scene_id"] for q in qs]
    strata = ["reworded" if q.get("is_paraphrased") else "direct" for q in qs]
    print(f"{len(qs)} questions, both models at DEFAULT ef (shallow fetch)\n")

    print("Measuring SMALL ...")
    small = measure_model(client, EMBEDDING_MODEL, "", qs, golds, strata)
    print("Measuring LARGE ...")
    large = measure_model(client, LARGE_MODEL, "_large", qs, golds, strata)

    # BM25 control (model-independent, ef-independent)
    bm = {"direct": [], "reworded": []}
    for q, g, st in zip(qs, golds, strata):
        bm[st].append(1 if (r := rank_of(bm25_scenes(q["question"]), g)) and r <= 5 else 0)
    bm_d = sum(bm["direct"]) / len(bm["direct"])
    bm_r = sum(bm["reworded"]) / len(bm["reworded"])

    print("\n=== SMALL vs LARGE at default ef (recall@5) ===")
    hdr = f"{'mode':<12}{'small D':>9}{'small R':>9}{'large D':>9}{'large R':>9}{'ΔD':>8}{'ΔR':>8}"
    print(hdr); print("-" * len(hdr))
    for mode in VEC_MODES:
        sD, sR = small[mode]["direct"], small[mode]["reworded"]
        lD, lR = large[mode]["direct"], large[mode]["reworded"]
        print(f"{mode:<12}{sD:>9.3f}{sR:>9.3f}{lD:>9.3f}{lR:>9.3f}{lD - sD:>+8.3f}{lR - sR:>+8.3f}")
    print(f"{'bm25 (ctl)':<12}{bm_d:>9.3f}{bm_r:>9.3f}{bm_d:>9.3f}{bm_r:>9.3f}{0.0:>+8.3f}{0.0:>+8.3f}")

    print("\nRegime check: small window should be ~0.440/0.472, small scene ~0.300/0.357, "
          "small utterance ~0.268/0.265 (matching the earlier default-ef run). "
          "BM25 is identical by construction (control).")


if __name__ == "__main__":
    main()
