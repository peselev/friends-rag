"""
scripts/fusion_experiment.py

Compare fusion strategies, fully offline.

Six modes: {naive, fixed, IDF} x {1+1, union}.
  1+1   = bm25 + window
  union = bm25 + window + scene + utterance

  naive = equal-weight RRF
  fixed = best of a small static-weight sweep (weights tuned on this set; see caveat)
  IDF   = query-adaptive weight (reproduces src/smart_hybrid: rare-term queries
          lean BM25, common-word queries lean vector), generalized to the union
          by splitting the vector mass across the three granularities by a fixed
          ratio.

Why offline: coverage_results.jsonl already stores each retriever's top-50
deduped SCENE list per question; fusion only recombines those ranked lists.
bm25_index.pkl supplies per-query IDF for the adaptive modes. No retrieval/API.

Metric note: scenes are the unit (lists are already deduped to parent scenes,
upstream, per retriever). We fuse scene lists with RRF, then read recall@k /
MRR off the fused order. recall@5 = quality if fusion is the final stage;
recall@10/20/50 = coverage of the set a reranker would receive at that depth.

Heavily-reworded questions are excluded (out of scope).

USAGE
-----
    python -m scripts.fusion_experiment
    python -m scripts.fusion_experiment --results data/processed/coverage_results.jsonl
"""
import argparse
import json
from pathlib import Path
from itertools import product

import numpy as np

from src.config import DATA_PROCESSED
from src.bm25_retriever import _get_index, _tokenize

# Retriever keys, exactly as stored in coverage_results.jsonl scene_lists.
BM25 = "bm25"
WINDOW = "vector_window_noheader"
SCENE = "vector"
UTT = "vector_utterance_noheader"

POOL_1P1 = [BM25, WINDOW]
POOL_UNION = [BM25, WINDOW, SCENE, UTT]

RRF_K = 60                 # matches production fusion
K_VALUES = [5, 10, 20, 50]
STRATA = ["direct", "reworded"]   # heavy excluded

# --- IDF adaptive weighting (mirrors src/smart_hybrid.py) -------------------
IDF_LOW, IDF_HIGH = 2.0, 5.0
BM25_W_BOUNDS = (0.2, 0.8)
# How the vector mass is split across granularities in the IDF-union mode.
VEC_SPLIT = {WINDOW: 0.60, SCENE: 0.20, UTT: 0.20}

# --- Fixed-weight sweeps ----------------------------------------------------
FIXED_1P1_BM25 = [0.3, 0.4, 0.5, 0.6, 0.7]          # window = 1 - bm25
FIXED_UNION_BM25 = [0.3, 0.4, 0.5]
FIXED_UNION_VECSPLIT = [                              # (window, scene, utt) of vector mass
    (0.60, 0.20, 0.20),
    (0.50, 0.25, 0.25),
    (0.70, 0.15, 0.15),
]


# --------------------------------------------------------------------------- #
# IDF
# --------------------------------------------------------------------------- #

_BM25_INDEX = None

def _bm25():
    global _BM25_INDEX
    if _BM25_INDEX is None:
        bm25, _chunks = _get_index()
        _BM25_INDEX = bm25
    return _BM25_INDEX


def query_avg_idf(question: str) -> float:
    bm25 = _bm25()
    toks = _tokenize(question)
    idfs = [bm25.idf[t] for t in toks if t in bm25.idf]
    return float(np.mean(idfs)) if idfs else 0.0


def bm25_weight_for(question: str) -> float:
    idf = query_avg_idf(question)
    lo, hi = BM25_W_BOUNDS
    if idf <= IDF_LOW:
        return lo
    if idf >= IDF_HIGH:
        return hi
    return lo + (idf - IDF_LOW) / (IDF_HIGH - IDF_LOW) * (hi - lo)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #

def weighted_rrf(scene_lists: dict, weights: dict) -> list[str]:
    """Weighted RRF over per-retriever scene lists -> fused scene ranking."""
    scores: dict[str, float] = {}
    for r, w in weights.items():
        if w <= 0:
            continue
        lst = scene_lists.get(r)
        if not isinstance(lst, list):
            continue
        for rank, sid in enumerate(lst, start=1):
            scores[sid] = scores.get(sid, 0.0) + w / (RRF_K + rank)
    return sorted(scores, key=lambda s: scores[s], reverse=True)


def weights_for_mode(mode: str, pool: list[str], question: str, fixed_cfg=None) -> dict:
    if mode == "naive":
        return {r: 1.0 for r in pool}

    if mode == "idf":
        bw = bm25_weight_for(question)
        vw = 1.0 - bw
        if pool == POOL_1P1:
            return {BM25: bw, WINDOW: vw}
        return {BM25: bw, WINDOW: vw * VEC_SPLIT[WINDOW],
                SCENE: vw * VEC_SPLIT[SCENE], UTT: vw * VEC_SPLIT[UTT]}

    if mode == "fixed":
        if pool == POOL_1P1:
            bw = fixed_cfg
            return {BM25: bw, WINDOW: 1.0 - bw}
        bw, (ww, sw, uw) = fixed_cfg
        vw = 1.0 - bw
        return {BM25: bw, WINDOW: vw * ww, SCENE: vw * sw, UTT: vw * uw}

    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def gold_rank(fused: list[str], gold: str) -> int | None:
    for i, sid in enumerate(fused, start=1):
        if sid == gold:
            return i
    return None


def score_mode(rows, pool, mode, fixed_cfg=None) -> dict:
    """Return {('recall', k): val, ('mrr',): val} over the given rows."""
    n = len(rows)
    hits = {k: 0 for k in K_VALUES}
    mrr = 0.0
    for row in rows:
        weights = weights_for_mode(mode, pool, row["question"], fixed_cfg)
        fused = weighted_rrf(row["scene_lists"], weights)
        r = gold_rank(fused, row["ground_truth_scene_id"])
        if r is not None:
            mrr += 1.0 / r
            for k in K_VALUES:
                if r <= k:
                    hits[k] += 1
    out = {("recall", k): hits[k] / n for k in K_VALUES}
    out[("mrr",)] = mrr / n
    return out


def best_fixed(rows_by_stratum, pool):
    """Sweep fixed weights; pick the config with best recall@5 on direct+reworded."""
    natural = rows_by_stratum["direct"] + rows_by_stratum["reworded"]
    configs = (FIXED_1P1_BM25 if pool == POOL_1P1
               else list(product(FIXED_UNION_BM25, FIXED_UNION_VECSPLIT)))
    best_cfg, best_r5 = None, -1.0
    for cfg in configs:
        r5 = score_mode(natural, pool, "fixed", cfg)[("recall", 5)]
        if r5 > best_r5:
            best_r5, best_cfg = r5, cfg
    return best_cfg


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DATA_PROCESSED / "coverage_results.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results)]
    rows = [r for r in rows if r.get("stratum") in STRATA]
    by = {s: [r for r in rows if r["stratum"] == s] for s in STRATA}
    by["natural"] = rows
    print(f"Loaded {len(rows)} questions (direct={len(by['direct'])}, reworded={len(by['reworded'])})\n")

    pools = {"1+1": POOL_1P1, "union": POOL_UNION}
    fixed_winners = {}

    rows_out = []  # (pool_label, mode_label, stratum, scores)
    for pool_label, pool in pools.items():
        fixed_cfg = best_fixed(by, pool)
        fixed_winners[pool_label] = fixed_cfg
        for mode in ["naive", "fixed", "idf"]:
            cfg = fixed_cfg if mode == "fixed" else None
            for stratum in ["direct", "reworded"]:
                scores = score_mode(by[stratum], pool, mode, cfg)
                rows_out.append((pool_label, mode, stratum, scores))

    # ---- report ----
    print("Fixed-weight winners (tuned on direct+reworded recall@5):")
    print(f"  1+1   bm25_weight = {fixed_winners['1+1']}")
    print(f"  union (bm25_w, (win,scene,utt)) = {fixed_winners['union']}\n")

    hdr = f"{'pool':<6} {'mode':<6} {'set':<9}" + "".join(f"{'R@'+str(k):>8}" for k in K_VALUES) + f"{'MRR':>8}"
    print(hdr)
    print("-" * len(hdr))
    last = None
    for pool_label, mode, stratum, sc in rows_out:
        tag = f"{pool_label:<6} {mode:<6}"
        if (pool_label, mode) == last:
            tag = " " * 13
        last = (pool_label, mode)
        cells = "".join(f"{sc[('recall', k)]:>8.3f}" for k in K_VALUES)
        print(f"{tag} {stratum:<9}{cells}{sc[('mrr',)]:>8.3f}")

    print("\nBars to beat: shipped naive 1+1 ~0.515/0.507 (direct/reworded recall@5); "
          "IDF 1+1 ~0.535/0.527. Caveat: fixed weights are tuned on the same set "
          "they're scored on, so treat fixed as an optimistic upper bound.")


if __name__ == "__main__":
    main()
