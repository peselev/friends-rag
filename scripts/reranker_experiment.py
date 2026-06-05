"""
scripts/reranker_experiment.py

Compare reranking models on top of the shipped naive hybrid (BM25 + window).

Modes:
    none    - no reranker; the fused top-5 (this is the baseline row)
    minilm  - cross-encoder/ms-marco-MiniLM-L-6-v2   (local, sentence-transformers)
    bge     - BAAI/bge-reranker-base                 (local, sentence-transformers)
    cohere  - rerank-v3.5 via Cohere API             (needs COHERE_API_KEY)
    all     - run every mode, print the writeup table

Why offline (mostly): like fusion_experiment.py, candidates are rebuilt from
coverage_results.jsonl, which stores each retriever's top-50 deduped SCENE list
per question. We fuse BM25 + window with equal-weight RRF (k=60) -- the exact
production hybrid_window fusion -- take the top-N as the candidate set, look up
each scene's full text from scenes.jsonl, and rerank (query, scene_text) pairs.
The ONLY external dependency is the reranker model itself: MiniLM/BGE download
from HuggingFace once (then cached), Cohere is an API call.

Candidate depth N (default 20) matches production: retrieve_hybrid_window(top_k=20)
feeds the reranker, which keeps the top 5. recall@5 is "is the gold scene in the
reranked top 5". The rerank ceiling at a given N is the fused recall@N (e.g. for
BM25+window at N=20, ~0.725 direct, from the coverage table).

Heavily-reworded questions are excluded (out of scope), matching fusion_experiment.

USAGE
-----
    # baseline only (no network, runs anywhere)
    python -m scripts.reranker_experiment --mode none

    # one local model (downloads weights from HF on first run)
    python -m scripts.reranker_experiment --mode bge

    # Cohere (set the key first)
    export COHERE_API_KEY=...        # or pass --cohere-key
    python -m scripts.reranker_experiment --mode cohere

    # everything, then the final 4-row table
    python -m scripts.reranker_experiment --mode all

    # quick smoke test on 25 questions per stratum
    python -m scripts.reranker_experiment --mode all --limit 25
"""
import argparse
import json
import os
import time
from pathlib import Path

from src.config import DATA_PROCESSED

# Retriever keys, exactly as stored in coverage_results.jsonl scene_lists.
BM25 = "bm25"
WINDOW = "vector_window_noheader"
POOL = [BM25, WINDOW]            # the shipped naive hybrid

RRF_K = 60                       # matches production fusion
STRATA = ["direct", "reworded"]  # heavy excluded
TOP_K = 5                        # final cut we score recall@5 on

MINILM_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_MODEL = "BAAI/bge-reranker-base"
COHERE_MODEL = "rerank-v3.5"     # current general-purpose Cohere reranker
                                 # (rerank-v4.0-fast / -pro also exist; v3.5 is the
                                 # documented sweet spot for chunk-sized docs)

MODE_LABELS = {
    "none": "Hybrid, no rerank",
    "minilm": "+ MS-MARCO MiniLM",
    "bge": "+ BGE",
    "cohere": "+ Cohere",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_rows(results_path: Path, limit: int | None) -> dict:
    rows = [json.loads(l) for l in open(results_path)]
    rows = [r for r in rows if r.get("stratum") in STRATA]
    by = {}
    for s in STRATA:
        sub = [r for r in rows if r["stratum"] == s]
        by[s] = sub[:limit] if limit else sub
    return by


def build_scene_text_index() -> dict:
    index = {}
    with open(DATA_PROCESSED / "scenes.jsonl") as f:
        for line in f:
            scene = json.loads(line)
            index[scene["scene_id"]] = scene["full_text"]
    return index


# --------------------------------------------------------------------------- #
# Candidate generation: reproduce production hybrid_window fusion from cache
# --------------------------------------------------------------------------- #

def fused_candidates(scene_lists: dict, depth: int) -> list[str]:
    """Equal-weight RRF over BM25 + window scene lists -> top `depth` scene ids."""
    scores: dict[str, float] = {}
    for r in POOL:
        lst = scene_lists.get(r) or []
        for rank, sid in enumerate(lst, start=1):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores, key=lambda s: scores[s], reverse=True)
    return ordered[:depth]


# --------------------------------------------------------------------------- #
# Rerankers. Each takes (query, [scene_texts]) and returns reordered indices.
# --------------------------------------------------------------------------- #

_ce_cache: dict = {}


def _crossencoder_order(model_name: str):
    """Return a reranker fn for a local sentence-transformers CrossEncoder."""
    from sentence_transformers import CrossEncoder
    if model_name not in _ce_cache:
        print(f"Loading cross-encoder {model_name} (first run downloads from HF)...")
        _ce_cache[model_name] = CrossEncoder(model_name)
    model = _ce_cache[model_name]

    def order(query: str, texts: list[str]) -> list[int]:
        scores = model.predict([(query, t) for t in texts])
        return sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)

    return order


def _cohere_order(api_key: str, sleep: float):
    """Return a reranker fn backed by Cohere rerank-v3.5 (ClientV2)."""
    import cohere
    client = cohere.ClientV2(api_key)

    def order(query: str, texts: list[str]) -> list[int]:
        # one call per query; retry a couple times on transient/rate errors
        for attempt in range(4):
            try:
                resp = client.rerank(
                    model=COHERE_MODEL,
                    query=query,
                    documents=texts,
                    top_n=len(texts),          # rank all, we cut to TOP_K ourselves
                )
                if sleep:
                    time.sleep(sleep)
                return [r.index for r in resp.results]
            except Exception as e:               # noqa: BLE001
                wait = 2 ** attempt
                print(f"  Cohere error ({type(e).__name__}): retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError("Cohere rerank failed after retries")

    return order


def get_reranker(mode: str, args):
    if mode == "minilm":
        return _crossencoder_order(MINILM_MODEL)
    if mode == "bge":
        return _crossencoder_order(BGE_MODEL)
    if mode == "cohere":
        key = args.cohere_key or os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
        if not key:
            raise SystemExit("Set COHERE_API_KEY (or pass --cohere-key) for --mode cohere")
        return _cohere_order(key, args.sleep)
    raise ValueError(mode)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def gold_rank(order: list[str], gold: str) -> int | None:
    for i, sid in enumerate(order, start=1):
        if sid == gold:
            return i
    return None


def score_mode(rows, scene_text, mode, reranker, depth) -> dict:
    """recall@5 and MRR for one mode over one stratum's rows."""
    n = len(rows)
    hits5 = 0
    mrr = 0.0
    for row in rows:
        gold = row["ground_truth_scene_id"]
        cands = fused_candidates(row["scene_lists"], depth)

        if mode == "none":
            final = cands[:TOP_K]
            ranked = cands                      # MRR over the fused order
        else:
            texts = [scene_text.get(sid, "") for sid in cands]
            order = reranker(row["question"], texts)
            ranked = [cands[i] for i in order]  # full reranked candidate list
            final = ranked[:TOP_K]

        if gold in final:
            hits5 += 1
        r = gold_rank(ranked, gold)
        if r is not None:
            mrr += 1.0 / r
    return {"recall@5": hits5 / n, "mrr": mrr / n}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODE_LABELS) + ["all"], default="all")
    ap.add_argument("--results", type=Path,
                    default=DATA_PROCESSED / "coverage_results.jsonl")
    ap.add_argument("--candidate-depth", type=int, default=20,
                    help="N candidates fed to the reranker (production: 20)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap questions per stratum (smoke test)")
    ap.add_argument("--cohere-key", type=str, default=None)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between Cohere calls (raise if rate-limited)")
    args = ap.parse_args()

    by = load_rows(args.results, args.limit)
    scene_text = build_scene_text_index()
    print(f"Loaded direct={len(by['direct'])}, reworded={len(by['reworded'])} "
          f"| candidate depth N={args.candidate_depth}\n")

    modes = list(MODE_LABELS) if args.mode == "all" else [args.mode]

    # results[mode] = {"direct": {...}, "reworded": {...}}
    results = {}
    for mode in modes:
        reranker = None if mode == "none" else get_reranker(mode, args)
        results[mode] = {}
        for stratum in STRATA:
            print(f"Scoring {MODE_LABELS[mode]:<22} [{stratum}] ...", flush=True)
            results[mode][stratum] = score_mode(
                by[stratum], scene_text, mode, reranker, args.candidate_depth
            )

    # ---- writeup table ----
    lines = []
    lines.append("| Retrieval mode | Direct (recall@5) | Reworded (recall@5) | MRR (Direct) |")
    lines.append("|---|---|---|---|")
    for mode in modes:
        d = results[mode]["direct"]
        w = results[mode]["reworded"]
        lines.append(f"| {MODE_LABELS[mode]} | {d['recall@5']:.3f} | "
                     f"{w['recall@5']:.3f} | {d['mrr']:.3f} |")
    table = "\n".join(lines)

    print("\n" + table + "\n")

    out = DATA_PROCESSED / f"reranker_experiment_n{args.candidate_depth}.md"
    with open(out, "w") as f:
        f.write("# Friends RAG - Reranker experiment\n\n")
        f.write(f"Candidate depth N={args.candidate_depth}, "
                f"direct={len(by['direct'])}, reworded={len(by['reworded'])} "
                f"(heavily_reworded excluded).\n\n")
        f.write(table + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
