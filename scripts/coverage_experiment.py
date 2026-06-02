"""
scripts/coverage_experiment.py

Union-coverage experiment.

PURPOSE
-------
Measure the retrieval *ceiling* of a pool of complementary retrievers, to bound
how much any reranker could recover. A reranker can only ever promote a scene
that some first-stage retriever already surfaced; it cannot conjure a scene that
no retriever returned. So the union of the pool's top-k lists is the hard upper
bound on recall@5 for a retrieve-then-rerank design.

We answer four questions, stratified across the three question sets:
  1. UNION coverage  - if we pool everyone's top-k scenes, is the gold scene in?
  2. CONSENSUS        - how many retrievers find the gold scene (1 vs all 4)?
  3. MARGINAL value   - which scenes does ONLY one retriever find (unique lift)?
  4. REDUNDANCY       - do the vector modes return the same scenes or different?

UNIT OF MEASUREMENT
-------------------
Everything is SCENE-LEVEL. Each retriever's raw chunks are deduplicated to their
parent scene_id (first occurrence wins), and k counts *unique scenes*. This is
the correct unit because (a) union is a set operation over scenes and (b) the
shipped hybrid hands the reranker deduped scenes, so the ceiling must be measured
in the same currency.

NOTE vs the writeup: for window/utterance modes, scene-rank reads slightly higher
than the chunk-rank numbers in the writeup (a gold scene at chunk #8 but unique
scene #3 scores as rank 3 here). For bm25 and scene-vector the two are identical.

COST
----
Retrieval is the only paid step (OpenAI query embeddings), and it is cached. The
script pre-warms the cache with ONE batched embedding pass over all unique
questions, then every vector query is a free cache hit. Re-running with
--analyze-only touches no API at all.

USAGE
-----
    python -m scripts.coverage_experiment                 # retrieve (if needed) + analyze
    python -m scripts.coverage_experiment --force         # recompute all retrieval
    python -m scripts.coverage_experiment --analyze-only  # re-analyze stored results, no API
    python -m scripts.coverage_experiment --no-prewarm    # skip the batch prewarm pass
"""
import argparse
import json
import itertools
from pathlib import Path

from tqdm import tqdm

from src.config import (
    DATA_PROCESSED,
    SCENE_COLLECTION,
    WINDOW_NOHEADER_COLLECTION,
    UTTERANCE_NOHEADER_COLLECTION,
)
from src.bm25_retriever import retrieve_bm25
from src.retriever import retrieve_from_collection

# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #

# The pool of complementary retrievers. All NO-HEADER where a header variant
# exists (header modes are disabled project-wide). Scene + window cover
# different granularities; bm25 covers exact rare tokens; utterance is the
# finest grain. Together they are the diverse pool whose union we measure.
POOL = [
    "bm25",
    "vector",                       # scene-level
    "vector_window_noheader",       # 3-utterance window
    "vector_utterance_noheader",    # single utterance
]

# Human labels for the report.
MODE_LABELS = {
    "bm25": "BM25 (keyword)",
    "vector": "Vector (scene)",
    "vector_window_noheader": "Vector (window)",
    "vector_utterance_noheader": "Vector (utterance)",
}

K_VALUES = [1, 5, 10, 20, 50]
K_MAX = max(K_VALUES)  # how many unique scenes to store per mode per question

# How many raw chunks to fetch before deduping to scenes. 1:1 modes need only
# K_MAX; finer modes emit many chunks per scene, so oversample generously to be
# sure we can reach K_MAX *unique* scenes.
FETCH_DEPTH = {
    "bm25": K_MAX,
    "vector": K_MAX,
    "vector_window_noheader": 600,
    "vector_utterance_noheader": 600,
}

# Sub-pools to report for the union table (each is an ordered subset of POOL).
SUB_POOLS = {
    "bm25 only (baseline)": ["bm25"],
    "bm25 + window (shipped pair)": ["bm25", "vector_window_noheader"],
    "bm25 + window + scene": ["bm25", "vector_window_noheader", "vector"],
    "FULL pool (all 4)": POOL,
}

DEFAULT_MAIN = DATA_PROCESSED / "eval_sample.jsonl"            # 800: direct + reworded
DEFAULT_LEXICAL = DATA_PROCESSED / "eval_lexical_paraphrases.jsonl"  # 200: heavily reworded
DEFAULT_RESULTS = DATA_PROCESSED / "coverage_results.jsonl"
REPORT_PATH = DATA_PROCESSED / "coverage_report.md"


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def _dedupe_to_scenes(results, limit: int) -> list[str]:
    """Collapse a chunk result list to an ordered list of unique parent scene_ids."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for r in results:
        sid = r.metadata.get("scene_id", r.scene_id)
        if sid not in seen_set:
            seen_set.add(sid)
            seen.append(sid)
            if len(seen) >= limit:
                break
    return seen


def ranked_scenes(query: str, mode: str, limit: int = K_MAX) -> list[str]:
    """Return up to `limit` unique parent scene_ids, in retrieval order, for one mode."""
    fetch = FETCH_DEPTH[mode]
    if mode == "bm25":
        results = retrieve_bm25(query, top_k=fetch)
    elif mode == "vector":
        results = retrieve_from_collection(query, fetch, SCENE_COLLECTION)
    elif mode == "vector_window_noheader":
        results = retrieve_from_collection(query, fetch, WINDOW_NOHEADER_COLLECTION)
    elif mode == "vector_utterance_noheader":
        results = retrieve_from_collection(query, fetch, UTTERANCE_NOHEADER_COLLECTION)
    else:
        raise ValueError(f"Unknown mode in POOL: {mode!r}")
    return _dedupe_to_scenes(results, limit)


# --------------------------------------------------------------------------- #
# Question loading
# --------------------------------------------------------------------------- #

def load_questions(main_path: Path, lexical_path: Path) -> list[dict]:
    """Load all questions, tagging each with its stratum.

    direct / reworded come from the main sample (split by is_paraphrased);
    heavily_reworded comes from the lexical-paraphrase file.
    """
    questions = []

    with open(main_path) as f:
        for line in f:
            q = json.loads(line)
            q["stratum"] = "reworded" if q.get("is_paraphrased") else "direct"
            questions.append(q)

    if lexical_path.exists():
        with open(lexical_path) as f:
            for line in f:
                q = json.loads(line)
                q["stratum"] = "heavily_reworded"
                questions.append(q)
    else:
        print(f"  (lexical file {lexical_path.name} not found - skipping heavily_reworded)")

    return questions


# --------------------------------------------------------------------------- #
# Cache prewarm
# --------------------------------------------------------------------------- #

def prewarm_cache(questions: list[dict], batch_size: int = 256) -> None:
    """Embed every not-yet-cached question in batched API calls, then persist.

    Turns ~1,000 sequential single embeds into a handful of batch calls, and
    means all subsequent vector retrieval is a free cache hit.
    """
    from src.embedder import embed_texts
    from src.embed_cache import _cache

    # Unique, order-preserving, only those not already cached.
    todo = []
    seen = set()
    for q in questions:
        text = q["question"]
        if text in seen:
            continue
        seen.add(text)
        if _cache.get(text) is None:
            todo.append(text)

    if not todo:
        print("  Cache already warm - nothing to embed.")
        return

    print(f"  Embedding {len(todo):,} uncached questions in batches of {batch_size}...")
    for i in tqdm(range(0, len(todo), batch_size), desc="  Prewarm"):
        batch = todo[i:i + batch_size]
        vectors = embed_texts(batch)
        for text, vec in zip(batch, vectors):
            _cache.set(text, vec)
    _cache.save()
    print(f"  Cache now holds {len(_cache):,} query embeddings.")


# --------------------------------------------------------------------------- #
# Phase 1: retrieve and store scene lists
# --------------------------------------------------------------------------- #

def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
                rows[row["qa_id"]] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return rows


def atomic_write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)


def retrieve_all(questions: list[dict], results_path: Path, force: bool) -> dict[str, dict]:
    """Run each pool mode for each question, storing top-K_MAX scene lists.

    Resumable: a question is skipped only if every POOL mode is already present
    in its stored scene_lists (unless --force).
    """
    existing = load_existing(results_path)
    results_by_id = dict(existing)

    def needs_work(q) -> list[str]:
        if force:
            return list(POOL)
        have = set(existing.get(q["qa_id"], {}).get("scene_lists", {}).keys())
        return [m for m in POOL if m not in have]

    work = [(q, needs_work(q)) for q in questions]
    work = [(q, modes) for q, modes in work if modes]

    print(f"Questions needing retrieval: {len(work):,} / {len(questions):,}")
    if not work:
        print("  All retrieval already cached in results file.")
        return results_by_id

    processed = 0
    for q, modes in tqdm(work, desc="Retrieving"):
        qa_id = q["qa_id"]
        scene_lists = results_by_id.get(qa_id, {}).get("scene_lists", {})
        for mode in modes:
            try:
                scene_lists[mode] = ranked_scenes(q["question"], mode, limit=K_MAX)
            except Exception as e:
                scene_lists[mode] = f"ERROR: {type(e).__name__}: {e}"

        results_by_id[qa_id] = {
            "qa_id": qa_id,
            "question": q["question"],
            "ground_truth_scene_id": q["ground_truth_scene_id"],
            "stratum": q["stratum"],
            "scene_lists": scene_lists,
        }
        processed += 1
        if processed % 100 == 0:
            atomic_write(results_path, list(results_by_id.values()))

    atomic_write(results_path, list(results_by_id.values()))
    print(f"  Wrote {results_path}")
    return results_by_id


# --------------------------------------------------------------------------- #
# Phase 2: analysis (pure, reads stored scene lists)
# --------------------------------------------------------------------------- #

def rank_of(row: dict, mode: str) -> int | None:
    """1-indexed scene-rank of the gold scene in `mode`'s list, or None."""
    gold = row["ground_truth_scene_id"]
    lst = row["scene_lists"].get(mode)
    if not isinstance(lst, list):
        return None
    for i, sid in enumerate(lst, start=1):
        if sid == gold:
            return i
    return None


def hit(row: dict, mode: str, k: int) -> bool:
    r = rank_of(row, mode)
    return r is not None and r <= k


def topk_scene_set(row: dict, mode: str, k: int) -> set[str]:
    lst = row["scene_lists"].get(mode)
    if not isinstance(lst, list):
        return set()
    return set(lst[:k])


STRATA_ORDER = ["all", "direct", "reworded", "heavily_reworded"]


def stratify(rows: list[dict]) -> dict[str, list[dict]]:
    out = {"all": rows}
    for s in ["direct", "reworded", "heavily_reworded"]:
        subset = [r for r in rows if r["stratum"] == s]
        if subset:
            out[s] = subset
    return out


def per_mode_recall(rows: list[dict]) -> dict:
    """{mode: {k: recall}} scene-level recall@k for each pool mode."""
    n = len(rows)
    return {
        mode: {k: sum(hit(r, mode, k) for r in rows) / n for k in K_VALUES}
        for mode in POOL
    }


def union_recall(rows: list[dict], pool: list[str]) -> dict:
    """{k: recall} where a question counts if ANY mode in `pool` hits@k."""
    n = len(rows)
    return {
        k: sum(any(hit(r, m, k) for m in pool) for r in rows) / n
        for k in K_VALUES
    }


def consensus_histogram(rows: list[dict], k: int) -> dict[int, int]:
    """{n_retrievers_hitting: count} over rows, for the full pool at depth k."""
    hist = {i: 0 for i in range(len(POOL) + 1)}
    for r in rows:
        c = sum(hit(r, m, k) for m in POOL)
        hist[c] += 1
    return hist


def marginal_contribution(rows: list[dict], k: int) -> dict[str, int]:
    """{mode: count} of questions where ONLY that mode hits@k (unique lift)."""
    out = {}
    for mode in POOL:
        others = [m for m in POOL if m != mode]
        out[mode] = sum(
            hit(r, mode, k) and not any(hit(r, o, k) for o in others)
            for r in rows
        )
    return out


def gold_cohit_jaccard(rows: list[dict], k: int) -> dict[tuple, float]:
    """For each mode pair: |both hit| / |either hits| at depth k.

    High value -> the two retrievers tend to find gold on the SAME questions
    (redundant on correctness). Low value -> complementary.
    """
    out = {}
    for a, b in itertools.combinations(POOL, 2):
        both = either = 0
        for r in rows:
            ha, hb = hit(r, a, k), hit(r, b, k)
            if ha or hb:
                either += 1
                if ha and hb:
                    both += 1
        out[(a, b)] = (both / either) if either else 0.0
    return out


def set_overlap_jaccard(rows: list[dict], k: int) -> dict[tuple, float]:
    """Mean Jaccard of the top-k *retrieved scene sets* between mode pairs.

    Measures redundancy of what each retriever returns, regardless of whether
    gold is among them. High -> the two return largely the same scenes.
    """
    out = {}
    for a, b in itertools.combinations(POOL, 2):
        total = 0.0
        for r in rows:
            sa, sb = topk_scene_set(r, a, k), topk_scene_set(r, b, k)
            union = sa | sb
            total += (len(sa & sb) / len(union)) if union else 0.0
        out[(a, b)] = total / len(rows)
    return out


# --------------------------------------------------------------------------- #
# Report formatting
# --------------------------------------------------------------------------- #

def _kcols() -> str:
    return " | ".join(f"@{k}" for k in K_VALUES)


def build_report(rows: list[dict]) -> str:
    L = []
    strata = stratify(rows)

    L.append("# Friends RAG - Union Coverage Experiment\n")
    L.append(f"Total questions: {len(rows)}  |  "
             + ", ".join(f"{s}={len(strata[s])}" for s in STRATA_ORDER if s in strata))
    L.append("\nUnit: scene-level. `k` counts unique parent scenes per retriever "
             "(chunks deduped to scenes). For window/utterance this reads slightly "
             "higher than the chunk-rank numbers in the writeup; for bm25 and "
             "scene-vector it is identical.\n")
    L.append(f"Pool: {', '.join(MODE_LABELS[m] for m in POOL)}\n")

    # Flag any retrieval errors.
    err = [(r["qa_id"], m) for r in rows for m in POOL
           if not isinstance(r["scene_lists"].get(m), list)]
    if err:
        L.append(f"\n⚠️ {len(err)} mode-results errored or are missing "
                 f"(e.g. {err[:3]}). They count as misses below.\n")

    for s in STRATA_ORDER:
        if s not in strata:
            continue
        rs = strata[s]
        L.append(f"\n\n## {s.upper()}  (n={len(rs)})\n")

        # 1. Per-mode recall
        L.append("### Per-mode recall@k (scene-level)\n")
        L.append(f"| Mode | {_kcols()} |")
        L.append("|" + "---|" * (len(K_VALUES) + 1))
        pmr = per_mode_recall(rs)
        for m in POOL:
            cells = " | ".join(f"{pmr[m][k]:.3f}" for k in K_VALUES)
            L.append(f"| {MODE_LABELS[m]} | {cells} |")

        # 2. Union coverage (the ceiling)
        L.append("\n### Union coverage@k  (ceiling a reranker could reach)\n")
        L.append(f"| Pool | {_kcols()} |")
        L.append("|" + "---|" * (len(K_VALUES) + 1))
        for name, pool in SUB_POOLS.items():
            u = union_recall(rs, pool)
            cells = " | ".join(f"{u[k]:.3f}" for k in K_VALUES)
            L.append(f"| {name} | {cells} |")

        # 3. Consensus at k=10 and k=20
        for k in (10, 20):
            if k not in K_VALUES:
                continue
            L.append(f"\n### Consensus @{k}: how many of the {len(POOL)} retrievers find gold\n")
            hist = consensus_histogram(rs, k)
            tot = len(rs)
            L.append("| # retrievers hitting | questions | share |")
            L.append("|---|---|---|")
            for nh in range(len(POOL) + 1):
                c = hist[nh]
                L.append(f"| {nh} | {c} | {c / tot:.1%} |")

        # 4. Marginal contribution at k=10
        L.append("\n### Marginal contribution @10: questions only this mode finds\n")
        marg = marginal_contribution(rs, 10)
        L.append("| Mode | unique-hit questions |")
        L.append("|---|---|")
        for m in POOL:
            L.append(f"| {MODE_LABELS[m]} | {marg[m]} |")

        # 5. Redundancy: gold co-hit + set overlap at k=10
        L.append("\n### Redundancy @10 (mode pairs)\n")
        L.append("gold co-hit = of questions where either finds gold, share where BOTH do "
                 "(high = redundant on correctness). "
                 "set overlap = mean Jaccard of the top-10 retrieved scene sets "
                 "(high = return the same scenes).\n")
        cohit = gold_cohit_jaccard(rs, 10)
        overlap = set_overlap_jaccard(rs, 10)
        L.append("| Pair | gold co-hit | set overlap |")
        L.append("|---|---|---|")
        for a, b in itertools.combinations(POOL, 2):
            L.append(f"| {MODE_LABELS[a]} + {MODE_LABELS[b]} | "
                     f"{cohit[(a, b)]:.3f} | {overlap[(a, b)]:.3f} |")

    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--questions-lexical", type=Path, default=DEFAULT_LEXICAL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--force", action="store_true",
                        help="Recompute all retrieval, ignoring stored results.")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip retrieval; analyze stored results (no API).")
    parser.add_argument("--no-prewarm", action="store_true",
                        help="Skip the batched embedding prewarm pass.")
    args = parser.parse_args()

    print("Loading questions...")
    questions = load_questions(args.questions_main, args.questions_lexical)
    counts = {}
    for q in questions:
        counts[q["stratum"]] = counts.get(q["stratum"], 0) + 1
    print(f"  Loaded {len(questions):,}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    if args.analyze_only:
        results_by_id = load_existing(args.results)
        if not results_by_id:
            print(f"No stored results at {args.results}. Run without --analyze-only first.")
            return
        rows = list(results_by_id.values())
    else:
        if not args.no_prewarm:
            print("\nPrewarming query-embedding cache...")
            prewarm_cache(questions)
        print("\nRetrieving scene lists...")
        results_by_id = retrieve_all(questions, args.results, args.force)
        rows = list(results_by_id.values())

    print("\nAnalyzing...")
    report = build_report(rows)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"\n\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
