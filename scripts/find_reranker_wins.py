"""
Mine existing eval results to find questions where the reranker helped most
(and where it hurt), using retrieval rank as the signal. No API calls.

A "reranker win" = reranked mode ranked the ground-truth scene higher
(smaller rank) than the fast mode. A "reranker loss" = the opposite (e.g. PIVOT).

Usage:
    python -m scripts.find_reranker_wins
    python -m scripts.find_reranker_wins --results data/processed/eval_results.jsonl
"""
import argparse
import json
from pathlib import Path

from src.config import DATA_PROCESSED

FAST = "hybrid_window_bm25"
RERANKED = "hybrid_window_bm25_reranked_bge"

# Treat a missing/None rank as "not found" = worse than any real rank.
NOT_FOUND = 999


def rank_val(ranks: dict, mode: str) -> int:
    v = ranks.get(mode)
    if v is None or isinstance(v, str):  # None or "ERROR: ..."
        return NOT_FOUND
    return int(v)


def main(results_path: Path):
    rows = []
    with open(results_path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    wins, losses, ties = [], [], []
    for row in rows:
        ranks = row.get("ranks", {})
        if FAST not in ranks or RERANKED not in ranks:
            continue
        fast_r = rank_val(ranks, FAST)
        rerank_r = rank_val(ranks, RERANKED)

        record = {
            "question": row["question"],
            "gt_scene": row["ground_truth_scene_id"],
            "is_paraphrased": row.get("is_paraphrased", False),
            "fast_rank": fast_r,
            "reranked_rank": rerank_r,
            "improvement": fast_r - rerank_r,  # positive = reranker better
        }
        if rerank_r < fast_r:
            wins.append(record)
        elif rerank_r > fast_r:
            losses.append(record)
        else:
            ties.append(record)

    # Sort wins by biggest improvement; prefer cases where reranked is top-1 or top-3
    wins.sort(key=lambda r: (r["reranked_rank"], -r["improvement"]))
    losses.sort(key=lambda r: (r["fast_rank"], -(- r["improvement"])))

    print(f"Total scored questions: {len(wins) + len(losses) + len(ties)}")
    print(f"  Reranker WINS:   {len(wins)}")
    print(f"  Reranker LOSSES: {len(losses)}")
    print(f"  Ties:            {len(ties)}")

    print("\n" + "=" * 80)
    print("TOP RERANKER WINS (reranked found it high, fast found it low/not at all)")
    print("Best demo candidates - reranked ranks it #1-3, fast misses or buries it")
    print("=" * 80)
    shown = 0
    for r in wins:
        # Strong demo candidate: reranked top-3, fast much worse
        if r["reranked_rank"] <= 3 and r["fast_rank"] >= 5:
            fr = "not found" if r["fast_rank"] >= NOT_FOUND else f"#{r['fast_rank']}"
            print(f"\n  Q: {r['question']}")
            print(f"     reranked: #{r['reranked_rank']}  |  fast: {fr}  "
                  f"|  {'paraphrased' if r['is_paraphrased'] else 'original'}")
            print(f"     ground-truth scene: {r['gt_scene']}")
            shown += 1
        if shown >= 25:
            break

    print("\n" + "=" * 80)
    print("TOP RERANKER LOSSES (fast found it high, reranked buried it - e.g. PIVOT)")
    print("Avoid these as demo questions")
    print("=" * 80)
    shown = 0
    for r in losses:
        if r["fast_rank"] <= 3 and r["reranked_rank"] >= 5:
            rr = "not found" if r["reranked_rank"] >= NOT_FOUND else f"#{r['reranked_rank']}"
            print(f"\n  Q: {r['question']}")
            print(f"     fast: #{r['fast_rank']}  |  reranked: {rr}  "
                  f"|  {'paraphrased' if r['is_paraphrased'] else 'original'}")
            shown += 1
        if shown >= 15:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=DATA_PROCESSED / "eval_results.jsonl")
    args = parser.parse_args()
    main(args.results)