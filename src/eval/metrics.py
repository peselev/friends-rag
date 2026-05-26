"""
Compute retrieval metrics from eval_results.jsonl.

Outputs:
    - Pretty-printed table to terminal
    - Markdown table to data/processed/eval_metrics.md (for the writeup)

Run with: python -m src.eval.metrics
"""
import json
from pathlib import Path

from src.config import DATA_PROCESSED
from src.unified_retriever import AVAILABLE_MODES

K_VALUES = [1, 3, 5, 10]


def load_results() -> list[dict]:
    path = DATA_PROCESSED / "eval_results.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def is_hit_at_k(rank, k: int) -> bool:
    """True if rank is a valid integer ≤ k (rank None or error string → miss)."""
    return isinstance(rank, int) and rank <= k


def reciprocal_rank(rank) -> float:
    """1/rank if hit, 0 if missed or errored."""
    if isinstance(rank, int) and rank > 0:
        return 1.0 / rank
    return 0.0


def compute_metrics(rows: list[dict]) -> dict:
    """
    For each mode, compute recall@k for each k in K_VALUES, plus MRR.
    Returns: {mode: {"recall@1": .., "recall@3": .., ..., "mrr": ..}}
    """
    if not rows:
        return {}

    metrics = {}
    for mode in AVAILABLE_MODES:
        ranks = [row["ranks"].get(mode) for row in rows]
        n = len(ranks)

        mode_metrics = {}
        for k in K_VALUES:
            hits = sum(1 for r in ranks if is_hit_at_k(r, k))
            mode_metrics[f"recall@{k}"] = hits / n

        mode_metrics["mrr"] = sum(reciprocal_rank(r) for r in ranks) / n
        metrics[mode] = mode_metrics

    return metrics


def stratify(rows: list[dict]) -> dict:
    """Split rows into all / originals / paraphrased."""
    return {
        "all": rows,
        "originals": [r for r in rows if not r["is_paraphrased"]],
        "paraphrased": [r for r in rows if r["is_paraphrased"]],
    }


def format_table(metrics_by_stratum: dict) -> str:
    """Build a markdown table comparing modes across strata."""
    lines = []

    for stratum_name, metrics in metrics_by_stratum.items():
        if not metrics:
            continue
        lines.append(f"\n### {stratum_name.title()}\n")
        header = "| Mode | " + " | ".join(f"R@{k}" for k in K_VALUES) + " | MRR |"
        sep = "|" + "|".join(["---"] * (len(K_VALUES) + 2)) + "|"
        lines.append(header)
        lines.append(sep)
        for mode in AVAILABLE_MODES:
            m = metrics[mode]
            row = (
                f"| {mode} | "
                + " | ".join(f"{m[f'recall@{k}']:.3f}" for k in K_VALUES)
                + f" | {m['mrr']:.3f} |"
            )
            lines.append(row)
    return "\n".join(lines)


def main():
    print("Loading eval results...")
    rows = load_results()
    print(f"  Loaded {len(rows):,} result rows\n")

    strata = stratify(rows)
    print(
        f"Stratum counts: "
        f"all={len(strata['all'])}, "
        f"originals={len(strata['originals'])}, "
        f"paraphrased={len(strata['paraphrased'])}"
    )

    metrics_by_stratum = {name: compute_metrics(rs) for name, rs in strata.items()}

    table = format_table(metrics_by_stratum)
    print(table)

    out_path = DATA_PROCESSED / "eval_metrics.md"
    with open(out_path, "w") as f:
        f.write("# Friends RAG — Retrieval Evaluation Results\n")
        f.write(f"\nSample size: {len(rows)} questions "
                f"(originals: {len(strata['originals'])}, "
                f"paraphrased: {len(strata['paraphrased'])})\n")
        f.write(table)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()