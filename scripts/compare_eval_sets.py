"""
Side-by-side comparison of retrieval metrics across two eval sets:
the standard Emory sample vs the LLM-generated lexical paraphrases.

Run with: python -m scripts.compare_eval_sets
"""
import json
from pathlib import Path

from src.config import DATA_PROCESSED
from src.unified_retriever import AVAILABLE_MODES
from src.eval.metrics import compute_metrics, K_VALUES

EMORY_FILE = DATA_PROCESSED / "eval_results.jsonl"
LEXICAL_FILE = DATA_PROCESSED / "eval_results_lexical.jsonl"


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    emory_rows = load(EMORY_FILE)
    lexical_rows = load(LEXICAL_FILE)

    # Stratify emory rows
    emory_originals = [r for r in emory_rows if not r["is_paraphrased"]]
    emory_paraphrased = [r for r in emory_rows if r["is_paraphrased"]]

    sets = {
        "Emory originals": emory_originals,
        "Emory paraphrased": emory_paraphrased,
        "LLM lexical paraphrases": lexical_rows,
    }

    print(f"\n{'Mode':<20}", end="")
    for set_name in sets:
        print(f"{set_name:>26}", end="")
    print()
    print("-" * 100)

    for mode in AVAILABLE_MODES:
        # recall@5 row
        print(f"{mode + ' R@5':<20}", end="")
        for set_name, rows in sets.items():
            m = compute_metrics(rows)[mode]
            print(f"{m['recall@5']:>26.3f}", end="")
        print()

    print()
    for mode in AVAILABLE_MODES:
        # MRR row
        print(f"{mode + ' MRR':<20}", end="")
        for set_name, rows in sets.items():
            m = compute_metrics(rows)[mode]
            print(f"{m['mrr']:>26.3f}", end="")
        print()


if __name__ == "__main__":
    main()