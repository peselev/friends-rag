"""
Run all 4 retrieval modes against eval questions, save results.

Examples:
    # Default: standard eval_sample (existing behavior)
    python -m src.eval.runner

    # Against LLM-generated lexical paraphrases
    python -m src.eval.runner \\
        --questions data/processed/eval_lexical_paraphrases.jsonl \\
        --results data/processed/eval_results_lexical.jsonl
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.config import DATA_PROCESSED
from src.unified_retriever import retrieve_unified, AVAILABLE_MODES

TOP_K_FOR_EVAL = 10
MAX_WORKERS = 8

DEFAULT_QUESTIONS = DATA_PROCESSED / "eval_sample.jsonl"
DEFAULT_RESULTS = DATA_PROCESSED / "eval_results.jsonl"


def find_rank(results, target_scene_id: str) -> int | None:
    """Return 1-indexed rank of target_scene_id in results, or None if absent."""
    for i, r in enumerate(results, start=1):
        if r.scene_id == target_scene_id:
            return i
        if r.metadata.get("started_in_scene") == target_scene_id:
            return i
    return None


def evaluate_one(question: dict) -> dict:
    target = question["ground_truth_scene_id"]
    row = {
        "qa_id": question["qa_id"],
        "question": question["question"],
        "ground_truth_scene_id": target,
        "is_paraphrased": question["is_paraphrased"],
        "ranks": {},
    }
    for mode in AVAILABLE_MODES:
        try:
            results = retrieve_unified(question["question"], mode=mode, top_k=TOP_K_FOR_EVAL)
            row["ranks"][mode] = find_rank(results, target)
        except Exception as e:
            row["ranks"][mode] = f"ERROR: {type(e).__name__}: {e}"
    return row


def load_already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["qa_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main(questions_path: Path, results_path: Path):
    with open(questions_path) as f:
        questions = [json.loads(line) for line in f]

    done = load_already_done(results_path)
    remaining = [q for q in questions if q["qa_id"] not in done]
    print(
        f"Questions file: {questions_path.name}\n"
        f"Results file:   {results_path.name}\n"
        f"Total: {len(questions):,}  |  Already done: {len(done):,}  |  Remaining: {len(remaining):,}"
    )
    if not remaining:
        print("Nothing to do.")
        return

    with open(results_path, "a") as out, ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_one, q): q for q in remaining}
        for future in tqdm(as_completed(futures), total=len(remaining), desc="Evaluating"):
            row = future.result()
            out.write(json.dumps(row) + "\n")
            out.flush()

    print(f"\nDone. Results in {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS,
                        help="Path to questions JSONL")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help="Path to results JSONL (created if missing, resumable)")
    args = parser.parse_args()
    main(args.questions, args.results)