"""
Run all 4 retrieval modes against the sampled eval questions, save results.

For each (question, mode), records the rank at which the ground-truth scene
appears in the top-10 retrieved results (or None if not in top-10).

Run with: python -m src.eval.runner
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.config import DATA_PROCESSED
from src.unified_retriever import retrieve_unified, AVAILABLE_MODES

TOP_K_FOR_EVAL = 10
MAX_WORKERS = 8


def find_rank(results, target_scene_id: str) -> int | None:
    """Return 1-indexed rank of target_scene_id in results, or None if absent."""
    for i, r in enumerate(results, start=1):
        # scene_id is the chunk id; for scene chunks this IS the scene_id.
        # For naive chunks, we need to check metadata.started_in_scene.
        if r.scene_id == target_scene_id:
            return i
        if r.metadata.get("started_in_scene") == target_scene_id:
            return i
    return None


def evaluate_one(question: dict) -> dict:
    """Run all modes on one question, return a result row."""
    target = question["ground_truth_scene_id"]
    row = {
        "qa_id": question["qa_id"],
        "question": question["question"],
        "ground_truth_scene_id": target,
        "is_paraphrased": question["is_paraphrased"],
        "ranks": {},  # mode -> rank or None
    }
    for mode in AVAILABLE_MODES:
        try:
            results = retrieve_unified(question["question"], mode=mode, top_k=TOP_K_FOR_EVAL)
            row["ranks"][mode] = find_rank(results, target)
        except Exception as e:
            row["ranks"][mode] = f"ERROR: {type(e).__name__}: {e}"
    return row


def load_already_done(path) -> set[str]:
    """Find qa_ids already in the output file (for resumability)."""
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


def main():
    sample_path = DATA_PROCESSED / "eval_sample.jsonl"
    results_path = DATA_PROCESSED / "eval_results.jsonl"

    with open(sample_path) as f:
        questions = [json.loads(line) for line in f]

    done = load_already_done(results_path)
    remaining = [q for q in questions if q["qa_id"] not in done]
    print(f"Total: {len(questions):,}  |  Already done: {len(done):,}  |  Remaining: {len(remaining):,}")
    if not remaining:
        print("Nothing to do.")
        return

    # Append mode - we write as we go for resilience.
    with open(results_path, "a") as out, ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_one, q): q for q in remaining}
        for future in tqdm(as_completed(futures), total=len(remaining), desc="Evaluating"):
            row = future.result()
            out.write(json.dumps(row) + "\n")
            out.flush()  # so partial results survive a crash

    print(f"\nDone. Results in {results_path}")


if __name__ == "__main__":
    main()