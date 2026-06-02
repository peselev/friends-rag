"""
Run retrieval modes against eval questions, save results incrementally.

Only computes modes that are MISSING for each question, so adding a new mode
to AVAILABLE_MODES and re-running will fill in just the new mode without
recomputing existing ones.

Examples:
    python -m src.eval.runner
    python -m src.eval.runner --questions data/processed/eval_lexical_paraphrases.jsonl \\
                              --results data/processed/eval_results_lexical.jsonl
    python -m src.eval.runner --force vector_window_noheader   # recompute one mode
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from src.config import DATA_PROCESSED
from src.unified_retriever import retrieve_unified, AVAILABLE_MODES

TOP_K_FOR_EVAL = 10          # how many unique SCENES we want to rank within
RETRIEVE_DEPTH = 80          # chunks to fetch before scene-dedup; generous so
                             # finer modes still yield TOP_K_FOR_EVAL scenes
MAX_WORKERS = 4

DEFAULT_QUESTIONS = DATA_PROCESSED / "eval_sample.jsonl"
DEFAULT_RESULTS = DATA_PROCESSED / "eval_results.jsonl"


def _parent_scene_id(r) -> str:
    """Canonical scene id for a result. Matches the dedup key used by the live
    pipeline (hybrid_window._parent_scene_id) and the coverage experiment, so
    'scene' means the same thing everywhere: the unit the bot actually serves."""
    return r.metadata.get("scene_id", r.scene_id)


def _matches_target(r, target_scene_id: str) -> bool:
    """Does this result belong to the gold scene? Keeps the original three-way
    match so boundary-crossing naive chunks (started_in_scene) still count."""
    return (
        r.scene_id == target_scene_id
        or r.metadata.get("scene_id") == target_scene_id
        or r.metadata.get("started_in_scene") == target_scene_id
    )


def find_rank(results, target_scene_id: str) -> int | None:
    """Return the 1-indexed rank of target_scene_id among UNIQUE scenes.

    Results are deduplicated to their parent scene (first occurrence wins)
    before ranking, because multi-chunk-per-scene modes (window, utterance)
    otherwise let several chunks of one scene occupy distinct rank slots --
    which depresses recall@k versus what the bot, which dedupes to scenes
    before serving, actually delivers. For bm25 and scene modes (one chunk
    per scene) this is identical to the old chunk-level rank.
    """
    seen: set[str] = set()
    rank = 0
    for r in results:
        sid = _parent_scene_id(r)
        if sid in seen:
            continue          # duplicate scene: does not consume a new rank slot
        seen.add(sid)
        rank += 1
        if _matches_target(r, target_scene_id):
            return rank
    return None


def run_modes_for_question(question: dict, modes: list[str]) -> dict:
    """Run the given modes for one question, return {mode: rank}."""
    target = question["ground_truth_scene_id"]
    ranks = {}
    for mode in modes:
        try:
            results = retrieve_unified(question["question"], mode=mode, top_k=RETRIEVE_DEPTH)
            rank = find_rank(results, target)
            # Only count hits within the first TOP_K_FOR_EVAL unique scenes;
            # deeper hits are misses for any k <= TOP_K_FOR_EVAL.
            ranks[mode] = rank if (rank is not None and rank <= TOP_K_FOR_EVAL) else None
        except Exception as e:
            ranks[mode] = f"ERROR: {type(e).__name__}: {e}"
    return ranks


def load_existing(path: Path) -> dict[str, dict]:
    """Load existing results keyed by qa_id. Empty dict if file absent."""
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


def atomic_write(path: Path, rows: list[dict]):
    """Write rows to a temp file, then atomically rename. Crash-safe."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(path)  # atomic on POSIX


def write_meta(results_path: Path, modes_run: list[str], n_questions: int):
    """Append a run record to a sidecar metadata file."""
    meta_path = results_path.with_suffix(".meta.json")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modes_computed": modes_run,
        "questions_touched": n_questions,
        "top_k": TOP_K_FOR_EVAL,
    }
    history = []
    if meta_path.exists():
        try:
            history = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            history = []
    history.append(record)
    meta_path.write_text(json.dumps(history, indent=2))


def main(questions_path: Path, results_path: Path, force_mode: str | None):
    with open(questions_path) as f:
        questions = [json.loads(line) for line in f]

    existing = load_existing(results_path)

    # Determine, per question, which modes still need to run
    work = []  # list of (question, [modes_to_run])
    for q in questions:
        qa_id = q["qa_id"]
        done_modes = set(existing.get(qa_id, {}).get("ranks", {}).keys())
        if force_mode:
            needed = [force_mode]
        else:
            needed = [m for m in AVAILABLE_MODES if m not in done_modes]
        if needed:
            work.append((q, needed))

    all_modes_needed = sorted({m for _, modes in work for m in modes})
    print(f"Questions file: {questions_path.name}")
    print(f"Results file:   {results_path.name}")
    print(f"Questions needing work: {len(work):,} / {len(questions):,}")
    print(f"Modes to compute: {all_modes_needed or '(none)'}")
    if not work:
        print("Nothing to do.")
        return

    # Build the result rows we'll write (start from existing, merge new)
    results_by_id = dict(existing)  # shallow copy

    def process(item):
        question, modes = item
        new_ranks = run_modes_for_question(question, modes)
        return question, new_ranks

    processed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, item): item for item in work}
        for future in tqdm(as_completed(futures), total=len(work), desc="Evaluating"):
            question, new_ranks = future.result()
            qa_id = question["qa_id"]

            if qa_id in results_by_id:
                results_by_id[qa_id]["ranks"].update(new_ranks)
            else:
                results_by_id[qa_id] = {
                    "qa_id": qa_id,
                    "question": question["question"],
                    "ground_truth_scene_id": question["ground_truth_scene_id"],
                    "is_paraphrased": question["is_paraphrased"],
                    "ranks": new_ranks,
                }

            processed += 1
            # Checkpoint every 100 questions
            if processed % 100 == 0:
                atomic_write(results_path, list(results_by_id.values()))

    # Final write
    atomic_write(results_path, list(results_by_id.values()))
    write_meta(results_path, all_modes_needed, len(work))
    print(f"\nDone. Results in {results_path}")
    print(f"Run metadata appended to {results_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--force", type=str, default=None,
                        help="Recompute this single mode even if results exist")
    args = parser.parse_args()
    main(args.questions, args.results, args.force)