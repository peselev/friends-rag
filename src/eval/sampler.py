"""
Sample evaluation questions stratified by is_paraphrased.

Run with: python -m src.eval.sampler
"""
import json
import random

from src.config import DATA_PROCESSED

SAMPLE_SIZE_PER_STRATUM = 400
RANDOM_SEED = 42  # Fixed for reproducibility - same sample every run


def load_questions() -> list[dict]:
    """Load all eval questions from JSONL."""
    path = DATA_PROCESSED / "eval_questions.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def stratified_sample(questions: list[dict], n_per_stratum: int, seed: int) -> list[dict]:
    """
    Split questions by is_paraphrased and sample n from each.
    Returns the combined sample (originals + paraphrased), interleaved order.
    """
    originals = [q for q in questions if not q["is_paraphrased"]]
    paraphrased = [q for q in questions if q["is_paraphrased"]]

    print(f"  Available: {len(originals):,} original, {len(paraphrased):,} paraphrased")

    rng = random.Random(seed)
    if len(originals) < n_per_stratum or len(paraphrased) < n_per_stratum:
        raise ValueError(
            f"Not enough questions to sample {n_per_stratum} per stratum"
        )

    sampled_orig = rng.sample(originals, n_per_stratum)
    sampled_para = rng.sample(paraphrased, n_per_stratum)

    return sampled_orig + sampled_para


def main():
    print("Loading eval questions...")
    questions = load_questions()
    print(f"  Loaded {len(questions):,} questions")

    print(f"\nSampling {SAMPLE_SIZE_PER_STRATUM} per stratum (seed={RANDOM_SEED})...")
    sample = stratified_sample(questions, SAMPLE_SIZE_PER_STRATUM, RANDOM_SEED)
    print(f"  Sample size: {len(sample):,}")

    out_path = DATA_PROCESSED / "eval_sample.jsonl"
    with open(out_path, "w") as f:
        for q in sample:
            f.write(json.dumps(q) + "\n")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()