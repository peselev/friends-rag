"""
Generate lexically-distinct paraphrases of original questions using Claude.

Uses single-threaded execution with rate limiting to stay under 50 req/min.
Resumable: re-running picks up where it left off.

Run with: python -m src.eval.paraphrase_generator
"""
import json
import random
import time

from anthropic import Anthropic
from tqdm import tqdm

from src.config import ANTHROPIC_API_KEY, DATA_PROCESSED

N_SAMPLES = 200
RANDOM_SEED = 42
MODEL = "claude-haiku-4-5"
DELAY_SECONDS = 1.3  # ~46 requests/min, safely under 50/min limit

_client = Anthropic(api_key=ANTHROPIC_API_KEY)


PARAPHRASE_PROMPT = """Rewrite this question so it asks for the exact same information, but using completely different wording.

Rules:
1. Replace proper nouns with descriptive equivalents (e.g., "Ross" → "the paleontologist", "Central Perk" → "the coffee shop").
2. Replace key content words with synonyms or paraphrases (e.g., "say" → "remark", "girlfriend" → "romantic partner").
3. Preserve no more than 2-3 content words from the original.
4. The new question must still ask for the same answer - do not change the meaning.
5. Output ONLY the rewritten question, no preamble or explanation.

Original question: {question}

Rewritten question:"""


def paraphrase_one(question: dict) -> dict:
    response = _client.messages.create(
        model=MODEL,
        max_tokens=200,
        temperature=0.7,
        messages=[{
            "role": "user",
            "content": PARAPHRASE_PROMPT.format(question=question["question"]),
        }],
    )
    new_question = response.content[0].text.strip()

    return {
        "qa_id": question["qa_id"] + "_LexicalParaphrase",
        "question": new_question,
        "original_question": question["question"],
        "ground_truth_scene_id": question["ground_truth_scene_id"],
        "answers": question["answers"],
        "is_paraphrased": True,
        "is_lexical_paraphrase": True,
    }


def load_already_done(path) -> set[str]:
    """Find qa_ids already in the output file (for resumability)."""
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                # Strip the suffix we added to get back the original qa_id
                row = json.loads(line)
                original_qa_id = row["qa_id"].replace("_LexicalParaphrase", "")
                done.add(original_qa_id)
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    sample_path = DATA_PROCESSED / "eval_sample.jsonl"
    out_path = DATA_PROCESSED / "eval_lexical_paraphrases.jsonl"

    with open(sample_path) as f:
        all_questions = [json.loads(line) for line in f]

    originals = [q for q in all_questions if not q["is_paraphrased"]]

    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(originals, N_SAMPLES)

    done = load_already_done(out_path)
    remaining = [q for q in sampled if q["qa_id"] not in done]

    print(f"Target: {N_SAMPLES}  |  Already done: {len(done)}  |  Remaining: {len(remaining)}")
    if not remaining:
        print("Nothing to do.")
        return

    # Append mode for resumability
    with open(out_path, "a") as f:
        for question in tqdm(remaining, desc="Paraphrasing"):
            try:
                result = paraphrase_one(question)
                f.write(json.dumps(result) + "\n")
                f.flush()
            except Exception as e:
                print(f"\n  Error on {question['qa_id']}: {type(e).__name__}: {e}")
                print("  Waiting 30s and continuing...")
                time.sleep(30)
                continue
            time.sleep(DELAY_SECONDS)

    print(f"\nDone. Output: {out_path}")


if __name__ == "__main__":
    main()