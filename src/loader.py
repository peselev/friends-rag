"""
Loader: read raw Emory character-mining JSON files, produce processed JSONL.

Outputs two files:
    data/processed/scenes.jsonl         - one scene per line, stripped to essentials
    data/processed/eval_questions.jsonl - span_qa Q&A pairs for evaluation

Run with: python -m src.loader
"""
import json
from pathlib import Path

from src.config import DATA_RAW, DATA_PROCESSED


def parse_scene_id(scene_id: str) -> tuple[int, int, int]:
    """
    Parse a scene_id like 's01_e02_c03' into (season=1, episode=2, scene_num=3).
    The Emory format uses 's' for season, 'e' for episode, 'c' for scene (chunk).
    """
    parts = scene_id.split("_")
    season = int(parts[0][1:])
    episode = int(parts[1][1:])
    scene_num = int(parts[2][1:])
    return season, episode, scene_num


def clean_utterance(utt: dict) -> dict:
    """
    Strip an utterance down to {speaker, text}.

    - Joins multiple speakers with ' & '.
    - Uses transcript_with_note when available (preserves action notes like '(to Ross)').
    - Empty speakers become '[stage direction]' (preserves context-setting lines).
    """
    speakers = utt.get("speakers", [])
    speaker = " & ".join(speakers) if speakers else "[stage direction]"

    text = utt.get("transcript_with_note") or utt.get("transcript", "")

    return {"speaker": speaker, "text": text}


def process_scene(raw_scene: dict) -> dict:
    """Transform one raw scene into our processed shape."""
    scene_id = raw_scene["scene_id"]
    season, episode, scene_num = parse_scene_id(scene_id)

    utterances = [clean_utterance(u) for u in raw_scene["utterances"]]

    # Build a single string representation - this is what we'll embed later.
    full_text = "\n".join(f"{u['speaker']}: {u['text']}" for u in utterances)

    return {
        "scene_id": scene_id,
        "season": season,
        "episode": episode,
        "scene_num": scene_num,
        "utterances": utterances,
        "full_text": full_text,
    }


def extract_eval_questions(raw_scene: dict) -> list[dict]:
    """
    Pull span_qa items out of a scene, attaching the scene_id so we can
    later check whether retrieval found the right scene.
    """
    qa_items = raw_scene.get("span_qa") or []
    questions = []
    for qa in qa_items:
        # Each qa has: id, question, answers (list of {answer_text, utterance_id, ...})
        questions.append({
            "qa_id": qa["id"],
            "question": qa["question"],
            "ground_truth_scene_id": raw_scene["scene_id"],
            "answers": [a["answer_text"] for a in qa.get("answers", [])],
            "is_paraphrased": qa["id"].endswith("_Paraphrased"),
        })
    return questions


def load_and_process_all():
    """Main entry point: read all 10 raw seasons, write processed JSONL files."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    scenes_path = DATA_PROCESSED / "scenes.jsonl"
    eval_path = DATA_PROCESSED / "eval_questions.jsonl"

    n_scenes = 0
    n_questions = 0

    with open(scenes_path, "w") as scenes_out, open(eval_path, "w") as eval_out:
        # Sort so seasons go in order 01..10
        for raw_file in sorted(DATA_RAW.glob("friends_season_*.json")):
            print(f"Processing {raw_file.name}...")
            with open(raw_file) as f:
                data = json.load(f)

            for episode in data["episodes"]:
                for raw_scene in episode["scenes"]:
                    # Write the processed scene
                    processed = process_scene(raw_scene)
                    scenes_out.write(json.dumps(processed) + "\n")
                    n_scenes += 1

                    # Write any eval questions attached to this scene
                    for q in extract_eval_questions(raw_scene):
                        eval_out.write(json.dumps(q) + "\n")
                        n_questions += 1

    print(f"\nDone.")
    print(f"  Scenes:    {n_scenes:,} written to {scenes_path}")
    print(f"  Questions: {n_questions:,} written to {eval_path}")


if __name__ == "__main__":
    load_and_process_all()
