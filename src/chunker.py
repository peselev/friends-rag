"""
Chunker: convert processed scenes into embed-ready chunks with metadata.

Reads:  data/processed/scenes.jsonl
Writes: data/processed/chunks.jsonl

Each chunk contains:
    - id: stable unique identifier (same as scene_id for now)
    - text: what we'll embed (includes a metadata header for stronger embeddings)
    - metadata: structured fields for filtering at query time

Run with: python -m src.chunker
"""
import json
from pathlib import Path

from src.config import DATA_PROCESSED


def build_chunk(scene: dict) -> dict:
    """
    Turn a scene into a chunk.

    The 'text' field is what gets embedded. We prefix it with a human-readable
    header ('Friends S01E02, Scene 3') because the embedding model will pick up
    semantic signal from those words too - useful for queries that mention
    season/episode context.
    """
    season = scene["season"]
    episode = scene["episode"]
    scene_num = scene["scene_num"]

    header = f"Friends S{season:02d}E{episode:02d}, Scene {scene_num}"
    body = scene["full_text"]
    text = f"{header}\n{body}"

    # Unique speakers in this scene (excluding stage directions),
    # preserving first-appearance order.
    speakers_seen = []
    for u in scene["utterances"]:
        sp = u["speaker"]
        if sp != "[stage direction]" and sp not in speakers_seen:
            speakers_seen.append(sp)

    return {
        "id": scene["scene_id"],
        "text": text,
        "metadata": {
            "scene_id": scene["scene_id"],
            "season": season,
            "episode": episode,
            "scene_num": scene_num,
            # Chroma metadata values must be primitives (str/int/float/bool),
            # not lists. So we join speakers into a comma-separated string.
            # For filtering, we can do substring matches like "Chandler Bing" in speakers.
            "speakers": ", ".join(speakers_seen),
            "n_utterances": len(scene["utterances"]),
        },
    }


def chunk_all_scenes():
    """Read scenes.jsonl, write chunks.jsonl."""
    scenes_path = DATA_PROCESSED / "scenes.jsonl"
    chunks_path = DATA_PROCESSED / "chunks.jsonl"

    if not scenes_path.exists():
        raise FileNotFoundError(
            f"{scenes_path} not found. Run `python -m src.loader` first."
        )

    n = 0
    with open(scenes_path) as scenes_in, open(chunks_path, "w") as chunks_out:
        for line in scenes_in:
            scene = json.loads(line)
            chunk = build_chunk(scene)
            chunks_out.write(json.dumps(chunk) + "\n")
            n += 1

    print(f"Wrote {n:,} chunks to {chunks_path}")


if __name__ == "__main__":
    chunk_all_scenes()
