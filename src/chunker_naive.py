"""
Naive chunker: fixed-window chunking that ignores scene structure.

This chunker exists as a counter-example. It demonstrates what happens when you
chunk a dialogue corpus the way generic RAG tutorials chunk PDFs: take all the
text, slide a fixed-size window over it, ignore semantic boundaries.

Reads:  data/processed/scenes.jsonl
Writes: data/processed/chunks_naive.jsonl

Run with: python -m src.chunker_naive
"""
import json

from src.config import DATA_PROCESSED

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # characters of overlap between consecutive chunks


def load_scenes() -> list[dict]:
    """Read scenes.jsonl into a list."""
    scenes_path = DATA_PROCESSED / "scenes.jsonl"
    if not scenes_path.exists():
        raise FileNotFoundError(
            f"{scenes_path} not found. Run `python -m src.loader` first."
        )
    with open(scenes_path) as f:
        return [json.loads(line) for line in f]


def build_naive_chunks(scenes: list[dict]) -> list[dict]:
    """
    Concatenate all scenes into one long string, then slide a fixed window.

    We keep track of which scene each character position originally came from,
    so each chunk can record which scene it 'started in' - this metadata is
    deliberately fragile (chunks often span scenes), but useful for evaluation.
    """
    # Build the giant string + a parallel array mapping char position -> scene_id
    parts = []
    pos_to_scene = []  # one entry per character

    for scene in scenes:
        header = f"Friends S{scene['season']:02d}E{scene['episode']:02d}, Scene {scene['scene_num']}"
        body = scene["full_text"]
        block = f"{header}\n{body}\n\n"
        parts.append(block)
        pos_to_scene.extend([scene["scene_id"]] * len(block))

    corpus = "".join(parts)

    # Slide the window
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP  # how far to advance between chunks
    chunk_idx = 0

    for start in range(0, len(corpus), step):
        end = start + CHUNK_SIZE
        text = corpus[start:end]
        if not text.strip():
            continue

        # Which scene did this chunk's start position fall in?
        scene_id_at_start = pos_to_scene[start] if start < len(pos_to_scene) else "unknown"

        chunks.append({
            "id": f"naive_{chunk_idx:05d}",
            "text": text,
            "metadata": {
                "started_in_scene": scene_id_at_start,
                "char_start": start,
                "char_end": min(end, len(corpus)),
            },
        })
        chunk_idx += 1

        if end >= len(corpus):
            break

    return chunks


def main():
    print("Loading scenes...")
    scenes = load_scenes()
    print(f"  Loaded {len(scenes):,} scenes")

    print(f"\nChunking with {CHUNK_SIZE}-char windows, {CHUNK_OVERLAP}-char overlap...")
    chunks = build_naive_chunks(scenes)
    print(f"  Produced {len(chunks):,} naive chunks")

    out_path = DATA_PROCESSED / "chunks_naive.jsonl"
    with open(out_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
