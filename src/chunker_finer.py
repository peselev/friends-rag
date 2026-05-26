"""
Two finer-grained chunkers for the granularity experiment:

- Utterance-level: 1 chunk per spoken line (~67K chunks)
- Window-level:    3 utterances per chunk, sliding by 1 (~67K chunks)

Both preserve scene_id in metadata for parent-child retrieval.

Reads:  data/processed/scenes.jsonl
Writes: data/processed/chunks_utterance.jsonl
        data/processed/chunks_window.jsonl

Run with: python -m src.chunker_finer
"""
import json

from src.config import DATA_PROCESSED

WINDOW_SIZE = 3   # utterances per window chunk
WINDOW_STRIDE = 1  # step between consecutive windows


def load_scenes() -> list[dict]:
    """Read scenes.jsonl into a list."""
    with open(DATA_PROCESSED / "scenes.jsonl") as f:
        return [json.loads(line) for line in f]


def scene_header(scene: dict) -> str:
    return f"Friends S{scene['season']:02d}E{scene['episode']:02d}, Scene {scene['scene_num']}"


def build_utterance_chunks(scenes: list[dict]) -> list[dict]:
    """One chunk per spoken line. Skip pure stage directions."""
    chunks = []
    for scene in scenes:
        header = scene_header(scene)
        for utt_idx, utt in enumerate(scene["utterances"]):
            if utt["speaker"] == "[stage direction]":
                continue
            text = f"{header}\n{utt['speaker']}: {utt['text']}"
            chunks.append({
                "id": f"{scene['scene_id']}_u{utt_idx:03d}",
                "text": text,
                "metadata": {
                    "scene_id": scene["scene_id"],
                    "season": scene["season"],
                    "episode": scene["episode"],
                    "scene_num": scene["scene_num"],
                    "speaker": utt["speaker"],
                    "utterance_index": utt_idx,
                },
            })
    return chunks


def build_window_chunks(scenes: list[dict]) -> list[dict]:
    """
    3-utterance sliding windows, stride 1. Stage directions included in text
    for context but windows are anchored on the *position*, not the speaker.
    """
    chunks = []
    for scene in scenes:
        header = scene_header(scene)
        utts = scene["utterances"]
        for start in range(0, max(1, len(utts) - WINDOW_SIZE + 1), WINDOW_STRIDE):
            window = utts[start:start + WINDOW_SIZE]
            body = "\n".join(f"{u['speaker']}: {u['text']}" for u in window)
            text = f"{header}\n{body}"
            chunks.append({
                "id": f"{scene['scene_id']}_w{start:03d}",
                "text": text,
                "metadata": {
                    "scene_id": scene["scene_id"],
                    "season": scene["season"],
                    "episode": scene["episode"],
                    "scene_num": scene["scene_num"],
                    "window_start": start,
                    "window_size": len(window),
                },
            })
    return chunks


def main():
    print("Loading scenes...")
    scenes = load_scenes()
    print(f"  Loaded {len(scenes):,} scenes\n")

    print("Building utterance-level chunks...")
    utt_chunks = build_utterance_chunks(scenes)
    print(f"  Produced {len(utt_chunks):,} chunks")

    print("Building 3-utterance window chunks...")
    win_chunks = build_window_chunks(scenes)
    print(f"  Produced {len(win_chunks):,} chunks\n")

    for name, chunks in [("utterance", utt_chunks), ("window", win_chunks)]:
        out_path = DATA_PROCESSED / f"chunks_{name}.jsonl"
        with open(out_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        print(f"Wrote {len(chunks):,} chunks to {out_path}")


if __name__ == "__main__":
    main()