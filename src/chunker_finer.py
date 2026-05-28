"""
Two finer-grained chunkers for the granularity experiment:

- Utterance-level: 1 chunk per spoken line
- Window-level:    3 utterances per chunk, sliding by 1

By default, each chunk's embed text is prefixed with a scene header
("Friends S01E01, Scene 1"). Pass --no-header to test the impact of
removing it.

Run with:
    python -m src.chunker_finer            # default with headers
    python -m src.chunker_finer --no-header
"""
import argparse
import json

from src.config import DATA_PROCESSED

WINDOW_SIZE = 3
WINDOW_STRIDE = 1


def load_scenes() -> list[dict]:
    with open(DATA_PROCESSED / "scenes.jsonl") as f:
        return [json.loads(line) for line in f]


def scene_header(scene: dict) -> str:
    return f"Friends S{scene['season']:02d}E{scene['episode']:02d}, Scene {scene['scene_num']}"


def build_utterance_chunks(scenes: list[dict], include_header: bool) -> list[dict]:
    chunks = []
    for scene in scenes:
        header = scene_header(scene) + "\n" if include_header else ""
        for utt_idx, utt in enumerate(scene["utterances"]):
            if utt["speaker"] == "[stage direction]":
                continue
            text = f"{header}{utt['speaker']}: {utt['text']}"
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


def build_window_chunks(scenes: list[dict], include_header: bool) -> list[dict]:
    chunks = []
    for scene in scenes:
        header = scene_header(scene) + "\n" if include_header else ""
        utts = scene["utterances"]
        for start in range(0, max(1, len(utts) - WINDOW_SIZE + 1), WINDOW_STRIDE):
            window = utts[start:start + WINDOW_SIZE]
            body = "\n".join(f"{u['speaker']}: {u['text']}" for u in window)
            text = f"{header}{body}"
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


def main(include_header: bool):
    print("Loading scenes...")
    scenes = load_scenes()
    print(f"  Loaded {len(scenes):,} scenes\n")

    suffix = "" if include_header else "_noheader"

    print(f"Building utterance-level chunks (header={'yes' if include_header else 'no'})...")
    utt_chunks = build_utterance_chunks(scenes, include_header)
    print(f"  Produced {len(utt_chunks):,} chunks")

    print(f"Building 3-utterance window chunks (header={'yes' if include_header else 'no'})...")
    win_chunks = build_window_chunks(scenes, include_header)
    print(f"  Produced {len(win_chunks):,} chunks\n")

    for name, chunks in [("utterance", utt_chunks), ("window", win_chunks)]:
        out_path = DATA_PROCESSED / f"chunks_{name}{suffix}.jsonl"
        with open(out_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        print(f"Wrote {len(chunks):,} chunks to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-header", action="store_true",
        help="Build header-less variants (saves with '_noheader' suffix)",
    )
    args = parser.parse_args()
    main(include_header=not args.no_header)