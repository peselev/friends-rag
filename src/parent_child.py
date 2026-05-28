"""
Parent-child retrieval: retrieve at fine granularity, return coarse-grained
parent units (scenes or episodes) as context.

Per Weekend 3 findings, vector_window is the best fine-grained retriever, so
we use it for the child step. The parent step fetches scene or episode
content, deduplicated by parent ID, preserving the order in which parents
first appeared in the child results.

Public functions:
    retrieve_parent_scene(query, top_k=5)    - returns up to top_k unique scenes
    retrieve_parent_episode(query, top_k=3)  - returns up to top_k unique episodes
"""
import json
from collections import OrderedDict

from src.config import DATA_PROCESSED
from src.retriever import RetrievalResult
from src.unified_retriever import retrieve_unified

# Child retrieval pulls many candidates so we have enough parent diversity
# after deduplication. 30 typically yields 8-15 unique parent scenes.
CHILD_RETRIEVAL_K = 30

# The underlying child retriever. vector_window was strongest in Weekend 3 eval.
CHILD_MODE = "vector_window"


# --- Parent lookup tables (built once at module load) ---

def _build_scene_index() -> dict:
    """Map scene_id -> full scene dict (for fast parent lookup)."""
    index = {}
    with open(DATA_PROCESSED / "scenes.jsonl") as f:
        for line in f:
            scene = json.loads(line)
            index[scene["scene_id"]] = scene
    return index


def _build_episode_index(scenes_by_id: dict) -> dict:
    """
    Map (season, episode) -> list of scenes in that episode, in scene_num order.
    """
    by_episode = {}
    for scene in scenes_by_id.values():
        key = (scene["season"], scene["episode"])
        by_episode.setdefault(key, []).append(scene)
    # Sort scenes within each episode
    for key in by_episode:
        by_episode[key].sort(key=lambda s: s["scene_num"])
    return by_episode


_SCENES = _build_scene_index()
_EPISODES = _build_episode_index(_SCENES)


# --- Builder helpers ---

def _make_scene_result(scene: dict, distance: float) -> RetrievalResult:
    """Wrap a full scene as a RetrievalResult for downstream use."""
    return RetrievalResult(
        scene_id=scene["scene_id"],
        text=scene["full_text"],
        metadata={
            "scene_id": scene["scene_id"],
            "season": scene["season"],
            "episode": scene["episode"],
            "scene_num": scene["scene_num"],
            "parent_type": "scene",
        },
        distance=distance,
    )


def _make_episode_result(season: int, episode: int, scenes: list, distance: float) -> RetrievalResult:
    """Wrap a full episode (joined scenes) as a RetrievalResult."""
    text = "\n\n---\n\n".join(
        f"Scene {s['scene_num']}:\n{s['full_text']}" for s in scenes
    )
    episode_id = f"s{season:02d}_e{episode:02d}"
    return RetrievalResult(
        scene_id=episode_id,  # using this field as the unique ID
        text=text,
        metadata={
            "season": season,
            "episode": episode,
            "n_scenes": len(scenes),
            "parent_type": "episode",
            # Provide scene_num so existing citation property still works
            "scene_num": 0,
        },
        distance=distance,
    )


# --- Public retrieval functions ---

def retrieve_parent_scene(query: str, top_k: int = 5) -> list[RetrievalResult]:
    """
    Retrieve windows, deduplicate by parent scene_id, return up to top_k full scenes.
    Parents are ordered by the rank at which their first child window appeared.
    """
    child_results = retrieve_unified(query, mode=CHILD_MODE, top_k=CHILD_RETRIEVAL_K)

    # OrderedDict preserves first-seen order across iteration
    parents = OrderedDict()
    for child in child_results:
        scene_id = child.metadata.get("scene_id")
        if scene_id and scene_id not in parents and scene_id in _SCENES:
            # Take the first (best) child's distance as the parent's score
            parents[scene_id] = (child.distance, _SCENES[scene_id])
        if len(parents) >= top_k:
            break

    return [_make_scene_result(scene, dist) for dist, scene in parents.values()]


def retrieve_parent_episode(query: str, top_k: int = 3) -> list[RetrievalResult]:
    """
    Retrieve windows, deduplicate by parent (season, episode), return full episodes.
    """
    child_results = retrieve_unified(query, mode=CHILD_MODE, top_k=CHILD_RETRIEVAL_K)

    parents = OrderedDict()
    for child in child_results:
        m = child.metadata
        if "season" not in m or "episode" not in m:
            continue
        key = (m["season"], m["episode"])
        if key not in parents:
            parents[key] = (child.distance, _EPISODES.get(key, []))
        if len(parents) >= top_k:
            break

    results = []
    for (season, episode), (dist, scenes) in parents.items():
        if not scenes:
            continue
        results.append(_make_episode_result(season, episode, scenes, dist))
    return results


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Ross yells PIVOT moving couch"
    mode = sys.argv[2] if len(sys.argv) > 2 else "scene"

    print(f"Mode: parent_{mode}  Query: {query!r}\n")

    if mode == "scene":
        results = retrieve_parent_scene(query)
    elif mode == "episode":
        results = retrieve_parent_episode(query)
    else:
        print(f"Unknown mode: {mode}. Use 'scene' or 'episode'.")
        sys.exit(1)

    for i, r in enumerate(results, start=1):
        print(f"{i}. {r.citation}  (distance: {r.distance:.3f})")
        text_preview = r.text[:200].replace("\n", " | ")
        print(f"   {text_preview}...")
        print(f"   ({len(r.text):,} chars total)")
        print()