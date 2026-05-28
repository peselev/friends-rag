"""
Hybrid retrieval fusing BM25 (scene-level) with vector_window_noheader
(window-level), via Reciprocal Rank Fusion.

Two wrinkles handled here:
1. The two retrievers operate at different granularities. BM25 returns scenes;
   window retrieval returns 3-utterance windows. We normalize both to the
   parent scene_id before fusing.
2. Multiple windows from one scene are deduplicated to a single parent (first
   occurrence wins) so one scene doesn't get over-weighted in the fusion.

Two modes exposed:
    retrieve_hybrid_window(query, top_k)        - equal-weight RRF
    retrieve_hybrid_window_smart(query, top_k)  - IDF-weighted RRF (in smart_hybrid.py)
"""
import json
from collections import OrderedDict

from src.bm25_retriever import retrieve_bm25
from src.config import DATA_PROCESSED, TOP_K, WINDOW_NOHEADER_COLLECTION
from src.retriever import RetrievalResult, retrieve_from_collection

RRF_K = 60
OVERSAMPLE_MULTIPLIER = 6  # window dedup loses many, so oversample more


# Scene text lookup (built once) - for returning full parent scene text
def _build_scene_text_index() -> dict:
    index = {}
    with open(DATA_PROCESSED / "scenes.jsonl") as f:
        for line in f:
            scene = json.loads(line)
            index[scene["scene_id"]] = scene
    return index


_SCENES = _build_scene_text_index()


def _parent_scene_id(result: RetrievalResult) -> str:
    """Return the parent scene_id for a result, whether scene- or window-level."""
    # Window results carry parent scene in metadata; BM25 results' scene_id IS the scene
    return result.metadata.get("scene_id", result.scene_id)


def _dedupe_windows_to_scenes(results: list[RetrievalResult]) -> list[str]:
    """Collapse window results to an ordered list of unique parent scene_ids."""
    seen = OrderedDict()
    for r in results:
        sid = _parent_scene_id(r)
        if sid not in seen:
            seen[sid] = True
    return list(seen.keys())


def _make_scene_result(scene_id: str, distance: float) -> RetrievalResult:
    """Build a RetrievalResult with full scene text for a scene_id."""
    scene = _SCENES[scene_id]
    return RetrievalResult(
        scene_id=scene_id,
        text=scene["full_text"],
        metadata={
            "scene_id": scene_id,
            "season": scene["season"],
            "episode": scene["episode"],
            "scene_num": scene["scene_num"],
        },
        distance=distance,
    )


def retrieve_hybrid_window(query: str, top_k: int = TOP_K) -> list[RetrievalResult]:
    """Equal-weight RRF fusion of BM25 and window_noheader, returning scenes."""
    n = top_k * OVERSAMPLE_MULTIPLIER

    bm25_results = retrieve_bm25(query, top_k=n)
    window_results = retrieve_from_collection(query, n, WINDOW_NOHEADER_COLLECTION)

    # Both reduced to ordered lists of parent scene_ids
    bm25_scenes = [r.scene_id for r in bm25_results]
    window_scenes = _dedupe_windows_to_scenes(window_results)

    rrf = {}
    for rank, sid in enumerate(bm25_scenes, start=1):
        rrf[sid] = rrf.get(sid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, sid in enumerate(window_scenes, start=1):
        rrf[sid] = rrf.get(sid, 0.0) + 1.0 / (RRF_K + rank)

    top = sorted(rrf.keys(), key=lambda s: rrf[s], reverse=True)[:top_k]
    return [_make_scene_result(sid, -rrf[sid]) for sid in top if sid in _SCENES]


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Who was Joey in love with?"
    print(f"Query: {query!r}\n")
    for i, r in enumerate(retrieve_hybrid_window(query), start=1):
        print(f"{i}. {r.citation}  (RRF score: {-r.distance:.4f})")
        print(f"   {r.text[:120].replace(chr(10), ' | ')}...")
        print()