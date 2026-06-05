"""
Run the holdout test queries through the two DEMO modes, end-to-end
(retrieval + generation). This is the sealed final test - run ONCE, record
honestly, do not tune against it.

The two modes mirror the deployed demo exactly:
  - hybrid_window_bm25                  : fast default (fused top-5, no rerank)
  - hybrid_window_bm25_reranked_cohere  : "higher accuracy" (fused top-20 -> Cohere -> top-5)

Cohere needs COHERE_API_KEY (and optionally COHERE_API_KEY_PROD) in .env. If
Cohere is unavailable, that mode is reported as skipped rather than crashing.

Run with: python -m scripts.run_holdout
"""
from src.generator import answer_question
from src.cohere_reranker import CohereUnavailable

# The holdout queries. Ground-truth expectations are in
# data/processed/holdout_queries.md (kept separate so this script just runs them).
HOLDOUT_QUERIES = [
    "Who was Joey in love with?",
    "Who did Joey have feelings for in season 8?",
    "What did Joey confess to Rachel at the restaurant?",
]

MODES = ["hybrid_window_bm25", "hybrid_window_bm25_reranked_cohere"]


def main():
    for query in HOLDOUT_QUERIES:
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        for mode in MODES:
            print(f"\n--- Mode: {mode} ---")
            try:
                result = answer_question(query, mode=mode)
            except CohereUnavailable as e:
                print(f"[Cohere unavailable — rerank skipped: {e}]")
                continue

            print("\nRetrieved scenes:")
            for i, src in enumerate(result.sources, start=1):
                print(f"  {i}. {src.citation}")

            print(f"\nClaude's answer:\n{result.answer}\n")


if __name__ == "__main__":
    main()