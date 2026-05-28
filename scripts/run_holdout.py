"""
Run the holdout test queries through the demo-candidate modes, end-to-end
(retrieval + generation). This is the sealed final test - run ONCE, record
honestly, do not tune against it.

Run with: python -m scripts.run_holdout
"""
from src.generator import answer_question

# The holdout queries. Ground-truth expectations are in
# data/processed/holdout_queries.md (kept separate so this script just runs them).
HOLDOUT_QUERIES = [
    "Who was Joey in love with?",
    "Who did Joey have feelings for in season 8?",
    "What did Joey confess to Rachel at the restaurant?",
]

MODES = ["hybrid_window_bm25", "hybrid_window_bm25_reranked_bge"]


def main():
    for query in HOLDOUT_QUERIES:
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        for mode in MODES:
            print(f"\n--- Mode: {mode} ---")
            result = answer_question(query, mode=mode)

            print("\nRetrieved scenes:")
            for i, src in enumerate(result.sources, start=1):
                print(f"  {i}. {src.citation}")

            print(f"\nClaude's answer:\n{result.answer}\n")


if __name__ == "__main__":
    main()