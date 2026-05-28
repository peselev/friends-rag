"""
Run candidate demo questions through both modes; print answers side by side.
Goal: find questions where the reranker gives a VISIBLY better answer AND
fast mode isn't embarrassingly wrong.

Usage: python -m scripts.compare_answers
"""
from src.generator import answer_question

FAST = "hybrid_window_bm25"
RERANKED = "hybrid_window_bm25_reranked_bge"

CANDIDATES = [
    # --- Clean rephrasings of confirmed miner-wins (reranked #1, fast missed) ---
    "What major did Ross choose in college?",          # s01_e09_c11
    "When did Carol want Ross to talk to the baby?",    # s01_e09_c09
    "What was Chandler doing in the bathroom?",         # s03_e21_c16
    "What is Monica showing off?",                      # s03_e20_c03
    "Where was Rachel reading?",                        # s04_e02_c09
    "What did Chandler have when he crossed the street?",  # s01_e13_c02
    "Who is catching an earlier flight?",               # s01_e10_c05
    "What credit card does Alessandro's not accept?",   # s04_e09_c07

    # --- Naturally memorable (test whether reranker helps or at least ties) ---
    "Why did Ross and Rachel go on a break?",
    "What job does Chandler actually have?",
    "How did Ross and Rachel first get together?",
]


def main():
    for q in CANDIDATES:
        print("\n" + "#" * 80)
        print(f"# {q}")
        print("#" * 80)
        for mode in (FAST, RERANKED):
            try:
                result = answer_question(q, mode=mode)
                label = "FAST" if mode == FAST else "RERANKED"
                print(f"\n----- {label} -----")
                print("Scenes:", ", ".join(s.citation for s in result.sources))
                print(result.answer)
            except Exception as e:
                print(f"\n----- {mode} ERROR: {e} -----")


if __name__ == "__main__":
    main()