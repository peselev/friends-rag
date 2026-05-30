"""
Streamlit UI for comparing four retrieval modes side-by-side.

Run with: streamlit run app_compare.py
"""
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from src.generator import answer_question

# The four modes this comparison UI is built around.
COMPARE_MODES = ["vector", "vector_naive", "bm25", "hybrid"]

MODE_LABELS = {
    "vector": "Vector (scene chunks)",
    "vector_naive": "Vector (naive chunks)",
    "bm25": "BM25 (keyword)",
    "hybrid": "Hybrid (BM25 + Vector)",
}

MODE_DESCRIPTIONS = {
    "vector": "Semantic search over complete scenes. Strong on paraphrase.",
    "vector_naive": "Same model, but 500-char fixed-window chunks ignoring scene boundaries.",
    "bm25": "Classic keyword search. Strong on exact terms and names.",
    "hybrid": "BM25 + Vector results fused via Reciprocal Rank Fusion.",
}


st.set_page_config(page_title="Friends RAG: Mode Comparison", page_icon="☕", layout="wide")

st.title("☕ Friends RAG — Mode Comparison")
st.caption(
    "Same question, four retrieval strategies. Compare how different chunking and "
    "ranking approaches affect what scenes are retrieved and what answer Claude generates."
)


@st.cache_resource
def warmup():
    # Force module imports so first query isn't cold
    from src.generator import answer_question  # noqa
    return True


warmup()


# --- Input ---
with st.form("ask_form"):
    question = st.text_input(
        "Ask a question about Friends:",
        placeholder="What does Ross yell when moving a couch?",
    )
    submit = st.form_submit_button("Compare modes", type="primary")


# --- Run all modes in parallel on submit ---
if submit and question:
    with st.spinner("Running all four modes in parallel..."):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                mode: executor.submit(answer_question, question, mode=mode)
                for mode in COMPARE_MODES
            }
            results = {mode: f.result() for mode, f in futures.items()}

    # --- Answers in 4 columns ---
    st.markdown("## Answers")
    cols = st.columns(len(COMPARE_MODES))
    for col, mode in zip(cols, COMPARE_MODES):
        with col:
            st.markdown(f"### {MODE_LABELS[mode]}")
            st.caption(MODE_DESCRIPTIONS[mode])
            st.markdown(results[mode].answer)

    # --- Sources, mode by mode, stacked ---
    st.markdown("## Sources retrieved per mode")
    for mode in COMPARE_MODES:
        with st.expander(f"{MODE_LABELS[mode]} — {len(results[mode].sources)} scenes"):
            for i, src in enumerate(results[mode].sources, start=1):
                st.markdown(f"**{i}. {src.citation}** · distance: `{src.distance:.3f}`")
                st.text(src.text[:600] + ("..." if len(src.text) > 600 else ""))
                st.markdown("---")
