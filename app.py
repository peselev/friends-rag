"""
Streamlit UI for the Friends RAG demo.

Run with: streamlit run app.py
"""
import streamlit as st

from src.generator import answer_question


# --- Page setup ---
st.set_page_config(
    page_title="Friends RAG",
    page_icon="☕",
    layout="centered",
)

st.title("☕ Friends RAG")
st.caption(
    "A semantic search + Claude system over all 236 episodes of Friends. "
    "Ask anything about the show — answers are grounded in actual scene transcripts."
)


# --- Cache the expensive setup so it runs once per session ---
@st.cache_resource
def warmup():
    """
    Import-time side effects (loading Chroma collection, initializing clients)
    happen on first call. Caching means subsequent queries are fast.
    """
    from src.generator import answer_question  # noqa
    return True


warmup()


# --- Input form: Enter OR button click both submit ---
with st.form("ask_form"):
    question = st.text_input(
        "Ask a question:",
        placeholder="What does Ross yell when moving a couch?",
    )
    submit = st.form_submit_button("Ask", type="primary")


# --- Run on submit ---
if submit and question:
    with st.spinner("Searching scenes and asking Claude..."):
        result = answer_question(question)

    # Answer
    st.markdown("### Answer")
    st.markdown(result.answer)

    # Sources
    st.markdown("### Sources")
    st.caption(
        "Scenes retrieved by semantic search. Lower distance = closer semantic match."
    )
    for i, src in enumerate(result.sources, start=1):
        with st.expander(
            f"{i}. {src.citation}  ·  distance: {src.distance:.3f}"
        ):
            st.text(src.text)
