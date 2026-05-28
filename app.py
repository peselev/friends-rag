"""
Friends RAG — grounded question-answering demo.

Run with: streamlit run app.py
"""
import os
import streamlit as st

from src.generator import answer_question

st.set_page_config(page_title="Friends — Grounded Q&A", page_icon="☕", layout="centered")

# --- Palette matched to peselev.com: white bg, #101828 ink, #4338ca indigo accent ---
st.markdown("""
<style>
    .stApp { max-width: 860px; margin: 0 auto; }

    /* Grounding note box - light indigo tint */
    .grounding-note {
        background: #eef2ff; border-left: 3px solid #4338ca;
        padding: 0.7rem 1rem; border-radius: 0.5rem; font-size: 0.9rem;
        color: #101828; margin: 0.5rem 0 1.5rem 0;
    }

    /* Make the text input read as clearly ACTIVE and distinct from buttons */
    .stTextInput > div > div > input {
        background: #ffffff;
        border: 2px solid #4338ca;
        border-radius: 0.5rem;
        color: #101828;
        font-size: 1.05rem;
        padding: 0.6rem 0.8rem;
    }
    .stTextInput > div > div > input::placeholder {
        color: #98a2b3;
    }

    /* Primary (Ask) button in indigo */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {
        background: #4338ca;
        border: none;
        color: #fff;
    }
    .stFormSubmitButton > button:hover {
        background: #3730a3;
        color: #fff;
    }

    /* Example-question buttons: white with indigo outline, distinct from input */
    .stButton > button {
        background: #ffffff;
        border: 1px solid #e0e7ff;
        color: #101828;
        text-align: left;
    }
    .stButton > button:hover {
        border-color: #4338ca;
        background: #eef2ff;
        color: #101828;
    }
</style>
""", unsafe_allow_html=True)

# --- Header image (save chosen image to assets/header.png) ---
if os.path.exists("assets/header.png"):
    st.image("assets/header.png", use_container_width=True)
else:
    st.title("☕ Friends — Grounded Q&A")

# --- Intro paragraph (restored) + grounding note (new text) ---
st.markdown(
    "Ask a question about *Friends*. Every answer is **grounded in actual scene "
    "transcripts** — with citations to the exact episode and scene. If the "
    "transcripts don't contain the answer, the system says so rather than making "
    "something up."
)
st.markdown(
    "<div class='grounding-note'>This is a retrieval-augmented Q&A system over all "
    "236 episodes. It only knows what's in the transcripts — it won't answer trivia "
    "the show never covered.</div>",
    unsafe_allow_html=True,
)


@st.cache_resource
def warmup():
    from src.generator import answer_question  # noqa
    return True

warmup()


def render_answer(result):
    st.markdown("### Answer")
    st.markdown(result.answer)
    st.markdown("### Sources")
    st.caption("Scenes retrieved and used to ground the answer above.")
    for i, src in enumerate(result.sources, start=1):
        with st.expander(f"{i}. {src.citation}"):
            st.text(src.text[:1500] + ("..." if len(src.text) > 1500 else ""))


def run_query(question: str, use_reranker: bool):
    mode = "hybrid_window_bm25_reranked_bge" if use_reranker else "hybrid_window_bm25"
    msg = ("Retrieving scenes, reranking, and asking Claude..."
           if use_reranker else "Retrieving scenes and asking Claude...")
    with st.spinner(msg):
        result = answer_question(question, mode=mode)
    render_answer(result)


EXAMPLES = [
    ("What does Ross yell when moving a couch?",
     "Iconic moment — tests exact-dialogue retrieval."),
    ("How did Chandler propose to Monica?",
     "Multi-scene answer — synthesizes across episodes."),
    ("What is the name of Ross's pet monkey?",
     "Deep-cut factual recall."),
    ("Which cities did Ross consider moving to?",
     "Try this with and without 'Higher accuracy' — the reranker shines here."),
    ("Who was Joey in love with?",
     "Hard: implied across many scenes, never stated plainly."),
    ("What is the capital of France?",
     "Should REFUSE — proves it only uses transcripts."),
]

# --- Session state ---
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "auto_ask" not in st.session_state:
    st.session_state.auto_ask = False
if "use_reranker" not in st.session_state:
    st.session_state.use_reranker = False

# --- Example questions: clicking sets the input value AND flags auto-ask ---
# (Buttons are created BEFORE the text_input, so writing its key here is allowed.)
st.markdown("**Try one of these, or write your own:**")
cols = st.columns(2)
for i, (q, note) in enumerate(EXAMPLES):
    with cols[i % 2]:
        if st.button(q, key=f"ex_{i}", use_container_width=True, help=note):
            st.session_state.question_input = q
            st.session_state.auto_ask = True

# --- Reranker toggle: keyed, no value= ---
use_reranker = st.checkbox(
    "Higher accuracy (slower)",
    key="use_reranker",
    help="Adds a second-stage cross-encoder reranker. ~3s vs ~200ms.",
)

# --- Input row: text field (keyed, no value=) + Ask button to the right ---
with st.form("ask_form"):
    c1, c2 = st.columns([5, 1])
    with c1:
        st.text_input(
            "Your question:",
            key="question_input",
            placeholder="Type your question here",
            label_visibility="collapsed",
        )
    with c2:
        submit = st.form_submit_button("Ask", type="primary", use_container_width=True)

# --- Decide whether to run ---
# Read the question straight from the widget's session_state key.
question = st.session_state.question_input

should_run = False
if submit and question:
    should_run = True
elif st.session_state.auto_ask and question:
    should_run = True
    st.session_state.auto_ask = False  # consume the flag

if should_run:
    reranker_on = st.session_state.get("use_reranker", False)
    st.markdown(f"**Q: {question}**")
    run_query(question, reranker_on)