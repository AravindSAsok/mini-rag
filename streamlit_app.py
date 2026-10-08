"""Small browser interface for the existing Mini RAG retriever."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import DEFAULT_DATA_DIR, ResearchAssistant  # noqa: E402


st.set_page_config(page_title="Mini RAG Assistant", page_icon="🔎")
st.title("Mini RAG Assistant")
st.write(
    "Ask a question about the documents in the `data` folder. "
    "This demo finds and displays the most relevant passages; it does not "
    "generate an answer with an AI chat model."
)


@st.cache_resource(show_spinner="Loading documents and building the search index...")
def load_assistant() -> ResearchAssistant:
    """Load the documents and embedding model once for this app process."""
    return ResearchAssistant(DEFAULT_DATA_DIR)


try:
    assistant = load_assistant()
except Exception as exc:
    st.error(
        "The documents or embedding model could not be loaded. Check that "
        "`data` contains a non-empty .txt or .md file and that the model "
        f"can be downloaded. Details: {exc}"
    )
    st.stop()

st.caption(
    f"Loaded {len(assistant.chunks)} passages. "
    "Similarity is a ranking score, not a probability or guarantee of correctness."
)

with st.form("search_form"):
    question = st.text_input(
        "Your question",
        placeholder="For example: Which members were in EXO-M?",
    )
    submitted = st.form_submit_button("Search passages")

if submitted:
    if not question.strip():
        st.warning("Type a question before searching.")
    else:
        try:
            results = assistant.search(question, top_k=3)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            st.subheader("Most relevant passages")
            for rank, result in enumerate(results, start=1):
                st.markdown(f"**Passage {rank} · similarity {result.score:.3f}**")
                st.caption(
                    f"Source: {result.chunk.source} · "
                    f"Chunk: {result.chunk.number}"
                )
                st.text(result.chunk.text)
                st.divider()
