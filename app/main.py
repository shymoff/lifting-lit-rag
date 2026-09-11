"""Streamlit app: search the strength-training literature corpus."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from rag.generate import (
    GUARDRAIL_MESSAGE,
    hits_to_sources,
    is_medical_question,
    stream_tokens,
)
from rag.retrieve import search

st.set_page_config(page_title="lifting-lit-rag", page_icon="\U0001f3cb", layout="wide")


@st.cache_data(show_spinner=False)
def cached_search(question: str, k: int):
    return search(question, k=k)


st.title("lifting-lit-rag")
st.caption(
    "A search tool over open-access strength-training research literature. "
    "Not medical advice and not a training plan - consult a qualified "
    "professional for those."
)

(tab_search,) = st.tabs(["Search"])

with tab_search:
    with st.sidebar:
        k = st.slider("Number of sources (k)", min_value=1, max_value=8, value=3)

    question = st.text_input(
        "Ask a question about strength training research",
        placeholder="e.g. Does training to failure maximize hypertrophy?",
    )

    if question:
        if is_medical_question(question):
            st.warning(GUARDRAIL_MESSAGE)
        else:
            with st.spinner("Searching the corpus..."):
                hits = cached_search(question, k)
            sources = hits_to_sources(hits)

            answer_placeholder = st.empty()
            full_text = ""
            for token in stream_tokens(question, hits):
                full_text += token
                answer_placeholder.markdown(full_text)

            if sources:
                with st.expander(f"Sources ({len(sources)})", expanded=False):
                    for i, s in enumerate(sources, 1):
                        st.markdown(
                            f"**[{i}] {s.title}**  \n"
                            f"{s.pmcid} · {s.section} · [View on PMC]({s.url})"
                        )
