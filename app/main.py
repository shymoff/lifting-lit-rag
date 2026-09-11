"""Streamlit app: search the strength-training literature corpus."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from app.query_log import fetch_all, log_query
from rag.generate import (
    GENERATOR_MODEL,
    GUARDRAIL_MESSAGE,
    hits_to_sources,
    is_medical_question,
    stream_tokens,
)
from rag.retrieve import search

st.set_page_config(page_title="lifting-lit-rag", page_icon="\U0001f3cb", layout="wide")

# Below this cosine-similarity score, a query is probably asking about
# something outside the corpus - calibrated in Phase 3: genuinely on-topic
# questions scored ~0.81-0.90, out-of-scope ones ~0.69-0.72.
LOW_SCORE_THRESHOLD = 0.75


@st.cache_data(show_spinner=False)
def cached_search(question: str, k: int):
    return search(question, k=k)


st.title("lifting-lit-rag")
st.caption(
    "A search tool over open-access strength-training research literature. "
    "Not medical advice and not a training plan - consult a qualified "
    "professional for those."
)

tab_search, tab_monitoring = st.tabs(["Search", "Monitoring"])

with tab_search:
    with st.sidebar:
        k = st.slider("Number of sources (k)", min_value=1, max_value=8, value=3)

    question = st.text_input(
        "Ask a question about strength training research",
        placeholder="e.g. Does training to failure maximize hypertrophy?",
    )

    if question:
        start = time.perf_counter()

        if is_medical_question(question):
            st.warning(GUARDRAIL_MESSAGE)
            log_query(
                query=question,
                retrieved_chunk_ids=[],
                top_score=None,
                latency_ms=(time.perf_counter() - start) * 1000,
                n_tokens_in=None,
                n_tokens_out=None,
                model=None,
                refused=True,
            )
        else:
            with st.spinner("Searching the corpus..."):
                hits = cached_search(question, k)
            sources = hits_to_sources(hits)

            answer_placeholder = st.empty()
            full_text = ""
            stats: dict = {}
            for token in stream_tokens(question, hits, stats=stats):
                full_text += token
                answer_placeholder.markdown(full_text)

            log_query(
                query=question,
                retrieved_chunk_ids=[h["id"] for h in hits],
                top_score=hits[0]["score"] if hits else None,
                latency_ms=(time.perf_counter() - start) * 1000,
                n_tokens_in=stats.get("n_tokens_in"),
                n_tokens_out=stats.get("n_tokens_out"),
                model=GENERATOR_MODEL,
                refused=False,
            )

            if hits and hits[0]["score"] < LOW_SCORE_THRESHOLD:
                st.info(
                    f"Top match score ({hits[0]['score']:.2f}) is low - this question "
                    "may be outside what the corpus covers well."
                )

            if sources:
                with st.expander(f"Sources ({len(sources)})", expanded=False):
                    for i, s in enumerate(sources, 1):
                        st.markdown(
                            f"**[{i}] {s.title}**  \n"
                            f"{s.pmcid} · {s.section} · [View on PMC]({s.url})"
                        )

with tab_monitoring:
    rows = fetch_all()

    if not rows:
        st.info("No queries logged yet - ask something in the Search tab.")
    else:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["refused"] = df["refused"].astype(bool)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total queries", len(df))
        col2.metric("p50 latency", f"{df['latency_ms'].quantile(0.50):.0f} ms")
        col3.metric("p95 latency", f"{df['latency_ms'].quantile(0.95):.0f} ms")
        col4.metric("Refusal rate", f"{df['refused'].mean() * 100:.0f}%")

        answered = df[~df["refused"]]
        if not answered.empty and answered["top_score"].notna().any():
            st.metric("Average top score", f"{answered['top_score'].mean():.2f}")

        st.subheader("Queries over time")
        by_day = df.set_index("timestamp").resample("D").size()
        st.bar_chart(by_day)

        st.subheader("Last 20 queries")
        recent = df.sort_values("id", ascending=False).head(20).copy()
        recent["low_score"] = recent["top_score"].apply(
            lambda s: bool(s is not None and s < LOW_SCORE_THRESHOLD)
        )

        def _flag(row):
            if row["refused"]:
                return "refused"
            if row["low_score"]:
                return "low score"
            return ""

        recent["flag"] = recent.apply(_flag, axis=1)
        st.dataframe(
            recent[["timestamp", "query", "top_score", "latency_ms", "flag"]],
            use_container_width=True,
            hide_index=True,
        )
