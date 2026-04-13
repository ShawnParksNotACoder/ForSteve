from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "corpus" / "corpus.json"

st.set_page_config(page_title="Buick Manual Assistant", page_icon="🔧", layout="wide")


def load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        return []
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def search_docs(query: str, docs: list[dict], top_k: int = 5) -> list[tuple[float, dict]]:
    texts = [d["text"] for d in docs]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(texts + [query])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = sorted(zip(sims, docs), key=lambda x: x[0], reverse=True)[:top_k]
    return ranked


def summarize_answer(query: str, hits: list[tuple[float, dict]]) -> str:
    if not hits:
        return "No matching reference found yet. Try adding more source pages or using a more specific question."

    best = hits[0][1]["text"] if isinstance(hits[0][1], dict) else hits[0][1]
    return (
        "Best matching reference text:\n\n"
        + best[:1200]
        + "\n\nUse the source cards below to confirm the exact manual language before turning a wrench."
    )


st.title("🔧 Buick Manual Assistant")
st.caption("Dark-mode manual lookup for Buick Regal / Grand National reference material")

with st.sidebar:
    st.header("Project status")
    docs = load_corpus()
    st.metric("Indexed chunks", len(docs))
    st.write("This starter uses local retrieval only. Later you can swap in embeddings and a hosted model.")
    st.markdown("### Suggested questions")
    st.write("- Where is the ECM wiring info?\n- Powermaster brake diagnostics\n- Starting and charging diagram\n- Cam sensor adjustment")

query = st.text_input("Ask a repair/reference question", placeholder="Example: Where is the starting and charging wiring diagram?")

if query:
    docs = load_corpus()
    if not docs:
        st.error("No corpus found. Run fetch_sources.py and build_corpus.py first.")
    else:
        hits = search_docs(query, docs, top_k=5)
        st.subheader("Answer")
        st.write(summarize_answer(query, hits))

        st.subheader("Top source matches")
        for score, doc in hits:
            with st.container(border=True):
                st.markdown(f"**{doc['title']}**")
                st.caption(f"Match score: {score:.3f} · Source file: {doc['source_file']}")
                st.write(doc["text"][:1400])

else:
    st.info("Enter a question to search the indexed Buick reference material.")
