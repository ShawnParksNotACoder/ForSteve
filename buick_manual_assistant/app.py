from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "corpus" / "corpus.json"

st.set_page_config(page_title="Buick Manual Assistant", page_icon="🔧", layout="wide")


@st.cache_resource
def load_corpus_and_vectorizer() -> tuple[list[dict], TfidfVectorizer, object]:
    """Load corpus once and fit vectorizer — cached across all sessions."""
    if not CORPUS_PATH.exists():
        return [], None, None
    docs = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    if not docs:
        return [], None, None
    texts = [d["text"] for d in docs]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(texts)
    return docs, vec, matrix


def search_docs(query: str, docs: list[dict], vec: TfidfVectorizer, matrix, top_k: int = 5) -> list[tuple[float, dict]]:
    query_vec = vec.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()
    ranked = sorted(zip(sims, docs), key=lambda x: x[0], reverse=True)[:top_k]
    return ranked


def summarize_answer(hits: list[tuple[float, dict]]) -> str:
    if not hits:
        return "No matching reference found yet. Try adding more source pages or using a more specific question."
    best = hits[0][1]["text"]
    return (
        "Best matching reference text:\n\n"
        + best[:1200]
        + "\n\n---\n_Always verify specs before turning a wrench._"
    )


st.title("🔧 Buick Grand National — Manual Assistant")
st.caption("Reference lookup for 1982–1987 Buick Regal / Grand National")

docs, vec, matrix = load_corpus_and_vectorizer()
source_count = len({d.get("source_file", "") for d in docs}) if docs else 0

with st.sidebar:
    st.header("Corpus status")
    st.metric("Indexed chunks", len(docs))
    st.metric("Source pages", source_count)
    if not docs:
        st.warning("No corpus found. Run `fetch_sources.py` then `build_corpus.py` first.")
    else:
        st.success("Corpus loaded and ready.")
    st.markdown("---")
    st.markdown("### Example questions")
    st.markdown(
        "- What is the turbo boost pressure?\n"
        "- Powermaster brake diagnostics\n"
        "- ECM wiring info\n"
        "- Fuel pump relay circuit\n"
        "- Wastegate adjustment\n"
        "- HEI distributor timing\n"
        "- TH200-4R transmission fluid\n"
        "- Spark plug gap spec"
    )
    st.markdown("---")
    st.caption("Always verify specs before turning a wrench.")

query = st.text_input(
    "Ask a repair or reference question",
    placeholder="Example: What is the stock turbo boost pressure on a Grand National?"
)

if query:
    if not docs:
        st.error("No corpus found. Run `fetch_sources.py` and `build_corpus.py` first.")
    else:
        hits = search_docs(query, docs, vec, matrix, top_k=5)
        st.subheader("Best match")
        st.write(summarize_answer(hits))

        st.subheader("Top source matches")
        for score, doc in hits:
            with st.container(border=True):
                st.markdown(f"**{doc['title']}**")
                st.caption(f"Match score: {score:.3f} \u00b7 Source file: {doc['source_file']}")
                st.write(doc["text"][:1400])
else:
    st.info("Enter a question above to search indexed Buick Grand National reference material.")
