from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "corpus" / "corpus.json"
SCRIPTS_DIR = BASE_DIR / "scripts"

st.set_page_config(page_title="Buick Manual Assistant", page_icon="🔧", layout="wide")


def _import_script(name: str):
    """Import a script from the scripts/ directory by name."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import importlib
    return importlib.import_module(name)


def ensure_corpus() -> None:
    """On first run, fetch sources and build corpus automatically."""
    if CORPUS_PATH.exists():
        return

    st.info("First run — building the Buick reference corpus. Hang tight, this takes about a minute...")

    with st.spinner("Fetching Buick reference pages from the web..."):
        fetch = _import_script("fetch_sources")
        fetch.main()

    with st.spinner("Building searchable corpus..."):
        build = _import_script("build_corpus")
        build.main()

    st.success("Corpus ready — loading the app now!")
    st.rerun()


@st.cache_resource
def load_corpus_and_vectorizer() -> tuple[list[dict], TfidfVectorizer | None, object]:
    """Load corpus once and fit vectorizer — cached for the lifetime of the server process."""
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
        return "No matching reference found. Try rephrasing or adding more source pages."
    best = hits[0][1]["text"]
    return (
        "Best matching reference text:\n\n"
        + best[:1200]
        + "\n\n---\n_Always verify specs before turning a wrench._"
    )


# --- Auto-build corpus on first deploy ---
ensure_corpus()

# --- Load corpus (cached) ---
docs, vec, matrix = load_corpus_and_vectorizer()
source_count = len({d.get("source_file", "") for d in docs}) if docs else 0

# --- UI ---
st.title("🔧 Buick Grand National — Manual Assistant")
st.caption("Reference lookup for 1982–1987 Buick Regal / Grand National")

with st.sidebar:
    st.header("Corpus status")
    st.metric("Indexed chunks", len(docs))
    st.metric("Source pages", source_count)
    if not docs:
        st.warning("Corpus not loaded yet — refresh in a moment.")
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
        st.warning("Corpus is still loading — please refresh in a moment.")
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
