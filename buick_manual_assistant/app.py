from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
DOCS_PATH = KB_DIR / "docs.jsonl"
ALIASES_PATH = KB_DIR / "aliases.json"

st.set_page_config(
    page_title="Grand National Shop Manual",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CATEGORY_BADGE = {
    "technical-service-bulletins": "🟡 TSB",
    "specifications": "🟢 Specs",
    "service-and-repair": "🔵 Service & Repair",
    "repair-and-diagnosis": "🔴 Repair & Diagnosis",
    "maintenance": "🟠 Maintenance",
}

QUICK_QUERIES = [
    "cranks but won't start",
    "no spark",
    "fuel pressure low",
    "boost issue",
    "rough idle",
    "charging problem",
    "spark plug gap",
    "TH200-4R transmission",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_frontmatter(content: str) -> dict:
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                return yaml.safe_load(parts[1]) or {}
            except Exception:
                pass
    return {}


def strip_frontmatter(content: str) -> str:
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return content.strip()


def clean_for_display(text: str) -> str:
    """Strip image refs and internal .md links for cleaner display."""
    text = re.sub(r"!\[.*?\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\.md[^)]*\)", r"**\1**", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Knowledge base loading ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading shop manual index…")
def load_knowledge_base():
    with open(DOCS_PATH, encoding="utf-8") as f:
        raw_docs = [json.loads(line) for line in f]

    with open(ALIASES_PATH, encoding="utf-8") as f:
        aliases = json.load(f)

    records = []
    for doc in raw_docs:
        content = doc["content"]
        fm = parse_frontmatter(content)
        body = strip_frontmatter(content)

        title = fm.get("title") or doc.get("filename", "Untitled")
        keywords = fm.get("keywords") or []
        category = fm.get("category", "")
        system = fm.get("system", "")

        search_text = " ".join([title] + keywords + [body])

        records.append({
            "path": doc["path"],
            "title": title,
            "keywords": keywords,
            "category": category,
            "system": system,
            "body": body,
            "search_text": search_text,
        })

    texts = [r["search_text"] for r in records]
    vec = TfidfVectorizer(stop_words="english", max_features=60000, ngram_range=(1, 2))
    matrix = vec.fit_transform(texts)

    return records, vec, matrix, aliases


def expand_query(query: str, aliases: dict) -> str:
    q_lower = query.lower()
    parts = [query]
    for key, terms in aliases.items():
        if key.lower() in q_lower:
            parts.extend(terms)
    return " ".join(parts)


def search(query: str, records, vec, matrix, top_k: int = 8):
    expanded = expand_query(query, st.session_state.get("_aliases", {}))
    q_vec = vec.transform([expanded])
    sims = cosine_similarity(q_vec, matrix).flatten()
    ranked = sorted(zip(sims, records), key=lambda x: x[0], reverse=True)
    return [(s, d) for s, d in ranked[:top_k] if s > 0.01]


# ── UI ────────────────────────────────────────────────────────────────────────

records, vec, matrix, aliases = load_knowledge_base()
st.session_state["_aliases"] = aliases

st.title("🔧 Grand National Shop Manual")
st.caption("1984 Buick Regal 3.8L Turbo V6 · VIN 9 · Repair & Diagnosis Reference")

# Quick-access diagnostic shortcuts
with st.expander("Quick diagnostic searches", expanded=False):
    cols = st.columns(2)
    for i, q in enumerate(QUICK_QUERIES):
        if cols[i % 2].button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state["_query"] = q

# Search box — pre-fill from quick-access buttons
default_query = st.session_state.pop("_query", "")
query = st.text_input(
    "Search the manual",
    value=default_query,
    placeholder="e.g. spark plug gap, boost pressure, fuel pump, timing",
    label_visibility="collapsed",
)

st.markdown("---")

if query:
    hits = search(query, records, vec, matrix, top_k=8)

    if not hits:
        st.warning("No results found. Try different keywords.")
    else:
        st.markdown(f"**{len(hits)} result(s)** for: _{query}_")
        st.markdown("")

        for score, doc in hits:
            badge = CATEGORY_BADGE.get(doc["category"], "📄")
            system_label = doc["system"].replace("-", " ").title() if doc["system"] else ""

            with st.container(border=True):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"### {doc['title']}")
                    if system_label:
                        st.caption(f"{badge}  ·  {system_label}")
                with col2:
                    st.caption(f"Score\n**{score:.2f}**")

                # Preview — first 600 chars of cleaned body
                preview = clean_for_display(doc["body"])[:600]
                if preview:
                    st.markdown(preview + ("…" if len(doc["body"]) > 600 else ""))

                with st.expander("Full document"):
                    st.markdown(clean_for_display(doc["body"]))

else:
    st.markdown("#### About this manual")
    st.markdown(
        "Full repair and diagnosis reference for the **1984 Buick Regal 3.8L Turbo V6** "
        "(Grand National platform). Covers engine, fuel, ignition, electrical, transmission, "
        "brakes, suspension, HVAC, and more.\n\n"
        "Use the search box above or tap a **Quick diagnostic search** to get started."
    )

    st.markdown("---")
    st.markdown("#### Popular searches")
    col1, col2 = st.columns(2)
    examples = [
        ("Spark plug gap", "spark plug gap"),
        ("Turbo boost pressure", "turbo boost pressure"),
        ("Fuel pump pressure", "fuel pump pressure"),
        ("Timing specs", "timing specifications"),
        ("ECM trouble codes", "ECM diagnostic trouble codes"),
        ("TH200-4R fluid", "TH200-4R transmission fluid"),
        ("Wastegate adjustment", "wastegate"),
        ("Ignition module", "ignition control module"),
    ]
    for i, (label, term) in enumerate(examples):
        col = col1 if i % 2 == 0 else col2
        if col.button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state["_query"] = term
            st.rerun()

st.markdown("---")
st.caption("Always verify specifications before turning a wrench. · 1984 Buick Regal 3.8L Turbo VIN 9")
