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
LOGO_PATH = BASE_DIR / "assets" / "ghost_rider.png"

st.set_page_config(
    page_title="Ghost Rider — GN Shop Manual",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Gloss black body */
  .stApp { background: #080808; }

  /* Cyan headings */
  h1, h2, h3 { color: #00D4FF !important; letter-spacing: 0.04em; }

  /* Result card: subtle cyan border, dark fill */
  [data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #00D4FF33 !important;
    background: #111111 !important;
    border-radius: 6px;
  }

  /* Orange primary buttons */
  .stButton > button[kind="primary"],
  .stButton > button {
    background: #FF6A00 !important;
    color: #080808 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 4px !important;
    letter-spacing: 0.05em;
  }
  .stButton > button:hover {
    background: #FF8C00 !important;
    box-shadow: 0 0 12px #FF6A0066;
  }

  /* Cyan outline for secondary/expander buttons */
  details > summary {
    color: #00D4FF !important;
    font-size: 0.85rem;
  }

  /* Expander border */
  [data-testid="stExpander"] {
    border: 1px solid #00D4FF22 !important;
    border-radius: 4px;
  }

  /* Score badge */
  .score-badge {
    color: #00D4FF;
    font-size: 0.75rem;
    font-family: monospace;
  }

  /* Text input glow */
  .stTextInput > div > div > input {
    background: #141414 !important;
    border: 1px solid #FF6A0066 !important;
    color: #E8E8E8 !important;
    border-radius: 4px !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #FF6A00 !important;
    box-shadow: 0 0 8px #FF6A0044 !important;
  }

  /* Caption text */
  .stCaption { color: #888 !important; }

  /* Divider */
  hr { border-color: #222 !important; }

  /* Sidebar logo area */
  [data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid #FF6A0033;
  }

  /* Hide Streamlit branding */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Splash screen ─────────────────────────────────────────────────────────────
if "splash_dismissed" not in st.session_state:
    st.session_state.splash_dismissed = False

if not st.session_state.splash_dismissed:
    st.markdown("<br><br>", unsafe_allow_html=True)

    if LOGO_PATH.exists():
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding: 3rem 0 1rem 0;'>
          <span style='
            font-size: 3.5rem;
            font-weight: 900;
            font-family: Georgia, serif;
            background: linear-gradient(90deg, #FF6A00 0%, #FFB347 50%, #FF6A00 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 0.08em;
          '>GHOST RIDER</span>
          <p style='color:#00D4FF; font-size:1rem; margin-top:0.5rem;
                    font-family:monospace; letter-spacing:0.1em;'>
            1984 BUICK GRAND NATIONAL
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button("🔧  Enter the Shop", use_container_width=True, type="primary"):
            st.session_state.splash_dismissed = True
            st.rerun()

    st.markdown("""
    <p style='text-align:center; color:#444; font-size:0.75rem;
              margin-top:3rem; font-family:monospace;'>
      1984 BUICK REGAL 3.8L TURBO V6 · SHOP MANUAL REFERENCE
    </p>
    """, unsafe_allow_html=True)
    st.stop()


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    text = re.sub(r"!\[.*?\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\.md[^)]*\)", r"**\1**", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Knowledge base ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading shop manual…")
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
        search_text = " ".join([title] + keywords + [body])
        records.append({
            "path": doc["path"],
            "title": title,
            "keywords": keywords,
            "category": fm.get("category", ""),
            "system": fm.get("system", ""),
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


def do_search(query: str, records, vec, matrix, aliases, top_k: int = 8):
    expanded = expand_query(query, aliases)
    q_vec = vec.transform([expanded])
    sims = cosine_similarity(q_vec, matrix).flatten()
    ranked = sorted(zip(sims, records), key=lambda x: x[0], reverse=True)
    return [(s, d) for s, d in ranked[:top_k] if s > 0.01]


CATEGORY_BADGE = {
    "technical-service-bulletins": ("🟡", "TSB"),
    "specifications":              ("🟢", "Specs"),
    "service-and-repair":          ("🔵", "Service & Repair"),
    "repair-and-diagnosis":        ("🔴", "Repair & Diagnosis"),
    "maintenance":                 ("🟠", "Maintenance"),
}

QUICK_QUERIES = [
    ("cranks but won't start", "cranks but won't start"),
    ("no spark",               "no spark"),
    ("fuel pressure",          "fuel pressure low"),
    ("boost issue",            "boost issue turbo"),
    ("rough idle",             "rough idle"),
    ("charging problem",       "charging problem alternator"),
    ("spark plug gap",         "spark plug gap"),
    ("transmission fluid",     "TH200-4R transmission fluid"),
]

POPULAR = [
    ("Spark plug gap",       "spark plug gap"),
    ("Turbo boost pressure", "turbo boost pressure"),
    ("Fuel pump pressure",   "fuel pump pressure"),
    ("Timing specs",         "timing specifications"),
    ("ECM trouble codes",    "ECM diagnostic trouble codes"),
    ("TH200-4R fluid",       "TH200-4R transmission fluid"),
    ("Wastegate adjustment", "wastegate"),
    ("Ignition module",      "ignition control module"),
]


# ── Load ──────────────────────────────────────────────────────────────────────
records, vec, matrix, aliases = load_knowledge_base()


# ── Sidebar (mini logo + nav) ─────────────────────────────────────────────────
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)
    else:
        st.markdown(
            "<p style='color:#FF6A00; font-weight:900; font-size:1.1rem;"
            " font-family:Georgia; letter-spacing:0.05em;'>👻 GHOST RIDER</p>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<p style='color:#00D4FF; font-size:0.7rem; font-family:monospace;"
        " letter-spacing:0.1em; margin-top:-0.5rem;'>GN SHOP MANUAL</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#888; font-size:0.75rem;'>3,266 manual pages indexed.<br>"
        "1984 Buick Regal 3.8L Turbo V6</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#FF6A00; font-size:0.7rem; font-family:monospace;'>"
        "⚠ Always verify specs<br>before turning a wrench.</p>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; font-size:1.6rem; letter-spacing:0.12em;'>"
    "👻 GHOST RIDER</h1>"
    "<p style='text-align:center; color:#888; font-size:0.78rem; "
    "font-family:monospace; letter-spacing:0.08em; margin-top:-0.8rem;'>"
    "1984 BUICK GRAND NATIONAL · SHOP MANUAL</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# Quick diagnostic shortcuts
with st.expander("⚡ Quick diagnostic searches", expanded=False):
    cols = st.columns(2)
    for i, (label, term) in enumerate(QUICK_QUERIES):
        if cols[i % 2].button(label, key=f"quick_{i}", use_container_width=True):
            st.session_state["_query"] = term

# Search box
default_query = st.session_state.pop("_query", "")
query = st.text_input(
    "search",
    value=default_query,
    placeholder="e.g.  spark plug gap · boost pressure · fuel pump · timing",
    label_visibility="collapsed",
)

st.markdown("---")

# ── Results ───────────────────────────────────────────────────────────────────
if query:
    hits = do_search(query, records, vec, matrix, aliases, top_k=8)

    if not hits:
        st.warning("No results. Try different keywords.")
    else:
        st.markdown(
            f"<p style='color:#888; font-size:0.8rem; font-family:monospace;'>"
            f"{len(hits)} results for: <span style='color:#00D4FF;'>{query}</span></p>",
            unsafe_allow_html=True,
        )

        for score, doc in hits:
            icon, label = CATEGORY_BADGE.get(doc["category"], ("📄", "Manual"))
            system_label = doc["system"].replace("-", " ").title()

            with st.container(border=True):
                col_title, col_score = st.columns([5, 1])
                with col_title:
                    st.markdown(
                        f"<p style='color:#00D4FF; font-size:1rem; font-weight:700;"
                        f" margin-bottom:0; font-family:monospace;'>{doc['title']}</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<p style='color:#FF6A00; font-size:0.72rem; margin-top:0;"
                        f" font-family:monospace;'>{icon} {label}"
                        + (f"  ·  {system_label}" if system_label else "")
                        + "</p>",
                        unsafe_allow_html=True,
                    )
                with col_score:
                    st.markdown(
                        f"<p class='score-badge' style='text-align:right;"
                        f" margin-top:0.3rem;'>{score:.2f}</p>",
                        unsafe_allow_html=True,
                    )

                preview = clean_for_display(doc["body"])[:500].strip()
                if preview:
                    st.markdown(preview + ("…" if len(doc["body"]) > 500 else ""))

                with st.expander("Full document"):
                    st.markdown(clean_for_display(doc["body"]))

# ── Home screen ───────────────────────────────────────────────────────────────
else:
    st.markdown(
        "<p style='color:#888; font-size:0.85rem;'>"
        "Full repair reference for the Grand National platform. Search above "
        "or tap a shortcut below.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Popular searches")
    col1, col2 = st.columns(2)
    for i, (label, term) in enumerate(POPULAR):
        col = col1 if i % 2 == 0 else col2
        if col.button(label, key=f"pop_{i}", use_container_width=True):
            st.session_state["_query"] = term
            st.rerun()

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#333; font-size:0.7rem; font-family:monospace;'>"
    "GHOST RIDER · 1984 BUICK REGAL 3.8L TURBO VIN 9 · SHOP MANUAL</p>",
    unsafe_allow_html=True,
)
