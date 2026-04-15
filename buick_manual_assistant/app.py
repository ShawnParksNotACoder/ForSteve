from __future__ import annotations

import base64
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
IMAGES_DIR = KB_DIR / "images" / "DM14Q313" / "gm100"
LOGO_PATH      = BASE_DIR / "assets" / "ghost_rider.png"
SMOOTH_GIF_PATH = BASE_DIR / "static" / "ghost_rider_smooth.gif"

st.set_page_config(
    page_title="Ghost Rider — GN Shop Manual",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── App base — animated flame background ────────────────────────── */
  html, body { background-color: #080808 !important; }
  /* Fixed pseudo-element sits below everything, visible through transparent containers */
  body::before {
    content: "" !important;
    position: fixed !important;
    inset: 0 !important;
    background-image:
      linear-gradient(to bottom,
        rgba(8,8,8,0.50) 0%,
        rgba(8,8,8,0.50) 50%,
        rgba(8,8,8,0.88) 72%,
        rgba(8,8,8,1.00) 85%
      ),
      url('/app/static/ghost_rider_flames_bg.gif') !important;
    background-size: cover, 100% auto !important;
    background-position: top center, top center !important;
    background-repeat: no-repeat, no-repeat !important;
    z-index: -1 !important;
    pointer-events: none !important;
  }
  /* Every Streamlit layer must be transparent or the GIF is hidden */
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  section.main {
    background: transparent !important;
  }
  h1, h2, h3 { color: #00D4FF !important; letter-spacing: 0.04em; }

  /* ── Responsive layout — centered layout, widened ───────────────── */
  div.block-container {
    max-width: 900px !important;
    padding: 1rem 2rem 3rem !important;
  }
  [role="tabpanel"] { padding: 0 !important; }
  @media (max-width: 768px) {
    div.block-container {
      max-width: 100% !important;
      padding: 0.75rem 0.75rem 2rem !important;
    }
  }


  /* ── Cards / containers ──────────────────────────────────────────── */
  [data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #00D4FF1A !important;
    background: #0e0e0e !important;
    border-radius: 16px !important;
  }

  /* ── Buttons ─────────────────────────────────────────────────────── */
  .stButton > button[kind="primary"],
  .stButton > button {
    background: #FF6A00 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    letter-spacing: 0.05em;
    padding: 0.45rem 1.2rem !important;
    transition: background 0.15s ease, color 0.15s ease, box-shadow 0.15s ease !important;
  }
  .stButton > button:hover {
    background: #FF8C00 !important;
    color: #080808 !important;
    box-shadow: 0 0 18px #FF6A0055 !important;
  }

  /* ── Expanders ───────────────────────────────────────────────────── */
  [data-testid="stExpander"] {
    border: 1px solid #00D4FF1A !important;
    border-radius: 14px !important;
    overflow: hidden;
  }
  /* Summary row: flex, centered, remove default marker */
  [data-testid="stExpander"] details summary {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.8rem !important;
    list-style: none !important;
    cursor: pointer !important;
  }
  [data-testid="stExpander"] details summary::-webkit-details-marker { display: none !important; }
  /* Hide Streamlit's default chevron — target both the SVG and its wrapping div */
  [data-testid="stExpander"] details summary svg { display: none !important; }
  [data-testid="stExpander"] details summary > div:last-child { display: none !important; }
  /* Text div: don't stretch, center its text */
  [data-testid="stExpander"] details summary > div:first-child {
    flex: 0 0 auto !important;
    text-align: center !important;
    color: #00D4FF !important;
    font-size: 0.85rem !important;
  }
  [data-testid="stExpander"] details summary p {
    color: #00D4FF !important;
    text-align: center !important;
    margin: 0 !important;
  }
  /* Flip triangles: ::before/::after become flex items flanking the text div */
  [data-testid="stExpander"] details:not([open]) summary::before,
  [data-testid="stExpander"] details:not([open]) summary::after {
    content: "▼" !important;
    color: #00D4FF !important;
    font-size: 1.4rem !important;
    line-height: 1 !important;
    flex-shrink: 0 !important;
  }
  [data-testid="stExpander"] details[open] summary::before,
  [data-testid="stExpander"] details[open] summary::after {
    content: "▲" !important;
    color: #00D4FF !important;
    font-size: 1.4rem !important;
    line-height: 1 !important;
    flex-shrink: 0 !important;
  }

  /* ── Text inputs ─────────────────────────────────────────────────── */
  .stTextInput > div > div > input {
    background: #141414 !important;
    border: 1px solid #FF6A0055 !important;
    color: #E8E8E8 !important;
    border-radius: 12px !important;
    padding: 0.5rem 1rem !important;
    text-align: center !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #FF6A00 !important;
    box-shadow: 0 0 12px #FF6A0033 !important;
  }

  /* ── Pills / filter chips — centered, wrapping ───────────────────── */
  [data-testid="stPills"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
  }
  [data-testid="stPills"] > div {
    display: flex !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
    gap: 6px !important;
  }
  [data-testid="stPills"] button {
    border-radius: 20px !important;
    border: 1px solid #FF6A0055 !important;
    background: #141414 !important;
    color: #E8E8E8 !important;
    font-size: 0.8rem !important;
    white-space: nowrap !important;
  }
  [data-testid="stPills"] button[aria-selected="true"] {
    background: #FF6A00 !important;
    color: #080808 !important;
    border-color: #FF6A00 !important;
  }

  /* ── Sweettart round nav tabs ────────────────────────────────────── */
  [data-testid="stTabs"] { overflow: visible !important; }
  [data-testid="stTabs"] [role="tablist"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 14px !important;
    border-bottom: 1px solid #1e1e1e !important;
    padding: 4px 0 50px !important;
    overflow: visible !important;
  }
  /* Outer circle */
  [data-testid="stTabs"] [role="tab"] {
    width: 80px !important;
    min-width: 80px !important;
    max-width: 80px !important;
    height: 80px !important;
    border-radius: 50% !important;
    background: #282828 !important;
    border: 3px solid #363636 !important;
    color: #bbb !important;
    font-size: 2.4rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    flex-shrink: 0 !important;
    font-family: unset !important;
    letter-spacing: 0 !important;
    transition: all 0.18s ease !important;
    position: relative !important;
    overflow: visible !important;
  }
  /* Hide Streamlit's indicator line div (not the content div) */
  [data-testid="stTabs"] [role="tab"] > div:not([data-testid="stMarkdownContainer"]) {
    display: none !important;
  }
  [data-testid="stTabs"] [role="tab"] p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
    color: inherit !important;
    font-size: 2.8rem !important;   /* explicit — not inherit, Streamlit overrides that */
  }
  /* Label text below each circle via CSS ::after */
  [data-testid="stTabs"] [role="tab"]::after {
    position: absolute !important;
    bottom: -34px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    font-size: 1rem !important;     /* was 0.58rem — 1.75x bigger */
    font-family: monospace !important;
    letter-spacing: 0.1em !important;
    color: #555 !important;
    white-space: nowrap !important;
  }
  [data-testid="stTabs"] [role="tab"]:nth-child(1)::after { content: "SEARCH" !important; }
  [data-testid="stTabs"] [role="tab"]:nth-child(2)::after { content: "DIAGRAMS" !important; }
  [data-testid="stTabs"] [role="tab"]:nth-child(3)::after { content: "SPECS" !important; }
  [data-testid="stTabs"] [role="tab"]:nth-child(4)::after { content: "CODES" !important; }
  [data-testid="stTabs"] [role="tab"]:nth-child(5)::after { content: "TSBs" !important; }
  /* Hover */
  [data-testid="stTabs"] [role="tab"]:hover {
    border-color: #FF6A00 !important;
    background: #2a1500 !important;
  }
  [data-testid="stTabs"] [role="tab"]:hover::after { color: #FF6A0099 !important; }
  /* Active / selected */
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #FF6A00 !important;
    border-color: #FF8C00 !important;
    color: #FFF !important;
    box-shadow: 0 0 30px #FF6A0088, 0 4px 18px rgba(0,0,0,0.65) !important;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {
    color: #FF6A00 !important;
  }
  /* Suppress any browser or Streamlit underline on tab labels */
  [data-testid="stTabs"] [role="tab"],
  [data-testid="stTabs"] [role="tab"]::after {
    text-decoration: none !important;
    border-bottom: none !important;
  }
  /* Mobile: scale circles to fit 5 across — still clearly readable */
  @media (max-width: 500px) {
    [data-testid="stTabs"] [role="tablist"] { gap: 8px !important; padding-bottom: 46px !important; }
    [data-testid="stTabs"] [role="tab"] {
      width: 60px !important;
      min-width: 60px !important;
      max-width: 60px !important;
      height: 60px !important;
      border-width: 2.5px !important;
    }
    [data-testid="stTabs"] [role="tab"] p {
      font-size: 2.2rem !important;
    }
    [data-testid="stTabs"] [role="tab"]::after {
      font-size: 0.77rem !important;  /* −10% from 0.85rem */
      bottom: -28px !important;
    }
  }

  /* ── Diagram images — pinch-zoomable ─────────────────────────────── */
  [data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #00D4FF1A;
    touch-action: pinch-zoom;
  }

  .stCaption { color: #777 !important; }
  hr { border-color: #1c1c1c !important; }

  /* ── Sidebar ─────────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: #0a0a0a !important;
    border-right: 1px solid #FF6A0022;
  }

  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Splash ────────────────────────────────────────────────────────────────────
if "splash_dismissed" not in st.session_state:
    st.session_state.splash_dismissed = False

if not st.session_state.splash_dismissed:
    # Embed smooth GIF as base64 — reliable, no dependency on static serving URL.
    if SMOOTH_GIF_PATH.exists():
        _s_b64 = base64.b64encode(SMOOTH_GIF_PATH.read_bytes()).decode()
        _splash_src = f"data:image/gif;base64,{_s_b64}"
    elif LOGO_PATH.exists():
        _s_b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        _splash_src = f"data:image/png;base64,{_s_b64}"
    else:
        _splash_src = ""

    _splash_img = (
        f'<img src="{_splash_src}" style="width:100%; border-radius:16px; display:block;">'
        if _splash_src else
        "<p style='color:#FF6A00; font-size:2rem; font-weight:900;'>👻 GHOST RIDER</p>"
    )

    st.markdown(f"""
    <div style="min-height:70vh; display:flex; flex-direction:column;
                align-items:center; justify-content:center; padding:2rem 1rem;">
      <div style="max-width:460px; width:100%;
        background:linear-gradient(145deg,rgba(255,255,255,0.08) 0%,rgba(255,255,255,0.02) 100%);
        backdrop-filter:blur(32px); -webkit-backdrop-filter:blur(32px);
        border:1px solid rgba(255,255,255,0.13);
        border-top:1px solid rgba(255,255,255,0.22);
        border-left:1px solid rgba(255,255,255,0.16);
        border-radius:28px;
        box-shadow:0 24px 56px rgba(0,0,0,0.75),0 8px 20px rgba(0,0,0,0.5),
          inset 0 1px 0 rgba(255,255,255,0.12),0 0 80px rgba(255,100,0,0.12);
        padding:1.75rem 1.75rem 1.25rem; text-align:center;">
        {_splash_img}
        <p style="color:#00D4FF; font-family:monospace; font-size:0.78rem;
                  letter-spacing:0.18em; margin:1rem 0 0.25rem;">
          1984 BUICK GRAND NATIONAL</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button("🔧  Enter the Shop", use_container_width=True, type="primary"):
            st.session_state.splash_dismissed = True
            st.rerun()
    st.markdown(
        "<p style='text-align:center; color:#333; font-size:0.7rem;"
        " font-family:monospace; margin-top:1.5rem;'>"
        "3.8L TURBO V6 · VIN 9 · SHOP MANUAL</p>",
        unsafe_allow_html=True,
    )
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


def extract_image_paths(text: str) -> list[Path]:
    """Resolve all PNG references in doc body to local file paths."""
    filenames = re.findall(r"!\[.*?\]\([^)]*?/([^/\s]+\.png)\)", text)
    return [IMAGES_DIR / fn for fn in filenames if (IMAGES_DIR / fn).exists()]


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
        system = fm.get("system", "")
        img_paths = extract_image_paths(body)
        search_text = " ".join([title] + keywords + [body])
        records.append({
            "path": doc["path"],
            "title": title,
            "keywords": keywords,
            "category": fm.get("category", ""),
            "system": system,
            "body": body,
            "search_text": search_text,
            "has_images": len(img_paths) > 0,
            "image_paths": [str(p) for p in img_paths],
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


def do_search(query: str, records, vec, matrix, aliases,
              system_filter: str | None = None, top_k: int = 10):
    expanded = expand_query(query, aliases)
    q_vec = vec.transform([expanded])
    sims = cosine_similarity(q_vec, matrix).flatten()
    ranked = sorted(zip(sims, records), key=lambda x: x[0], reverse=True)
    hits = [(s, d) for s, d in ranked if s > 0.01]
    if system_filter:
        hits = [(s, d) for s, d in hits if d["system"] == system_filter]
    return hits[:top_k]


# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORY_BADGE = {
    "technical-service-bulletins": ("🟡", "TSB"),
    "specifications":              ("🟢", "Specs"),
    "service-and-repair":          ("🔵", "Service & Repair"),
    "repair-and-diagnosis":        ("🔴", "Repair & Diag"),
    "maintenance":                 ("🟠", "Maintenance"),
}

SYSTEM_FILTER_OPTIONS = [
    ("All",          None),
    ("Electrical",   "starting-and-charging"),
    ("Transmission", "transmission-and-drivetrain"),
    ("Suspension",   "steering-and-suspension"),
    ("Specs",        "specifications"),
    ("TSBs",         "technical-service-bulletins"),
    ("Sensors",      "sensors-and-switches"),
]
SYSTEM_FILTER_LABELS = [label for label, _ in SYSTEM_FILTER_OPTIONS]
SYSTEM_FILTER_MAP   = {label: val for label, val in SYSTEM_FILTER_OPTIONS}

DIAGRAM_SYSTEMS = {
    "All Diagrams":   None,
    "Electrical":     "starting-and-charging",
    "Transmission":   "transmission-and-drivetrain",
    "Suspension":     "steering-and-suspension",
    "Sensors":        "sensors-and-switches",
    "Wipers/Windows": "wiper-and-washer-systems",
}

SPEC_SUBCATS = {
    "⚙️  Mechanical":           "mechanical-specifications",
    "⚡  Electrical":            "electrical-specifications",
    "🧪  Capacities":            "capacity-specifications",
    "🌡️  Pressure & Temp":      "pressure-vacuum-and-temperature-specifications",
    "💧  Fluid Types":           "fluid-type-specifications",
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    img_count = sum(1 for r in records if r["has_images"])
    st.markdown(
        f"<p style='color:#888; font-size:0.75rem;'>"
        f"3,266 manual pages<br>"
        f"{img_count:,} pages with diagrams<br>"
        f"1984 Buick Regal 3.8L Turbo V6</p>",
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
    "<p style='text-align:center; color:#888; font-size:0.78rem;"
    " font-family:monospace; letter-spacing:0.08em; margin-top:-0.8rem;'>"
    "1984 BUICK GRAND NATIONAL · SHOP MANUAL</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_search, tab_diagrams, tab_specs, tab_codes, tab_tsbs = st.tabs([
    "🔍", "📐", "📋", "⚠️", "📣",
])


# ════════════════════════════════════════════════════════════════════════════
# SEARCH TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_search:
    with st.expander("⚡  QUICK DIAGNOSTIC SEARCHES  ⚡", expanded=False):
        cols = st.columns(2)
        for i, (label, term) in enumerate(QUICK_QUERIES):
            if cols[i % 2].button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state["_query"] = term

    default_query = st.session_state.pop("_query", "")
    query = st.text_input(
        "search",
        value=default_query,
        placeholder="Search the shop manual…",
        label_visibility="collapsed",
    )

    # System filter pills with label — columns force reliable centering
    st.markdown(
        "<p style='color:#888; font-size:0.75rem; font-family:monospace;"
        " letter-spacing:0.1em; margin-bottom:0.2rem; text-align:center;'>FILTER:</p>",
        unsafe_allow_html=True,
    )
    _pl, _pm, _pr = st.columns([1, 6, 1])
    with _pm:
        selected_filter = st.pills(
            "Filter",
            options=SYSTEM_FILTER_LABELS,
            default="All",
            label_visibility="collapsed",
        )
    system_filter = SYSTEM_FILTER_MAP.get(selected_filter)

    st.markdown("---")

    if query:
        hits = do_search(query, records, vec, matrix, aliases,
                         system_filter=system_filter, top_k=10)

        if not hits:
            st.warning("No results. Try different keywords or clear the filter.")
        else:
            st.markdown(
                f"<p style='color:#888; font-size:0.8rem; font-family:monospace;'>"
                f"{len(hits)} results for: <span style='color:#00D4FF;'>{query}</span></p>",
                unsafe_allow_html=True,
            )

            for score, doc in hits:
                icon, badge_label = CATEGORY_BADGE.get(doc["system"],
                    CATEGORY_BADGE.get(doc["category"], ("📄", "Manual")))
                system_label = doc["system"].replace("-", " ").title()
                img_paths = [Path(p) for p in doc["image_paths"]]

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
                            f" font-family:monospace;'>{icon} {system_label}</p>",
                            unsafe_allow_html=True,
                        )
                    with col_score:
                        st.markdown(
                            f"<p style='color:#00D4FF; font-size:0.75rem; font-family:monospace;"
                            f" text-align:right; margin-top:0.3rem;'>{score:.2f}</p>",
                            unsafe_allow_html=True,
                        )

                    body_clean = clean_for_display(doc["body"])
                    PREVIEW = 400
                    if len(body_clean) <= PREVIEW:
                        st.markdown(body_clean)
                    else:
                        break_at = body_clean.rfind("\n", 0, PREVIEW)
                        if break_at < 100:
                            break_at = body_clean.rfind(" ", 0, PREVIEW)
                        if break_at < 0:
                            break_at = PREVIEW
                        st.markdown(body_clean[:break_at].strip() + "…")
                        with st.expander("Read more ↓"):
                            st.markdown(body_clean[break_at:].strip())

                    # Show diagrams inline (cap at 3 per result)
                    if img_paths:
                        with st.expander(f"📐 Diagrams ({len(img_paths)})", expanded=False):
                            for p in img_paths[:3]:
                                st.image(str(p), use_container_width=True)
                            if len(img_paths) > 3:
                                st.caption(f"+ {len(img_paths) - 3} more diagrams in full document.")

    else:
        st.markdown(
            "<p style='color:#888; font-size:0.85rem; text-align:center;'>"
            "Full repair reference for the Grand National platform. Search above "
            "or tap a shortcut below.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='text-align:center;'>Popular searches</h4>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        for i, (label, term) in enumerate(POPULAR):
            col = col1 if i % 2 == 0 else col2
            if col.button(label, key=f"pop_{i}", use_container_width=True):
                st.session_state["_query"] = term
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# DIAGRAMS TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_diagrams:
    st.markdown(
        "<p style='color:#888; font-size:0.85rem;'>"
        "Browse wiring diagrams, component locations, and schematics by system.</p>",
        unsafe_allow_html=True,
    )

    diag_system = st.selectbox(
        "System",
        options=list(DIAGRAM_SYSTEMS.keys()),
        label_visibility="collapsed",
    )
    selected_sys = DIAGRAM_SYSTEMS[diag_system]

    # Filter to docs that have images, optionally by system
    diagram_docs = [
        r for r in records
        if r["has_images"] and (selected_sys is None or r["system"] == selected_sys)
    ]

    # Further filter to docs with "diagram" in path (actual diagram pages first),
    # then fall back to any doc with images
    explicit_diagrams = [d for d in diagram_docs if "diagram" in d["path"]]
    other_with_images = [d for d in diagram_docs if "diagram" not in d["path"]]
    ordered = explicit_diagrams + other_with_images

    st.markdown(
        f"<p style='color:#888; font-size:0.78rem; font-family:monospace;'>"
        f"{len(ordered)} pages with diagrams</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not ordered:
        st.info("No diagrams found for this system.")
    else:
        for doc in ordered[:60]:   # cap for performance
            img_paths = [Path(p) for p in doc["image_paths"]]
            if not img_paths:
                continue
            system_label = doc["system"].replace("-", " ").title()
            with st.container(border=True):
                st.markdown(
                    f"<p style='color:#00D4FF; font-size:0.95rem; font-weight:700;"
                    f" margin-bottom:0; font-family:monospace;'>{doc['title']}</p>"
                    f"<p style='color:#FF6A00; font-size:0.7rem; margin-top:0;"
                    f" font-family:monospace;'>{system_label}</p>",
                    unsafe_allow_html=True,
                )
                # Show first image always — full width for mobile readability
                st.image(str(img_paths[0]), use_container_width=True)

                # Remaining images + text in expander
                if len(img_paths) > 1 or clean_for_display(doc["body"]).strip():
                    with st.expander("More ↓"):
                        for p in img_paths[1:]:
                            st.image(str(p), use_container_width=True)
                        body_clean = clean_for_display(doc["body"])
                        if body_clean:
                            st.markdown(body_clean)

# ════════════════════════════════════════════════════════════════════════════
# SPECS TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_specs:
    st.markdown(
        "<p style='color:#888; font-size:0.85rem;'>"
        "Factory specifications grouped by type. Expand a category to browse all docs.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    for section_label, subcat_key in SPEC_SUBCATS.items():
        section_docs = [
            r for r in records
            if r["system"] == "specifications"
            and subcat_key in r["path"]
        ]
        if not section_docs:
            continue

        with st.expander(f"{section_label}  ·  {len(section_docs)} docs", expanded=False):
            for doc in section_docs:
                body_clean = clean_for_display(doc["body"])
                img_paths = [Path(p) for p in doc["image_paths"]]
                with st.container(border=True):
                    st.markdown(
                        f"<p style='color:#00D4FF; font-size:0.95rem; font-weight:700;"
                        f" margin-bottom:0; font-family:monospace;'>{doc['title']}</p>",
                        unsafe_allow_html=True,
                    )
                    # Show images if present (specs often have tables as images)
                    for p in img_paths[:2]:
                        st.image(str(p), use_container_width=True)
                    if body_clean:
                        SPEC_PREVIEW = 300
                        if len(body_clean) <= SPEC_PREVIEW:
                            st.markdown(body_clean)
                        else:
                            break_at = body_clean.rfind("\n", 0, SPEC_PREVIEW)
                            if break_at < 50:
                                break_at = SPEC_PREVIEW
                            st.markdown(body_clean[:break_at].strip() + "…")
                            with st.expander("Full spec ↓"):
                                st.markdown(body_clean[break_at:].strip())
                                for p in img_paths[2:]:
                                    st.image(str(p), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# CODES TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_codes:
    st.markdown(
        "<p style='color:#888; font-size:0.85rem;'>"
        "Diagnostic Trouble Codes (DTCs) and MIL-related service information.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Collect code docs — deduplicate by title
    seen_titles: set[str] = set()
    code_docs = []
    for r in records:
        path_lower = r["path"].lower()
        title_lower = r["title"].lower()
        if (
            "diagnostic-trouble-code" in path_lower
            or "dtc" in path_lower
            or "/dtc" in path_lower
            or "trouble-code" in path_lower
            or "dtc" in title_lower
            or "trouble code" in title_lower
            or "mil " in title_lower
        ):
            if r["title"] not in seen_titles:
                seen_titles.add(r["title"])
                code_docs.append(r)

    if not code_docs:
        st.info("No trouble code documents found.")
    else:
        st.markdown(
            f"<p style='color:#888; font-size:0.78rem; font-family:monospace;'>"
            f"{len(code_docs)} code-related documents</p>",
            unsafe_allow_html=True,
        )
        for doc in code_docs:
            body_clean = clean_for_display(doc["body"])
            img_paths = [Path(p) for p in doc["image_paths"]]
            with st.container(border=True):
                st.markdown(
                    f"<p style='color:#00D4FF; font-size:0.95rem; font-weight:700;"
                    f" margin-bottom:0; font-family:monospace;'>{doc['title']}</p>",
                    unsafe_allow_html=True,
                )
                for p in img_paths[:2]:
                    st.image(str(p), use_container_width=True)
                if body_clean:
                    PREVIEW = 400
                    if len(body_clean) <= PREVIEW:
                        st.markdown(body_clean)
                    else:
                        break_at = body_clean.rfind("\n", 0, PREVIEW)
                        if break_at < 50:
                            break_at = PREVIEW
                        st.markdown(body_clean[:break_at].strip() + "…")
                        with st.expander("Full doc ↓"):
                            st.markdown(body_clean[break_at:].strip())
                            for p in img_paths[2:]:
                                st.image(str(p), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TSBs TAB
# ════════════════════════════════════════════════════════════════════════════
with tab_tsbs:
    st.markdown(
        "<p style='color:#888; font-size:0.85rem;'>"
        "Technical Service Bulletins. Search by keyword or browse all.</p>",
        unsafe_allow_html=True,
    )

    tsb_search = st.text_input(
        "Filter TSBs",
        placeholder="e.g.  A/C · ignition · fuel · brakes",
        label_visibility="collapsed",
        key="tsb_search",
    )
    st.markdown("---")

    tsb_docs = [r for r in records if r["system"] == "technical-service-bulletins"]

    # Deduplicate by title
    seen: set[str] = set()
    unique_tsbs = []
    for r in tsb_docs:
        if r["title"] not in seen:
            seen.add(r["title"])
            unique_tsbs.append(r)

    if tsb_search:
        q = tsb_search.lower()
        unique_tsbs = [r for r in unique_tsbs
                       if q in r["title"].lower() or q in r["body"].lower()]

    st.markdown(
        f"<p style='color:#888; font-size:0.78rem; font-family:monospace;'>"
        f"{len(unique_tsbs)} bulletins</p>",
        unsafe_allow_html=True,
    )

    for doc in unique_tsbs[:80]:   # cap for performance
        body_clean = clean_for_display(doc["body"])
        img_paths = [Path(p) for p in doc["image_paths"]]
        with st.container(border=True):
            st.markdown(
                f"<p style='color:#00D4FF; font-size:0.9rem; font-weight:700;"
                f" margin-bottom:0; font-family:monospace;'>{doc['title']}</p>",
                unsafe_allow_html=True,
            )
            for p in img_paths[:1]:
                st.image(str(p), use_container_width=True)
            if body_clean:
                PREVIEW = 350
                if len(body_clean) <= PREVIEW:
                    st.markdown(body_clean)
                else:
                    break_at = body_clean.rfind("\n", 0, PREVIEW)
                    if break_at < 50:
                        break_at = PREVIEW
                    st.markdown(body_clean[:break_at].strip() + "…")
                    with st.expander("Read full bulletin ↓"):
                        st.markdown(body_clean[break_at:].strip())
                        for p in img_paths[1:]:
                            st.image(str(p), use_container_width=True)

    if len(unique_tsbs) > 80:
        st.caption(f"Showing 80 of {len(unique_tsbs)}. Use the search box to narrow results.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#333; font-size:0.7rem; font-family:monospace;'>"
    "GHOST RIDER · 1984 BUICK REGAL 3.8L TURBO VIN 9 · SHOP MANUAL</p>",
    unsafe_allow_html=True,
)
