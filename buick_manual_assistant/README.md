# Buick Manual Assistant

A dark-mode starter app for a Buick Regal / Grand National manual assistant.

This project is designed for a non-developer-friendly path:
- Store public manual/reference pages as source URLs
- Ingest them into a local JSON corpus
- Search them with simple vector-style retrieval (TF-IDF now; embeddings later)
- Ask questions in a dark-mode web app

## What this package includes
- `app.py` — Streamlit app with dark theme
- `scripts/fetch_sources.py` — downloads approved public pages listed in `config/sources.json`
- `scripts/build_corpus.py` — converts downloaded HTML into a searchable JSON corpus
- `scripts/chat_local.py` — optional local CLI search test
- `config/sources.json` — starting list of public source URLs
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — dark theme settings

## What this package does not include
- No copyrighted service manual PDFs are redistributed here.
- No API keys are included.
- No hosted deployment is created automatically.

## Easiest path
1. Create a new GitHub repo.
2. Upload all files from this folder.
3. On your computer, install Python 3.11 or newer.
4. Run the fetch and build steps.
5. Run the Streamlit app locally, or deploy from GitHub to Streamlit Community Cloud.

## Local setup
Open a terminal in this project folder and run:

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt
python scripts/fetch_sources.py
python scripts/build_corpus.py
streamlit run app.py
```

## Deploy from GitHub
A simple option is Streamlit Community Cloud. It deploys directly from a GitHub repo and updates on push.

## Future upgrade
This starter uses TF-IDF retrieval so it works without API costs. You can later switch retrieval to OpenAI embeddings and a vector index.
