from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "sources.json"
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BuickManualAssistant/1.0)"
}


def safe_name(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "index"
    host = parsed.netloc.replace(".", "_")
    return f"{host}__{path}.html"


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        sources = json.load(f)

    for item in sources:
        url = item["url"]
        title = item.get("title", url)
        out_path = RAW_DIR / safe_name(url)
        print(f"Fetching: {title}\n  {url}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            out_path.write_text(resp.text, encoding="utf-8")
            print(f"  Saved -> {out_path.name}")
        except Exception as exc:
            print(f"  Failed: {exc}")


if __name__ == "__main__":
    main()
