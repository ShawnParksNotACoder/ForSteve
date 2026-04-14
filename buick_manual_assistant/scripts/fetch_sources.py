from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

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


def extract_sub_links(html: str, base_url: str, max_links: int = 30) -> list[str]:
    """Return same-domain links found on the page, capped at max_links."""
    soup = BeautifulSoup(html, "lxml")
    base_domain = urlparse(base_url).netloc
    seen: set[str] = set()
    links: list[str] = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith("#"):
            continue
        absolute = urljoin(base_url, href).split("#")[0]
        if urlparse(absolute).netloc != base_domain:
            continue
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
        if len(links) >= max_links:
            break

    return links


def fetch_and_save(url: str, fetched: set[str]) -> str | None:
    """Fetch url and save to RAW_DIR. Returns html on success, None on failure."""
    if url in fetched:
        return None
    out_path = RAW_DIR / safe_name(url)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        html = resp.text
        out_path.write_text(html, encoding="utf-8")
        fetched.add(url)
        print(f"    Saved -> {out_path.name}")
        return html
    except Exception as exc:
        print(f"    Failed: {exc}")
        fetched.add(url)  # Don't retry failures
        return None


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        sources = json.load(f)

    fetched: set[str] = set()

    for item in sources:
        url = item["url"]
        title = item.get("title", url)
        print(f"\nFetching: {title}\n  {url}")

        html = fetch_and_save(url, fetched)
        if html is None:
            continue

        sub_links = extract_sub_links(html, url, max_links=30)
        if sub_links:
            print(f"  Crawling {len(sub_links)} sub-pages...")
            for sub_url in sub_links:
                fetch_and_save(sub_url, fetched)

    print(f"\nDone. {len(fetched)} pages saved to {RAW_DIR}")


if __name__ == "__main__":
    main()
