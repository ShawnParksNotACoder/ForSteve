from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_PATH = BASE_DIR / "corpus" / "corpus.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def extract_html(file_path: Path) -> dict:
    html = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(" ", strip=True) if soup.title else file_path.name

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = normalize(soup.get_text(" ", strip=True))
    chunks = chunk_text(text)

    return {
        "source_file": file_path.name,
        "title": title,
        "chunks": chunks,
    }


def main() -> None:
    records = []
    for path in sorted(RAW_DIR.glob("*.html")):
        try:
            record = extract_html(path)
            for i, chunk in enumerate(record["chunks"]):
                records.append(
                    {
                        "id": f"{path.stem}-{i}",
                        "title": record["title"],
                        "source_file": record["source_file"],
                        "text": chunk,
                    }
                )
            print(f"Indexed {path.name}")
        except Exception as exc:
            print(f"Failed to parse {path.name}: {exc}")

    OUT_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved corpus -> {OUT_PATH}")


if __name__ == "__main__":
    main()
