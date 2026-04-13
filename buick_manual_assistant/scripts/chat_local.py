from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[1]
CORPUS_PATH = BASE_DIR / "corpus" / "corpus.json"


def load_corpus() -> list[dict]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def search(query: str, top_k: int = 5) -> list[tuple[float, dict]]:
    docs = load_corpus()
    texts = [d["text"] for d in docs]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(texts + [query])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = sorted(zip(sims, docs), key=lambda x: x[0], reverse=True)[:top_k]
    return ranked


def main() -> None:
    print("Buick Manual Assistant - local search")
    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        for score, doc in search(q):
            print(f"\n[{score:.3f}] {doc['title']} ({doc['source_file']})")
            print(doc["text"][:700])


if __name__ == "__main__":
    main()
