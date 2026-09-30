
import argparse
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pymongo.operations import SearchIndexModel

from core import EMBED_DIM, db, embed
from personas import PERSONAS

USER_AGENT = "VoiceNG-KB-Bot/1.0"
SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf", ".zip",
            ".mp4", ".mp3", ".avi", ".mov", ".doc", ".docx", ".xls", ".xlsx")


def normalize(url: str) -> str:
    return urlparse(url)._replace(fragment="").geturl().rstrip("/")


def extract(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    parts = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        parts.append(f"## {text}" if el.name.startswith("h") else text)

    links = [normalize(urljoin(base_url, a["href"])) for a in soup.find_all("a", href=True)]
    return title, "\n".join(parts), links


def crawl(persona: str) -> list[dict]:
    cfg = PERSONAS[persona]
    seed = cfg["seed"]
    if not seed:
        raise RuntimeError(f"No seed URL configured for {persona}")

    host = urlparse(seed).netloc
    queue, visited, pages = deque([normalize(seed)]), set(), []
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    while queue and len(pages) < cfg["max_pages"]:
        url = queue.popleft()
        if url in visited or url.lower().endswith(SKIP_EXT):
            continue
        visited.add(url)

        print(f"[crawl] {len(pages) + 1}/{cfg['max_pages']} {url}")
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
        except Exception as exc:
            print(f"[crawl] failed: {exc}")
            continue

        if "text/html" not in r.headers.get("content-type", "").lower():
            continue

        title, text, links = extract(r.text, r.url)
        if len(text.split()) >= 30:
            pages.append({"url": normalize(r.url), "title": title, "text": text})

        for link in links:
            if urlparse(link).netloc == host and link not in visited:
                queue.append(link)
        time.sleep(0.15)

    return pages


def split_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def index_pages(persona: str, pages: list[dict]):
    records = []
    for page in pages:
        for chunk in split_text(page["text"]):
            records.append({
                "persona": persona,
                "title": page["title"],
                "url": page["url"],
                "text": f"Title: {page['title']}\n\n{chunk}",
            })

    if not records:
        print("Nothing to index.")
        return

    print(f"Embedding {len(records)} chunks...")
    for i in range(0, len(records), 50):
        batch = records[i:i + 50]
        vectors = embed([r["text"] for r in batch], "RETRIEVAL_DOCUMENT")
        for rec, vec in zip(batch, vectors):
            rec["embedding"] = vec
        time.sleep(1)  # stay inside free-tier rate limits

    db.chunks.delete_many({"persona": persona})
    db.chunks.insert_many(records)
    print(f"Inserted {len(records)} chunks for {persona}")


def ensure_vector_index():
    if list(db.chunks.list_search_indexes(name="vector_index")):
        print("Vector index exists.")
        return
    db.chunks.create_search_index(
        SearchIndexModel(
            name="vector_index",
            type="vectorSearch",
            definition={
                "fields": [
                    {"type": "vector", "path": "embedding",
                     "numDimensions": EMBED_DIM, "similarity": "cosine"},
                    {"type": "filter", "path": "persona"},
                ]
            },
        )
    )
    print("Created vector_index (takes about a minute to become active).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--persona", choices=["voiceng", "fia", "both"], default="both")
    args = p.parse_args()

    targets = ["voiceng", "fia"] if args.persona == "both" else [args.persona]
    for name in targets:
        print(f"=== {name} ===")
        index_pages(name, crawl(name))
    ensure_vector_index()
