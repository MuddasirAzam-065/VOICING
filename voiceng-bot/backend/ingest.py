import argparse
import os
import sys
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pymongo.operations import SearchIndexModel

from core import EMBED_DIM, db, embed
from personas import PERSONAS

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf", ".zip",
            ".mp4", ".mp3", ".avi", ".mov", ".doc", ".docx", ".xls", ".xlsx")
BATCH = 16


def normalize(url: str) -> str:
    return urlparse(url)._replace(fragment="").geturl().rstrip("/")


def canonical_host(host: str) -> str:
    host = host.lower().split(":", 1)[0]
    return host[4:] if host.startswith("www.") else host


def is_internal_host(host: str, root_host: str) -> bool:
    host = canonical_host(host)
    return host == root_host or host.endswith(f".{root_host}")


def extract(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    parts = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "address"]):
        text = el.get_text(" ", strip=True)
        if text:
            parts.append(f"## {text}" if el.name.startswith("h") else text)
    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.lower().startswith(("javascript:", "data:")):
            continue
        link = normalize(urljoin(base_url, href))
        links.append(link)
        if link.lower().startswith(("mailto:", "tel:")):
            label = anchor.get_text(" ", strip=True)
            parts.append(f"Contact: {label} ({link})" if label else f"Contact: {link}")
    text = "\n".join(dict.fromkeys(parts))

    # fallback for pages built from plain <div>/<span> elements
    if len(text.split()) < 30:
        lines = [l.strip() for l in main.get_text("\n").splitlines() if l.strip()]
        text = "\n".join(dict.fromkeys(lines))
    return title, text, list(dict.fromkeys(links))


def crawl(persona: str) -> list[dict]:
    cfg = PERSONAS[persona]
    seed = cfg["seed"]
    if not seed:
        raise RuntimeError(f"No seed URL configured for {persona}")

    render = cfg.get("render", False)
    host = canonical_host(urlparse(seed).netloc)
    queue, visited, pages = deque([normalize(seed)]), set(), []
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })

    pw = browser = page = None
    if render:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        browser = pw.chromium.launch()
        page = browser.new_page(user_agent=USER_AGENT)

    try:
        while queue and len(pages) < cfg["max_pages"]:
            url = queue.popleft()
            if url in visited or url.lower().endswith(SKIP_EXT):
                continue
            visited.add(url)

            print(f"[crawl] {len(pages) + 1}/{cfg['max_pages']} {url}")
            try:
                if render:
                    page.goto(url, wait_until="networkidle", timeout=45000)
                    page.wait_for_timeout(1500)
                    html, final_url = page.content(), page.url
                else:
                    r = None
                    for attempt in range(3):
                        try:
                            r = session.get(url, timeout=60)
                            r.raise_for_status()
                            break
                        except requests.RequestException as exc:
                            if attempt == 2:
                                raise
                            print(f"[crawl] retry {attempt + 1} for {url}: {repr(exc)[:80]}")
                            time.sleep(5 * (attempt + 1))
                    if "text/html" not in r.headers.get("content-type", "").lower():
                        continue
                    html, final_url = r.text, r.url
            except Exception as exc:
                print(f"[crawl] failed: {exc}")
                continue

            title, text, links = extract(html, final_url)
            words = len(text.split())
            print(f"[crawl]   {words} words")
            if words >= 30:
                pages.append({"url": normalize(final_url), "title": title, "text": text})

            for link in links:
                parsed = urlparse(link)
                if parsed.scheme not in {"http", "https"}:
                    continue
                if is_internal_host(parsed.netloc, host) and link not in visited:
                    queue.append(link)
            time.sleep(0.15)
    finally:
        if browser:
            browser.close()
        if pw:
            pw.stop()

    return pages


def split_text(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def embed_with_retry(texts: list[str]) -> list[list[float]]:
    for attempt in range(6):
        try:
            return embed(texts, "document")
        except Exception as exc:
            if attempt == 5:
                raise
            wait = 15 * (attempt + 1)
            print(f"Embedding failed ({repr(exc)[:120]}), retrying in {wait}s...")
            time.sleep(wait)


def index_pages(persona: str, pages: list[dict]) -> bool:
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
        print(f"::error::No usable text found for {persona}. Old data kept.")
        return False

    total = len(records)
    if os.getenv("HF_TOKEN"):
        print(f"Embedding {total} chunks...")
        for i in range(0, total, BATCH):
            batch = records[i:i + BATCH]
            vectors = embed_with_retry([r["text"] for r in batch])
            for rec, vec in zip(batch, vectors):
                rec["embedding"] = vec
            print(f"  {min(i + BATCH, total)}/{total}")
    else:
        print("HF_TOKEN is not set; storing text chunks for keyword retrieval.")

    db.chunks.delete_many({"persona": persona})
    db.chunks.insert_many(records)
    print(f"Inserted {total} chunks for {persona}")
    return True


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
    failed = []
    for name in targets:
        print(f"=== {name} ===")
        try:
            if not index_pages(name, crawl(name)):
                failed.append(name)
        except Exception as exc:
            print(f"::error::{name} failed: {repr(exc)[:300]}")
            failed.append(name)

    ensure_vector_index()
    if failed:
        sys.exit(f"Failed personas: {', '.join(failed)}")
