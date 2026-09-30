import argparse
import datetime
import re
import time
from collections import deque
from urllib.parse import (
    urljoin,
    urlparse,
)

import requests
from bs4 import BeautifulSoup
from pymongo.operations import SearchIndexModel

from .kb import db, embedder
from .personas import PERSONAS


USER_AGENT = "VoiceNG-KB-Bot/1.0"

BINARY_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".pdf",
    ".zip",
    ".mp4",
    ".mp3",
    ".avi",
    ".mov",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
}


def normalize_url(url: str) -> str:

    parsed = urlparse(url)

    clean = parsed._replace(
        fragment=""
    ).geturl()

    return clean.rstrip("/")


def is_same_domain(
    base_url: str,
    target_url: str,
) -> bool:

    base = urlparse(base_url).netloc
    target = urlparse(target_url).netloc

    return base == target


def is_binary(url: str) -> bool:

    path = urlparse(url).path.lower()

    return any(
        path.endswith(ext)
        for ext in BINARY_EXTENSIONS
    )


def extract_page(
    url: str,
    html: str,
) -> tuple[str, list[tuple[str, str]]]:

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "form",
            "aside",
        ]
    ):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    title = ""

    if soup.title:
        title = soup.title.get_text(
            " ",
            strip=True,
        )

    sections = []

    current_heading = title

    for element in main.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "p",
            "li",
            "td",
        ]
    ):

        text = element.get_text(
            " ",
            strip=True,
        )

        if not text:
            continue

        if element.name in {
            "h1",
            "h2",
            "h3",
            "h4",
        }:
            current_heading = text

            continue

        sections.append(
            (
                current_heading,
                text,
            )
        )

    links = []

    for anchor in soup.find_all("a", href=True):

        href = anchor["href"]

        absolute = normalize_url(
            urljoin(url, href)
        )

        links.append(
            (
                absolute,
                anchor.get_text(
                    " ",
                    strip=True,
                ),
            )
        )

    return title, sections, links


def crawl(
    persona: str,
) -> list[dict]:

    config = PERSONAS[persona]

    seeds = [
        s
        for s in config["seeds"]
        if s
    ]

    if not seeds:
        raise RuntimeError(
            f"No seed URL configured for {persona}"
        )

    max_pages = config["max_pages"]

    queue = deque(
        normalize_url(url)
        for url in seeds
    )

    visited = set()

    pages = []

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
        }
    )

    while queue and len(pages) < max_pages:

        url = queue.popleft()

        if url in visited:
            continue

        visited.add(url)

        if is_binary(url):
            continue

        if not any(
            is_same_domain(seed, url)
            for seed in seeds
        ):
            continue

        print(
            f"[crawl] {len(pages)+1}/{max_pages} {url}"
        )

        try:

            response = session.get(
                url,
                timeout=20,
                allow_redirects=True,
            )

            response.raise_for_status()

        except Exception as exc:

            print(
                f"[crawl] failed: {url} -> {exc}"
            )

            continue

        content_type = response.headers.get(
            "content-type",
            "",
        ).lower()

        if "text/html" not in content_type:
            continue

        title, sections, links = extract_page(
            response.url,
            response.text,
        )

        words = sum(
            len(text.split())
            for _, text in sections
        )

        if words >= 30:

            pages.append(
                {
                    "url": normalize_url(
                        response.url
                    ),
                    "title": title,
                    "sections": sections,
                }
            )

        for link, _ in links:

            if link in visited:
                continue

            if not is_same_domain(
                seeds[0],
                link,
            ):
                continue

            if is_binary(link):
                continue

            queue.append(link)

        time.sleep(0.15)

    return pages


def slugify(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "-",
        text,
    )

    return text.strip("-")[:100] or "document"


def save_documents(
    persona: str,
    pages: list[dict],
):

    db.documents.delete_many(
        {
            "persona": persona,
        }
    )

    for index, page in enumerate(pages):

        title = (
            page["title"]
            or f"Document {index + 1}"
        )

        sections = page["sections"]

        content_parts = []

        for heading, text in sections:

            if heading:
                content_parts.append(
                    f"## {heading}"
                )

            content_parts.append(text)

        content = "\n\n".join(
            content_parts
        )

        db.documents.insert_one(
            {
                "persona": persona,
                "slug": slugify(title),
                "title": title,
                "url": page["url"],
                "content": content,
                "created_at": datetime.datetime.now(
                    datetime.timezone.utc
                ),
            }
        )


def split_text(
    text: str,
    size: int = 200,
    overlap: int = 40,
) -> list[str]:

    words = text.split()

    if not words:
        return []

    chunks = []

    step = max(
        1,
        size - overlap,
    )

    for start in range(
        0,
        len(words),
        step,
    ):

        chunk = words[
            start:start + size
        ]

        if not chunk:
            break

        chunks.append(
            " ".join(chunk)
        )

        if start + size >= len(words):
            break

    return chunks


def index_documents(
    persona: str,
):

    db.chunks.delete_many(
        {
            "persona": persona,
        }
    )

    documents = db.documents.find(
        {
            "persona": persona,
        }
    )

    records = []

    for document in documents:

        title = document.get(
            "title",
            "",
        )

        url = document.get(
            "url",
            "",
        )

        content = document.get(
            "content",
            "",
        )

        chunks = split_text(content)

        for chunk_index, chunk in enumerate(
            chunks
        ):

            text = (
                f"Title: {title}\n"
                f"URL: {url}\n\n"
                f"{chunk}"
            )

            records.append(
                {
                    "persona": persona,
                    "title": title,
                    "url": url,
                    "section": (
                        f"chunk-{chunk_index}"
                    ),
                    "text": text,
                }
            )

    if not records:
        print(
            f"No documents found for {persona}"
        )
        return

    texts = [
        record["text"]
        for record in records
    ]

    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )

    for record, embedding in zip(
        records,
        embeddings,
    ):

        record["embedding"] = (
            embedding.tolist()
        )

    db.chunks.insert_many(records)

    print(
        f"Inserted {len(records)} chunks "
        f"for {persona}"
    )


def ensure_vector_index():

    index_name = "vector_index"

    existing = list(
        db.chunks.list_search_indexes(
            name=index_name
        )
    )

    if existing:
        print(
            f"Vector index '{index_name}' exists."
        )
        return

    model = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 384,
                    "similarity": "cosine",
                },
                {
                    "type": "filter",
                    "path": "persona",
                },
            ]
        },
        name=index_name,
        type="vectorSearch",
    )

    db.chunks.create_search_index(
        model
    )

    print(
        "Created vector_index."
    )


def run(persona: str):

    print(
        f"Starting ingestion for {persona}"
    )

    pages = crawl(persona)

    print(
        f"Crawled {len(pages)} pages."
    )

    save_documents(
        persona,
        pages,
    )

    index_documents(
        persona,
    )

    ensure_vector_index()

    print(
        f"Finished ingestion for {persona}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--persona",
        choices=[
            "voiceng",
            "fia",
            "both",
        ],
        default="both",
    )

    args = parser.parse_args()

    if args.persona == "both":

        run("voiceng")
        run("fia")

    else:

        run(args.persona)