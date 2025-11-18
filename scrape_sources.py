"""Scrape parenting documents from UNICEF and CDC and save each as a text file in data/.

- Crawls a small, safe subset of pages starting from seed URLs (UNICEF parenting, CDC parenting).
- If a link points to a PDF, download it directly.
- If a link is HTML, fetch its main text and save as a .txt file for easy reading.

Usage (PowerShell):
    python scrape_sources.py --max-pages 30 --max-docs 20 --delay 1.0

Notes:
- This is a conservative scraper: small crawl, polite delay.
- Text files are saved with UTF-8 encoding for maximum compatibility.
"""
from __future__ import annotations
import os
import re
import time
import math
import hashlib
import argparse
from pathlib import Path
from typing import Set, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import DATA_DIR

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ParentingRAGBot/1.0; +https://github.com/GuermaziAhmed/Parenting-Child-Psychology-RAG-Pipeline)"
}

SEEDS = [
    # UNICEF parenting hub
    "https://www.unicef.org/parenting",
    # CDC parenting resources
    "https://www.cdc.gov/parents/essentials/index.html",
    "https://www.cdc.gov/ncbddd/childdevelopment/positiveparenting/index.html",
]

ALLOWED_DOMAINS = {"www.unicef.org", "unicef.org", "www.cdc.gov", "cdc.gov"}

# Path keywords to keep scope on parenting/children topics
KEYWORDS = [
    "parent", "child", "children", "development", "positive-parenting", "toddl", "teen",
    "behav", "discipline", "emotional", "support", "family", "caregiver", "education"
]

PDF_EXTENSIONS = (".pdf",)


def sanitize_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    return name[:150] if len(name) > 150 else name


def is_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path.lower().endswith(PDF_EXTENSIONS)


def same_domain(url: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    return netloc in ALLOWED_DOMAINS


def keyword_relevant(url: str) -> bool:
    lower = url.lower()
    return any(k in lower for k in KEYWORDS)


def extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.startswith("javascript:"):
            continue
        full = urljoin(base_url, href)
        # Stay on allowed domains and parenting-relevant
        if same_domain(full) and keyword_relevant(full):
            links.append(full)
    return list(dict.fromkeys(links))  # dedupe preserve order


def fetch(url: str, timeout: float = 20.0) -> Tuple[int, str]:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    return resp.status_code, resp.text


def download_pdf(url: str, out_path: Path) -> bool:
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download PDF {url}: {e}")
        return False


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Prefer article/main content if present
    article = soup.find("article") or soup.find("main")
    container = article or soup.body or soup
    paragraphs = []
    for p in container.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
        text = p.get_text(separator=" ", strip=True)
        if len(text) > 0:
            paragraphs.append(text)
    text = "\n\n".join(paragraphs)
    # Collapse overlong whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def save_text_to_file(text: str, out_path: Path, title: str | None = None):
    """Save text content to a plain .txt file with UTF-8 encoding."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if title:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
        f.write(text)


def unique_output_path(base_name: str, extension: str = ".txt") -> Path:
    # Ensure unique name by adding a hash suffix
    h = hashlib.md5(base_name.encode("utf-8")).hexdigest()[:8]
    fname = sanitize_filename(base_name)
    return DATA_DIR / f"{fname}-{h}{extension}"


def crawl_and_save(max_pages: int, max_docs: int, delay: float) -> List[Path]:
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    visited: Set[str] = set()
    queue: List[str] = []
    queue.extend(SEEDS)

    saved: List[Path] = []
    pages_processed = 0

    while queue and pages_processed < max_pages and len(saved) < max_docs:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            print(f"Fetching: {url}")
            if is_pdf_url(url):
                # Direct PDF link - download as PDF
                base_name = Path(urlparse(url).path).name or "document"
                out_path = unique_output_path(base_name, extension=".pdf")
                if download_pdf(url, out_path):
                    print(f"Saved PDF: {out_path}")
                    saved.append(out_path)
            else:
                status, html = fetch(url)
                if status != 200:
                    print(f"Non-200 status {status} for {url}")
                else:
                    text = extract_text_from_html(html)
                    if len(text) > 300:  # avoid tiny pages
                        # Title if any
                        soup = BeautifulSoup(html, "html.parser")
                        title_tag = soup.find("title")
                        title = title_tag.get_text(strip=True) if title_tag else ""
                        base_name = title or urlparse(url).path.split("/")[-1] or "document"
                        out_path = unique_output_path(base_name, extension=".txt")
                        save_text_to_file(text, out_path, title=title)
                        print(f"Saved text: {out_path}\nFrom: {url}")
                        saved.append(out_path)

                    # Enqueue next links
                    for link in extract_links(url, html):
                        if link not in visited and link not in queue:
                            queue.append(link)
        except Exception as e:
            print(f"Error processing {url}: {e}")
        finally:
            pages_processed += 1
            if delay > 0:
                time.sleep(delay)

    return saved


def main():
    parser = argparse.ArgumentParser(description="Scrape UNICEF and CDC parenting documents to text files")
    parser.add_argument("--max-pages", type=int, default=30, help="Max pages to fetch during crawl")
    parser.add_argument("--max-docs", type=int, default=20, help="Max documents to save")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay seconds between requests")
    args = parser.parse_args()

    print(f"Output directory: {DATA_DIR}")
    saved = crawl_and_save(max_pages=args.max_pages, max_docs=args.max_docs, delay=args.delay)
    print(f"\nCompleted. Saved {len(saved)} documents:")
    for p in saved:
        print("-", p)

if __name__ == "__main__":
    main()
