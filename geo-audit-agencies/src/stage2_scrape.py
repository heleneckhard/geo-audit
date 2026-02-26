#!/usr/bin/env python3
"""
Stage 2, Step 2: Web scraping of brand content.

Reads the top N brands from the normalized entities file, fetches their
curated pages from data/brand_urls.yaml, cleans the HTML to marketing-copy
text, and stores results as JSONL.

Output is one record per scraped page. Already-scraped URLs are skipped
on re-runs (resume-safe).

Usage:
    python src/stage2_scrape.py --session SESSION_ID
    python src/stage2_scrape.py --session SESSION_ID --dry-run
    python src/stage2_scrape.py --session SESSION_ID --top 10
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Tags that contain navigation/boilerplate rather than brand-voice content
NOISE_TAGS = [
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "iframe", "svg", "button", "cookie",
]

# Minimum characters for a text block to be kept
MIN_BLOCK_LENGTH = 40


# ── Text cleaning ─────────────────────────────────────────────────────────────

def extract_text(html: str, url: str) -> str:
    """
    Parse HTML and return cleaned marketing-copy text.
    Strips noise tags, collapses whitespace, removes short boilerplate lines.
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(NOISE_TAGS):
        tag.decompose()

    # Prefer explicit main content containers
    main = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find("article")
        or soup.find(id=lambda x: x and "content" in x.lower())
        or soup.find(class_=lambda x: x and "content" in " ".join(x).lower())
        or soup.body
    )

    if not main:
        return ""

    lines = []
    for element in main.stripped_strings:
        text = element.strip()
        if len(text) >= MIN_BLOCK_LENGTH:
            lines.append(text)

    return "\n\n".join(lines)


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Brand resolution ──────────────────────────────────────────────────────────

def get_top_brands(session_id: str, top_n: int) -> list[str]:
    """Return top N brands by positive mention count from normalized entities."""
    path = OUTPUTS_DIR / "stage2" / f"session_{session_id}_entities_normalized.jsonl"
    if not path.exists():
        print(f"Error: normalized entities not found at {path}")
        sys.exit(1)

    records  = load_jsonl(path)
    positive = [r["canonical"] for r in records if r.get("canonical") and r.get("sentiment") == "positive"]
    counts   = Counter(positive)
    return [brand for brand, _ in counts.most_common(top_n)]

def load_brand_urls() -> dict[str, list[str]]:
    data = yaml.safe_load((DATA_DIR / "brand_urls.yaml").read_text())
    return {brand: entry["urls"] for brand, entry in data["brands"].items()}


# ── Main runner ───────────────────────────────────────────────────────────────

def run_scrape(session_id: str, top_n: int, dry_run: bool = False) -> None:
    cfg_path    = ROOT / "config.yaml"
    cfg         = yaml.safe_load(cfg_path.read_text())
    output_path = OUTPUTS_DIR / "stage2" / f"session_{session_id}_brand_content.jsonl"

    top_brands  = get_top_brands(session_id, top_n)
    brand_urls  = load_brand_urls()

    # Build work list: (brand, url) pairs not yet scraped
    existing_urls = {r["url"] for r in load_jsonl(output_path)}

    work = []
    missing_from_map = []
    for brand in top_brands:
        if brand not in brand_urls:
            missing_from_map.append(brand)
            continue
        for url in brand_urls[brand]:
            if url not in existing_urls:
                work.append((brand, url))

    print(f"\nSession     : {session_id}")
    print(f"Top brands  : {top_n}  →  {len(top_brands)} found in entities")
    print(f"URLs to scrape : {len(work)}  |  Already done: {len(existing_urls)}")

    if missing_from_map:
        print(f"\nWarning: these brands have no entry in brand_urls.yaml:")
        for b in missing_from_map:
            print(f"  - {b}")
        print("Add them to data/brand_urls.yaml to include their content.")

    if dry_run:
        print("\n[dry-run] Work list:")
        for brand, url in work:
            print(f"  {brand:30s}  {url}")
        return

    if not work:
        print("Nothing to scrape.")
        return

    errors  = []
    delay   = 2.0  # seconds between requests — polite scraping

    with tqdm(total=len(work), desc="Scraping", unit="page") as pbar:
        for i, (brand, url) in enumerate(work):
            if i > 0:
                time.sleep(delay)
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
                resp.raise_for_status()

                cleaned_text = extract_text(resp.text, url)

                if len(cleaned_text) < 100:
                    tqdm.write(f"  WARN {brand} — {url} — very little text extracted ({len(cleaned_text)} chars)")

                append_jsonl({
                    "session_id":   session_id,
                    "brand":        brand,
                    "url":          url,
                    "status_code":  resp.status_code,
                    "scraped_at":   datetime.now(timezone.utc).isoformat(),
                    "char_count":   len(cleaned_text),
                    "cleaned_text": cleaned_text,
                }, output_path)

            except Exception as e:
                errors.append({"brand": brand, "url": url, "error": str(e)})
                tqdm.write(f"  ERROR {brand} — {url}: {e}")
                # Write a failed sentinel so we can track it
                append_jsonl({
                    "session_id":   session_id,
                    "brand":        brand,
                    "url":          url,
                    "status_code":  None,
                    "scraped_at":   datetime.now(timezone.utc).isoformat(),
                    "char_count":   0,
                    "cleaned_text": None,
                    "error":        str(e),
                }, output_path)

            pbar.update(1)

    success = len(work) - len(errors)
    print(f"\nDone. {success}/{len(work)} pages scraped → {output_path}")

    if errors:
        print(f"\n{len(errors)} failed pages:")
        for e in errors:
            print(f"  {e['brand']:30s}  {e['url']}")
            print(f"    {e['error']}")
        print("\nFailed URLs can be retried by re-running (they are not marked complete).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEO Audit — Stage 2 brand content scraper")
    parser.add_argument("--session", required=True, metavar="SESSION_ID")
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Override top_brands from config (default: use config.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg   = yaml.safe_load((ROOT / "config.yaml").read_text())
    top_n = args.top or cfg["stage2"]["top_brands"]

    run_scrape(args.session, top_n=top_n, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
