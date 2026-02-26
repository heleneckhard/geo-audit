#!/usr/bin/env python3
"""
Stage 2, Step 2b: Load Screaming Frog exports into brand_content.jsonl.

Reads *_internal.csv files from outputs/stage2/sf_exports/, filters and
cleans each page's content, and writes one JSONL record per page.

The _custom.csv files (links exports) are ignored entirely.
Custom extraction content is read from the Extractor columns in the
internal CSV, which SF includes automatically when custom extractions
are configured.

Usage:
    python src/load_sf_exports.py --session SESSION_ID
    python src/load_sf_exports.py --session SESSION_ID --dry-run
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).parent.parent
EXPORTS_DIR = ROOT / "outputs" / "stage2" / "sf_exports"
OUTPUTS_DIR = ROOT / "outputs" / "stage2"

# ── Filename stem → canonical brand name ──────────────────────────────────────
# Add entries here if you crawl additional brands.

FILENAME_MAP = {
    # Phase 2: Marketing Agencies
    # File stem (no spaces, no special chars) → canonical brand name
    "Scorpion":            "Scorpion",
    "RefineLabs":          "Refine Labs",
    "Wonderist":           "Wonderist Agency",
    "Directive":           "Directive Consulting",
    "KickStart":           "KickStart Dental Marketing",
    "Firegang":            "Firegang Dental Marketing",
    "Location3":           "Location3",
    "PainFree":            "Pain-Free Dental Marketing",
    "SOCi":                "SOCi",
    "SmartBug":            "SmartBug Media",
    "Cardinal":            "Cardinal Digital Marketing",
    "PoweredBySearch":     "Powered by Search",
    "BrandMuscle":         "BrandMuscle",
    "Ansira":              "BrandMuscle",   # rebranded from BrandMuscle
    "NoGood":              "NoGood",
    "GreatDentalWebsites": "Great Dental Websites",
    # Added in forthcoming Phase 2 update (missing from original top-15)
    "CuriousJane":              "Curious Jane",
    "FranchiseFastLane":        "Franchise FastLane",
    "FranchisePerformanceGroup": "Franchise Performance Group",
}

# ── URL patterns to exclude ───────────────────────────────────────────────────
# Pages matching any of these substrings are dropped before processing.

EXCLUDE_URL_PATTERNS = [
    "/career",
    "/careers",
    "/job",
    "/jobs",
    "/site-map",
    "/sitemap",
    "/privacy",
    "/legal",
    "/terms",
    "/contact",
    "/investor",
    "/press-release",
    "/tag/",             # blog tag archive pages
    "/category/",        # blog category pages
    "/author/",          # blog author pages
    "/wp-admin",
    "/wp-login",
    "?s=",               # search result pages
    "/page/",            # paginated archives
]

MAX_CRAWL_DEPTH  = 2    # 0 = homepage, 1 = one click, 2 = two clicks
MIN_WORD_COUNT   = 40   # skip near-empty pages
MIN_CONTENT_LEN  = 40   # minimum chars for an extractor block to be kept


# ── Content building ──────────────────────────────────────────────────────────

def clean_extractor_blocks(row: pd.Series, ext_cols: list[str]) -> str:
    """
    Pull non-empty extractor fields, collapse whitespace, deduplicate,
    and return as a single joined string.
    """
    seen    = set()
    blocks  = []
    for col in ext_cols:
        raw = str(row.get(col, ""))
        # Collapse all whitespace
        text = re.sub(r"\s+", " ", raw).strip()
        if len(text) >= MIN_CONTENT_LEN and text not in seen:
            seen.add(text)
            blocks.append(text)
    return "\n\n".join(blocks)


def build_page_text(row: pd.Series, ext_cols: list[str]) -> str:
    """
    Combine structured fields (title, meta, H1, H2s) with extractor body copy.
    Structured fields are the cleanest brand-voice signal; extractor adds depth.
    """
    parts = []

    for field in ["Title 1", "Meta Description 1", "H1-1", "H2-1", "H2-2"]:
        val = str(row.get(field, "")).strip()
        if val and val.lower() != "nan":
            parts.append(val)

    extractor_text = clean_extractor_blocks(row, ext_cols)
    if extractor_text:
        parts.append(extractor_text)

    return "\n\n".join(parts)


def should_exclude(url: str) -> bool:
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in EXCLUDE_URL_PATTERNS)


# ── Per-brand loader ──────────────────────────────────────────────────────────

def load_brand(csv_path: Path, brand_name: str) -> list[dict]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    ext_cols = [c for c in df.columns if c.startswith("Extractor")]

    # Filter
    df = df[df["Status Code"] == 200]
    df = df[df["Indexability"].str.strip() == "Indexable"]
    df = df[df["Crawl Depth"] <= MAX_CRAWL_DEPTH]
    df = df[df["Word Count"] >= MIN_WORD_COUNT]

    # Pandas drops all columns when apply() is called on an empty DataFrame.
    # Return early to avoid the KeyError in sort_values.
    if df.empty:
        return []

    df = df.copy()  # ensure a clean independent DataFrame before further ops
    df = df[~df["Address"].apply(should_exclude)]

    if df.empty:
        return []

    # Sort: shallowest first, then most content
    df = df.sort_values(["Crawl Depth", "Word Count"], ascending=[True, False])

    records = []
    for _, row in df.iterrows():
        text = build_page_text(row, ext_cols)
        if not text.strip():
            continue
        records.append({
            "brand":        brand_name,
            "url":          row["Address"],
            "crawl_depth":  int(row["Crawl Depth"]),
            "word_count":   int(row["Word Count"]),
            "title":        str(row.get("Title 1", "")).strip(),
            "meta_desc":    str(row.get("Meta Description 1", "")).strip(),
            "h1":           str(row.get("H1-1", "")).strip(),
            "cleaned_text": text,
            "char_count":   len(text),
            "source":       "screaming_frog",
            "scraped_at":   datetime.now(timezone.utc).isoformat(),
        })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session_id: str, dry_run: bool = False) -> None:
    output_path = OUTPUTS_DIR / f"session_{session_id}_brand_content.jsonl"

    # Find all internal CSVs
    internal_csvs = sorted(EXPORTS_DIR.glob("*_internal.csv"))
    if not internal_csvs:
        print(f"Error: no *_internal.csv files found in {EXPORTS_DIR}")
        sys.exit(1)

    # Map filenames to canonical brand names
    brand_files: list[tuple[str, Path]] = []
    unrecognised = []
    for csv_path in internal_csvs:
        stem = csv_path.stem.replace("_internal", "")
        if stem in FILENAME_MAP:
            brand_files.append((FILENAME_MAP[stem], csv_path))
        else:
            unrecognised.append(stem)

    if unrecognised:
        print(f"Warning: unrecognised file stems (add to FILENAME_MAP): {unrecognised}")

    print(f"\nSession  : {session_id}")
    print(f"Output   : {output_path}")
    print(f"Brands   : {len(brand_files)}")

    if dry_run:
        print("\n[dry-run] Would process:")
        for brand, path in brand_files:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            print(f"  {brand:30s}  {len(df):4d} rows in CSV  →  {path.name}")
        return

    all_records = []
    for brand, csv_path in brand_files:
        records = load_brand(csv_path, brand)
        all_records.extend(records)
        print(f"  {brand:30s}  {len(records):3d} pages loaded")

    # Write output (overwrite — this is a deterministic transform, safe to rerun)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    total_chars = sum(r["char_count"] for r in all_records)
    print(f"\n{len(all_records)} pages written → {output_path}")
    print(f"Total content: {total_chars:,} chars across {len(brand_files)} brands")


def main():
    parser = argparse.ArgumentParser(description="Load Screaming Frog exports into brand_content.jsonl")
    parser.add_argument("--session", required=True, metavar="SESSION_ID")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.session, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
