#!/usr/bin/env python3
"""
Brand name normalization — patches the canonical field in an entities JSONL file.

Two-pass normalization:
  1. Unicode: curly apostrophes → straight, strip extra whitespace
  2. Alias map: case-insensitive lookup against data/brand_aliases.yaml

Writes a new *_normalized.jsonl file and prints a summary of changes.
The original file is not modified.

Usage:
    python src/normalize_brands.py --session SESSION_ID
    python src/normalize_brands.py --session SESSION_ID --preview
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs" / "stage2"


# ── Normalization logic ───────────────────────────────────────────────────────

def load_aliases() -> dict[str, str]:
    path = DATA_DIR / "brand_aliases.yaml"
    raw  = yaml.safe_load(path.read_text())["aliases"]
    # Keys are already lowercase in the file; normalize values too for safety
    return {k.strip(): v for k, v in raw.items()}


def normalize_key(name: str) -> str:
    """Produce a lookup key: lowercase, straight apostrophes, collapsed whitespace."""
    name = name.replace("\u2019", "'").replace("\u2018", "'")  # curly → straight
    name = name.replace("\u2013", "-").replace("\u2014", "-")  # em/en dash → hyphen
    name = re.sub(r"\s+", " ", name).strip().lower()
    name = name.replace("'", "")   # strip apostrophes for fuzzy matching
    return name


def canonical(name: str | None, aliases: dict[str, str]) -> str | None:
    if not name:
        return name
    key = normalize_key(name)
    return aliases.get(key, name)  # fall back to original if no alias match


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session_id: str, preview: bool = False) -> None:
    aliases     = load_aliases()
    input_path  = OUTPUTS_DIR / f"session_{session_id}_entities.jsonl"
    output_path = OUTPUTS_DIR / f"session_{session_id}_entities_normalized.jsonl"

    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        sys.exit(1)

    records = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    changes   = Counter()
    normed    = []

    for r in records:
        original  = r.get("canonical")
        corrected = canonical(original, aliases)

        if corrected != original:
            changes[(original, corrected)] += 1

        normed.append({**r, "canonical": corrected})

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\nRecords   : {len(records)}")
    print(f"Changes   : {sum(changes.values())} across {len(changes)} distinct mappings\n")

    if changes:
        print("Mappings applied:")
        for (before, after), count in sorted(changes.items(), key=lambda x: -x[1]):
            print(f"  {count:4d}  {before!r:30s} → {after!r}")
    else:
        print("No changes needed.")

    if preview:
        print("\n[preview] No files written.")
        return

    # ── Write normalized file ─────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in normed:
            f.write(json.dumps(r) + "\n")

    print(f"\nNormalized file → {output_path}")

    # ── Post-normalization frequency check ────────────────────────────────────
    positive = [r["canonical"] for r in normed if r.get("canonical") and r.get("sentiment") == "positive"]
    print("\nTop brands after normalization (positive mentions):")
    for brand, count in Counter(positive).most_common(20):
        print(f"  {count:4d}  {brand}")


def main():
    parser = argparse.ArgumentParser(description="Normalize brand canonical names in entities JSONL")
    parser.add_argument("--session", required=True, metavar="SESSION_ID")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show what would change without writing any files",
    )
    args = parser.parse_args()
    run(args.session, preview=args.preview)


if __name__ == "__main__":
    main()
