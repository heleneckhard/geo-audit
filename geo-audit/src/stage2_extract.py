#!/usr/bin/env python3
"""
Stage 2, Step 1: Entity extraction from Stage 1 responses.

Uses GPT-4.1 structured output to extract brand mentions with sentiment,
recommendation rank, and cited attributes from each Stage 1 raw response.

Output is written in long format — one row per brand mention — which makes
downstream groupby operations in pandas straightforward.

Usage:
    python src/stage2_extract.py --session SESSION_ID
    python src/stage2_extract.py --session SESSION_ID --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

ROOT        = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs"

# ── Structured output schema ──────────────────────────────────────────────────

EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "brand_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "brands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "brand":      {"type": "string"},
                            "canonical":  {"type": "string"},
                            "sentiment":  {"type": "string", "enum": ["positive", "neutral", "negative"]},
                            "rank":       {"type": "string", "enum": ["primary", "secondary", "mentioned"]},
                            "attributes": {"type": "array", "items": {"type": "string"}},
                            "quote":      {"type": "string"},
                        },
                        "required":              ["brand", "canonical", "sentiment", "rank", "attributes", "quote"],
                        "additionalProperties":  False,
                    },
                }
            },
            "required":             ["brands"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = """You are a brand extraction assistant. Given a restaurant or food \
recommendation response, extract every food/restaurant brand mention.

For each brand return:
- brand: the name as it appears in the text
- canonical: normalized display name (e.g. "Mickey D's" → "McDonald's", "CFA" → "Chick-fil-A")
- sentiment: positive (recommended or praised), negative (warned against or criticized), \
neutral (mentioned without clear valence)
- rank: primary (the main recommendation), secondary (an alternative or runner-up), \
mentioned (referenced in passing)
- attributes: qualities cited about this brand from this list where applicable — \
value, speed, consistency, quality, health, service, variety, taste, convenience, \
portability, protein, family-friendly, deals, app, loyalty, innovation
- quote: the exact phrase or sentence from the text that supports your classification

Only extract food service brands (restaurants, fast food chains, coffee shops, \
food delivery services, etc.). Return an empty brands array if none are present."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def load_config() -> dict:
    return load_yaml(ROOT / "config.yaml")

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def get_processed_keys(records: list[dict]) -> set[tuple]:
    """Keys already written — includes both brand rows and no-brand sentinels."""
    return {(r["persona_id"], r["question_id"], r["run_number"]) for r in records}

def make_retry_decorator(cfg: dict):
    rl = cfg["rate_limiting"]
    return retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
        ),
        wait=wait_exponential(
            multiplier=1,
            min=rl["retry_wait_min_seconds"],
            max=rl["retry_wait_max_seconds"],
        ),
        stop=stop_after_attempt(rl["retry_attempts"]),
        reraise=True,
    )


# ── API call ──────────────────────────────────────────────────────────────────

def extract_brands(client: OpenAI, raw_response: str, cfg: dict, retry_fn) -> list[dict]:
    @retry_fn
    def _call():
        return client.chat.completions.create(
            model=cfg["stage2"]["extraction_model"],
            temperature=0,  # deterministic extraction
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": raw_response},
            ],
            response_format=EXTRACTION_SCHEMA,
        )

    response = _call()
    return json.loads(response.choices[0].message.content)["brands"]


# ── Main runner ───────────────────────────────────────────────────────────────

def run_extraction(session_id: str, dry_run: bool = False) -> None:
    cfg         = load_config()
    input_path  = OUTPUTS_DIR / "stage1" / f"session_{session_id}.jsonl"
    output_path = OUTPUTS_DIR / "stage2" / f"session_{session_id}_entities.jsonl"

    if not input_path.exists():
        print(f"Error: Stage 1 output not found at {input_path}")
        sys.exit(1)

    stage1_records = load_jsonl(input_path)
    existing       = load_jsonl(output_path)
    processed_keys = get_processed_keys(existing)
    pending        = [
        r for r in stage1_records
        if (r["persona_id"], r["question_id"], r["run_number"]) not in processed_keys
    ]

    total_mentions = sum(1 for r in existing if r.get("brand") is not None)

    print(f"\nSession        : {session_id}")
    print(f"Input          : {input_path} ({len(stage1_records)} records)")
    print(f"Output         : {output_path}")
    print(f"Pending        : {len(pending)}  |  Already done: {len(processed_keys)}  |  Mentions so far: {total_mentions}")

    if dry_run:
        print("\n[dry-run] First 3 pending records:")
        for r in pending[:3]:
            preview = r["raw_response"][:140].replace("\n", " ").strip()
            print(f"  {r['persona_id']} × {r['question_id']} run {r['run_number']}")
            print(f"  {preview}...")
        print(f"\n[dry-run] Estimated API calls: {len(pending)}")
        return

    if not pending:
        print("Nothing to do — all records already processed.")
        return

    client   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    retry_fn = make_retry_decorator(cfg)
    errors   = []

    with tqdm(total=len(pending), desc="Extracting", unit="response") as pbar:
        for record in pending:
            try:
                brands = extract_brands(client, record["raw_response"], cfg, retry_fn)

                base = {
                    "session_id":        record["session_id"],
                    "persona_id":        record["persona_id"],
                    "persona_name":      record["persona_name"],
                    "persona_archetype": record["persona_archetype"],
                    "question_id":       record["question_id"],
                    "question_category": record["question_category"],
                    "run_number":        record["run_number"],
                }

                if brands:
                    for brand in brands:
                        append_jsonl({**base, **brand}, output_path)
                else:
                    # Sentinel row so this record is skipped on resume
                    append_jsonl({
                        **base,
                        "brand": None, "canonical": None, "sentiment": None,
                        "rank": None, "attributes": [], "quote": None,
                    }, output_path)

            except Exception as e:
                errors.append({
                    "persona_id":  record["persona_id"],
                    "question_id": record["question_id"],
                    "run_number":  record["run_number"],
                    "error":       str(e),
                })
                tqdm.write(f"  ERROR {record['persona_id']} × {record['question_id']} run {record['run_number']}: {e}")

            pbar.update(1)

    final = load_jsonl(output_path)
    mention_count = sum(1 for r in final if r.get("brand") is not None)
    print(f"\nDone. {mention_count} brand mentions extracted → {output_path}")

    if errors:
        error_path = OUTPUTS_DIR / "stage2" / f"session_{session_id}_extract_errors.json"
        with open(error_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"{len(errors)} errors → {error_path}. Re-run to retry.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEO Audit — Stage 2 entity extraction")
    parser.add_argument(
        "--session",
        required=True,
        metavar="SESSION_ID",
        help="Stage 1 session ID to process (e.g. 20240218_143022)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without making API calls",
    )
    args = parser.parse_args()
    run_extraction(args.session, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
