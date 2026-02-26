#!/usr/bin/env python3
"""
Stage 1: Persona × Question recommendation audit.

Queries the OpenAI API with each persona × question combination for a
configurable number of runs, storing every output as a JSONL record.

Usage:
    python src/stage1_query.py                       # start a new session
    python src/stage1_query.py --resume SESSION_ID   # resume an interrupted session
    python src/stage1_query.py --dry-run             # preview cells without API calls
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
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
DATA_DIR    = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs" / "stage1"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def load_config() -> dict:
    return load_yaml(ROOT / "config.yaml")

def load_personas() -> list[dict]:
    return load_yaml(DATA_DIR / "personas.yaml")["personas"]

def load_questions() -> list[dict]:
    return load_yaml(DATA_DIR / "questions.yaml")["questions"]


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Run management ────────────────────────────────────────────────────────────

def get_completed_keys(records: list[dict]) -> set[tuple]:
    """Return the set of (persona_id, question_id, run_number) already on disk."""
    return {(r["persona_id"], r["question_id"], r["run_number"]) for r in records}

def build_cells(personas, questions, num_runs) -> list[tuple]:
    """Cartesian product of personas × questions × run numbers."""
    return [
        (persona, question, run_num)
        for persona   in personas
        for question  in questions
        for run_num   in range(1, num_runs + 1)
    ]


# ── API call ──────────────────────────────────────────────────────────────────

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

def call_api(
    client:   OpenAI,
    persona:  dict,
    question: dict,
    run_num:  int,
    cfg:      dict,
    retry_fn,
) -> dict:
    stage_cfg = cfg["stage1"]

    @retry_fn
    def _call():
        kwargs = dict(
            model       = stage_cfg["model"],
            temperature = stage_cfg["temperature"],
            max_tokens  = stage_cfg["max_tokens"],
            messages    = [
                {"role": "system", "content": persona["system_prompt"]},
                {"role": "user",   "content": question["text"]},
            ],
        )
        if stage_cfg.get("seed") is not None:
            kwargs["seed"] = stage_cfg["seed"]

        return client.chat.completions.create(**kwargs)

    response = _call()

    return {
        "session_id":        None,            # filled by caller
        "persona_id":        persona["id"],
        "persona_name":      persona["name"],
        "persona_archetype": persona["archetype"],
        "question_id":       question["id"],
        "question_category": question["category"],
        "question_text":     question["text"],
        "run_number":        run_num,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "model":             response.model,
        "temperature":       stage_cfg["temperature"],
        "seed":              stage_cfg.get("seed"),
        "raw_response":      response.choices[0].message.content,
        "finish_reason":     response.choices[0].finish_reason,
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens":      response.usage.total_tokens,
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_stage1(session_id: str, dry_run: bool = False) -> None:
    cfg       = load_config()
    personas  = load_personas()
    questions = load_questions()
    num_runs  = cfg["stage1"]["num_runs"]
    rpm_limit = cfg["rate_limiting"]["requests_per_minute"]

    output_path = OUTPUTS_DIR / f"session_{session_id}.jsonl"

    existing  = load_jsonl(output_path)
    completed = get_completed_keys(existing)
    all_cells = build_cells(personas, questions, num_runs)
    pending   = [(p, q, r) for (p, q, r) in all_cells if (p["id"], q["id"], r) not in completed]

    total = len(all_cells)
    done  = total - len(pending)

    print(f"\nSession : {session_id}")
    print(f"Output  : {output_path}")
    print(f"Cells   : {total} total  |  {done} done  |  {len(pending)} remaining")
    print(f"Model   : {cfg['stage1']['model']}  |  temp={cfg['stage1']['temperature']}  |  runs={num_runs}")

    if dry_run:
        print("\n[dry-run] First 5 pending cells:")
        for (p, q, r) in pending[:5]:
            print(f"  {p['id']} × {q['id']} × run {r} — {q['text'][:70].strip()}...")
        print(f"\n[dry-run] Estimated API calls: {len(pending)}")
        return

    if not pending:
        print("\nNothing to do — all cells already complete.")
        return

    client      = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    retry_fn    = make_retry_decorator(cfg)
    min_interval = 60.0 / rpm_limit
    errors      = []
    last_call_t = 0.0

    with tqdm(total=len(pending), desc="Stage 1", unit="call") as pbar:
        for persona, question, run_num in pending:
            # Respect rate limit
            wait = min_interval - (time.monotonic() - last_call_t)
            if wait > 0:
                time.sleep(wait)

            try:
                record = call_api(client, persona, question, run_num, cfg, retry_fn)
                record["session_id"] = session_id
                append_jsonl(record, output_path)
                last_call_t = time.monotonic()
            except Exception as e:
                errors.append({
                    "persona_id":  persona["id"],
                    "question_id": question["id"],
                    "run_number":  run_num,
                    "error":       str(e),
                })
                tqdm.write(f"  ERROR {persona['id']} × {question['id']} run {run_num}: {e}")

            pbar.update(1)

    success = len(pending) - len(errors)
    print(f"\n{success}/{len(pending)} records written → {output_path}")

    if errors:
        error_path = OUTPUTS_DIR / f"session_{session_id}_errors.json"
        with open(error_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"{len(errors)} errors logged → {error_path}")
        print("Re-run with --resume to retry failed cells.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEO Audit — Stage 1 query runner")
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume an interrupted session (e.g. 20240115_143022)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview pending cells without making any API calls",
    )
    args = parser.parse_args()

    if args.resume:
        session_id  = args.resume
        output_path = OUTPUTS_DIR / f"session_{session_id}.jsonl"
        if not output_path.exists():
            print(f"Error: no session file found for ID '{session_id}'")
            sys.exit(1)
    else:
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    run_stage1(session_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
