#!/usr/bin/env python3
"""
Stage 2, Step 3: Generate embeddings for brand content and persona responses.

Embeds two corpora:
  - Brand pages from brand_content.jsonl (chunked where > token limit)
  - Persona responses from Stage 1 JSONL

Outputs per corpus:
  - <name>_embeddings.npy  — float32 numpy array, shape (N, embedding_dim)
  - <name>_meta.jsonl      — one record per row with source metadata

These are loaded together by stage2_analyze.py for cosine similarity.

Usage:
    python src/stage2_embed.py --session SESSION_ID
    python src/stage2_embed.py --session SESSION_ID --dry-run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
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
OUTPUTS_DIR = ROOT / "outputs" / "stage2"

# text-embedding-3-large max context is 8192 tokens.
# Use 7500 as the ceiling to leave a small safety margin.
MAX_TOKENS  = 7_500
BATCH_SIZE  = 20        # embeddings per API call (reduced for larger SF content)
SKIP_BRANDS = {"KFC"}   # brands with no usable content

# tiktoken encoder for text-embedding-3-large (uses cl100k_base)
_ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """
    Split text into chunks <= MAX_TOKENS, breaking on paragraph boundaries
    where possible. Uses tiktoken for accurate token counting.
    Returns a list of at least one chunk.
    """
    if count_tokens(text) <= MAX_TOKENS:
        return [text]

    paragraphs = text.split("\n\n")
    chunks      = []
    current     = []
    current_tok = 0

    for para in paragraphs:
        para_tok = count_tokens(para)
        if current_tok + para_tok > MAX_TOKENS and current:
            chunks.append("\n\n".join(current))
            current     = []
            current_tok = 0
        # If a single paragraph is too long, split by proportional char slices
        # (rare edge case — only for enormous continuous text blocks)
        if para_tok > MAX_TOKENS:
            para_chars = len(para)
            step = max(1, para_chars * MAX_TOKENS // (para_tok + 1))
            for i in range(0, para_chars, step):
                chunks.append(para[i : i + step])
        else:
            current.append(para)
            current_tok += para_tok

    if current:
        chunks.append("\n\n".join(current))

    return chunks or [text]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def load_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())

def make_retry_decorator(cfg: dict):
    rl = cfg["rate_limiting"]
    return retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
        ),
        wait=wait_exponential(multiplier=1, min=rl["retry_wait_min_seconds"],
                              max=rl["retry_wait_max_seconds"]),
        stop=stop_after_attempt(rl["retry_attempts"]),
        reraise=True,
    )


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_batch(client: OpenAI, texts: list[str], model: str, retry_fn) -> list[list[float]]:
    """Embed a batch of texts, returning a list of embedding vectors."""
    @retry_fn
    def _call():
        return client.embeddings.create(model=model, input=texts)

    response = _call()
    # API returns embeddings sorted by index
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def embed_corpus(
    client:    OpenAI,
    items:     list[dict],   # list of {"text": str, **metadata}
    model:     str,
    retry_fn,
    label:     str,
) -> tuple[np.ndarray, list[dict]]:
    """
    Embed a corpus of items in batches.
    Returns (embeddings array, metadata list).
    """
    all_embeddings = []
    all_meta       = []

    batches = [items[i : i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]

    with tqdm(total=len(items), desc=label, unit="item") as pbar:
        for batch in batches:
            texts = [item["text"] for item in batch]
            vecs  = embed_batch(client, texts, model, retry_fn)

            for item, vec in zip(batch, vecs):
                meta = {k: v for k, v in item.items() if k != "text"}
                all_meta.append(meta)
                all_embeddings.append(vec)

            pbar.update(len(batch))
            time.sleep(0.1)  # light throttle between batches

    return np.array(all_embeddings, dtype=np.float32), all_meta


# ── Corpus builders ───────────────────────────────────────────────────────────

def build_brand_corpus(session_id: str) -> list[dict]:
    """
    Load brand_content.jsonl, filter skipped brands, chunk long pages.
    Returns list of {text, brand, url, chunk_index, chunk_total, char_count}.
    """
    path    = OUTPUTS_DIR / f"session_{session_id}_brand_content.jsonl"
    records = load_jsonl(path)

    items = []
    for r in records:
        if r["brand"] in SKIP_BRANDS:
            continue
        if not r.get("cleaned_text"):
            continue

        chunks = chunk_text(r["cleaned_text"])
        for i, chunk in enumerate(chunks):
            items.append({
                "text":        chunk,
                "brand":       r["brand"],
                "url":         r["url"],
                "chunk_index": i,
                "chunk_total": len(chunks),
                "char_count":  len(chunk),
            })

    return items


def build_persona_corpus(session_id: str) -> list[dict]:
    """
    Load Stage 1 JSONL responses.
    Returns list of {text, session_id, persona_id, persona_name, ...}.
    """
    path    = ROOT / "outputs" / "stage1" / f"session_{session_id}.jsonl"
    records = load_jsonl(path)

    return [
        {
            "text":             r["raw_response"],
            "session_id":       r["session_id"],
            "persona_id":       r["persona_id"],
            "persona_name":     r["persona_name"],
            "persona_archetype":r["persona_archetype"],
            "question_id":      r["question_id"],
            "question_category":r["question_category"],
            "run_number":       r["run_number"],
        }
        for r in records
    ]


# ── Save / load ───────────────────────────────────────────────────────────────

def save(name: str, session_id: str, embeddings: np.ndarray, meta: list[dict]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    npy_path  = OUTPUTS_DIR / f"session_{session_id}_{name}_embeddings.npy"
    meta_path = OUTPUTS_DIR / f"session_{session_id}_{name}_meta.jsonl"

    np.save(npy_path, embeddings)
    with open(meta_path, "w") as f:
        for record in meta:
            f.write(json.dumps(record) + "\n")

    print(f"  Saved {embeddings.shape[0]} vectors ({embeddings.shape[1]}d) → {npy_path.name}")
    print(f"  Metadata → {meta_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session_id: str, dry_run: bool = False) -> None:
    cfg      = load_config()
    model    = cfg["stage2"]["embedding_model"]

    brand_corpus   = build_brand_corpus(session_id)
    persona_corpus = build_persona_corpus(session_id)

    brands_present = sorted({i["brand"] for i in brand_corpus})
    total_tokens_est = sum(len(i["text"]) for i in brand_corpus + persona_corpus) // 4

    print(f"\nSession       : {session_id}")
    print(f"Model         : {model}")
    print(f"Brand chunks  : {len(brand_corpus)}  ({len(brands_present)} brands)")
    print(f"Persona items : {len(persona_corpus)}")
    print(f"Est. tokens   : {total_tokens_est:,}  (~${total_tokens_est/1_000_000*0.13:.4f})")

    if dry_run:
        print("\n[dry-run] Brands to embed:")
        from collections import Counter
        counts = Counter(i["brand"] for i in brand_corpus)
        for brand in sorted(counts):
            print(f"  {brand:30s}  {counts[brand]:3d} chunks")
        return

    client   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    retry_fn = make_retry_decorator(cfg)

    print("\nEmbedding brand content...")
    brand_emb, brand_meta = embed_corpus(client, brand_corpus, model, retry_fn, "Brand pages")
    save("brand", session_id, brand_emb, brand_meta)

    print("\nEmbedding persona responses...")
    persona_emb, persona_meta = embed_corpus(client, persona_corpus, model, retry_fn, "Persona responses")
    save("persona", session_id, persona_emb, persona_meta)

    print(f"\nDone.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEO Audit — Stage 2 embedding pipeline")
    parser.add_argument("--session", required=True, metavar="SESSION_ID")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.session, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
