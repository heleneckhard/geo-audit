#!/usr/bin/env python3
"""
Stage 2, Step 4: Cosine similarity analysis and content gap scoring.

Loads brand and persona embeddings, computes semantic affinity between
each persona and each brand, cross-references with entity mention data,
and produces a content gap analysis.

Outputs (all in outputs/stage2/):
  - session_{id}_affinity_matrix.csv      — persona × brand cosine similarities
  - session_{id}_mention_counts.csv       — persona × brand positive mention counts
  - session_{id}_category_affinity.csv    — similarity broken down by question category
  - session_{id}_content_gaps.csv         — full (persona, brand) pair gap analysis
  - session_{id}_analysis_summary.txt     — human-readable findings

Usage:
    python src/stage2_analyze.py --session SESSION_ID
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs" / "stage2"

# Gap score thresholds (percentile difference within a persona's brand ranking)
GAP_THRESHOLD  = 0.30   # |gap| > this → label as content_gap / missed_opportunity


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every row of A and every row of B.
    Returns shape (len(A), len(B)).
    """
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-10)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-10)
    return A_norm @ B_norm.T


def pct_rank(series: pd.Series) -> pd.Series:
    """Percentile-rank a series [0, 1]. Higher value → higher rank."""
    return series.rank(pct=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session_id: str) -> None:

    # ── Load data ─────────────────────────────────────────────────────────────
    brand_emb    = np.load(OUTPUTS_DIR / f"session_{session_id}_brand_embeddings.npy")
    persona_emb  = np.load(OUTPUTS_DIR / f"session_{session_id}_persona_embeddings.npy")
    brand_meta   = load_jsonl(OUTPUTS_DIR / f"session_{session_id}_brand_meta.jsonl")
    persona_meta = load_jsonl(OUTPUTS_DIR / f"session_{session_id}_persona_meta.jsonl")
    entities     = load_jsonl(OUTPUTS_DIR / f"session_{session_id}_entities_normalized.jsonl")

    brand_df    = pd.DataFrame(brand_meta)
    persona_df  = pd.DataFrame(persona_meta)
    entity_df   = pd.DataFrame(entities) if entities else pd.DataFrame()

    brands   = sorted(brand_df["brand"].unique())
    personas = sorted(persona_df["persona_id"].unique())

    # Build lookup: persona_id → human name
    persona_names = (
        persona_df[["persona_id", "persona_name"]]
        .drop_duplicates()
        .set_index("persona_id")["persona_name"]
        .to_dict()
    )
    categories = sorted(persona_df["question_category"].unique())

    print(f"\nSession  : {session_id}")
    print(f"Brands   : {len(brands)}")
    print(f"Personas : {len(personas)}")
    print(f"Brand emb shape   : {brand_emb.shape}")
    print(f"Persona emb shape : {persona_emb.shape}")

    # ── Brand and persona centroids ───────────────────────────────────────────
    brand_centroids = {
        brand: brand_emb[brand_df[brand_df["brand"] == brand].index].mean(axis=0)
        for brand in brands
    }
    persona_centroids = {
        pid: persona_emb[persona_df[persona_df["persona_id"] == pid].index].mean(axis=0)
        for pid in personas
    }

    P = np.stack([persona_centroids[pid]   for pid   in personas])  # (5, D)
    B = np.stack([brand_centroids[brand]   for brand in brands  ])  # (17, D)

    # ── 1. Affinity matrix ────────────────────────────────────────────────────
    sim_matrix  = cosine_sim_matrix(P, B)                           # (5, 17)
    affinity_df = pd.DataFrame(sim_matrix, index=personas, columns=brands)
    affinity_df.index.name = "persona_id"
    affinity_df.insert(0, "persona_name", [persona_names[p] for p in personas])

    aff_path = OUTPUTS_DIR / f"session_{session_id}_affinity_matrix.csv"
    affinity_df.to_csv(aff_path)
    print(f"\nAffinity matrix    → {aff_path.name}")

    # ── 2. Category-level affinity ────────────────────────────────────────────
    cat_rows = []
    for pid in personas:
        for cat in categories:
            mask = (
                (persona_df["persona_id"] == pid) &
                (persona_df["question_category"] == cat)
            )
            idx = persona_df[mask].index
            if len(idx) == 0:
                continue
            centroid = persona_emb[idx].mean(axis=0)
            sims     = cosine_sim_matrix(centroid.reshape(1, -1), B)[0]
            for brand, sim in zip(brands, sims):
                cat_rows.append({
                    "persona_id":        pid,
                    "persona_name":      persona_names[pid],
                    "question_category": cat,
                    "brand":             brand,
                    "similarity":        float(sim),
                })

    cat_df   = pd.DataFrame(cat_rows)
    cat_path = OUTPUTS_DIR / f"session_{session_id}_category_affinity.csv"
    cat_df.to_csv(cat_path, index=False)
    print(f"Category affinity  → {cat_path.name}")

    # ── 3. Mention counts ─────────────────────────────────────────────────────
    if not entity_df.empty and "sentiment" in entity_df.columns:
        pos = entity_df[entity_df["sentiment"] == "positive"]
        mention_counts = (
            pos.groupby(["persona_id", "canonical"])
            .size()
            .unstack(fill_value=0)
        )
        mention_counts = mention_counts.reindex(
            index=personas, columns=brands, fill_value=0
        )
    else:
        mention_counts = pd.DataFrame(0, index=personas, columns=brands)

    mc_path = OUTPUTS_DIR / f"session_{session_id}_mention_counts.csv"
    mention_counts.to_csv(mc_path)
    print(f"Mention counts     → {mc_path.name}")

    # ── 4. Content gap analysis ───────────────────────────────────────────────
    # gap_score = mention_percentile − similarity_percentile  (within each persona)
    # Positive  → brand over-recommended vs. content alignment  (content gap)
    # Negative  → brand under-recommended vs. content alignment (missed opportunity)

    gap_rows = []
    for pid in personas:
        sim_series     = affinity_df.loc[pid, brands].astype(float)
        mention_series = mention_counts.loc[pid, brands].astype(float)

        sim_pct     = pct_rank(sim_series)
        mention_pct = pct_rank(mention_series)

        for brand in brands:
            gap = float(mention_pct[brand] - sim_pct[brand])
            gap_rows.append({
                "persona_id":    pid,
                "persona_name":  persona_names[pid],
                "brand":         brand,
                "similarity":    round(float(sim_series[brand]), 6),
                "sim_pct":       round(float(sim_pct[brand]), 4),
                "mention_count": int(mention_counts.loc[pid, brand]),
                "mention_pct":   round(float(mention_pct[brand]), 4),
                "gap_score":     round(gap, 4),
                "gap_type": (
                    "content_gap"        if gap >  GAP_THRESHOLD else
                    "missed_opportunity" if gap < -GAP_THRESHOLD else
                    "aligned"
                ),
            })

    gap_df = (
        pd.DataFrame(gap_rows)
        .sort_values("gap_score", ascending=False)
        .reset_index(drop=True)
    )
    gap_path = OUTPUTS_DIR / f"session_{session_id}_content_gaps.csv"
    gap_df.to_csv(gap_path, index=False)
    print(f"Content gaps       → {gap_path.name}")

    # ── 5. Human-readable summary ─────────────────────────────────────────────
    lines = [
        "GEO AUDIT — ANALYSIS SUMMARY",
        f"Session: {session_id}",
        "=" * 70,
    ]

    # --- Affinity table -------------------------------------------------------
    lines += [
        "",
        "AFFINITY MATRIX  (cosine similarity — higher = stronger content alignment)",
        "-" * 70,
    ]
    display = affinity_df.drop(columns=["persona_name"]).copy().astype(float)
    display.index = [persona_names[p] for p in personas]
    # Truncate brand names to 14 chars for table width
    display.columns = [b[:14] for b in brands]
    lines.append(display.round(4).to_string())

    # --- Top brand per persona ------------------------------------------------
    lines += ["", "", "TOP 5 BRANDS BY PERSONA  (cosine similarity to persona responses)"]
    lines.append("-" * 70)
    for pid in personas:
        name = persona_names[pid]
        sims = affinity_df.loc[pid, brands].astype(float).sort_values(ascending=False)
        lines.append(f"\n  {name}:")
        for brand, sim in sims.head(5).items():
            lines.append(f"    {brand:<30s}  {sim:.4f}")

    # --- Top mentioned brands per persona (entity data) -----------------------
    lines += ["", "", "TOP 5 BRANDS BY POSITIVE MENTION COUNT PER PERSONA"]
    lines.append("-" * 70)
    for pid in personas:
        name    = persona_names[pid]
        counts  = mention_counts.loc[pid].sort_values(ascending=False)
        lines.append(f"\n  {name}:")
        for brand, cnt in counts.head(5).items():
            lines.append(f"    {brand:<30s}  {cnt} mentions")

    # --- Content gaps ---------------------------------------------------------
    lines += [
        "", "",
        "CONTENT GAPS  (brand over-recommended relative to content alignment)",
        "A positive gap means the AI recommends this brand to this persona more",
        "than the brand's content similarity would predict.  Opportunity: add",
        "persona-relevant messaging to the brand's web content.",
        "-" * 70,
    ]
    top_gaps = gap_df[gap_df["gap_type"] == "content_gap"].head(20)
    if top_gaps.empty:
        lines.append("  (none above threshold)")
    else:
        for _, row in top_gaps.iterrows():
            lines.append(
                f"  {row['persona_name']:<28s}  {row['brand']:<22s}"
                f"  gap={row['gap_score']:+.3f}"
                f"  mentions={row['mention_count']}"
                f"  sim={row['similarity']:.4f}"
            )

    # --- Missed opportunities -------------------------------------------------
    lines += [
        "", "",
        "MISSED OPPORTUNITIES  (strong content alignment, low mention count)",
        "A negative gap means the brand's content resonates with this persona",
        "but the AI is not recommending them.  Could indicate a GEO or",
        "discoverability gap — the content is there but not surfacing.",
        "-" * 70,
    ]
    missed = (
        gap_df[gap_df["gap_type"] == "missed_opportunity"]
        .sort_values("gap_score")
        .head(20)
    )
    if missed.empty:
        lines.append("  (none above threshold)")
    else:
        for _, row in missed.iterrows():
            lines.append(
                f"  {row['persona_name']:<28s}  {row['brand']:<22s}"
                f"  gap={row['gap_score']:+.3f}"
                f"  mentions={row['mention_count']}"
                f"  sim={row['similarity']:.4f}"
            )

    # --- Category breakdown ---------------------------------------------------
    lines += ["", "", "BEST-ALIGNED BRAND BY PERSONA × QUESTION CATEGORY"]
    lines.append("-" * 70)
    for pid in personas:
        name = persona_names[pid]
        lines.append(f"\n  {name}:")
        for cat in categories:
            sub = cat_df[
                (cat_df["persona_id"] == pid) &
                (cat_df["question_category"] == cat)
            ]
            if sub.empty:
                continue
            top_brand = sub.loc[sub["similarity"].idxmax(), "brand"]
            top_sim   = sub["similarity"].max()
            lines.append(f"    {cat:<38s}  {top_brand:<25s}  ({top_sim:.4f})")

    summary_text = "\n".join(lines)
    summary_path = OUTPUTS_DIR / f"session_{session_id}_analysis_summary.txt"
    summary_path.write_text(summary_text)
    print(f"Summary            → {summary_path.name}")

    print(f"\n{'=' * 70}")
    print(summary_text)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEO Audit — Stage 2 cosine similarity analysis")
    parser.add_argument("--session", required=True, metavar="SESSION_ID")
    args = parser.parse_args()
    run(args.session)


if __name__ == "__main__":
    main()
