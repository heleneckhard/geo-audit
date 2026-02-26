#!/usr/bin/env python3
"""
Stage 2, Step 5: Statistical significance testing.

For each brand–persona pair, answers:
  - How consistent were recommendations across the 3 runs?
    (per-run mention rates, mean ± 95% CI using t-distribution, df=2)
  - Is this persona's mention rate significantly different from the brand's
    overall base rate across all 5 personas?
    (two-sided binomial test, Bonferroni-corrected for 75 comparisons)
  - Is the distribution of recommendations across personas non-random?
    (chi-square test per brand, null = uniform distribution)
  - How stable is cosine similarity across runs?
    (per-run similarity variance, CV as an AI consistency signal)

Outputs (all in outputs/stage2/):
  - session_{id}_stats.csv            — per (persona, brand): rates, CI, binomial test
  - session_{id}_stats_chisquare.csv  — per brand: chi-square test
  - session_{id}_run_similarity.csv   — per (persona, brand): per-run similarities

Usage:
    python src/stage2_stats.py --session SESSION_ID
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT        = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs" / "stage2"

N_QUESTIONS = 25
N_RUNS      = 3
N_PER_PERSONA = N_QUESTIONS * N_RUNS   # 75 responses per persona
N_PERSONAS  = 5
N_BRANDS    = 15   # all brands have Screaming Frog content
N_TESTS_BINOM = N_PERSONAS * N_BRANDS  # 75 for Bonferroni correction
N_TESTS_CHI   = N_BRANDS              # one chi-square per brand
T_CRIT = stats.t.ppf(0.975, df=N_RUNS - 1)   # ≈ 4.303


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-10)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-10)
    return A_norm @ B_norm.T


def sig_stars(p_corrected: float) -> str:
    if p_corrected < 0.001: return "***"
    if p_corrected < 0.01:  return "**"
    if p_corrected < 0.05:  return "*"
    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session_id: str) -> None:

    # ── Load data ─────────────────────────────────────────────────────────────
    entities = load_jsonl(OUTPUTS_DIR / f"session_{session_id}_entities_normalized.jsonl")
    entity_df = pd.DataFrame(entities) if entities else pd.DataFrame()

    # Brand list from the affinity matrix (embedded brands only)
    aff_df = pd.read_csv(OUTPUTS_DIR / f"session_{session_id}_affinity_matrix.csv", index_col=0)
    if "persona_name" in aff_df.columns:
        aff_df = aff_df.drop(columns=["persona_name"])
    brands   = sorted(aff_df.columns.tolist())
    personas = sorted(aff_df.index.tolist())

    if entity_df.empty:
        print("No entity data found — cannot run statistical tests.")
        return

    # Positive mentions only, embedded brands only
    pos = entity_df[entity_df["sentiment"] == "positive"].copy()
    pos["brand"] = pos.get("canonical", pos.get("brand", pd.Series()))
    pos = pos[pos["brand"].isin(brands)]

    # ── Binary response-level mentions ────────────────────────────────────────
    # One row per (persona, question, run, brand) — deduplicated so multiple
    # mentions of the same brand in one response count as one.
    binary = (
        pos.groupby(["persona_id", "question_id", "run_number", "brand"])
        .size()
        .reset_index(name="_")[["persona_id", "question_id", "run_number", "brand"]]
    )

    # Base rate: fraction of all 375 responses that positively mention each brand
    brand_base_rates = {}
    for brand in brands:
        n_responses_mentioning = len(
            binary[binary["brand"] == brand]
            [["persona_id", "question_id", "run_number"]].drop_duplicates()
        )
        brand_base_rates[brand] = n_responses_mentioning / (N_PER_PERSONA * N_PERSONAS)

    # ── 1 & 2: Per-run mention rates + binomial tests ─────────────────────────
    stats_rows = []
    for pid in personas:
        for brand in brands:
            # Per-run rates
            run_rates = []
            for r in range(1, N_RUNS + 1):
                mask = (
                    (binary["persona_id"] == pid) &
                    (binary["brand"] == brand) &
                    (binary["run_number"] == r)
                )
                n_mentioned = binary[mask]["question_id"].nunique()
                run_rates.append(n_mentioned / N_QUESTIONS)

            mean_r = float(np.mean(run_rates))
            std_r  = float(np.std(run_rates, ddof=1)) if len(set(run_rates)) > 1 else 0.0
            se_r   = std_r / np.sqrt(N_RUNS)
            ci_low  = max(0.0, mean_r - T_CRIT * se_r)
            ci_high = min(1.0, mean_r + T_CRIT * se_r)

            # Consistency: 1 − coefficient_of_variation
            cv = (std_r / mean_r) if mean_r > 0 else 0.0
            consistency = max(0.0, 1.0 - cv)

            # Binomial test: k out of 75 vs base_rate
            k  = binary[(binary["persona_id"] == pid) & (binary["brand"] == brand)].shape[0]
            p0 = brand_base_rates[brand]
            if p0 == 0 or p0 == 1:
                p_val = 1.0
            else:
                p_val = float(stats.binomtest(k, N_PER_PERSONA, p0,
                                               alternative="two-sided").pvalue)
            p_corr  = min(1.0, p_val * N_TESTS_BINOM)
            stars   = sig_stars(p_corr)
            direction = "over" if mean_r > p0 else ("under" if mean_r < p0 else "—")

            stats_rows.append({
                "persona_id":    pid,
                "brand":         brand,
                "run1_rate":     round(run_rates[0], 4),
                "run2_rate":     round(run_rates[1], 4),
                "run3_rate":     round(run_rates[2], 4),
                "mean_rate":     round(mean_r, 4),
                "std_rate":      round(std_r,  4),
                "ci95_low":      round(ci_low,  4),
                "ci95_high":     round(ci_high, 4),
                "consistency":   round(consistency, 3),
                "k_mentions":    k,
                "n_responses":   N_PER_PERSONA,
                "base_rate":     round(p0, 4),
                "p_value":       round(p_val, 5),
                "p_corrected":   round(p_corr, 5),
                "sig_stars":     stars,
                "direction":     direction,
            })

    stats_df = pd.DataFrame(stats_rows)
    stats_path = OUTPUTS_DIR / f"session_{session_id}_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Stats (binomial) → {stats_path.name}")

    # ── 3: Chi-square per brand: is persona distribution non-uniform? ─────────
    chi_rows = []
    for brand in brands:
        observed = []
        for pid in personas:
            k = int(stats_df[
                (stats_df["persona_id"] == pid) & (stats_df["brand"] == brand)
            ]["k_mentions"].iloc[0])
            observed.append(k)

        total = sum(observed)
        if total < 5:
            chi_rows.append({
                "brand": brand, "total_mentions": total,
                "chi2_stat": None, "p_value": None, "p_corrected": None,
                "sig_stars": "", "significant": False,
                "dominant_persona": personas[int(np.argmax(observed))],
                "note": "Too few mentions for reliable test (n<5)",
            })
            continue

        expected = [total / N_PERSONAS] * N_PERSONAS
        chi2, p = stats.chisquare(observed, f_exp=expected)
        p_corr = min(1.0, p * N_TESTS_CHI)
        chi_rows.append({
            "brand":            brand,
            "total_mentions":   total,
            "chi2_stat":        round(chi2, 3),
            "p_value":          round(p, 5),
            "p_corrected":      round(p_corr, 5),
            "sig_stars":        sig_stars(p_corr),
            "significant":      p_corr < 0.05,
            "dominant_persona": personas[int(np.argmax(observed))],
            "note":             "",
        })

    chi_df = (
        pd.DataFrame(chi_rows)
        .sort_values("p_value", na_position="last")
        .reset_index(drop=True)
    )
    chi_path = OUTPUTS_DIR / f"session_{session_id}_stats_chisquare.csv"
    chi_df.to_csv(chi_path, index=False)
    print(f"Chi-square tests → {chi_path.name}")

    # ── 4: Per-run cosine similarity variance ─────────────────────────────────
    brand_emb    = np.load(OUTPUTS_DIR / f"session_{session_id}_brand_embeddings.npy")
    persona_emb  = np.load(OUTPUTS_DIR / f"session_{session_id}_persona_embeddings.npy")
    brand_meta   = load_jsonl(OUTPUTS_DIR / f"session_{session_id}_brand_meta.jsonl")
    persona_meta = load_jsonl(OUTPUTS_DIR / f"session_{session_id}_persona_meta.jsonl")

    brand_df   = pd.DataFrame(brand_meta)
    persona_df = pd.DataFrame(persona_meta)

    # Brand centroids (same as stage2_analyze.py)
    brand_centroids = {
        brand: brand_emb[brand_df[brand_df["brand"] == brand].index].mean(axis=0)
        for brand in brands
    }
    B      = np.stack([brand_centroids[b] for b in brands])
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-10)

    sim_rows = []
    for pid in personas:
        run_sims = {}
        for r in range(1, N_RUNS + 1):
            mask = (
                (persona_df["persona_id"] == pid) &
                (persona_df["run_number"] == r)
            )
            idx = persona_df[mask].index
            if len(idx) == 0:
                continue
            centroid = persona_emb[idx].mean(axis=0)
            c_norm   = centroid / np.linalg.norm(centroid).clip(min=1e-10)
            run_sims[r] = c_norm @ B_norm.T   # shape (17,)

        for j, brand in enumerate(brands):
            vals = [run_sims[r][j] for r in range(1, N_RUNS + 1) if r in run_sims]
            mean_s = float(np.mean(vals))
            std_s  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            cv_s   = std_s / abs(mean_s) if abs(mean_s) > 1e-10 else 0.0
            sim_rows.append({
                "persona_id":  pid,
                "brand":       brand,
                "run1_sim":    round(float(vals[0]), 4) if len(vals) > 0 else None,
                "run2_sim":    round(float(vals[1]), 4) if len(vals) > 1 else None,
                "run3_sim":    round(float(vals[2]), 4) if len(vals) > 2 else None,
                "mean_sim":    round(mean_s, 4),
                "sim_std":     round(std_s, 4),
                "sim_cv":      round(cv_s, 4),  # low CV = stable AI signal
            })

    run_sim_df = pd.DataFrame(sim_rows)
    run_sim_path = OUTPUTS_DIR / f"session_{session_id}_run_similarity.csv"
    run_sim_df.to_csv(run_sim_path, index=False)
    print(f"Run similarity   → {run_sim_path.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_sig_raw  = (stats_df["p_value"]    < 0.05).sum()
    n_sig_corr = (stats_df["p_corrected"] < 0.05).sum()
    n_chi_sig  = chi_df["significant"].sum()

    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}")
    print(f"\nBinomial tests (n={len(stats_df)} pairs):")
    print(f"  Raw p<0.05:              {n_sig_raw}")
    print(f"  Bonferroni-corrected:    {n_sig_corr}  (α = 0.05/{N_TESTS_BINOM})")

    print(f"\nChi-square (n={len(chi_df)} brands tested, α corrected for {N_TESTS_CHI}):")
    sig_chi = chi_df[chi_df["significant"] == True]
    print(f"  Non-uniform distributions: {len(sig_chi)}")
    for _, row in sig_chi.iterrows():
        if row["chi2_stat"] is not None:
            print(f"    {row['brand']:<25}  χ²={row['chi2_stat']:.1f}  "
                  f"p_corr={row['p_corrected']:.4f}{row['sig_stars']}"
                  f"  → dominant: {row['dominant_persona']}")

    print(f"\nMost consistent brand-persona pairs (mean_rate>0.10):")
    top = (
        stats_df[stats_df["mean_rate"] > 0.10]
        .sort_values("consistency", ascending=False)
        .head(8)
    )
    for _, row in top.iterrows():
        print(f"  {row['persona_id']} × {row['brand']:<26} "
              f"rate={row['mean_rate']:.3f}  "
              f"consistency={row['consistency']:.3f}  "
              f"CI=[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
              f"  {row['sig_stars']}")

    print(f"\nMost AI-inconsistent similarity pairs (high CV = noisy signal):")
    noisy = run_sim_df[run_sim_df["mean_sim"] > 0.35].sort_values(
        "sim_cv", ascending=False
    ).head(6)
    for _, row in noisy.iterrows():
        print(f"  {row['persona_id']} × {row['brand']:<26} "
              f"mean_sim={row['mean_sim']:.4f}  CV={row['sim_cv']:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEO Audit — statistical significance")
    parser.add_argument("--session", required=True, metavar="SESSION_ID")
    args = parser.parse_args()
    run(args.session)


if __name__ == "__main__":
    main()
