#!/usr/bin/env python3
"""
Stage 2 (Optional Step): Brand authority correlation analysis.

Tests whether traditional SEO authority metrics (Domain Authority, referring
domains, organic traffic) predict AI recommendation performance.

Key hypothesis: brands with higher web authority are recommended more often
by ChatGPT — and their content aligns more closely with what ChatGPT says
about them.

Requires data/brand_authority.csv to be populated. Get metrics from:
  - Moz Link Explorer  → Domain Authority, referring domains
  - Ahrefs             → Ahrefs DR, referring domains
  - SEMrush / Ahrefs   → organic keywords, estimated monthly traffic

Outputs:
  - outputs/stage2/session_{id}_authority_correlation.csv
  - outputs/reports/figures/session_{id}/authority_spearman_heatmap.png
  - outputs/reports/figures/session_{id}/authority_scatter_da_mentions.png
  - outputs/reports/figures/session_{id}/authority_scatter_da_similarity.png
  - outputs/reports/figures/session_{id}/authority_scatter_rd_mentions.png

Usage:
    python src/stage2_authority.py --session SESSION_ID
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT           = Path(__file__).parent.parent
OUTPUTS_DIR    = ROOT / "outputs" / "stage2"
REPORTS_DIR    = ROOT / "outputs" / "reports"
DATA_DIR       = ROOT / "data"
AUTHORITY_PATH = DATA_DIR / "brand_authority.csv"

AUTHORITY_LABELS = {
    "domain_authority":    "Domain Authority (Moz)",
    "referring_domains":   "Referring Domains",
    "organic_keywords":    "Organic Keywords",
    "organic_traffic_est": "Est. Monthly Organic Traffic",
    "ahrefs_dr":           "Ahrefs Domain Rating",
}

PERFORMANCE_LABELS = {
    "total_mentions":   "Total Positive Mentions",
    "primary_mentions": "Primary Recommendations",
    "mention_rate":     "Mention Rate (mentions / 375)",
    "mean_cosine_sim":  "Mean Content Match Score (0–100)",
}

FIG_BG     = "#FAFAFA"
DPI        = 150
C_SCATTER  = "#2471A3"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ── Data loading ──────────────────────────────────────────────────────────────

def load_authority() -> pd.DataFrame:
    if not AUTHORITY_PATH.exists():
        raise FileNotFoundError(
            f"Authority file not found: {AUTHORITY_PATH}\n"
            "Fill in data/brand_authority.csv with DA, referring domains, etc. "
            "from Moz, Ahrefs, or SEMrush before running this script."
        )
    df = pd.read_csv(AUTHORITY_PATH)
    # Check that at least one authority column has data (only cols present in the file)
    all_auth_cols = ["domain_authority", "referring_domains", "organic_keywords",
                     "organic_traffic_est", "ahrefs_dr"]
    auth_cols = [c for c in all_auth_cols if c in df.columns]
    if not auth_cols or df[auth_cols].isna().all().all():
        raise ValueError(
            "data/brand_authority.csv exists but all metric columns are empty.\n"
            "Please fill in at least one column (e.g. domain_authority) for each brand."
        )
    return df


def build_performance_df(session_id: str, brands: list[str]) -> pd.DataFrame:
    """Aggregate AI performance metrics per brand across all personas."""
    entities = load_jsonl(OUTPUTS_DIR / f"session_{session_id}_entities_normalized.jsonl")
    entity_df = pd.DataFrame(entities) if entities else pd.DataFrame()

    aff_df = pd.read_csv(OUTPUTS_DIR / f"session_{session_id}_affinity_matrix.csv", index_col=0)
    if "persona_name" in aff_df.columns:
        aff_df = aff_df.drop(columns=["persona_name"])

    rows = []
    for brand in brands:
        total_m   = 0
        primary_m = 0
        if not entity_df.empty:
            pos = entity_df[entity_df["sentiment"] == "positive"]
            brand_pos = pos[pos.get("canonical", pos.get("brand", pd.Series())).isin([brand])]
            if "canonical" in pos.columns:
                brand_pos = pos[pos["canonical"] == brand]
            elif "brand" in pos.columns:
                brand_pos = pos[pos["brand"] == brand]
            total_m   = len(brand_pos)
            primary_m = (brand_pos["rank"] == "primary").sum() if "rank" in brand_pos.columns else 0

        mean_sim = float(aff_df[brand].mean()) * 100   # scale to 0–100

        rows.append({
            "brand":           brand,
            "total_mentions":  total_m,
            "primary_mentions": int(primary_m),
            "mention_rate":    round(total_m / 375, 4),
            "mean_cosine_sim": round(mean_sim, 2),
        })

    return pd.DataFrame(rows)


# ── Analysis ──────────────────────────────────────────────────────────────────

def spearman_matrix(
    merged: pd.DataFrame,
    authority_cols: list[str],
    performance_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman rho and p-value between every authority × performance pair."""
    rhos = pd.DataFrame(index=authority_cols, columns=performance_cols, dtype=float)
    pvals = pd.DataFrame(index=authority_cols, columns=performance_cols, dtype=float)

    for a in authority_cols:
        for p in performance_cols:
            valid = merged[[a, p]].dropna()
            if len(valid) < 5:
                rhos.loc[a, p]  = np.nan
                pvals.loc[a, p] = np.nan
            else:
                rho, pv = stats.spearmanr(valid[a], valid[p])
                rhos.loc[a, p]  = round(float(rho), 3)
                pvals.loc[a, p] = round(float(pv),  4)

    return rhos, pvals


# ── Figures ───────────────────────────────────────────────────────────────────

def scatter_plot(
    merged: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
) -> None:
    valid = merged[["brand", x_col, y_col]].dropna()
    if len(valid) < 5:
        print(f"  Skipping {x_col} vs {y_col} — fewer than 5 complete rows")
        return

    rho, pv = stats.spearmanr(valid[x_col], valid[y_col])
    x = valid[x_col].values.astype(float)
    y = valid[y_col].values.astype(float)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.grid(color="#E5E5E5", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#E5E5E5")

    ax.scatter(x, y, s=80, color=C_SCATTER, alpha=0.8,
               edgecolors="white", linewidths=0.7, zorder=3)

    # Regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#E74C3C", lw=1.5,
            ls="--", alpha=0.7, zorder=2)

    # Label each point
    for _, row in valid.iterrows():
        short = (row["brand"]
                 .replace(" Bread", "").replace(" King", " K.")
                 .replace(" in the Box", " Box").replace(" Burger", ""))
        ax.annotate(short, (row[x_col], row[y_col]),
                    fontsize=8, xytext=(5, 3), textcoords="offset points",
                    color="#333333")

    n = len(valid)
    ax.set_xlabel(AUTHORITY_LABELS.get(x_col, x_col), fontsize=11)
    ax.set_ylabel(PERFORMANCE_LABELS.get(y_col, y_col), fontsize=11)
    ax.set_title(
        f"{AUTHORITY_LABELS.get(x_col, x_col)} vs. "
        f"{PERFORMANCE_LABELS.get(y_col, y_col)}\n"
        f"Spearman ρ = {rho:.3f}  (p = {pv:.3f}{sig_stars(pv)},  n = {n})",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.text(0.01, 0.01,
            f"ρ = {rho:.3f}  {sig_stars(pv)}  |  n={n}  |  "
            f"Interpret cautiously: small sample, wide CI on ρ",
            transform=ax.transAxes, fontsize=8, color="#888",
            va="bottom")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def spearman_heatmap(
    rhos: pd.DataFrame,
    pvals: pd.DataFrame,
    out_path: Path,
) -> None:
    # Build annotation matrix: rho + stars
    annot = pd.DataFrame(index=rhos.index, columns=rhos.columns, dtype=str)
    for a in rhos.index:
        for p in rhos.columns:
            rho = rhos.loc[a, p]
            pv  = pvals.loc[a, p]
            if pd.isna(rho):
                annot.loc[a, p] = "n/a"
            else:
                annot.loc[a, p] = f"{rho:.2f}{sig_stars(float(pv))}"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(FIG_BG)

    sns.heatmap(
        rhos.astype(float),
        ax=ax,
        xticklabels=[PERFORMANCE_LABELS.get(c, c) for c in rhos.columns],
        yticklabels=[AUTHORITY_LABELS.get(c, c) for c in rhos.index],
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
    )
    ax.set_title(
        "Spearman Correlations: Web Authority vs. AI Recommendation Performance\n"
        "* p<0.05  ** p<0.01  *** p<0.001  (uncorrected; n=17 brands)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.tick_params(axis="x", rotation=15, labelsize=9.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=9.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=FIG_BG)
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session_id: str) -> None:
    authority_df = load_authority()

    # Determine which brands we have embeddings for
    aff_path = OUTPUTS_DIR / f"session_{session_id}_affinity_matrix.csv"
    aff_check = pd.read_csv(aff_path, index_col=0)
    if "persona_name" in aff_check.columns:
        aff_check = aff_check.drop(columns=["persona_name"])
    brands = sorted(aff_check.columns.tolist())

    perf_df = build_performance_df(session_id, brands)
    merged  = authority_df.merge(perf_df, on="brand", how="inner")

    n_brands_matched = len(merged)
    print(f"\nAuthority correlation analysis")
    print(f"Brands with both authority data and embeddings: {n_brands_matched}")

    authority_cols   = [c for c in AUTHORITY_LABELS if c in merged.columns
                        and merged[c].notna().sum() >= 5]
    performance_cols = list(PERFORMANCE_LABELS.keys())

    if not authority_cols:
        print("\nNo authority columns have sufficient data. "
              "Fill in data/brand_authority.csv and re-run.")
        return

    # Spearman correlation matrix
    rhos, pvals = spearman_matrix(merged, authority_cols, performance_cols)

    # Save merged data
    corr_path = OUTPUTS_DIR / f"session_{session_id}_authority_correlation.csv"
    merged.to_csv(corr_path, index=False)
    print(f"Merged data → {corr_path.name}")

    # Figures
    figures_dir = REPORTS_DIR / "figures" / f"session_{session_id}"

    spearman_heatmap(rhos, pvals,
                     figures_dir / "authority_spearman_heatmap.png")

    if "domain_authority" in authority_cols:
        scatter_plot(merged, "domain_authority", "total_mentions",
                     figures_dir / "authority_scatter_da_mentions.png")
        scatter_plot(merged, "domain_authority", "mean_cosine_sim",
                     figures_dir / "authority_scatter_da_similarity.png")

    if "referring_domains" in authority_cols:
        scatter_plot(merged, "referring_domains", "total_mentions",
                     figures_dir / "authority_scatter_rd_mentions.png")

    # Print correlation summary
    print(f"\n{'='*60}")
    print("SPEARMAN CORRELATION SUMMARY")
    print(f"n = {n_brands_matched} brands  "
          f"(interpret cautiously — small sample, wide CI on ρ)")
    print(f"{'='*60}")
    print(f"\n{'Authority Metric':<30}  {'vs. Mentions':>14}  "
          f"{'vs. Primary':>14}  {'vs. Sim':>12}")
    print("-" * 74)
    for a in authority_cols:
        rho_m  = rhos.loc[a, "total_mentions"]
        pv_m   = pvals.loc[a, "total_mentions"]
        rho_pr = rhos.loc[a, "primary_mentions"]
        pv_pr  = pvals.loc[a, "primary_mentions"]
        rho_s  = rhos.loc[a, "mean_cosine_sim"]
        pv_s   = pvals.loc[a, "mean_cosine_sim"]
        def fmt(r, p):
            if pd.isna(r): return "   n/a"
            return f"{r:+.3f}{sig_stars(float(p)):<3}"
        print(f"  {AUTHORITY_LABELS[a]:<28}  {fmt(rho_m, pv_m):>14}  "
              f"{fmt(rho_pr, pv_pr):>14}  {fmt(rho_s, pv_s):>12}")
    print()
    print("Interpretation guide:")
    print("  |ρ| > 0.60 → strong association")
    print("  |ρ| 0.40–0.59 → moderate")
    print("  |ρ| < 0.40 → weak (or noise given n=17)")
    print("  With n=17, even ρ=0.48 is required to reach p<0.05 (uncorrected)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GEO Audit — brand authority correlation analysis"
    )
    parser.add_argument("--session", required=True, metavar="SESSION_ID")
    args = parser.parse_args()
    run(args.session)


if __name__ == "__main__":
    main()
