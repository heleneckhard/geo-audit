#!/usr/bin/env python3
"""
GEO Audit — Figure generator.

Produces publication-quality PNG figures for the key report findings:

  1. affinity_heatmap.png          — Cosine similarity: persona × brand
  2. mention_heatmap.png           — Positive mention counts: persona × brand
  3. content_gap_scatter.png       — Gap analysis scatter (5 persona subplots)
  4. recommendation_landscape.png  — Brand mention totals with primary split
  5. persona_top_brands.png        — Top 8 brands per persona (5 subplots)
  6. top_content_gaps.png          — Ranked content gap scores
  7. category_affinity.png         — Question-category alignment for top brands

Also writes an illustrated version of the report with images embedded:
  outputs/reports/session_{id}_report_illustrated.md

Usage:
    python src/generate_figures.py --session SESSION_ID
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

ROOT        = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs" / "stage2"
REPORTS_DIR = ROOT / "outputs" / "reports"
DATA_DIR    = ROOT / "data"

# ── Style constants ────────────────────────────────────────────────────────────
BRAND_FONT   = "DejaVu Sans"
DPI          = 150
FIG_BG       = "#FAFAFA"
GRID_COLOR   = "#E5E5E5"

C_GAP        = "#C0392B"   # content gap   (over-recommended)
C_MISSED     = "#2471A3"   # missed opp    (under-recommended)
C_ALIGNED    = "#7F8C8D"   # aligned

PERSONA_COLORS = {
    "P1": "#1A5276",   # Marcus  — deep navy
    "P2": "#196F3D",   # Jenna   — forest green
    "P3": "#7D3C98",   # Tyler   — purple
    "P4": "#B7950B",   # Priya   — gold
    "P5": "#922B21",   # Dale    — brick red
}

CATEGORY_SHORT = {
    "comparative_evaluative": "Comparative",
    "discovery_trend":        "Discovery",
    "need_constraint":        "Need/Constraint",
    "spontaneous_occasion":   "Spontaneous",
    "strategic_planning":     "Strategic",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def savefig(fig, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor=FIG_BG, **kwargs)
    plt.close(fig)
    print(f"  Saved → {path.name}")


def styled_ax(ax):
    """Apply consistent grid/spine style to an axis."""
    ax.set_facecolor(FIG_BG)
    ax.grid(axis="both", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    return ax


# ── Figure 1: Affinity Heatmap ─────────────────────────────────────────────

def fig_affinity_heatmap(aff_df: pd.DataFrame, pid_to_name: dict, figures_dir: Path) -> Path:
    """Cosine similarity heatmap — persona × brand."""
    brands   = aff_df.columns.tolist()
    personas = aff_df.index.tolist()

    # Build display matrix
    data = aff_df.values.astype(float)
    row_labels = [pid_to_name[p] for p in personas]

    # Abbreviate brand names for display
    col_labels = [b.replace(" Bread", "").replace(" King", " K.")
                   .replace(" in the Box", " Box")
                   .replace(" Burger", "")
                   .replace("-N-Out", "-N-Out")
                   for b in brands]

    fig, ax = plt.subplots(figsize=(15, 4.5))
    fig.patch.set_facecolor(FIG_BG)

    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(
        data,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 8.5, "weight": "bold"},
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.7},
        vmin=data.min() - 0.01,
        vmax=data.max() + 0.01,
    )

    ax.set_title(
        "Brand Content Alignment by Persona\n"
        "(Cosine similarity between persona response centroid and brand content centroid)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Brand", fontsize=11, labelpad=8)
    ax.set_ylabel("Persona", fontsize=11, labelpad=8)
    ax.tick_params(axis="x", rotation=35, labelsize=9.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=10)

    plt.tight_layout()
    out = figures_dir / "affinity_heatmap.png"
    savefig(fig, out)
    return out


# ── Figure 2: Mention Count Heatmap ───────────────────────────────────────────

def fig_mention_heatmap(mention_matrix: pd.DataFrame, pid_to_name: dict,
                        figures_dir: Path) -> Path:
    """Positive mention count heatmap — persona × brand."""
    brands   = mention_matrix.columns.tolist()
    personas = mention_matrix.index.tolist()

    data = mention_matrix.values.astype(float)
    row_labels = [pid_to_name[p] for p in personas]
    col_labels = [b.replace(" Bread", "").replace(" King", " K.")
                   .replace(" in the Box", " Box").replace(" Burger", "")
                   for b in brands]

    fig, ax = plt.subplots(figsize=(15, 4.5))
    fig.patch.set_facecolor(FIG_BG)

    sns.heatmap(
        data,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"label": "Positive Mentions", "shrink": 0.7},
        vmin=0,
    )

    ax.set_title(
        "Positive Brand Mentions by Persona\n"
        "(Extracted from 375 ChatGPT responses)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Brand", fontsize=11, labelpad=8)
    ax.set_ylabel("Persona", fontsize=11, labelpad=8)
    ax.tick_params(axis="x", rotation=35, labelsize=9.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=10)

    plt.tight_layout()
    out = figures_dir / "mention_heatmap.png"
    savefig(fig, out)
    return out


# ── Figure 3: Content Gap Scatter ─────────────────────────────────────────────

def fig_content_gap_scatter(aff_df: pd.DataFrame,
                             mention_matrix: pd.DataFrame,
                             gap_df: pd.DataFrame,
                             pid_to_name: dict,
                             figures_dir: Path) -> Path:
    """
    5-panel scatter (one per persona).
    X = cosine similarity, Y = positive mention count.
    Quadrant lines at the across-brand medians for that persona.
    Points colored by gap_type.
    """
    brands   = aff_df.columns.tolist()
    personas = sorted(aff_df.index.tolist())

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor(FIG_BG)
    axes = axes.flatten()

    for i, pid in enumerate(personas):
        ax = axes[i]
        styled_ax(ax)

        sims    = aff_df.loc[pid].values.astype(float)
        counts  = mention_matrix.loc[pid].values.astype(float)
        pname   = pid_to_name[pid]
        pcolor  = PERSONA_COLORS[pid]

        # Get gap types
        gap_types = {}
        for _, row in gap_df[gap_df["persona_id"] == pid].iterrows():
            gap_types[row["brand"]] = row["gap_type"]

        # Reference lines at per-persona medians
        med_sim   = np.median(sims)
        med_count = np.median(counts)
        ax.axvline(med_sim,   color="#AAAAAA", lw=1.2, ls="--", zorder=1)
        ax.axhline(med_count, color="#AAAAAA", lw=1.2, ls="--", zorder=1)

        # Quadrant labels
        x_min = sims.min() - 0.01
        x_max = sims.max() + 0.01
        y_max = counts.max()
        ax.text(x_min + 0.003, y_max * 0.97, "Content Gap ▲",
                fontsize=7.5, color=C_GAP, alpha=0.7, ha="left", va="top")
        ax.text(x_max - 0.003, med_count * 0.25, "Missed Opp. ►",
                fontsize=7.5, color=C_MISSED, alpha=0.7, ha="right", va="bottom")

        for j, brand in enumerate(brands):
            gt   = gap_types.get(brand, "aligned")
            col  = C_GAP if gt == "content_gap" else \
                   C_MISSED if gt == "missed_opportunity" else C_ALIGNED
            sz   = max(55, counts[j] * 4.5)
            ax.scatter(sims[j], counts[j], s=sz, color=col,
                       alpha=0.82, edgecolors="white", linewidths=0.8,
                       zorder=3)

            # Label significant points
            is_top_mention = counts[j] >= np.percentile(counts, 60)
            is_gap         = abs(gap_df.loc[
                (gap_df["persona_id"] == pid) & (gap_df["brand"] == brand), "gap_score"
            ].values[0] if len(gap_df[(gap_df["persona_id"] == pid) &
                                       (gap_df["brand"] == brand)]) else [0]) > 0.28
            if is_top_mention or is_gap:
                short = (brand.replace(" Bread", "")
                              .replace(" King", " K.")
                              .replace(" in the Box", " Box")
                              .replace(" Burger", ""))
                ax.annotate(
                    short,
                    (sims[j], counts[j]),
                    fontsize=7.5,
                    xytext=(5, 4),
                    textcoords="offset points",
                    color="#222222",
                )

        ax.set_title(pname, fontsize=11, fontweight="bold", pad=6, color=pcolor)
        ax.set_xlabel("Content Alignment (Cosine Similarity)", fontsize=9)
        ax.set_ylabel("Positive Mentions", fontsize=9)
        ax.tick_params(labelsize=8.5)

    # Legend
    axes[-1].set_visible(False)
    legend_elements = [
        mpatches.Patch(color=C_GAP,     label="Content Gap  (over-recommended)"),
        mpatches.Patch(color=C_MISSED,  label="Missed Opportunity  (under-recommended)"),
        mpatches.Patch(color=C_ALIGNED, label="Aligned"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.05),
        fontsize=10,
        framealpha=0.9,
        edgecolor=GRID_COLOR,
    )
    fig.suptitle(
        "Content Gap Analysis: Similarity vs. Mention Count by Persona\n"
        "Dashed lines show per-persona medians. Bubble size ∝ mention count.",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = figures_dir / "content_gap_scatter.png"
    savefig(fig, out)
    return out


# ── Figure 4: Recommendation Landscape ────────────────────────────────────────

def fig_recommendation_landscape(brand_totals: Counter, brand_primary: Counter,
                                  top_n: int, figures_dir: Path) -> Path:
    """Horizontal stacked bar: total vs. primary mentions, top N brands."""
    top_brands = [b for b, _ in brand_totals.most_common(top_n)]
    totals  = [brand_totals[b]  for b in top_brands]
    prims   = [brand_primary.get(b, 0) for b in top_brands]
    others  = [t - p for t, p in zip(totals, prims)]

    # Sort ascending for horizontal bar (so top brand is at top)
    idx = list(range(len(top_brands)))
    idx.sort(key=lambda i: totals[i])
    top_brands = [top_brands[i] for i in idx]
    prims      = [prims[i]  for i in idx]
    others     = [others[i] for i in idx]

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(FIG_BG)
    styled_ax(ax)
    ax.grid(axis="x")
    ax.grid(axis="y", color="white")  # hide horizontal gridlines

    y = range(len(top_brands))
    bar_h = 0.62

    # Primary (darker)
    bars_p = ax.barh(y, prims,  height=bar_h, color="#1A5276", label="Primary rec",  zorder=3)
    # Secondary/mentioned (lighter)
    bars_o = ax.barh(y, others, height=bar_h, left=prims, color="#AED6F1",
                     label="Secondary / mentioned", zorder=3)

    ax.set_yticks(list(y))
    ax.set_yticklabels(top_brands, fontsize=10)
    ax.set_xlabel("Number of Positive Mentions", fontsize=11)
    ax.set_title(
        f"Brand Recommendation Landscape — Top {top_n} Brands (All Personas)\n"
        "Dark segment = Primary recommendation; Light = Secondary / mentioned",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Value labels on right edge
    total_list = [p + o for p, o in zip(prims, others)]
    for bar_p, bar_o, tot in zip(bars_p, bars_o, total_list):
        x_end = bar_p.get_x() + bar_p.get_width() + bar_o.get_width()
        ax.text(x_end + 1, bar_p.get_y() + bar_p.get_height() / 2,
                str(tot), va="center", ha="left", fontsize=9, color="#333333")

    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, max(total_list) * 1.12)
    plt.tight_layout()
    out = figures_dir / "recommendation_landscape.png"
    savefig(fig, out)
    return out


# ── Figure 5: Per-Persona Top Brands ──────────────────────────────────────────

def fig_persona_top_brands(entity_stats: dict, brands: list[str],
                            personas: list[str], pid_to_name: dict,
                            figures_dir: Path) -> Path:
    """5-panel horizontal bar chart — top 8 brands per persona."""
    top_n = 8

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor(FIG_BG)
    axes = axes.flatten()

    for i, pid in enumerate(personas):
        ax = axes[i]
        styled_ax(ax)
        ax.grid(axis="x")
        ax.grid(axis="y", color="white")

        pname  = pid_to_name[pid]
        pcolor = PERSONA_COLORS[pid]

        brand_counts = {b: entity_stats[(pid, b)]["count"] for b in brands}
        top = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        top = [(b, c) for b, c in top if c > 0][:top_n]
        top = list(reversed(top))   # ascending for horizontal bar

        b_labels = [b.replace(" Bread", "").replace(" King", " K.")
                     .replace(" in the Box", " Box").replace(" Burger", "")
                     for b, _ in top]
        b_vals   = [c for _, c in top]
        prims    = [entity_stats[(pid, b)]["primary"] for b, _ in top]
        secs     = [v - p for v, p in zip(b_vals, prims)]

        y = range(len(b_labels))
        ax.barh(y, prims, height=0.6, color=pcolor,      label="Primary",  zorder=3)
        ax.barh(y, secs,  height=0.6, left=prims,
                color=pcolor, alpha=0.35, label="Secondary/Mentioned", zorder=3)

        ax.set_yticks(list(y))
        ax.set_yticklabels(b_labels, fontsize=9.5)
        ax.set_xlabel("Positive Mentions", fontsize=9)
        ax.set_title(pname, fontsize=11, fontweight="bold", pad=6, color=pcolor)
        ax.tick_params(labelsize=9)

        for bar, val in zip(ax.patches[:len(b_labels)], b_vals):
            pass
        # Label total at right edge
        for j, (val, bar) in enumerate(zip(b_vals, ax.patches[:len(b_labels)])):
            ax.text(
                b_vals[j] + 0.4, j,
                str(b_vals[j]),
                va="center", ha="left", fontsize=8.5, color="#333333",
            )

    axes[-1].set_visible(False)
    fig.legend(
        handles=[
            mpatches.Patch(color="#555", label="Primary recommendation"),
            mpatches.Patch(color="#AAAAAA", label="Secondary / mentioned"),
        ],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.04),
        fontsize=10,
        framealpha=0.9,
    )
    fig.suptitle(
        "Top Brand Recommendations by Persona\n"
        "Dark = primary recommendation; light = secondary or mentioned",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = figures_dir / "persona_top_brands.png"
    savefig(fig, out)
    return out


# ── Figure 6: Ranked Content Gaps ─────────────────────────────────────────────

def fig_top_gaps(gap_df: pd.DataFrame, pid_to_name: dict, figures_dir: Path) -> Path:
    """
    Dual horizontal bar chart:
      Top panel  — content gaps  (gap_score > 0.3)
      Bottom panel — missed opps (gap_score < -0.3)
    """
    content_gaps = gap_df[gap_df["gap_type"] == "content_gap"].nlargest(12, "gap_score")
    missed_opps  = gap_df[gap_df["gap_type"] == "missed_opportunity"].nsmallest(12, "gap_score")

    def make_labels(df):
        return [f"{r['brand']}\n→ {r['persona_name']}" for _, r in df.iterrows()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor(FIG_BG)

    # -- Content Gaps --
    styled_ax(ax1)
    ax1.grid(axis="x")
    ax1.grid(axis="y", color="white")

    cg_labels = list(reversed(make_labels(content_gaps)))
    cg_vals   = list(reversed(content_gaps["gap_score"].tolist()))
    cg_cnts   = list(reversed(content_gaps["mention_count"].tolist()))

    y1 = range(len(cg_labels))
    bars1 = ax1.barh(y1, cg_vals, height=0.65, color=C_GAP, zorder=3)
    ax1.set_yticks(list(y1))
    ax1.set_yticklabels(cg_labels, fontsize=9)
    ax1.set_xlabel("Gap Score (mention_pct − sim_pct)", fontsize=10)
    ax1.set_title("Content Gaps\n(over-recommended vs. content alignment)",
                  fontsize=11, fontweight="bold", color=C_GAP, pad=8)
    for bar, cnt in zip(bars1, cg_cnts):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{cnt} mentions", va="center", ha="left", fontsize=8.5, color="#444")

    # -- Missed Opportunities --
    styled_ax(ax2)
    ax2.grid(axis="x")
    ax2.grid(axis="y", color="white")

    mo_labels = make_labels(missed_opps)
    mo_vals   = [abs(v) for v in missed_opps["gap_score"].tolist()]
    mo_sims   = missed_opps["similarity"].tolist()

    y2 = range(len(mo_labels))
    bars2 = ax2.barh(y2, mo_vals, height=0.65, color=C_MISSED, zorder=3)
    ax2.set_yticks(list(y2))
    ax2.set_yticklabels(mo_labels, fontsize=9)
    ax2.set_xlabel("|Gap Score|  (sim_pct − mention_pct)", fontsize=10)
    ax2.set_title("Missed Opportunities\n(aligned content, low recommendations)",
                  fontsize=11, fontweight="bold", color=C_MISSED, pad=8)
    for bar, sim in zip(bars2, mo_sims):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"sim {sim:.3f}", va="center", ha="left", fontsize=8.5, color="#444")

    fig.suptitle(
        "Content Gap Rankings: Where Recommendations and Content Alignment Diverge",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = figures_dir / "top_content_gaps.png"
    savefig(fig, out)
    return out


# ── Figure 7: Category Affinity Heatmap ───────────────────────────────────────

def fig_category_affinity(cat_df: pd.DataFrame, pid_to_name: dict,
                           top_brands: list[str], figures_dir: Path) -> Path:
    """
    Heatmap of per-category cosine similarity for top brands, one panel per persona.
    Shows which question contexts drive alignment to which brands.
    """
    categories = sorted(cat_df["question_category"].unique())
    personas   = sorted(cat_df["persona_id"].unique())

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
    fig.patch.set_facecolor(FIG_BG)
    axes = axes.flatten()

    cat_labels = [CATEGORY_SHORT.get(c, c) for c in categories]

    for i, pid in enumerate(personas):
        ax    = axes[i]
        pname = pid_to_name[pid]
        sub   = cat_df[cat_df["persona_id"] == pid]

        matrix = pd.DataFrame(index=top_brands, columns=categories, dtype=float)
        for _, row in sub.iterrows():
            if row["brand"] in top_brands:
                matrix.loc[row["brand"], row["question_category"]] = row["similarity"]

        matrix = matrix.fillna(0)

        # Abbreviate row (brand) labels
        row_labels = [b.replace(" Bread", "").replace(" King", " K.")
                       .replace(" in the Box", " Box").replace(" Burger", "")
                       for b in top_brands]

        sns.heatmap(
            matrix.values.astype(float),
            ax=ax,
            xticklabels=cat_labels,
            yticklabels=row_labels,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            linewidths=0.4,
            linecolor="white",
            annot_kws={"size": 8},
            cbar=False,
            vmin=matrix.values.min() - 0.005,
            vmax=matrix.values.max() + 0.005,
        )
        ax.set_title(pname, fontsize=11, fontweight="bold",
                     color=PERSONA_COLORS[pid], pad=6)
        ax.tick_params(axis="x", rotation=25, labelsize=8.5)
        ax.tick_params(axis="y", rotation=0,  labelsize=8.5)

    axes[-1].set_visible(False)
    fig.suptitle(
        "Brand Content Alignment by Question Category and Persona\n"
        "(Cosine similarity between persona-category centroid and brand content centroid)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = figures_dir / "category_affinity.png"
    savefig(fig, out)
    return out


# ── Illustrated Report ────────────────────────────────────────────────────────

def write_illustrated_report(session_id: str, figures_dir: Path,
                              report_path: Path) -> None:
    """
    Reads the base markdown report and inserts image references at the
    appropriate sections, writing a new _illustrated.md file.
    """
    rel_dir = figures_dir.relative_to(REPORTS_DIR)

    def img(fname, caption):
        return f"\n![{caption}]({rel_dir}/{fname})\n"

    text = report_path.read_text()

    # Insert after the recommendation landscape table note
    text = text.replace(
        "\n---\n\n## Persona-by-Persona Analysis",
        img("recommendation_landscape.png", "Brand Recommendation Landscape") +
        img("mention_heatmap.png", "Mention Count Heatmap") +
        "\n---\n\n## Persona-by-Persona Analysis",
    )

    # Insert after Persona-by-Persona section header paragraph
    text = text.replace(
        "the question contexts driving recommendations, and the gap assessment.\n",
        "the question contexts driving recommendations, and the gap assessment.\n" +
        img("persona_top_brands.png", "Top Brands by Persona"),
    )

    # Insert content gap scatter before brand-by-brand section
    text = text.replace(
        "\n---\n\n## Brand Content Alignment Analysis",
        img("content_gap_scatter.png", "Content Gap Scatter — Similarity vs. Mentions") +
        "\n---\n\n## Brand Content Alignment Analysis",
    )

    # Insert top gaps chart before strategic recommendations
    text = text.replace(
        "\n---\n\n## Strategic Recommendations",
        img("top_content_gaps.png", "Ranked Content Gap Scores") +
        "\n---\n\n## Strategic Recommendations",
    )

    # Insert affinity heatmap before appendix A
    text = text.replace(
        "\n## Appendix A: Full Affinity Matrix",
        img("affinity_heatmap.png", "Affinity Matrix Heatmap") +
        "\n## Appendix A: Full Affinity Matrix",
    )

    # Insert category affinity before appendix C
    text = text.replace(
        "\n## Appendix C: Persona Profiles",
        img("category_affinity.png", "Category-Level Brand Alignment") +
        "\n## Appendix C: Persona Profiles",
    )

    out_path = REPORTS_DIR / f"session_{session_id}_report_illustrated.md"
    out_path.write_text(text)
    print(f"  Illustrated report → {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session_id: str) -> None:
    # ── Load ─────────────────────────────────────────────────────────────────
    personas_cfg = yaml.safe_load((DATA_DIR / "personas.yaml").read_text())
    entities     = load_jsonl(OUTPUTS_DIR / f"session_{session_id}_entities_normalized.jsonl")

    aff_df  = pd.read_csv(OUTPUTS_DIR / f"session_{session_id}_affinity_matrix.csv",  index_col=0)
    gap_df  = pd.read_csv(OUTPUTS_DIR / f"session_{session_id}_content_gaps.csv")
    cat_df  = pd.read_csv(OUTPUTS_DIR / f"session_{session_id}_category_affinity.csv")

    entity_df = pd.DataFrame(entities) if entities else pd.DataFrame()

    if "persona_name" in aff_df.columns:
        aff_df = aff_df.drop(columns=["persona_name"])

    brands   = sorted(aff_df.columns.tolist())
    personas = sorted(aff_df.index.tolist())

    pid_to_name = {p["id"]: p["name"] for p in personas_cfg["personas"]}

    # ── Entity counts ─────────────────────────────────────────────────────────
    entity_stats = defaultdict(lambda: {"count": 0, "primary": 0, "weighted": 0,
                                         "attrs": Counter(), "quotes": []})
    brand_totals = Counter()
    brand_primary = Counter()

    if not entity_df.empty:
        pos = entity_df[entity_df["sentiment"] == "positive"]
        for _, row in pos.iterrows():
            pid   = row["persona_id"]
            brand = row.get("canonical") or row.get("brand", "")
            if not brand:
                continue
            key = (pid, brand)
            entity_stats[key]["count"]    += 1
            entity_stats[key]["weighted"] += 2 if row.get("rank") == "primary" else 1
            brand_totals[brand]           += 1
            if row.get("rank") == "primary":
                entity_stats[key]["primary"] += 1
                brand_primary[brand]          += 1

    # Build mention matrix (only the 17 embedded brands)
    mention_data = {
        brand: [entity_stats[(pid, brand)]["count"] for pid in personas]
        for brand in brands
    }
    mention_matrix = pd.DataFrame(mention_data, index=personas)

    # ── Figures directory ─────────────────────────────────────────────────────
    figures_dir = REPORTS_DIR / "figures" / f"session_{session_id}"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Pick the top 10 brands by total mentions for the category heatmap
    top10 = [b for b, _ in sorted(brand_totals.items(),
                                   key=lambda x: -x[1]) if b in brands][:10]

    print(f"\nGenerating figures → {figures_dir}")

    fig_affinity_heatmap(aff_df, pid_to_name, figures_dir)
    fig_mention_heatmap(mention_matrix, pid_to_name, figures_dir)
    fig_content_gap_scatter(aff_df, mention_matrix, gap_df, pid_to_name, figures_dir)
    fig_recommendation_landscape(brand_totals, brand_primary, top_n=18, figures_dir=figures_dir)
    fig_persona_top_brands(entity_stats, brands, personas, pid_to_name, figures_dir)
    fig_top_gaps(gap_df, pid_to_name, figures_dir)
    fig_category_affinity(cat_df, pid_to_name, top_brands=top10, figures_dir=figures_dir)

    # ── Illustrated report ────────────────────────────────────────────────────
    report_path = REPORTS_DIR / f"session_{session_id}_report.md"
    if report_path.exists():
        write_illustrated_report(session_id, figures_dir, report_path)
    else:
        print(f"  Warning: base report not found at {report_path}, skipping illustration.")

    print(f"\nDone. {len(list(figures_dir.glob('*.png')))} figures in {figures_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEO Audit — figure generator")
    parser.add_argument("--session", required=True, metavar="SESSION_ID")
    args = parser.parse_args()
    run(args.session)


if __name__ == "__main__":
    main()
