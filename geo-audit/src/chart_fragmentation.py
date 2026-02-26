#!/usr/bin/env python3
"""
Two supplementary charts for the GEO Audit report:
  1. Market fragmentation — donut charts showing Top 3 / Top 10 / Rest share
  2. Persona specificity — share of brands appearing for 1 / 2–3 / 4–5 personas
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT      = Path(__file__).parent.parent
REPORT    = ROOT.parent / "revised_report"
OUT_DIR   = REPORT


# ── Load data ─────────────────────────────────────────────────────────────────

def load_csv(path, brand_col):
    brands = []
    with open(path) as f:
        reader = csv.DictReader(f)
        persona_cols = [c for c in reader.fieldnames if c not in (brand_col, 'total')]
        for row in reader:
            per_persona = [int(row[p]) for p in persona_cols]
            n_personas  = sum(1 for m in per_persona if m > 0)
            brands.append({'name': row[brand_col], 'total': int(row['total']), 'n_personas': n_personas})
    return brands


qsr    = load_csv(REPORT / "QSR_phase1_all_mentions.csv",    "brand")
agency = load_csv(REPORT / "agency_phase1_all_mentions.csv", "agency")


# ── Shared style ──────────────────────────────────────────────────────────────

QSR_COLOR    = "#2563EB"   # blue
AGENCY_COLOR = "#059669"   # emerald
GRID_COLOR   = "#E5E7EB"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.8,
})


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Fragmentation donuts: Top 3 / Top 4–10 / Rest
# ─────────────────────────────────────────────────────────────────────────────

def concentration_slices(brands):
    """Return (top3_pct, next7_pct, rest_pct, top3_names) sorted by total desc."""
    ranked = sorted(brands, key=lambda b: -b['total'])
    total  = sum(b['total'] for b in ranked)
    top3   = ranked[:3]
    next7  = ranked[3:10]
    rest   = ranked[10:]
    return (
        sum(b['total'] for b in top3)  / total * 100,
        sum(b['total'] for b in next7) / total * 100,
        sum(b['total'] for b in rest)  / total * 100,
        [b['name'] for b in top3],
    )


qsr_s3, qsr_s7, qsr_sr, qsr_top3       = concentration_slices(qsr)
ag_s3,  ag_s7,  ag_sr,  agency_top3    = concentration_slices(agency)

# Colours: dark → mid → light
C_DARK  = "#1E3A5F"
C_MID   = "#3B82F6"
C_LIGHT = "#BFDBFE"
COLORS_DONUT = [C_DARK, C_MID, C_LIGHT]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
fig.suptitle("Market fragmentation: share of AI mentions by brand tier",
             fontsize=13, y=1.01)

for ax, slices, top3_names, title, n_brands in [
    (axes[0], [qsr_s3, qsr_s7, qsr_sr],  qsr_top3,    "QSR",    68),
    (axes[1], [ag_s3,  ag_s7,  ag_sr],   agency_top3, "Agency", 407),
]:
    wedges, _ = ax.pie(
        slices,
        colors=COLORS_DONUT,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
    )

    # Centre label: total brand count
    ax.text(0, 0.08, str(n_brands), ha="center", va="center",
            fontsize=26, fontweight="bold", color="#1E3A5F")
    ax.text(0, -0.22, "brands\nmentioned", ha="center", va="center",
            fontsize=10, color="#6B7280")

    # Percentage labels on each wedge
    cumulative = 0
    for i, (wedge, pct) in enumerate(zip(wedges, slices)):
        angle = (wedge.theta1 + wedge.theta2) / 2
        rad   = np.deg2rad(angle)
        r     = 0.72
        x     = r * np.cos(rad)
        y     = r * np.sin(rad)
        label_color = "white" if i < 2 else C_DARK
        ax.text(x, y, f"{pct:.0f}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color=label_color)

    ax.set_title(title, fontsize=14, pad=14, fontweight="semibold")

    # Brand name list below each donut
    lines = [
        f"● Top 3:  {', '.join(top3_names)}",
        f"● #4–10:  next 7 brands",
        f"● Rest:    remaining {n_brands - 10} brands",
    ]
    for j, (line, color) in enumerate(zip(lines, [C_DARK, C_MID, C_LIGHT])):
        ax.text(0, -1.28 - j * 0.22, line, ha="center", va="top",
                fontsize=8.5, color="#374151",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color + "33",
                          edgecolor="none"))

    ax.set_aspect("equal")

plt.tight_layout()
out1 = OUT_DIR / "chart_fragmentation_donuts.png"
fig.savefig(out1, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved → {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Persona specificity breakdown
# ─────────────────────────────────────────────────────────────────────────────

def persona_buckets(brands):
    n = len(brands)
    only1   = sum(1 for b in brands if b['n_personas'] == 1) / n * 100
    mid     = sum(1 for b in brands if 2 <= b['n_personas'] <= 3) / n * 100
    broad   = sum(1 for b in brands if b['n_personas'] >= 4) / n * 100
    return only1, mid, broad


qsr_b1, qsr_b2, qsr_b3       = persona_buckets(qsr)
agency_b1, agency_b2, agency_b3 = persona_buckets(agency)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)

labels  = ["Agency\n(407 brands)", "QSR\n(68 brands)"]
b1_vals = [agency_b1, qsr_b1]
b2_vals = [agency_b2, qsr_b2]
b3_vals = [agency_b3, qsr_b3]

COLORS = ["#1E3A5F", "#3B82F6", "#93C5FD"]   # dark→mid→light blue

bars1 = ax.barh(labels, b1_vals, color=COLORS[0], label="1 persona only (highly specific)")
bars2 = ax.barh(labels, b2_vals, left=b1_vals,    color=COLORS[1], label="2–3 personas (moderate)")
bars3 = ax.barh(labels, b3_vals,
                left=[a + b for a, b in zip(b1_vals, b2_vals)],
                color=COLORS[2], label="4–5 personas (near-ubiquitous)")

# Value labels inside each segment
def label_segments(bars, vals, lefts=None):
    for bar, val, left in zip(bars, vals, lefts or [0]*len(vals)):
        if val >= 5:
            cx = left + val / 2
            ax.text(cx, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")

label_segments(bars1, b1_vals)
label_segments(bars2, b2_vals, b1_vals)
label_segments(bars3, b3_vals, [a + b for a, b in zip(b1_vals, b2_vals)])

ax.set_xlim(0, 100)
ax.set_xlabel("Share of brands (%)", labelpad=8)
ax.set_title("Persona specificity: how many buyer types does each brand reach?\nShare of brands by number of personas they appear for", fontsize=11, pad=12)
ax.tick_params(axis='y', labelsize=11)
ax.tick_params(axis='x', labelsize=9)
ax.xaxis.set_visible(False)

legend = ax.legend(loc="lower right", framealpha=0.9, fontsize=9,
                   bbox_to_anchor=(1.0, -0.25), ncol=3)

plt.tight_layout()
out2 = OUT_DIR / "chart_persona_specificity.png"
fig.savefig(out2, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved → {out2}")
