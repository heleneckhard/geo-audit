"""
GEO Audit Revised Report — Chart Generation Script
Produces 5 publication-ready charts for the revised report.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = "/Users/heleneckhard/geo/revised_report/charts"
os.makedirs(OUT, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": "#222222",
    "axes.labelcolor": "#555555",
})

TOP3_COLOR   = "#1A1A2E"   # very dark navy — top 3
REST_COLOR   = "#A8B4C8"   # muted blue-grey — the rest
ACCENT_COLOR = "#E63946"   # red accent for callouts
LIGHT_GREY   = "#E8ECF0"

# ═════════════════════════════════════════════════════════════════════════════
# CHART 1 — Horizontal bar: Total QSR mentions by brand
# ═════════════════════════════════════════════════════════════════════════════
brands_mentions = [
    ("Wendy's",         237),
    ("Chick-fil-A",     189),
    ("McDonald's",      151),
    ("Chipotle",        129),
    ("Taco Bell",       121),
    ("Subway",          112),
    ("Panera Bread",    101),
    ("Starbucks",        47),
    ("Burger King",      42),
    ("Arby's",           33),
    ("Jack in the Box",  31),
    ("Culver's",         27),
    ("Domino's",         15),
    ("Dunkin'",          18),
    ("In-N-Out Burger",  18),
    ("Sweetgreen",       12),
    ("Wawa",             12),
]
# Sort descending
brands_mentions.sort(key=lambda x: x[1], reverse=True)
names   = [b[0] for b in brands_mentions]
counts  = [b[1] for b in brands_mentions]
colors  = [TOP3_COLOR if i < 3 else REST_COLOR for i in range(len(names))]

fig, ax = plt.subplots(figsize=(8, 6.5))
bars = ax.barh(names[::-1], counts[::-1], color=colors[::-1],
               height=0.65, zorder=2)

# Value labels
for bar, val in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", fontsize=9,
            color="#222222", fontweight="normal")

ax.set_xlim(0, 280)
ax.set_xlabel("Total AI mentions (375 queries per persona × 5 personas)", fontsize=9, color="#666666")
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=9)
ax.spines["bottom"].set_color("#CCCCCC")
ax.spines["bottom"].set_linewidth(0.8)

# Legend
top3_patch = mpatches.Patch(color=TOP3_COLOR, label="Top 3 brands")
rest_patch  = mpatches.Patch(color=REST_COLOR, label="All other brands")
ax.legend(handles=[top3_patch, rest_patch], loc="lower right",
          fontsize=9, frameon=False)

ax.set_title("Total AI mentions by QSR brand", fontsize=13,
             fontweight="bold", pad=14, loc="left")

fig.tight_layout()
fig.savefig(f"{OUT}/chart_01_qsr_mentions.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Chart 1 done")


# ═════════════════════════════════════════════════════════════════════════════
# CHART 2 — Donut: Mention concentration (top 3 vs rest)
# ═════════════════════════════════════════════════════════════════════════════
top3_total   = 237 + 189 + 151  # 577
rest_total   = sum(c for _, c in brands_mentions) - top3_total  # 718
total_all    = top3_total + rest_total

fig, ax = plt.subplots(figsize=(5.5, 5.5))
sizes   = [top3_total, rest_total]
labels  = [f"Top 3 brands\n({top3_total:,} mentions)", f"14 other brands\n({rest_total:,} mentions)"]
colors_d = [TOP3_COLOR, LIGHT_GREY]
explode  = (0.03, 0)

wedges, texts = ax.pie(
    sizes, labels=None, colors=colors_d, startangle=90,
    explode=explode, wedgeprops=dict(width=0.48, edgecolor="white", linewidth=2)
)

# Centre annotation
pct = top3_total / total_all * 100
ax.text(0, 0.06, f"{pct:.0f}%", ha="center", va="center",
        fontsize=28, fontweight="bold", color=TOP3_COLOR)
ax.text(0, -0.22, "of all mentions\ngo to 3 brands", ha="center", va="center",
        fontsize=10, color="#555555")

# External labels
ax.annotate(labels[0], xy=(-0.55, 0.55), fontsize=10,
            color=TOP3_COLOR, fontweight="bold", ha="center")
ax.annotate(labels[1], xy=(0.62, -0.55), fontsize=10,
            color="#777777", ha="center")

ax.set_title("Mention concentration — QSR study", fontsize=13,
             fontweight="bold", pad=8, loc="center")

fig.tight_layout()
fig.savefig(f"{OUT}/chart_02_concentration_donut.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Chart 2 done")


# ═════════════════════════════════════════════════════════════════════════════
# CHART 3 — 2×2 Quadrant scatter: The two levers (QSR)
# ═════════════════════════════════════════════════════════════════════════════
# Data: (brand, DA, mean_content_alignment 0-100, total_mentions)
brands_scatter = [
    ("Wendy's",         88, 56.61, 237),
    ("Chick-fil-A",     90, 57.63, 189),
    ("McDonald's",      90, 57.50, 151),
    ("Chipotle",        90, 48.48, 129),
    ("Taco Bell",       89, 58.17, 121),
    ("Subway",          90, 51.50, 112),
    ("Panera Bread",    89, 41.35, 101),
    ("Starbucks",       92, 32.58,  47),
    ("Burger King",     89, 39.42,  42),
    ("Arby's",          85, 42.99,  33),
    ("Jack in the Box", 84, 44.71,  31),
    ("Culver's",        83, 46.79,  27),
    ("Dunkin'",         89, 38.01,  18),
    ("In-N-Out Burger", 86, 46.84,  18),
    ("Domino's",        89, 30.00,  15),
    ("Sweetgreen",      87, 45.86,  12),
    ("Wawa",            83, 47.68,  12),
]

das       = np.array([b[1] for b in brands_scatter])
aligns    = np.array([b[2] for b in brands_scatter])
mentions  = np.array([b[3] for b in brands_scatter])
bnames    = [b[0] for b in brands_scatter]

# Quadrant lines at medians
med_da    = float(np.median(das))    # 89
med_align = float(np.median(aligns)) # ~46.79

# Bubble size scaled
bubble_sizes = (mentions / mentions.max()) * 2200 + 80

# Quadrant colours (subtle fill)
q_colors = []
for da, al in zip(das, aligns):
    if da >= med_da and al >= med_align:
        q_colors.append("#1A3A5C")     # top-right: Authority + Alignment
    elif da >= med_da and al < med_align:
        q_colors.append("#9B2335")     # top-left: Coasting on Brand
    elif da < med_da and al >= med_align:
        q_colors.append("#2E7D32")     # bottom-right: Punching Above Weight
    else:
        q_colors.append("#888888")     # bottom-left: Invisible

fig, ax = plt.subplots(figsize=(9, 7))

# Quadrant background fills
ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] < med_da else 80, med_da,
           ymin=0, ymax=1, color="#F7F9FC", zorder=0)
ax.axvspan(med_da, 94, ymin=0.5, ymax=1, color="#FFF5F5", zorder=0)
ax.axvspan(med_da, 94, ymin=0, ymax=0.5, color="#F5FFF5", zorder=0)
ax.axvspan(80, med_da, ymin=0, ymax=0.5, color="#F9F9F9", zorder=0)
ax.axvspan(80, med_da, ymin=0.5, ymax=1, color="#F9F9F9", zorder=0)

# Quadrant lines
ax.axvline(med_da,    color="#CCCCCC", linewidth=1.2, linestyle="--", zorder=1)
ax.axhline(med_align, color="#CCCCCC", linewidth=1.2, linestyle="--", zorder=1)

# Scatter
sc = ax.scatter(das, aligns, s=bubble_sizes, c=q_colors,
                alpha=0.82, edgecolors="white", linewidths=1.5, zorder=3)

# Brand labels — manual nudge for readability
offsets = {
    "Wendy's":         (-2.8, -2.2),
    "Chick-fil-A":     ( 0.2,  1.0),
    "McDonald's":      (-3.8, -2.0),
    "Chipotle":        ( 0.2,  0.9),
    "Taco Bell":       ( 0.2, -2.2),
    "Subway":          ( 0.2,  0.8),
    "Panera Bread":    ( 0.2,  0.8),
    "Starbucks":       ( 0.2,  0.8),
    "Burger King":     (-3.5, -2.0),
    "Arby's":          (-0.3, -2.0),
    "Jack in the Box": ( 0.2,  0.8),
    "Culver's":        (-3.0, -2.2),
    "Dunkin'":         ( 0.2,  0.8),
    "In-N-Out Burger": ( 0.2,  0.8),
    "Domino's":        ( 0.2,  0.8),
    "Sweetgreen":      ( 0.2, -2.2),
    "Wawa":            ( 0.2,  0.8),
}
for name, da, al in zip(bnames, das, aligns):
    dx, dy = offsets.get(name, (0.2, 0.8))
    ax.text(da + dx, al + dy, name, fontsize=7.8, ha="left",
            color="#222222", zorder=4)

# Quadrant labels
kw = dict(fontsize=9.5, fontweight="bold", alpha=0.7, ha="center", va="center")
ax.text(91.5, 55.5, "Authority\n+ Alignment", color="#1A3A5C", **kw)
ax.text(91.5, 35.5, "Coasting\non Brand",     color="#9B2335", **kw)
ax.text(84.5, 55.5, "Punching\nAbove Weight", color="#2E7D32", **kw)
ax.text(84.5, 35.5, "Invisible",               color="#888888", **kw)

ax.set_xlabel("Domain Authority →", fontsize=10, labelpad=8)
ax.set_ylabel("Content Alignment Score →", fontsize=10, labelpad=8)
ax.set_xlim(80.5, 93.5)
ax.set_ylim(26, 63)
ax.spines["bottom"].set_color("#CCCCCC")
ax.spines["left"].set_color("#CCCCCC")

# Bubble size legend
for size_val, label in [(237, "237 mentions"), (100, "100"), (20, "20")]:
    ax.scatter([], [], s=(size_val / mentions.max()) * 2200 + 80,
               color="#999999", alpha=0.5, label=label)
ax.legend(title="Bubble = total mentions", title_fontsize=8,
          fontsize=8, frameon=False, loc="lower left",
          labelspacing=0.8, handletextpad=0.4)

ax.set_title("The two levers: Domain Authority vs. Content Alignment",
             fontsize=13, fontweight="bold", pad=14, loc="left")
fig.tight_layout()
fig.savefig(f"{OUT}/chart_03_two_levers_quadrant.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Chart 3 done")


# ═════════════════════════════════════════════════════════════════════════════
# CHART 4 — Bar: Franchise development vacancy (Derek persona)
# ═════════════════════════════════════════════════════════════════════════════
fd_agencies = [
    ("Scorpion",    30),
    ("Location3",    1),
    ("SOCi",         1),
    ("All others\n(12 agencies)", 0),
]
fd_names  = [a[0] for a in fd_agencies]
fd_counts = [a[1] for a in fd_agencies]
fd_colors = [TOP3_COLOR, REST_COLOR, REST_COLOR, "#E8ECF0"]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.bar(fd_names, fd_counts, color=fd_colors,
              width=0.52, zorder=2, edgecolor="white", linewidth=0)

# Value labels
for bar, val in zip(bars, fd_counts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            str(val), ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#222222")

# "Zero" call-out on last bar
ax.text(bars[-1].get_x() + bars[-1].get_width() / 2, 1.8,
        "0", ha="center", va="bottom",
        fontsize=14, fontweight="bold", color="#AAAAAA")

ax.set_ylim(0, 36)
ax.set_ylabel("AI mentions (375 queries)", fontsize=9, color="#666666")
ax.tick_params(axis="x", labelsize=10.5, pad=5)
ax.tick_params(axis="y", labelsize=9)
ax.spines["bottom"].set_color("#CCCCCC")
ax.spines["left"].set_visible(False)
ax.yaxis.set_visible(False)

ax.set_title("Who gets recommended for franchise development?\n(375 queries, franchise dev director persona)",
             fontsize=12, fontweight="bold", pad=10, loc="left")

fig.tight_layout()
fig.savefig(f"{OUT}/chart_04_franchise_vacancy.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Chart 4 done")


# ═════════════════════════════════════════════════════════════════════════════
# CHART 5 — Heatmap: Content alignment by brand × persona (QSR)
# ═════════════════════════════════════════════════════════════════════════════
# Rows = brands, Cols = personas
# Values from affinity_matrix.csv (cosine similarity × 100)
persona_labels = ["Marcus\n(Busy Pro)", "Jenna\n(Family)", "Tyler\n(Budget)", "Priya\n(Fitness)", "Dale\n(Tradesperson)"]
hm_brands = [
    "Taco Bell", "Chick-fil-A", "McDonald's", "Wendy's",
    "Subway", "Chipotle", "Wawa", "In-N-Out Burger",
    "Culver's", "Sweetgreen", "Jack in the Box",
    "Panera Bread", "Arby's", "Dunkin'", "Burger King",
    "Starbucks", "Domino's",
]
# Matrix: [brand][persona] — from affinity_matrix.csv, multiplied by 100
# Order: P1(Marcus), P2(Jenna), P3(Tyler), P4(Priya), P5(Dale)
raw = {
    "Taco Bell":        [58.6, 54.6, 65.6, 56.2, 55.9],
    "Chick-fil-A":      [60.3, 61.0, 55.8, 57.1, 54.0],
    "McDonald's":       [58.1, 57.0, 61.3, 53.6, 57.5],
    "Wendy's":          [57.8, 55.7, 58.4, 50.5, 60.6],
    "Subway":           [54.1, 51.7, 50.5, 50.9, 50.3],
    "Chipotle":         [48.9, 49.3, 49.2, 51.7, 43.4],
    "Wawa":             [50.7, 46.2, 48.9, 47.2, 45.4],
    "In-N-Out Burger":  [46.6, 46.3, 47.9, 43.9, 49.6],
    "Culver's":         [46.7, 49.2, 47.5, 43.7, 46.8],
    "Sweetgreen":       [47.4, 47.6, 44.7, 50.0, 39.7],
    "Jack in the Box":  [45.7, 41.7, 51.3, 41.0, 43.9],
    "Panera Bread":     [45.4, 45.2, 38.4, 39.1, 38.6],
    "Arby's":           [44.7, 41.6, 42.8, 40.0, 45.9],
    "Dunkin'":          [40.6, 36.9, 41.7, 34.8, 36.1],
    "Burger King":      [40.7, 37.4, 41.3, 36.1, 41.7],
    "Starbucks":        [34.4, 31.6, 33.5, 33.9, 29.5],
    "Domino's":         [30.8, 30.8, 32.1, 27.9, 28.3],
}

matrix = np.array([raw[b] for b in hm_brands])

import matplotlib.colors as mcolors

fig, ax = plt.subplots(figsize=(7.5, 7.5))

# Custom colormap: cool (low) → warm (high)
cmap = plt.cm.RdYlGn
im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=27, vmax=66)

ax.set_xticks(range(len(persona_labels)))
ax.set_xticklabels(persona_labels, fontsize=9.5)
ax.set_yticks(range(len(hm_brands)))
ax.set_yticklabels(hm_brands, fontsize=9.5)
ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
ax.tick_params(left=True, right=False)

# Cell annotations
for i in range(len(hm_brands)):
    for j in range(len(persona_labels)):
        val = matrix[i, j]
        text_color = "white" if val > 57 or val < 34 else "#222222"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                fontsize=8.5, color=text_color, fontweight="normal")

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Content alignment score (0–100)", fontsize=8.5, color="#555555")
cbar.ax.tick_params(labelsize=8)

ax.set_title("Content alignment by brand and customer persona",
             fontsize=12, fontweight="bold", pad=12, loc="left")
ax.set_xlabel("Customer persona →", fontsize=9.5, labelpad=8)

fig.tight_layout()
fig.savefig(f"{OUT}/chart_05_heatmap.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Chart 5 done")

print("\nAll charts generated in:", OUT)
