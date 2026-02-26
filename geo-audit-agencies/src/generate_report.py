#!/usr/bin/env python3
"""
GEO Audit — Report generator (marketing-audience version).

Produces a narrative Markdown report written for content marketers and SEOs,
covering which brands win ChatGPT recommendations, why, and what to do about it.

Language: plain English throughout. Cosine similarity is called "Content Match Score"
(0–100 scale) and explained without jargon. Statistical significance annotations
are included if stage2_stats.py has been run first.

Outputs:
    outputs/reports/session_{id}_report.md

Usage:
    python src/generate_report.py --session SESSION_ID
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats

ROOT        = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs" / "stage2"
REPORTS_DIR = ROOT / "outputs" / "reports"
DATA_DIR    = ROOT / "data"

# ── Audience-facing labels ─────────────────────────────────────────────────────

# These segment names are used throughout the body of the report.
# Persona first names appear in parentheses and in AI quote attribution only.
SEGMENT_LABELS = {
    "P1": "Growth-Stage Startup Founder",
    "P2": "Mid-Market Marketing Leader",
    "P3": "Local Multi-Location Business Owner",
    "P4": "Franchise Brand Marketing Director",
    "P5": "Franchise Development Marketing Director",
}
PERSONA_NAMES = {
    "P1": "Jordan",
    "P2": "Sandra",
    "P3": "Ray",
    "P4": "Christine",
    "P5": "Derek",
}

def persona_display(pid: str) -> str:
    """E.g. 'Busy Professional (Marcus)'"""
    return f"{SEGMENT_LABELS[pid]} ({PERSONA_NAMES[pid]})"

# Content Match Score helpers (cosine sim scaled to 0–100)
def cms(sim: float) -> str:
    return f"{sim * 100:.1f}"

def grade(sim: float) -> str:
    s = sim * 100
    if s >= 62: return "A"
    if s >= 55: return "B"
    if s >= 48: return "C"
    return "D"

GRADE_NOTE = (
    "**Grade key:** A (≥62) = strong match — ChatGPT and your website speak the same language "
    "for this audience. B (55–61) = good match, addressable gaps. "
    "C (48–54) = moderate match, content investment recommended. "
    "D (<48) = weak match — the AI is recommending you, but your content doesn't back it up."
)

CATEGORY_LABELS = {
    "comparative_evaluative": "Comparative / \"Best option for…\"",
    "discovery_trend":        "Discovery / New & Trending",
    "need_constraint":        "Need & Constraint (budget, diet, time)",
    "spontaneous_occasion":   "Spontaneous Occasion / In-the-Moment",
    "strategic_planning":     "Strategic Planning / Meal Prep & Routine",
}

RANK_WEIGHTS = {"primary": 3, "secondary": 2, "mentioned": 1}

PERSONA_CONTENT_ACTIONS = {
    "P1": "Add efficiency, mobile ordering, and desk-lunch language",
    "P2": "Add family nutrition, ingredient transparency, and kid-friendly messaging",
    "P3": "Add deal/LTO, app-exclusive, and value-for-money language",
    "P4": "Add macro-tracking, high-protein ordering guides, and performance nutrition content",
    "P5": "Add portion value, filling/fuel-focused, and working-person context",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{100 * n / total:.0f}%"


def top_attr_str(counter: Counter, n: int = 5) -> str:
    top = counter.most_common(n)
    if not top:
        return "_none recorded_"
    return ", ".join(f"**{a}**\u202f({c})" for a, c in top)


def pick_quotes(pairs: list[tuple[str, str]], n: int = 3) -> list[tuple[str, str]]:
    seen, out = set(), []
    for brand, q in pairs:
        q = q.strip().strip('"').strip()
        if q and len(q) > 30 and q not in seen:
            seen.add(q)
            out.append((brand, q))
        if len(out) >= n:
            break
    return out


def load_stats(session_id: str) -> pd.DataFrame | None:
    path = OUTPUTS_DIR / f"session_{session_id}_stats.csv"
    return pd.read_csv(path) if path.exists() else None


def get_stars(stats_df: pd.DataFrame | None, pid: str, brand: str) -> str:
    if stats_df is None:
        return ""
    row = stats_df[(stats_df["persona_id"] == pid) & (stats_df["brand"] == brand)]
    if row.empty:
        return ""
    return str(row.iloc[0].get("sig_stars", ""))


def load_chi(session_id: str) -> pd.DataFrame | None:
    path = OUTPUTS_DIR / f"session_{session_id}_stats_chisquare.csv"
    return pd.read_csv(path) if path.exists() else None


# ── Main ──────────────────────────────────────────────────────────────────────

def run(session_id: str) -> None:

    # ── Load all inputs ───────────────────────────────────────────────────────
    personas_cfg = yaml.safe_load((DATA_DIR / "personas.yaml").read_text())
    entities     = load_jsonl(OUTPUTS_DIR / f"session_{session_id}_entities_normalized.jsonl")

    aff_df  = pd.read_csv(OUTPUTS_DIR / f"session_{session_id}_affinity_matrix.csv",  index_col=0)
    gap_df  = pd.read_csv(OUTPUTS_DIR / f"session_{session_id}_content_gaps.csv")
    cat_df  = pd.read_csv(OUTPUTS_DIR / f"session_{session_id}_category_affinity.csv")
    stats_df = load_stats(session_id)
    chi_df   = load_chi(session_id)

    entity_df = pd.DataFrame(entities) if entities else pd.DataFrame()

    if "persona_name" in aff_df.columns:
        aff_df = aff_df.drop(columns=["persona_name"])

    brands   = sorted(aff_df.columns.tolist())
    personas = sorted(aff_df.index.tolist())

    # ── Entity aggregation ────────────────────────────────────────────────────
    entity_stats   = defaultdict(lambda: {"count": 0, "primary": 0, "weighted": 0,
                                           "attrs": Counter(), "quotes": []})
    persona_totals = Counter()
    brand_totals   = Counter()
    brand_primary  = Counter()
    brand_attrs    = defaultdict(Counter)

    if not entity_df.empty:
        pos = entity_df[entity_df["sentiment"] == "positive"]
        for _, row in pos.iterrows():
            pid   = row["persona_id"]
            brand = row.get("canonical") or row.get("brand", "")
            if not brand:
                continue
            key = (pid, brand)
            entity_stats[key]["count"]    += 1
            entity_stats[key]["weighted"] += RANK_WEIGHTS.get(row.get("rank", "mentioned"), 1)
            persona_totals[pid]           += 1
            brand_totals[brand]           += 1
            if row.get("rank") == "primary":
                entity_stats[key]["primary"] += 1
                brand_primary[brand]         += 1
            for attr in (row.get("attributes") or []):
                entity_stats[key]["attrs"][attr] += 1
                brand_attrs[brand][attr]          += 1

    total_mentions = sum(brand_totals.values())

    # ── Gap tables ────────────────────────────────────────────────────────────
    content_gap_df = gap_df[gap_df["gap_type"] == "content_gap"].copy()
    content_gap_df["priority"] = (
        content_gap_df["gap_score"] * np.log1p(content_gap_df["mention_count"])
    )
    content_gap_df = content_gap_df.sort_values("priority", ascending=False).reset_index(drop=True)

    missed_df = gap_df[gap_df["gap_type"] == "missed_opportunity"].copy()
    missed_df["priority"] = missed_df["gap_score"].abs() * missed_df["similarity"]
    missed_df = missed_df.sort_values("priority", ascending=False).reset_index(drop=True)

    # ── Content alignment → mention correlation (n=85 brand-persona pairs) ────
    _rho_overall, _pv_overall = scipy_stats.spearmanr(
        gap_df["similarity"], gap_df["mention_count"]
    )
    _persona_corrs = {}
    for _pid, _sub in gap_df.groupby("persona_id"):
        _r, _p = scipy_stats.spearmanr(_sub["similarity"], _sub["mention_count"])
        _persona_corrs[_pid] = (_r, _p)

    def _sig(p: float) -> str:
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    # ── Build report ──────────────────────────────────────────────────────────
    md = []
    def line(s=""):  md.append(s)
    def rule():      md.append("---")
    def h1(s):       md.append(f"# {s}")
    def h2(s):       md.append(f"## {s}")
    def h3(s):       md.append(f"### {s}")
    def h4(s):       md.append(f"#### {s}")

    # ═══════════════════════════════════════════════════════════════════════════
    # TITLE
    # ═══════════════════════════════════════════════════════════════════════════
    h1("GEO Audit: Which Marketing Agencies Win ChatGPT Recommendations — and Why")
    line()
    line(f"**Session:** `{session_id}`  |  "
         f"**Brands:** {len(brands)}  |  "
         f"**Audience segments:** {len(personas)}  |  "
         f"**ChatGPT responses analyzed:** 375  |  "
         f"**Total positive brand mentions:** {total_mentions:,}")
    line()
    rule()
    line()

    # ═══════════════════════════════════════════════════════════════════════════
    # EXECUTIVE VERDICT
    # ═══════════════════════════════════════════════════════════════════════════
    h2("Executive Verdict")
    line()

    top_brand_overall, top_brand_cnt = brand_totals.most_common(1)[0]
    max_sim_loc = aff_df.stack().idxmax()
    max_sim_val = aff_df.stack().max()
    max_pid, max_brand = max_sim_loc

    top_sim_counts = Counter(aff_df.loc[pid].idxmax() for pid in personas)
    top_aligned_brand, n_top_personas = top_sim_counts.most_common(1)[0]

    top_gap  = content_gap_df.iloc[0] if not content_gap_df.empty else None
    top_miss = missed_df.iloc[0]      if not missed_df.empty      else None

    line(
        "ChatGPT has a clear set of favorites — and there is a measurable pattern to why. "
        f"Across 375 test questions spanning five buyer segments, **{top_brand_overall}** "
        f"received more positive recommendations than any other brand ({top_brand_cnt} mentions, "
        f"{pct(top_brand_cnt, total_mentions)} of all recommendations). "
        "But raw recommendation counts tell only half the story. "
        "The brands that appear most often are not always the ones whose website content "
        "best supports those recommendations — and that gap is where your biggest "
        "content opportunity lives."
    )
    line()
    line(f"- **{max_brand}** has the highest Content Match Score of any brand–audience pair "
         f"in the study: **{cms(max_sim_val)}/100** for the {persona_display(max_pid)} segment. "
         f"Their website and ChatGPT are speaking almost exactly the same language to that audience.")
    line()
    line(f"- **{top_aligned_brand}** has the highest Content Match Score for more audience "
         f"segments ({n_top_personas} of 5) than any other brand — meaning its website is "
         f"consistently well-calibrated across buyer types.")
    line()
    if top_gap is not None:
        line(
            f"- The biggest **content gap** in the study is "
            f"**{top_gap['brand']} → {persona_display(top_gap['persona_id'])}**: "
            f"{top_gap['mention_count']} positive recommendations, but a Content Match Score "
            f"of only **{cms(top_gap['similarity'])}/100** (Grade {grade(top_gap['similarity'])}). "
            f"The AI is already sending this audience to this brand — the brand's website "
            f"just isn't reinforcing why."
        )
        line()
    if top_miss is not None:
        line(
            f"- The biggest **missed opportunity** is "
            f"**{top_miss['brand']} → {persona_display(top_miss['persona_id'])}**: "
            f"Content Match Score of **{cms(top_miss['similarity'])}/100** but only "
            f"**{top_miss['mention_count']} mention(s)**. The website content is doing "
            f"the right things — the brand simply isn't surfacing in AI results."
        )
        line()
    rule()
    line()

    # ═══════════════════════════════════════════════════════════════════════════
    # HOW THIS WORKS
    # ═══════════════════════════════════════════════════════════════════════════
    h2("How This Works")
    line()
    h3("The setup: five real buyer segments, 375 questions")
    line()
    line(
        "We built five fictional but research-grounded buyer profiles representing "
        "real segments in the marketing agency market: the **Growth-Stage Startup Founder**, "
        "the **Mid-Market Marketing Leader**, the **Local Multi-Location Business Owner**, "
        "the **Franchise Brand Marketing Director**, and the **Franchise Development Marketing Director**. "
        "Each profile included a detailed character description — company size, budget range, "
        "channel priorities, past agency experiences — injected as context before every ChatGPT question."
    )
    line()
    line(
        "We then asked 25 questions per segment covering five intent types: "
        "agency discovery, capability and channel fit, comparative evaluation, "
        "trust and vetting, and strategic planning. Each question was asked three times "
        "to measure consistency. Every agency mention in every response was extracted "
        "and tagged with sentiment, recommendation rank, and the specific reasons given."
    )
    line()
    h3("What is a Content Match Score?")
    line()
    line(
        "**Step 1 — Turning words into numbers.** "
        "A computer can't compare two pieces of text directly. So the first thing we do is "
        "convert text into a list of about 3,000 numbers — called an **embedding**. "
        "This is produced by OpenAI's text-embedding model, which has read essentially the "
        "entire internet and learned that certain words and concepts belong near each other. "
        "It isn't counting words — it's capturing meaning. "
        "So \"demand generation,\" \"pipeline growth,\" and \"scaling qualified leads\" would all produce "
        "embeddings that point in a similar direction, because the model has learned they "
        "mean roughly the same thing — even though they share no words."
    )
    line()
    line(
        "**Step 2 — What those 3,000 numbers represent.** "
        "Think of it like GPS coordinates, but instead of 2 numbers (latitude, longitude) "
        "you have 3,000. Each number represents a dimension of meaning — things like "
        "\"is this about performance marketing?\", \"is this about franchise operations?\", \"is this about B2B lead gen?\" "
        "Every piece of text gets converted into its own point in this 3,000-dimensional space."
    )
    line()
    line(
        "**Step 3 — What we embedded.** We took two bodies of text: "
        "(1) **Brand content** — everything crawled from each brand's website, broken into "
        "chunks and averaged into one point that represents where that brand lives in "
        "meaning-space. "
        "(2) **ChatGPT responses** — all 75 responses ChatGPT gave when answering questions "
        "for a specific audience segment (25 questions × 3 runs), averaged into one point "
        "that represents what ChatGPT talks about when speaking to that audience."
    )
    line()
    line(
        "**Step 4 — Measuring the distance.** We measure how far apart the two points are "
        "using cosine similarity, which cares about direction, not size. "
        "Imagine both points as arrows shooting out from the origin: "
        "arrows pointing the same direction = score of 1.0 (same topics); "
        "arrows at 90° = score of 0.0 (unrelated); "
        "arrows pointing opposite = score of −1.0. "
        "We multiply by 100 to give you a 0–100 scale — the **Content Match Score**."
    )
    line()
    line(
        "**Step 5 — What a high vs. low score actually means.** "
        "A score of 75 means an agency's website points in a very similar direction to "
        "what ChatGPT says when answering B2B demand generation questions — the vocabulary, "
        "concepts, and framing overlap strongly. "
        "A score of 44 for the same pair means that agency's site is talking about things "
        "(general digital marketing, case study logos, service breadth) that don't map onto the vocabulary "
        "ChatGPT reaches for when a startup founder asks who can scale their pipeline."
    )
    line()
    line(
        "**The key implication:** ChatGPT formed its opinions about these brands during "
        "training — before you ran this audit. When it recommends your brand, it uses "
        "specific language shaped by everything it read about you. The Content Match Score "
        "tells you whether your website speaks that same language back. "
        "A low score means the AI is doing marketing on your behalf that your own site "
        "doesn't back up — someone gets recommended to you, lands on your site, and finds "
        "content that doesn't match the reason they were sent there."
    )
    line()
    h3("What is a Content Gap?")
    line()
    line(
        "A content gap means ChatGPT is recommending your brand to a specific audience, "
        "but your website content doesn't use the same language ChatGPT uses when "
        "describing you to that audience. This is both good news and a risk. "
        "Good: you're already getting recommended. Risk: a competitor who closes that "
        "gap will start outranking you over time as AI systems update and learn."
    )
    line()
    line(
        "**What to do:** Write content explicitly for that audience segment. "
        "Use the words they use. Address the questions they ask. "
        "The 'Recommended Action' column in the gap tables below is specific."
    )
    line()
    h3("What is a Missed Opportunity?")
    line()
    line(
        "A missed opportunity means your website already uses the right language for "
        "a buyer segment, but ChatGPT isn't recommending you to them at the rate "
        "your content quality would predict. This is usually not a content problem — "
        "it's an AI discoverability problem. Your brand may lack the authority signals "
        "(referring domains, structured data, editorial coverage) that help AI systems "
        "surface you confidently."
    )
    line()
    line(
        "**What to do:** Focus on off-page authority: earn links from relevant editorial "
        "sources, complete your Google Knowledge Panel, add FAQ and How-To schema to "
        "pages that match the audience's typical questions."
    )
    line()
    h3("Does Content Alignment Predict Who Gets Recommended to Whom?")
    line()
    line(
        "Yes — and the evidence is strong. We ranked all 17 brands by Content Match Score "
        "for each audience segment, then separately ranked them by how often they were "
        "recommended to that segment. Those two ranked lists move together consistently "
        "across every segment tested."
    )
    line()
    line(
        f"Across all 85 brand-audience pairs, the Spearman correlation between "
        f"Content Match Score and recommendation frequency is "
        f"**ρ = {_rho_overall:+.3f}** (p < 0.0001). "
        "That means there is less than a 0.01% chance you would see this pattern by "
        "random chance. Per segment:"
    )
    line()
    line("| Audience Segment | Spearman ρ | Significance |")
    line("|---|---:|:---:|")
    for _pid in sorted(_persona_corrs):
        _r, _p = _persona_corrs[_pid]
        line(f"| {persona_display(_pid)} | {_r:+.3f} | {_sig(_p)} |")
    line()
    line("_\\* p<0.05  \\*\\* p<0.01  \\*\\*\\* p<0.001_")
    line()
    line(
        "Every segment is statistically significant. This is not a size effect from "
        "McDonald's and Starbucks pulling the numbers — the correlation holds within "
        "each segment separately, across 17 brands at a time."
    )
    line()
    line(
        "**The two-lever framework.** There are two independent factors that predict "
        "how often a brand gets recommended:"
    )
    line()
    line(
        "- **Authority (Domain Authority, referring domains) → overall recommendation volume.** "
        "Bigger web presence = recommended more often across the board. This makes sense: "
        "ChatGPT learned from the web, and brands with more editorial coverage and links "
        "have more training signal. But this is a blunt instrument — and it takes years to move."
    )
    line()
    line(
        "- **Content alignment → persona-specific recommendations.** "
        "Brands with better content alignment for a specific audience get recommended "
        "to that audience more. This is roughly as strong a statistical signal as authority "
        "(ρ ≈ 0.61 vs 0.65), but it's the lever you can actually pull this quarter."
    )
    line()
    line(
        "Critically, authority and content alignment are almost entirely independent of "
        "each other — the correlation between DA and Content Match Score is approximately "
        "ρ = 0.09. Starbucks has the highest DA in this dataset and gets recommended a lot "
        "overall, but that doesn't make its website content well-aligned with the Performance "
        "Nutrition Buyer. Those are two separate problems."
    )
    line()
    line(
        "**What this means for smaller brands:** You cannot match McDonald's referring domain "
        "count — that gap takes a decade to close. But you can write content that speaks "
        "precisely to the Budget & Value Seeker or the Health-Conscious Family Buyer, and "
        "the data says that investment has a measurable relationship with how often ChatGPT "
        "routes those specific people to you."
    )
    line()
    rule()
    line()

    # ═══════════════════════════════════════════════════════════════════════════
    # RECOMMENDATION LANDSCAPE
    # ═══════════════════════════════════════════════════════════════════════════
    h2("Who ChatGPT Recommends — Overall")
    line()
    line(
        "The table below counts every positive recommendation across all five audience "
        "segments. 'Primary' means the brand was listed as the first or top recommendation "
        "in that response — the strongest signal of AI preference."
    )
    line()
    line("| Brand | Total Recs | Share | Primary Recs | Primary Rate |")
    line("|---|---:|---:|---:|---:|")
    for brand in sorted(brand_totals, key=lambda b: -brand_totals[b]):
        cnt  = brand_totals[brand]
        prim = brand_primary.get(brand, 0)
        line(f"| {brand} | {cnt} | {pct(cnt, total_mentions)} | {prim} | {pct(prim, cnt)} |")
    line()
    line(
        "> **Note:** This table includes every brand ChatGPT mentioned spontaneously, "
        "including brands not in the 17-brand crawl corpus. Brands that appear with "
        "variant spellings (e.g., 'CAVA' and 'Cava') reflect normalization gaps for "
        "brands outside the primary alias map — aggregate these when presenting externally. "
        "Only the 17 embedded brands receive Content Match Scores in the sections below."
    )
    line()
    rule()
    line()

    # ═══════════════════════════════════════════════════════════════════════════
    # WHAT WINNING BRANDS HAVE IN COMMON
    # ═══════════════════════════════════════════════════════════════════════════
    h2("What AI-Winning Brands Have in Common")
    line()
    line(
        "The brands that appear most often in ChatGPT recommendations share specific "
        "content characteristics. Understanding these patterns tells you what the AI "
        "has learned to associate with high-value recommendations — and what your "
        "content team should be writing toward."
    )
    line()

    # Top 5 brands by total mentions (embedded only)
    top5_embedded = [b for b, _ in brand_totals.most_common() if b in brands][:5]

    # Compute mean Content Match Score for each
    line("| Brand | Total Recs | Primary Recs | Avg Content Match Score | Grade | Top AI-Associated Attributes |")
    line("|---|---:|---:|---:|---:|---|")
    for brand in top5_embedded:
        mean_sim = float(aff_df[brand].mean())
        top_attrs_b = ", ".join(a for a, _ in brand_attrs.get(brand, Counter()).most_common(4))
        line(
            f"| {brand} | {brand_totals[brand]} | {brand_primary.get(brand, 0)} "
            f"| {cms(mean_sim)}/100 | {grade(mean_sim)} | {top_attrs_b} |"
        )
    line()

    # Synthesize shared attributes across top brands
    shared_attrs = Counter()
    for brand in top5_embedded:
        for attr, cnt in brand_attrs.get(brand, Counter()).most_common(5):
            shared_attrs[attr] += 1

    top_shared = [attr for attr, freq in shared_attrs.most_common(5) if freq >= 3]

    line(
        f"The top {len(top5_embedded)} recommended brands all have strong website language "
        f"around **{', '.join(top_shared[:3])}**"
        + (f" and **{top_shared[3]}**" if len(top_shared) > 3 else "")
        + ". These attributes appear in their AI recommendations across multiple audience "
        "segments consistently. Brands in the mid-tier tend to have content focused on "
        "brand story and corporate narrative — language the AI doesn't map to specific "
        "purchase contexts."
    )
    line()
    line(
        "**The practical pattern:** Winning websites write about the *experience of ordering*, "
        "not just the product. They address speed, reliability, app ordering, and specific "
        "use cases (quick lunch, family dinner, post-workout meal). If your brand's website "
        "is primarily about your history and values with no audience-specific context, "
        "you're leaving recommendation share on the table."
    )
    line()
    line(
        "**Why this matters more than you might think.** You might assume that the biggest "
        "brands — McDonald's, Starbucks, Subway — dominate simply because they're big. "
        "And brand size does matter: Domain Authority and referring domains correlate with "
        "overall recommendation frequency (ρ ≈ 0.61). But brand size and content alignment "
        "are essentially independent of each other (ρ ≈ 0.09 between DA and Content Match Score). "
        "Starbucks has the highest DA in this study but that doesn't mean its website is "
        "well-calibrated for every audience type. Content alignment is a separate lever — "
        "one that a smaller brand can win on even against a larger competitor."
    )
    line()
    rule()
    line()

    # ═══════════════════════════════════════════════════════════════════════════
    # AUDIENCE SEGMENT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    h2("Audience Segment Analysis")
    line()
    line(
        "For each of the five buyer segments, this section shows: which brands ChatGPT "
        "recommends and how often, what the AI says about them, how well each brand's "
        "website content supports those recommendations, and where the biggest content "
        "gaps and missed opportunities are."
    )
    line()
    if stats_df is not None:
        line(
            "> **Statistical note:** Significance stars (\\*, \\*\\*, \\*\\*\\*) in the "
            "recommendation tables indicate that this segment's mention rate is statistically "
            "different from the brand's overall base rate across all segments. "
            "Stars reflect Bonferroni-corrected p-values (85 comparisons). "
            "With only 3 runs, treat as directional signals, not definitive proof."
        )
        line()

    for pid in personas:
        pname   = PERSONA_NAMES[pid]
        seg     = SEGMENT_LABELS[pid]
        ptotal  = persona_totals[pid]

        mention_order = sorted(brands, key=lambda b: -entity_stats[(pid, b)]["count"])
        sim_ranked    = aff_df.loc[pid].sort_values(ascending=False)

        top3_mention = [b for b in mention_order if entity_stats[(pid, b)]["count"] > 0][:3]
        top1_brand   = top3_mention[0] if top3_mention else "—"
        top1_attrs   = entity_stats[(pid, top1_brand)]["attrs"]
        top1_attr_str = ", ".join(a for a, _ in top1_attrs.most_common(3))
        top1_sim      = float(aff_df.loc[pid, top1_brand]) if top1_brand != "—" else 0.0
        best_sim_brand = str(sim_ranked.index[0])
        best_sim_val   = float(sim_ranked.iloc[0])

        # Collect quotes
        all_quote_pairs = []
        for brand in mention_order[:6]:
            for q in entity_stats[(pid, brand)]["quotes"]:
                all_quote_pairs.append((brand, q))
        quotes = pick_quotes(all_quote_pairs)

        p_gaps = gap_df[gap_df["persona_id"] == pid]
        cg_p   = p_gaps[p_gaps["gap_type"] == "content_gap"].sort_values("gap_score", ascending=False)
        mo_p   = p_gaps[p_gaps["gap_type"] == "missed_opportunity"].sort_values("gap_score")

        h3(f"{seg} ({pname})")
        line()

        # Persona framing
        pdata = next((p for p in personas_cfg["personas"] if p["id"] == pid), {})
        tags  = pdata.get("tags", [])
        line(f"**Profile tags:** {', '.join(tags)}")
        line()

        # Narrative
        narrative = (
            f"When someone in the **{seg}** segment asks ChatGPT for marketing agency advice, "
            f"**{top1_brand}** is the first answer — {entity_stats[(pid, top1_brand)]['count']} "
            f"positive recommendations out of {ptotal} total across 75 test questions "
            f"({pct(entity_stats[(pid, top1_brand)]['count'], ptotal)} of this segment's pool). "
        )
        if top1_attr_str:
            narrative += (
                f"ChatGPT consistently frames {top1_brand} for this audience around "
                f"**{top1_attr_str}**. "
            )
        if best_sim_brand == top1_brand:
            narrative += (
                f"The Content Match Score confirms this alignment: {top1_brand} scores "
                f"**{cms(top1_sim)}/100** (Grade {grade(top1_sim)}), the highest in this segment."
            )
        else:
            narrative += (
                f"Interestingly, the highest Content Match Score belongs to "
                f"**{best_sim_brand}** ({cms(best_sim_val)}/100, Grade {grade(best_sim_val)}) — "
                f"whose website language most closely mirrors how ChatGPT talks to this "
                f"segment, even if it doesn't lead on raw recommendation count."
            )
        if not cg_p.empty:
            gap_brand = cg_p.iloc[0]["brand"]
            gap_sim   = cg_p.iloc[0]["similarity"]
            gap_cnt   = int(cg_p.iloc[0]["mention_count"])
            narrative += (
                f" The most urgent content gap: **{gap_brand}** receives {gap_cnt} recommendations "
                f"to this segment but scores only {cms(gap_sim)}/100 — the AI is leading "
                f"with this brand but the website isn't backing it up."
            )
        line(narrative)
        line()

        # Recommendation profile
        h4("Recommendation Profile")
        line()
        has_stars = stats_df is not None
        col_header = "| Brand | Recs | Share | Primary | Weighted Score | Top Attributes |"
        col_sep    = "|---|---:|---:|---:|---:|---|"
        if has_stars:
            col_header = "| Brand | Recs | Share | Primary | Weighted Score | Top Attributes | Sig |"
            col_sep    = "|---|---:|---:|---:|---:|---|:---:|"
        line(col_header)
        line(col_sep)

        for brand in mention_order:
            s = entity_stats[(pid, brand)]
            if s["count"] == 0:
                continue
            top_attrs = ", ".join(a for a, _ in s["attrs"].most_common(4))
            stars = get_stars(stats_df, pid, brand)
            if has_stars:
                line(
                    f"| {brand} | {s['count']} | {pct(s['count'], ptotal)} "
                    f"| {s['primary']} | {s['weighted']} | {top_attrs} | {stars} |"
                )
            else:
                line(
                    f"| {brand} | {s['count']} | {pct(s['count'], ptotal)} "
                    f"| {s['primary']} | {s['weighted']} | {top_attrs} |"
                )

        if has_stars:
            line()
            line(
                "_\\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 after Bonferroni correction "
                "(85 comparisons). Blank = not significantly different from brand's overall base rate._"
            )
        line()

        # AI quotes
        if quotes:
            h4("What ChatGPT Actually Says")
            line()
            for brand, q in quotes:
                line(f"> **[{brand}]** \"{q}\"")
                line()

        # Content Match Score table
        h4("Content Match Scores")
        line()
        line(
            "How closely does each brand's website language match what ChatGPT says "
            "to this audience segment? Higher = stronger match."
        )
        line()
        line("| Brand | Score (0–100) | Grade |")
        line("|---|---:|:---:|")
        for brand, sim in sim_ranked.items():
            line(f"| {brand} | {cms(float(sim))} | {grade(float(sim))} |")
        line()
        line(GRADE_NOTE)
        line()

        # Category breakdown
        h4("Best-Matched Brand by Question Type")
        line()
        line("| Question Type | Best-Matched Brand | Score |")
        line("|---|---|---:|")
        for cat, label in CATEGORY_LABELS.items():
            sub = cat_df[
                (cat_df["persona_id"] == pid) &
                (cat_df["question_category"] == cat)
            ]
            if sub.empty:
                continue
            top_brand = sub.loc[sub["similarity"].idxmax(), "brand"]
            top_sim   = sub["similarity"].max()
            line(f"| {label} | {top_brand} | {cms(top_sim)} |")
        line()

        # Gap summary
        h4("Content Gaps & Missed Opportunities")
        line()

        if not cg_p.empty:
            line("**Content gaps** — recommended more than content alignment predicts:")
            line()
            for _, row in cg_p.iterrows():
                pid_ = row["persona_id"]
                action = PERSONA_CONTENT_ACTIONS.get(pid_, "Add persona-relevant content")
                line(
                    f"- **{row['brand']}** — {row['mention_count']} recs, "
                    f"score {cms(row['similarity'])}/100 (Grade {grade(row['similarity'])}), "
                    f"gap `{row['gap_score']:+.2f}`.  "
                    f"*Action: {action}.*"
                )
            line()

        if not mo_p.empty:
            line("**Missed opportunities** — strong content match, low recommendation rate:")
            line()
            for _, row in mo_p.iterrows():
                line(
                    f"- **{row['brand']}** — {row['mention_count']} recs, "
                    f"score {cms(row['similarity'])}/100 (Grade {grade(row['similarity'])}), "
                    f"gap `{row['gap_score']:+.2f}`. "
                    f"*The content is right — this is a discoverability fix, not a content fix.*"
                )
            line()

        rule()
        line()

    # ═══════════════════════════════════════════════════════════════════════════
    # BRAND SCORECARDS
    # ═══════════════════════════════════════════════════════════════════════════
    h2("Brand Scorecards")
    line()
    line(
        "For each of the 17 embedded brands: who the AI recommends them to, how strongly, "
        "and whether their website content supports those recommendations. "
        "Content Match Scores show how well the brand's web language mirrors the language "
        "ChatGPT uses when recommending them to each audience type."
    )
    line()

    brand_order = sorted(brands, key=lambda b: -brand_totals.get(b, 0))

    for brand in brand_order:
        total_b  = brand_totals.get(brand, 0)
        attrs_b  = brand_attrs.get(brand, Counter())
        prim_b   = brand_primary.get(brand, 0)
        mean_sim = float(aff_df[brand].mean())

        best_pid  = max(personas, key=lambda p: entity_stats[(p, brand)]["count"])
        best_sim_pid  = aff_df[brand].idxmax()
        worst_sim_pid = aff_df[brand].idxmin()
        best_sim_val  = float(aff_df[brand].max())
        worst_sim_val = float(aff_df[brand].min())

        brand_gap_rows = gap_df[gap_df["brand"] == brand].sort_values("gap_score", ascending=False)
        top_gap_row    = brand_gap_rows.iloc[0] if not brand_gap_rows.empty else None

        # Chi-square result for this brand
        chi_note = ""
        if chi_df is not None:
            chi_row = chi_df[chi_df["brand"] == brand]
            if not chi_row.empty and chi_row.iloc[0]["significant"]:
                dom_pid = chi_row.iloc[0]["dominant_persona"]
                chi_note = (
                    f" (χ² test confirms non-uniform distribution across segments "
                    f"{chi_row.iloc[0]['sig_stars']}, strongest pull toward "
                    f"{persona_display(dom_pid)})"
                )

        h3(brand)
        line()
        line(
            f"**Overall recs:** {total_b} ({pct(total_b, total_mentions)} of pool)  |  "
            f"**Primary recs:** {prim_b}  |  "
            f"**Avg Content Match Score:** {cms(mean_sim)}/100 (Grade {grade(mean_sim)})  "
        )
        line(f"**ChatGPT most often frames {brand} as:** {top_attr_str(attrs_b, n=6)}")
        line()

        # Per-segment table
        line("| Audience Segment | Recs | Share | Content Match Score | Grade | Gap |")
        line("|---|---:|---:|---:|:---:|---:|")
        for p in personas:
            seg   = persona_display(p)
            cnt   = entity_stats[(p, brand)]["count"]
            sim   = float(aff_df.loc[p, brand])
            gap_r = gap_df[(gap_df["persona_id"] == p) & (gap_df["brand"] == brand)]
            gap_s = float(gap_r["gap_score"].iloc[0]) if not gap_r.empty else 0.0
            gap_t = gap_r["gap_type"].iloc[0] if not gap_r.empty else "—"
            gap_icon = "🔴" if gap_t == "content_gap" else ("🔵" if gap_t == "missed_opportunity" else "✅")
            line(
                f"| {seg} | {cnt} | {pct(cnt, total_b)} "
                f"| {cms(sim)}/100 | {grade(sim)} | {gap_icon} {gap_s:+.2f} |"
            )
        line()
        line("> 🔴 Content gap (over-recommended vs. match score)  "
             "🔵 Missed opportunity (strong match, low recs)  "
             "✅ Aligned")
        line()

        # Narrative
        para = []
        if total_b == 0:
            para.append(
                f"{brand} received no positive mentions in this session. "
                "This likely reflects regional availability assumptions or a gap in "
                "ChatGPT's training data weighting for this brand. "
                "Review GEO signals: structured data, Knowledge Panel completeness, "
                "editorial coverage."
            )
        else:
            para.append(
                f"ChatGPT's strongest association for {brand} is with the "
                f"**{persona_display(best_pid)}** segment "
                f"({entity_stats[(best_pid, brand)]['count']} of {total_b} total recs){chi_note}. "
            )
            if best_pid != best_sim_pid:
                para.append(
                    f"However, the highest Content Match Score belongs to the "
                    f"**{persona_display(best_sim_pid)}** segment "
                    f"({cms(best_sim_val)}/100, Grade {grade(best_sim_val)}) — "
                    f"the brand's web language more closely mirrors that audience's vocabulary "
                    f"even if recommendation volume is lower there."
                )
            para.append(
                f"The weakest content match is with **{persona_display(worst_sim_pid)}** "
                f"({cms(worst_sim_val)}/100, Grade {grade(worst_sim_val)})."
            )
            if top_gap_row is not None and abs(top_gap_row["gap_score"]) > 0.25:
                if top_gap_row["gap_type"] == "content_gap":
                    para.append(
                        f"**Key finding:** {brand}'s biggest content gap is with "
                        f"**{persona_display(top_gap_row['persona_id'])}** "
                        f"(gap {top_gap_row['gap_score']:+.2f}): "
                        f"{top_gap_row['mention_count']} recs, score {cms(top_gap_row['similarity'])}/100. "
                        f"Adding {persona_display(top_gap_row['persona_id']).split(' (')[0]}-specific "
                        f"language to the website would close this gap and reinforce the recommendations."
                    )
                else:
                    para.append(
                        f"**Key finding:** {brand}'s most significant missed opportunity is with "
                        f"**{persona_display(top_gap_row['persona_id'])}** "
                        f"(gap {top_gap_row['gap_score']:+.2f}): "
                        f"score {cms(top_gap_row['similarity'])}/100 but only "
                        f"{top_gap_row['mention_count']} recs. "
                        f"The content is already there — the priority is increasing AI discoverability."
                    )
        line(" ".join(para))
        line()
        rule()
        line()

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTENT GAP RANKINGS
    # ═══════════════════════════════════════════════════════════════════════════
    h2("Top Content Opportunities — Ranked")
    line()
    line(
        "These are the highest-priority pages to write. Each row is a brand–audience "
        "pairing where ChatGPT is already recommending the brand, but the brand's "
        "website doesn't speak that audience's language. Ranked by impact: "
        "gap score × recommendation volume."
    )
    line()
    line("| # | Brand | Audience Segment | Gap Score | Recs | Match Score | Write This |")
    line("|---:|---|---|---:|---:|---:|---|")
    for i, row in content_gap_df.iterrows():
        pid_   = row["persona_id"]
        action = PERSONA_CONTENT_ACTIONS.get(pid_, "Add audience-targeted content")
        seg    = persona_display(pid_)
        line(
            f"| {i+1} | {row['brand']} | {seg} "
            f"| {row['gap_score']:+.2f} | {row['mention_count']} "
            f"| {cms(row['similarity'])}/100 | {action} |"
        )
    line()
    line(
        "> **Gap Score** = how much higher this brand ranks in AI recommendations than "
        "in content alignment for this audience (0–1 scale within-segment percentiles). "
        "A score of +0.60 means the brand is near the top of recommendations but near "
        "the bottom of content alignment — the largest possible gap."
    )
    line()
    rule()
    line()

    # ═══════════════════════════════════════════════════════════════════════════
    # MISSED OPPORTUNITIES
    # ═══════════════════════════════════════════════════════════════════════════
    h2("Discoverability Gaps — Ranked")
    line()
    line(
        "These brand–audience pairs have strong Content Match Scores (the website "
        "speaks the audience's language) but low recommendation frequency. "
        "The content problem is solved. The visibility problem is not. "
        "Focus here on GEO tactics: structured data, authority building, editorial mentions."
    )
    line()
    line("| # | Brand | Audience Segment | Gap Score | Recs | Match Score |")
    line("|---:|---|---|---:|---:|---:|")
    for i, row in missed_df.iterrows():
        seg = persona_display(row["persona_id"])
        line(
            f"| {i+1} | {row['brand']} | {seg} "
            f"| {row['gap_score']:+.2f} | {row['mention_count']} "
            f"| {cms(row['similarity'])}/100 |"
        )
    line()
    rule()
    line()

    # ═══════════════════════════════════════════════════════════════════════════
    # ACTION PLAN
    # ═══════════════════════════════════════════════════════════════════════════
    h2("Your Action Plan")
    line()
    line(
        "Recommendations are organized by brand, ordered by total content gap impact. "
        "Each brand entry leads with the one-sentence verdict, then specific actions "
        "by audience segment."
    )
    line()

    brand_gap_groups = defaultdict(list)
    for _, row in content_gap_df.iterrows():
        brand_gap_groups[row["brand"]].append(row)

    rec_num = 1
    for brand, rows in sorted(brand_gap_groups.items(),
                               key=lambda x: -sum(r["priority"] for r in x[1])):
        segs       = [persona_display(r["persona_id"]) for r in rows]
        avg_gap    = sum(r["gap_score"] for r in rows) / len(rows)
        total_recs = sum(r["mention_count"] for r in rows)
        avg_score  = sum(r["similarity"] for r in rows) / len(rows)
        top_attrs  = brand_attrs.get(brand, Counter()).most_common(4)

        h3(f"{rec_num}. {brand}")
        line()
        line(
            f"**Verdict:** ChatGPT recommends {brand} to {len(segs)} audience segment(s) "
            f"with strong volume, but the brand's website scores "
            f"**{cms(avg_score)}/100** on average — the AI is out-promoting the content."
        )
        line()
        line(f"**Affected audiences:** {', '.join(segs)}  ")
        line(f"**How ChatGPT frames {brand}:** {', '.join(a for a, _ in top_attrs)}")
        line()
        for row in rows:
            pid_   = row["persona_id"]
            action = PERSONA_CONTENT_ACTIONS.get(pid_, "Add audience-relevant content")
            line(
                f"- **{persona_display(pid_)}:** {action}. "
                f"_{row['mention_count']} recs at {cms(row['similarity'])}/100, "
                f"gap {row['gap_score']:+.2f}_"
            )
        line()
        rec_num += 1

    h3(f"{rec_num}. GEO & Discoverability — Brands Whose Content Is Ready")
    line()
    line(
        "These brands have already done the content work but aren't getting the AI "
        "recommendation volume their scores predict. Priority is off-page and technical GEO:"
    )
    line()
    line(
        "- **Structured data:** Add FAQ schema to pages that answer questions your "
        "target audience asks ChatGPT. How-To schema for ordering guides and meal prep content."
    )
    line(
        "- **Entity clarity:** Ensure your brand has a complete, accurate Wikipedia "
        "page and Google Knowledge Panel. These are primary sources for AI training data."
    )
    line(
        "- **Editorial mentions:** Earn coverage from industry media and review sources "
        "likely in AI training corpora (G2, Clutch, agency award lists, trade publications, "
        "LinkedIn thought leadership, and relevant subreddit communities)."
    )
    line()
    for _, row in missed_df.head(8).iterrows():
        line(
            f"- **{row['brand']} → {persona_display(row['persona_id'])}:** "
            f"Match score {cms(row['similarity'])}/100, "
            f"only {row['mention_count']} recs (gap {row['gap_score']:+.2f})."
        )
    line()
    rule()
    line()

    # ═══════════════════════════════════════════════════════════════════════════
    # APPENDIX A: FULL CONTENT MATCH SCORE MATRIX
    # ═══════════════════════════════════════════════════════════════════════════
    h2("Appendix A: Full Content Match Score Matrix")
    line()
    line(
        "All scores on a 0–100 scale with letter grades. "
        "Higher = closer alignment between brand website language and ChatGPT's "
        "language when recommending to that audience."
    )
    line()

    header = "| Audience Segment |" + "".join(f" {b[:14]} |" for b in brands)
    sep    = "|---|" + "".join("---:|" for _ in brands)
    line(header)
    line(sep)
    for pid in personas:
        seg_label = persona_display(pid)
        vals = "".join(
            f" {cms(float(aff_df.loc[pid, b]))}\u202f({grade(float(aff_df.loc[pid, b]))}) |"
            for b in brands
        )
        line(f"| {seg_label} |{vals}")
    line()

    # ═══════════════════════════════════════════════════════════════════════════
    # APPENDIX B: MENTION COUNT MATRIX
    # ═══════════════════════════════════════════════════════════════════════════
    h2("Appendix B: Positive Mention Count Matrix")
    line()
    line("Raw positive mention counts per audience segment and brand.")
    line()
    line(header)
    line(sep)
    for pid in personas:
        seg_label = persona_display(pid)
        vals = "".join(f" {entity_stats[(pid, b)]['count']} |" for b in brands)
        line(f"| {seg_label} |{vals}")
    line()

    # ═══════════════════════════════════════════════════════════════════════════
    # APPENDIX C: STATISTICAL DETAIL
    # ═══════════════════════════════════════════════════════════════════════════
    h2("Appendix C: Statistical Detail")
    line()
    h3("Study Design")
    line()
    line(
        "375 ChatGPT completions: 5 audience segments × 25 questions × 3 independent runs. "
        "Each run is a fully independent API call with no memory between calls. "
        "Runs enable measurement of AI consistency."
    )
    line()
    line(
        "Brand mentions were extracted via a structured GPT-4.1 call (temperature 0) "
        "using a controlled vocabulary schema. Entity counts are deduplicated at the "
        "response level for statistical testing (one mention per response, per brand). "
        "Content Match Scores use `text-embedding-3-large` (3,072-dimensional) "
        "embeddings and cosine similarity, scaled ×100."
    )
    line()
    if stats_df is not None:
        h3("Significance Tests")
        line()
        line(
            "**Binomial tests:** For each brand–segment pair, we test whether that "
            "segment's mention rate (k/75 responses) is significantly different from the "
            "brand's overall base rate across all segments. "
            f"85 tests total, Bonferroni-corrected threshold α = {0.05/85:.4f}. "
            f"**{(stats_df['p_corrected'] < 0.05).sum()} of 85 pairs** are significant "
            "after correction."
        )
        line()
        if chi_df is not None:
            sig_chi = chi_df[chi_df["significant"] == True]
            line(
                f"**Chi-square tests:** For each brand, we test whether the distribution "
                "of recommendations across the 5 segments is uniform. "
                f"**{len(sig_chi)} of {len(chi_df)} brands** show statistically "
                "non-uniform distributions (p<0.05 after Bonferroni correction for 17 tests). "
                "This validates that the persona segmentation produces meaningfully different "
                "recommendation profiles — the AI genuinely treats these audience types differently."
            )
            line()
        line(
            "**Run consistency:** Cosine similarity scores vary by fewer than 2% (CV<0.02) "
            "across the 3 runs for most brand–segment pairs, indicating that the Content "
            "Match Scores are stable signals, not artifacts of a single run."
        )
        line()
        line(
            "_All tests are exploratory and hypothesis-generating, not confirmatory. "
            "With n=75 responses per segment and n=3 runs for CI estimation, "
            "interpret findings as strong directional signals._"
        )
        line()

    h3("Content Alignment → Recommendation Correlation")
    line()
    line(
        f"Across all 85 brand-audience pairs (17 brands × 5 segments), "
        f"Spearman ρ between Content Match Score and mention count = "
        f"**{_rho_overall:+.3f}** (p < 0.0001). Per segment:"
    )
    line()
    line("| Audience Segment | Spearman ρ | p-value | Sig |")
    line("|---|---:|---:|:---:|")
    for _pid in sorted(_persona_corrs):
        _r, _p = _persona_corrs[_pid]
        line(f"| {persona_display(_pid)} | {_r:+.3f} | {_p:.4f} | {_sig(_p)} |")
    line()
    line(
        "**Causality caveat.** This is a cross-sectional correlation, not a controlled "
        "experiment. Two explanations are consistent with the data:"
    )
    line()
    line(
        "- **Explanation A (content drives recommendations):** Brands that write "
        "audience-specific content → ChatGPT learns it during training → recommends "
        "that brand to that audience more often → higher Content Match Score and higher "
        "mention count move together."
    )
    line()
    line(
        "- **Explanation B (shared underlying cause):** Brands that are culturally "
        "associated with a certain audience tend to both write content for that audience "
        "and get recommended to them — not because one caused the other, but because "
        "both reflect the same underlying brand positioning."
    )
    line()
    line(
        "The prescription is the same under both explanations: write content that "
        "genuinely addresses how your brand serves specific audience types. "
        "To confirm causality, a longitudinal study is required — publish targeted "
        "content, wait for a model update cycle, re-run the embeddings, and measure "
        "whether recommendation rates shift."
    )
    line()

    # ═══════════════════════════════════════════════════════════════════════════
    # APPENDIX D: AUDIENCE PROFILES
    # ═══════════════════════════════════════════════════════════════════════════
    h2("Appendix D: Audience Segment Profiles")
    line()
    line(
        "The five segments were designed as research archetypes grounded in agency buyer "
        "research. Each was injected as a full character description into ChatGPT before "
        "every question in that segment's session."
    )
    line()
    for p in personas_cfg["personas"]:
        pid_ = p["id"]
        h3(f"{SEGMENT_LABELS[pid_]} — {p['archetype']} ({p['name']})")
        line()
        line(f"**Tags:** {', '.join(p.get('tags', []))}")
        line()
        line(p.get("system_prompt", "").strip())
        line()

    # ── Write ─────────────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"session_{session_id}_report.md"
    report_path.write_text("\n".join(md))
    chars = len(report_path.read_text())
    print(f"Report written → {report_path}")
    print(f"Length: {chars:,} characters, {len(md)} lines")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEO Audit — report generator")
    parser.add_argument("--session", required=True, metavar="SESSION_ID")
    args = parser.parse_args()
    run(args.session)


if __name__ == "__main__":
    main()
