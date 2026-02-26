#!/usr/bin/env python3
"""
Build a single comprehensive PDF containing all study outputs:

  Section 01  Cross-Study Narrative
  Section 02  Phase 2 Full Report (Marketing Agencies)
  Section 03  Phase 1 Full Report (QSR Brands)
  Appendix A  Analysis Tables — all CSVs for both studies
  Appendix B  Raw ChatGPT Responses — every response, both studies
  Appendix C  Extracted Entity Mentions — both studies

Usage:
    python src/build_pdf.py
    python src/build_pdf.py --output /path/to/custom.pdf
"""

import argparse
import html as html_lib
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import markdown

ROOT     = Path(__file__).parent.parent
QSR_ROOT = ROOT.parent / "geo-audit"

# ── Report markdown files ─────────────────────────────────────────────────────
COMBINED_NARRATIVE = ROOT / "outputs" / "reports" / "combined_narrative.md"
AGENCY_REPORT      = ROOT / "outputs" / "reports" / "session_20260220_001321_report.md"
QSR_REPORT         = QSR_ROOT / "outputs" / "reports" / "session_20260218_141526_report.md"

# ── Stage 1: raw ChatGPT responses ───────────────────────────────────────────
AGENCY_STAGE1 = ROOT    / "outputs" / "stage1" / "session_20260220_001321.jsonl"
QSR_STAGE1    = QSR_ROOT / "outputs" / "stage1" / "session_20260218_141526.jsonl"

# ── Stage 2: analysis outputs ─────────────────────────────────────────────────
def s2(study_root, session, filename):
    return study_root / "outputs" / "stage2" / f"session_{session}_{filename}"

A_SID = "20260220_001321"
Q_SID = "20260218_141526"

AGENCY_CSVS = [
    ("Mention Counts by Persona",              s2(ROOT, A_SID, "mention_counts.csv")),
    ("Authority &amp; Content Alignment",      s2(ROOT, A_SID, "authority_correlation.csv")),
    ("Content Gaps &amp; Missed Opportunities",s2(ROOT, A_SID, "content_gaps.csv")),
    ("Affinity Matrix (Cosine Similarity)",    s2(ROOT, A_SID, "affinity_matrix.csv")),
    ("Per Brand-Persona Stats (Binomial)",     s2(ROOT, A_SID, "stats.csv")),
    ("Chi-Square Tests",                       s2(ROOT, A_SID, "stats_chisquare.csv")),
    ("Category Affinity",                      s2(ROOT, A_SID, "category_affinity.csv")),
    ("Run-Level Similarity Variance",          s2(ROOT, A_SID, "run_similarity.csv")),
]

QSR_CSVS = [
    ("Mention Counts by Persona",              s2(QSR_ROOT, Q_SID, "mention_counts.csv")),
    ("Authority &amp; Content Alignment",      s2(QSR_ROOT, Q_SID, "authority_correlation.csv")),
    ("Content Gaps &amp; Missed Opportunities",s2(QSR_ROOT, Q_SID, "content_gaps.csv")),
    ("Affinity Matrix (Cosine Similarity)",    s2(QSR_ROOT, Q_SID, "affinity_matrix.csv")),
    ("Per Brand-Persona Stats (Binomial)",     s2(QSR_ROOT, Q_SID, "stats.csv")),
    ("Chi-Square Tests",                       s2(QSR_ROOT, Q_SID, "stats_chisquare.csv")),
    ("Category Affinity",                      s2(QSR_ROOT, Q_SID, "category_affinity.csv")),
    ("Run-Level Similarity Variance",          s2(QSR_ROOT, Q_SID, "run_similarity.csv")),
]

AGENCY_ENTITIES = s2(ROOT,     A_SID, "entities_normalized.jsonl")
QSR_ENTITIES    = s2(QSR_ROOT, Q_SID, "entities_normalized.jsonl")

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
*, *::before, *::after { box-sizing: border-box; }

@page {
    size: Letter;
    margin: 0.8in 0.75in 0.8in 0.75in;
    @bottom-right {
        content: "Page " counter(page);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 8pt;
        color: #94a3b8;
    }
    @bottom-left {
        content: "GEO Audit — Full Report Package";
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 8pt;
        color: #94a3b8;
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 10pt;
    line-height: 1.65;
    color: #1e293b;
}

/* ── Cover ── */
.cover {
    min-height: 9in;
    display: flex;
    flex-direction: column;
    justify-content: center;
    page-break-after: always;
}
.cover h1 {
    font-size: 30pt;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 12pt 0;
    border: none;
    padding: 0;
    line-height: 1.15;
}
.cover .subtitle { font-size: 13pt; color: #475569; margin: 0 0 24pt 0; }
.cover .toc { margin: 0 0 24pt 0; }
.cover .toc-item {
    display: flex;
    align-items: baseline;
    gap: 8pt;
    padding: 5pt 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 10pt;
}
.cover .toc-num { font-weight: 700; color: #3b82f6; min-width: 24pt; }
.cover .toc-title { color: #1e293b; font-weight: 500; }
.cover .toc-desc { color: #64748b; font-size: 9pt; }
.cover .meta {
    font-size: 8.5pt;
    color: #94a3b8;
    border-top: 1px solid #e2e8f0;
    padding-top: 10pt;
    line-height: 1.8;
}

/* ── Section dividers ── */
.divider {
    page-break-before: always;
    page-break-after: always;
    min-height: 9in;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.divider .num { font-size: 80pt; font-weight: 700; color: #e2e8f0; line-height: 1; margin: 0; }
.divider h2 { font-size: 24pt; font-weight: 700; color: #0f172a; margin: 8pt 0 12pt 0; border: none; padding: 0; }
.divider .desc { font-size: 11pt; color: #64748b; max-width: 5.5in; line-height: 1.6; }
.divider .stats { margin-top: 20pt; display: flex; gap: 24pt; }
.divider .stat { }
.divider .stat-val { font-size: 20pt; font-weight: 700; color: #1e293b; }
.divider .stat-label { font-size: 9pt; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Body content ── */
.section { page-break-before: always; }

h1 {
    font-size: 18pt; font-weight: 700; color: #0f172a;
    margin: 0 0 8pt 0; padding-bottom: 6pt;
    border-bottom: 2px solid #e2e8f0;
    page-break-after: avoid;
}
h2 {
    font-size: 14pt; font-weight: 600; color: #1e293b;
    margin: 20pt 0 6pt 0; page-break-after: avoid;
}
h3 {
    font-size: 11.5pt; font-weight: 600; color: #334155;
    margin: 14pt 0 4pt 0; page-break-after: avoid;
}
h4 {
    font-size: 10pt; font-weight: 600; color: #475569;
    margin: 10pt 0 3pt 0; page-break-after: avoid;
}
p { margin: 0 0 7pt 0; }
ul, ol { margin: 0 0 7pt 18pt; padding: 0; }
li { margin-bottom: 3pt; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 14pt 0; }
blockquote {
    margin: 10pt 0; padding: 8pt 12pt;
    border-left: 3px solid #3b82f6;
    background: #f0f7ff; color: #334155;
}
code {
    font-family: "SF Mono", "Fira Code", "Courier New", monospace;
    font-size: 8.5pt; background: #f1f5f9;
    padding: 1pt 3pt; border-radius: 2pt;
}
pre {
    background: #f1f5f9; padding: 10pt; border-radius: 4pt;
    font-size: 8pt; line-height: 1.5; margin: 8pt 0;
    page-break-inside: avoid; overflow: hidden;
}
pre code { background: none; padding: 0; }
strong { font-weight: 600; }
a { color: #2563eb; }

/* ── Tables ── */
table {
    width: 100%; border-collapse: collapse;
    font-size: 9pt; margin: 8pt 0 12pt 0;
    page-break-inside: avoid;
}
thead th {
    background: #0f172a; color: #f8fafc;
    font-weight: 600; padding: 5pt 7pt;
    text-align: left; font-size: 8.5pt;
}
tbody tr:nth-child(even) { background: #f8fafc; }
tbody td { padding: 4pt 7pt; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
/* Numeric columns right-aligned */
.numeric { text-align: right; font-variant-numeric: tabular-nums;
           font-family: "SF Mono", monospace; font-size: 8.5pt; }

/* ── Raw response cards ── */
.persona-block { page-break-before: always; }
.persona-header {
    background: #0f172a; color: #f8fafc;
    padding: 10pt 14pt; border-radius: 4pt 4pt 0 0;
    margin-bottom: 0;
}
.persona-header h2 {
    color: #f8fafc; margin: 0; font-size: 14pt; border: none; padding: 0;
}
.persona-header .meta { font-size: 9pt; color: #94a3b8; margin-top: 3pt; }

.q-block { margin: 14pt 0; page-break-inside: avoid; }
.q-header {
    background: #1e293b; color: #e2e8f0;
    padding: 6pt 10pt; border-radius: 3pt 3pt 0 0;
    font-size: 9pt; font-weight: 600;
}
.q-text {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-top: none; border-bottom: none;
    padding: 8pt 10pt;
    font-size: 10pt; font-style: italic; color: #334155;
}
.run-block {
    border: 1px solid #e2e8f0;
    border-top: none;
}
.run-block:last-child { border-radius: 0 0 3pt 3pt; }
.run-header {
    background: #f1f5f9; padding: 4pt 10pt;
    font-size: 8.5pt; font-weight: 600; color: #64748b;
    border-bottom: 1px solid #e2e8f0;
    display: flex; justify-content: space-between;
}
.run-body {
    padding: 8pt 10pt;
    font-size: 9.5pt; line-height: 1.65; color: #1e293b;
    white-space: pre-wrap; word-break: break-word;
}

/* ── Entity mention cards ── */
.entity-table { font-size: 8.5pt; }
.entity-table tbody td { padding: 3pt 6pt; }
.sentiment-positive { color: #16a34a; font-weight: 600; }
.sentiment-negative { color: #dc2626; font-weight: 600; }
.sentiment-neutral  { color: #64748b; }

/* ── Appendix headings ── */
.appendix-section { page-break-before: always; }
.appendix-section > h2 {
    font-size: 12pt; color: #475569;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4pt; margin-bottom: 8pt;
}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def csv_to_html_table(csv_path: Path) -> str:
    import csv
    if not csv_path.exists():
        return f"<p><em>File not found: {csv_path.name}</em></p>"
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            rows.append(row)
    if not rows:
        return "<p><em>Empty file</em></p>"
    headers, data = rows[0], rows[1:]

    # Detect numeric columns by sniffing first data row
    numeric_cols = set()
    if data:
        for i, val in enumerate(data[0]):
            try:
                float(val)
                numeric_cols.add(i)
            except ValueError:
                pass

    th = "".join(f"<th>{html_lib.escape(h)}</th>" for h in headers)
    body = ""
    for row in data:
        tds = ""
        for i, cell in enumerate(row):
            cls = ' class="numeric"' if i in numeric_cols else ""
            tds += f"<td{cls}>{html_lib.escape(cell)}</td>"
        body += f"<tr>{tds}</tr>\n"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def md_to_html(md_path: Path) -> str:
    if not md_path.exists():
        return f"<p><em>Report not found: {md_path}</em></p>"
    text = md_path.read_text(encoding="utf-8")
    return markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "nl2br", "toc", "attr_list"],
    )


def build_responses_section(stage1_path: Path, study_label: str) -> str:
    records = load_jsonl(stage1_path)
    if not records:
        return f"<p><em>No response data found at {stage1_path}</em></p>"

    # Group: persona_id → question_id → run_number → record
    by_persona = defaultdict(lambda: defaultdict(dict))
    persona_names = {}
    question_texts = {}
    question_categories = {}

    for r in records:
        pid = r["persona_id"]
        qid = r["question_id"]
        run = r["run_number"]
        by_persona[pid][qid][run] = r
        persona_names[pid] = f"{r['persona_name']} — {r['persona_archetype']}"
        question_texts[qid] = r["question_text"]
        question_categories[qid] = r.get("question_category", "")

    parts = []
    for pid in sorted(by_persona):
        name = persona_names[pid]
        questions = by_persona[pid]
        n_questions = len(questions)
        n_responses = sum(len(runs) for runs in questions.values())

        parts.append(f"""
<div class="persona-block">
  <div class="persona-header">
    <h2>{pid}: {html_lib.escape(name)}</h2>
    <div class="meta">{n_questions} questions &nbsp;·&nbsp; {n_responses} responses</div>
  </div>
""")
        for qid in sorted(questions):
            runs = questions[qid]
            q_text = question_texts[qid]
            q_cat  = question_categories[qid].replace("_", " ").title()
            parts.append(f"""
  <div class="q-block">
    <div class="q-header">{html_lib.escape(qid)} &nbsp;·&nbsp; {html_lib.escape(q_cat)}</div>
    <div class="q-text">{html_lib.escape(q_text)}</div>
""")
            for run_num in sorted(runs):
                rec = runs[run_num]
                response_text = rec.get("raw_response", "").strip()
                tokens = rec.get("total_tokens", "—")
                model = rec.get("model", "")
                parts.append(f"""
    <div class="run-block">
      <div class="run-header">
        <span>Run {run_num}</span>
        <span>{html_lib.escape(model)} &nbsp;·&nbsp; {tokens} tokens</span>
      </div>
      <div class="run-body">{html_lib.escape(response_text)}</div>
    </div>
""")
            parts.append("  </div>")  # close q-block
        parts.append("</div>")  # close persona-block

    return f"""
<div class="section">
  <h1>Appendix B: Raw ChatGPT Responses — {html_lib.escape(study_label)}</h1>
  <p>All {len(records)} responses, organized by persona then question. Each question shows all 3 runs.</p>
  {"".join(parts)}
</div>
"""


def build_entities_section(entities_path: Path, study_label: str) -> str:
    records = load_jsonl(entities_path)
    if not records:
        return f"<p><em>No entity data found at {entities_path}</em></p>"

    headers = ["persona_id", "question_id", "run_number", "canonical", "raw_name",
               "sentiment", "confidence"]
    th = "".join(f"<th>{h}</th>" for h in headers)

    rows_html = ""
    for r in records:
        sentiment = r.get("sentiment", "")
        cls = f"sentiment-{sentiment}" if sentiment in ("positive", "negative", "neutral") else ""
        cells = []
        for h in headers:
            val = html_lib.escape(str(r.get(h, "")))
            if h == "sentiment":
                val = f'<span class="{cls}">{val}</span>'
            cells.append(f"<td>{val}</td>")
        rows_html += f"<tr>{''.join(cells)}</tr>\n"

    return f"""
<div class="appendix-section">
  <h2>Appendix C: Extracted Entity Mentions — {html_lib.escape(study_label)}</h2>
  <p>{len(records):,} entity mention records (normalized canonical names, all sentiments).</p>
  <table class="entity-table">
    <thead><tr>{th}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
"""


# ── HTML assembly ─────────────────────────────────────────────────────────────

def build_html() -> str:
    parts = []

    # Cover
    parts.append("""
<div class="cover">
  <h1>GEO Audit<br>Full Report Package</h1>
  <div class="subtitle">How ChatGPT Decides Who to Recommend — and Why<br>
    QSR Brands (Phase 1) &amp; Marketing Agencies (Phase 2)</div>

  <div class="toc">
    <div class="toc-item"><span class="toc-num">01</span><span class="toc-title">Cross-Study Narrative</span><span class="toc-desc">Combined findings, the two-lever framework, franchise white space</span></div>
    <div class="toc-item"><span class="toc-num">02</span><span class="toc-title">Phase 2 Full Report</span><span class="toc-desc">Marketing Agencies · 15 brands · 5 personas · 1,875 queries</span></div>
    <div class="toc-item"><span class="toc-num">03</span><span class="toc-title">Phase 1 Full Report</span><span class="toc-desc">QSR Brands · 17 chains · 5 personas · 1,875 queries</span></div>
    <div class="toc-item"><span class="toc-num">A</span><span class="toc-title">Appendix A: Analysis Tables</span><span class="toc-desc">All 16 CSV outputs — mention counts, authority, gaps, stats, affinity, similarity</span></div>
    <div class="toc-item"><span class="toc-num">B</span><span class="toc-title">Appendix B: Raw ChatGPT Responses</span><span class="toc-desc">All 750 responses verbatim (375 per study), organized by persona and question</span></div>
    <div class="toc-item"><span class="toc-num">C</span><span class="toc-title">Appendix C: Extracted Entity Mentions</span><span class="toc-desc">Every brand mention extracted from every response, with sentiment and confidence</span></div>
  </div>

  <div class="meta">
    Phase 1 · QSR Brands · Session 20260218_141526 · GPT-4o · 17 brands · 5 personas<br>
    Phase 2 · Marketing Agencies · Session 20260220_001321 · GPT-4.1 · 15 brands · 5 personas<br>
    Embeddings: text-embedding-3-large (3,072 dimensions) · Similarity: cosine scaled 0–100<br>
    Significance: Bonferroni-corrected binomial tests + chi-square per brand
  </div>
</div>
""")

    # 01 Combined narrative
    parts.append("""
<div class="divider">
  <div class="num">01</div>
  <h2>Cross-Study Narrative</h2>
  <div class="desc">
    The overarching story: how authority, content alignment, and market specialization
    determine who ChatGPT recommends. Includes the two-lever framework, the franchise
    white space finding, and the Powered by Search anomaly.
  </div>
  <div class="stats">
    <div class="stat"><div class="stat-val">ρ = +0.61</div><div class="stat-label">DA → mentions (QSR)</div></div>
    <div class="stat"><div class="stat-val">ρ = +0.01</div><div class="stat-label">DA → mentions (Agencies)</div></div>
    <div class="stat"><div class="stat-val">10 → 0</div><div class="stat-label">Franchise dev: Scorpion, then silence</div></div>
  </div>
</div>
""")
    parts.append(f'<div class="section">{md_to_html(COMBINED_NARRATIVE)}</div>')

    # 02 Agency report
    parts.append(f"""
<div class="divider">
  <div class="num">02</div>
  <h2>Phase 2 Full Report<br><span style="font-size:15pt;font-weight:400;color:#64748b">Marketing Agencies</span></h2>
  <div class="desc">Session {A_SID} · 15 agencies · 5 buyer personas · 1,875 ChatGPT queries · Screaming Frog content extraction</div>
  <div class="stats">
    <div class="stat"><div class="stat-val">487</div><div class="stat-label">Pages crawled</div></div>
    <div class="stat"><div class="stat-val">5.4M</div><div class="stat-label">Characters indexed</div></div>
    <div class="stat"><div class="stat-val">15/15</div><div class="stat-label">Chi-square significant</div></div>
  </div>
</div>
""")
    parts.append(f'<div class="section">{md_to_html(AGENCY_REPORT)}</div>')

    # 03 QSR report
    parts.append(f"""
<div class="divider">
  <div class="num">03</div>
  <h2>Phase 1 Full Report<br><span style="font-size:15pt;font-weight:400;color:#64748b">QSR Brands</span></h2>
  <div class="desc">Session {Q_SID} · 17 QSR chains · 5 consumer personas · 1,875 ChatGPT queries</div>
  <div class="stats">
    <div class="stat"><div class="stat-val">1,294</div><div class="stat-label">Total brand mentions</div></div>
    <div class="stat"><div class="stat-val">ρ = +0.647</div><div class="stat-label">Content → persona routing</div></div>
    <div class="stat"><div class="stat-val">17/17</div><div class="stat-label">Chi-square significant</div></div>
  </div>
</div>
""")
    if QSR_REPORT.exists():
        parts.append(f'<div class="section">{md_to_html(QSR_REPORT)}</div>')

    # Appendix A: All CSVs
    parts.append("""
<div class="divider">
  <div class="num">A</div>
  <h2>Appendix A: Analysis Tables</h2>
  <div class="desc">
    All 16 CSV output files from both studies. Eight tables per study:
    mention counts, authority correlation, content gaps, affinity matrix,
    binomial stats, chi-square tests, category affinity, and run-level similarity variance.
  </div>
</div>
""")

    parts.append('<div class="section"><h1>Appendix A · Phase 2: Marketing Agencies</h1></div>')
    for title, path in AGENCY_CSVS:
        parts.append(f'<div class="appendix-section"><h2>A · {title}</h2>{csv_to_html_table(path)}</div>')

    parts.append('<div class="appendix-section"><h1>Appendix A · Phase 1: QSR Brands</h1></div>')
    for title, path in QSR_CSVS:
        parts.append(f'<div class="appendix-section"><h2>A · {title}</h2>{csv_to_html_table(path)}</div>')

    # Appendix B: Raw responses
    parts.append("""
<div class="divider">
  <div class="num">B</div>
  <h2>Appendix B: Raw ChatGPT Responses</h2>
  <div class="desc">
    Every response verbatim. 375 responses per study (25 questions × 3 runs × 5 personas),
    750 total. Organized by persona, then question, with all 3 runs shown for each question.
  </div>
  <div class="stats">
    <div class="stat"><div class="stat-val">750</div><div class="stat-label">Total responses</div></div>
    <div class="stat"><div class="stat-val">375</div><div class="stat-label">Per study</div></div>
    <div class="stat"><div class="stat-val">3</div><div class="stat-label">Runs per question</div></div>
  </div>
</div>
""")
    parts.append(build_responses_section(AGENCY_STAGE1, "Phase 2: Marketing Agencies"))
    parts.append(build_responses_section(QSR_STAGE1,    "Phase 1: QSR Brands"))

    # Appendix C: Entity mentions
    parts.append("""
<div class="divider">
  <div class="num">C</div>
  <h2>Appendix C: Extracted Entity Mentions</h2>
  <div class="desc">
    Every brand entity extracted from every ChatGPT response, with canonical name resolution,
    sentiment classification, and confidence score. Used to compute all mention counts and
    base rates in the statistical analysis.
  </div>
</div>
""")
    parts.append(build_entities_section(AGENCY_ENTITIES, "Phase 2: Marketing Agencies"))
    parts.append(build_entities_section(QSR_ENTITIES,    "Phase 1: QSR Brands"))

    body = "\n".join(parts)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GEO Audit — Full Report Package</title>
<style>{CSS}</style>
</head>
<body>{body}</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build comprehensive PDF report package")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    out_dir = ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(args.output) if args.output else out_dir / "geo_audit_full_report.pdf"

    print("Building HTML...")
    html = build_html()
    print(f"  HTML size: {len(html):,} chars")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html)
        html_path = f.name
    print(f"  Temp file: {html_path}")

    print(f"Generating PDF → {pdf_path}")
    result = subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-extensions",
            "--disable-dev-shm-usage",
            "--run-all-compositor-stages-before-draw",
            f"--print-to-pdf={pdf_path}",
            "--print-to-pdf-no-header",
            f"file://{html_path}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print("Chrome stderr:", result.stderr[:2000])
        raise RuntimeError(f"Chrome exited with code {result.returncode}")

    Path(html_path).unlink(missing_ok=True)

    size_mb = pdf_path.stat().st_size / 1_048_576
    print(f"\nDone.")
    print(f"  PDF : {pdf_path}")
    print(f"  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
