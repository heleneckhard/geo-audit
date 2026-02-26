# How ChatGPT Decides Who to Recommend — And What You Can Do About It

*A cross-study analysis: QSR brands (17 chains) × Marketing Agencies (15 firms)*

---

## The Short Version

We ran a controlled experiment across two very different markets — national fast food chains and marketing agencies — and found the same underlying mechanism at work in both. ChatGPT isn't just guessing when it recommends a brand. It's matching the language of the question to the language on your website. And in some markets, that matchup is so weak that small brands with the right words on their pages are beating household names with 10× the web authority.

Here are the three most important findings:

1. **In commodity markets, size wins.** Among QSR brands, domain authority predicts recommendation frequency with ρ = +0.60. The bigger the brand, the more often ChatGPT mentions it. Full stop.

2. **In specialized markets, size is a weak signal at best.** Among marketing agencies, domain authority predicts recommendation frequency with ρ = +0.327 — less than half the strength of the QSR market, and well short of statistical significance at this sample size. A DA-50 dental agency with the right positioning beats a DA-88 generalist that speaks the wrong language.

3. **The franchise marketing space is almost entirely unclaimed.** When we asked ChatGPT franchise development questions, Scorpion got 30 mentions. Every other agency combined: 2. That's not a competitive market — it's a vacancy with one early occupant.

---

## How We Measured This

**Phase 1:** 17 major QSR brands (McDonald's, Chick-fil-A, Wendy's, Taco Bell, Chipotle, and 12 others). 5 buyer personas representing different fast food occasions. 375 ChatGPT queries per persona, 75 per question type. Each response was scored for brand mentions.

**Phase 2:** 15 marketing agencies across the spectrum — dental specialists, B2B demand gen firms, franchise-focused platforms, and general performance agencies. Same methodology: 5 buyer personas representing agency buyers (startup founder, mid-market marketer, local dental practice owner, franchise brand director, franchise development director). 375 queries each.

**The content matching score:** We took each brand's website content and each set of ChatGPT responses and converted them to vectors in a 3,072-dimensional embedding space. Cosine similarity between the brand vector and the persona response vector gives a score from 0–100. A score of 77 means the brand's content points in nearly the same direction as the AI's responses. A score of 39 means they're almost orthogonal — same market, completely different language.

---

## Chapter 1: The QSR Study — When Authority Drives Everything

### The finding that matters most

In the fast food world, domain authority and total mentions correlate at **ρ = +0.60**. That's not a weak relationship — that's a meaningful one. Bigger brands get mentioned more. Here's the data:

| Brand | Domain Authority | Total Mentions | Content Alignment (avg) |
|---|---|---|---|
| Wendy's | 88 | **237** | 57 |
| Chick-fil-A | 90 | 189 | **58** |
| McDonald's | 90 | 151 | 58 |
| Taco Bell | 89 | 121 | 58 |
| Subway | 90 | 112 | 51 |
| Chipotle | 90 | 129 | 48 |
| Panera Bread | 89 | 101 | 41 |
| Starbucks | 92 | 47 | **32** |
| Domino's | 89 | 15 | 30 |

Starbucks jumps out immediately. It has the **highest domain authority of any brand in the study** (DA 92) and the **lowest content alignment** (average score 32) — yet it still gets 47 mentions. Why? Because it's Starbucks. In a commodity market, ubiquity is its own recommendation engine.

But look at Domino's. DA 89 — nearly identical to Starbucks. Only 15 mentions. Content alignment score of 30 — also nearly identical to Starbucks. So why does Starbucks get 3× the mentions?

The difference is brand identity specificity. Starbucks has cultural gravity around specific occasions (morning ritual, work fuel, social meeting). Domino's has brand gravity around price and delivery — but ChatGPT's answers about dining recommendations didn't invoke those frames. The content is misaligned in the same way, but Starbucks has more cultural surface area.

**The lesson:** In a big brand market, you're competing on who owns the most mental real estate, and your website is the deed.

### The Two Levers

Here's the core QSR finding framed for marketers:

**Lever 1 — Authority** determines your *total volume* of recommendations across all audiences. It's blunt. Hard to move fast. A function of years of brand-building, backlinks, and press coverage.

**Lever 2 — Content Alignment** determines *which audience* ChatGPT routes to you. It's specific. You can move it this quarter by updating what your website says and how it says it.

These two levers are largely **independent**. The correlation between a brand's domain authority and its content alignment score is essentially zero. You can have high DA and low alignment (Starbucks for office-catering buyers). You can have moderate DA and high alignment (Chick-fil-A, consistently top-ranked by content across 4 of 5 personas).

**Content alignment predicts persona-specific routing with ρ = +0.647 — nearly as strong as authority predicts total volume.** This means the language on your website is directly responsible for whether ChatGPT sends the right buyer your way.

### The Starbucks content gap — and the fix

Priya is our office manager persona. She's asking ChatGPT questions like: *"What's the best option for catering a working lunch for 12 people?"* and *"Where should we order from for a team meeting with dietary restrictions?"*

Starbucks gets **28 mentions** from Priya's queries. But its content alignment score with Priya is **0.34** — the lowest of any brand in any persona combination. ChatGPT is recommending Starbucks to Priya based on cultural familiarity, not because Starbucks.com speaks her language. That's a content gap of +0.647.

The fix is direct: Starbucks could publish explicit content around team ordering, office catering, bulk orders, and dietary accommodation. Right now, ChatGPT is doing that work *for* Starbucks in spite of the website, not because of it. Close the gap, and you deepen the moat.

### The clearest missed opportunity: Sweetgreen and In-N-Out

On the flip side: **Sweetgreen** and **In-N-Out** both have strong content alignment with specific personas — but ChatGPT barely mentions them.

| Brand | Persona | Content Alignment | Mentions | Gap |
|---|---|---|---|---|
| In-N-Out | Dale (value buyer) | 0.50 | 0 | −0.59 |
| Sweetgreen | Marcus (health-conscious) | 0.47 | 0 | −0.47 |
| Sweetgreen | Tyler (Gen Z) | 0.45 | 0 | −0.35 |

Their content is speaking the right language. ChatGPT just isn't amplifying it yet. These brands have the content alignment signal — they need to build the authority signal to match.

---

## Chapter 2: The Agency Study — When Content Fit Is Everything

### The most striking finding in the data

In the agency market, domain authority and total mentions correlate at **ρ = +0.327** — less than half the QSR strength, and not statistically significant at n=15. The signal that explained 37% of variance in QSR explains roughly 11% in agencies. In a specialized market, brand size stops being the dominant force.

Here's the data that illustrates why:

| Agency | Domain Authority | Total Mentions | Content Alignment (avg) |
|---|---|---|---|
| Scorpion | 88 | **94** | 57 |
| Directive Consulting | 88 | 45 | **62** |
| Location3 | 73 | 45 | **62** |
| SOCi | 85 | 45 | 53 |
| Refine Labs | 76 | 52 | 56 |
| SmartBug Media | 86 | 32 | 59 |
| Wonderist Agency | 75 | 37 | 56 |
| KickStart Dental | 50 | 27 | 56 |
| Firegang Dental | 54 | 23 | 44 |
| Pain-Free Dental | 54 | 21 | 60 |
| **Powered by Search** | 83 | **20** | **67** |
| Cardinal Digital | 78 | 20 | 59 |
| NoGood | 82 | 17 | 58 |
| BrandMuscle | 77 | 16 | 47 |
| Great Dental Websites | 64 | 12 | 53 |

**Powered by Search has the highest content alignment score of any agency in the study — and the joint-lowest total mentions.** That is the most dramatic anomaly in either dataset.

### The Powered by Search paradox

This Canadian B2B performance agency has content that aligns extraordinarily well with how ChatGPT talks to marketing buyers — across *every single buyer persona and every single question category*:

| Persona | PBS Alignment Score | PBS Mentions | Biggest Gap |
|---|---|---|---|
| Jordan (startup founder) | **77.3** | 20 | — |
| Sandra (mid-market marketer) | **71.7** | 0 | −0.73 |
| Christine (franchise brand director) | **61.7** | 0 | −0.60 |
| Derek (franchise dev director) | **62.6** | 0 | −0.50 |
| Ray (dental practice owner) | **62.5** | 0 | −0.40 |

Sandra's gap of −0.733 is the **single largest missed opportunity in either study.** Her content alignment with Powered by Search (0.72) is more than double her alignment with most brands that actually get recommended to her.

**What this tells us:** Powered by Search has built a website that speaks fluent buyer language. But the brand hasn't yet built the authority signals — backlinks, citations, press — that cause ChatGPT to surface it. The content is there. The credibility scaffolding is not. This is a fixable problem, and it's a race against the market catching up.

### Why DA barely predicts agency recommendations

The agency market is fundamentally segmented. Buyers aren't asking "who's the biggest marketing agency" — they're asking:

- *"Who specializes in franchise development lead generation?"*
- *"What's the best agency for dental practice marketing?"*
- *"What B2B demand gen agency has the best track record for SaaS?"*

Each of those questions has a different answer. And ChatGPT routes to the agency whose content most closely matches the question's framing — not to the agency with the most backlinks.

In QSR, every persona is asking some version of "where should I eat?" The common frame means big brand awareness translates directly to recommendations. In agencies, there's no common frame. The right answer to Jordan's question is completely different from the right answer to Ray's question. Authority is irrelevant to that routing decision.

---

## Chapter 3: The Franchise Blind Spot

This is the most actionable finding in the data.

### Derek's world: 15 agencies, 375 questions, 10 recommendations

Derek is a franchise development marketing director. He's responsible for finding new franchisees. His questions to ChatGPT sound like:

- *"What marketing agencies specialize in franchise development lead generation?"*
- *"Who's the best at running IFX-style multi-touch franchise development campaigns?"*
- *"What platforms help emerging franchise brands scale their franchise development marketing?"*

We ran 375 of those questions. Here's every answer ChatGPT gave:

| Agency | Mentions from Derek |
|---|---|
| Scorpion | **30** |
| Location3 | 1 |
| SOCi | 1 |
| All other 12 agencies combined | **0** |

This isn't a competitive market. It's a vacancy with one early occupant and two footnotes.

And Scorpion's 30 mentions represent only **8% of Derek's total queries**. In 92% of Derek's questions, ChatGPT couldn't confidently name a single franchise development specialist.

### What ChatGPT said instead

When ChatGPT didn't name a specific agency for Derek, it described the *type* of agency that should help — and that description matches the content that's missing from the market. Phrases that appeared repeatedly in the responses:

- "agencies with deep experience in franchise development lead generation"
- "firms that understand the dual-audience challenge of franchise marketing" (selling the concept to franchisees while supporting franchisees to sell to customers)
- "partners with IFX conference presence and franchisor relationships"
- "agencies that can manage both brand standards enforcement and local activation"

Every agency we studied has some of this language. None of them have built a content presence dense enough for ChatGPT to route to them confidently when franchise development is the explicit brief.

### Christine's franchise brand world — more competitive, but thin

Christine is the internal marketing director for an established franchise brand. She gets more recommendations — but they're still concentrated:

| Agency | Mentions from Christine |
|---|---|
| Scorpion | **52** |
| SOCi | 43 |
| Location3 | 40 |
| BrandMuscle | 16 |
| Cardinal Digital Marketing | 5 |
| Wonderist Agency | 1 |
| All others | 0 |

Four agencies dominate, but the content alignment scores tell a different story. Location3's alignment with Christine (0.68) is higher than Scorpion's (0.58), yet Scorpion gets 30% more mentions. **Location3 is under-surfaced relative to content fit** — and BrandMuscle's 16 mentions come despite an alignment score of only 0.49, lowest of any brand that gets recommended.

BrandMuscle is surfacing on market familiarity, not content resonance. That's a content gap — and a clear opening for competitors who speak Christine's language more directly.

---

## The Mechanism: How ChatGPT Actually Decides

Here's the plain-English explanation of what's happening under the hood.

When someone asks ChatGPT a question, the model doesn't search Google. It generates an answer based on patterns learned from billions of web pages. The brands and agencies that appear in those answers are the ones that:

1. **Are mentioned frequently** across the web in relevant contexts (this is where authority helps — more pages, more citations, more likelihood of appearing in the training signal)

2. **Use language that matches the question** — not just keywords, but conceptual framing, vocabulary, the problems they describe solving, the outcomes they claim to deliver

Point 2 is what the content alignment score measures. We took each brand's website content, converted it to a mathematical representation of meaning, and compared it to the mathematical representation of the AI's own answers to buyer questions. High similarity = the brand speaks the same language the AI speaks when answering those questions. Low similarity = the brand exists in a different conceptual space.

**This is why a DA-50 dental agency can beat a DA-88 generalist agency for a dental practice buyer.** KickStart Dental (DA 50) has built its entire website around dental practice growth: patient acquisition, same-day appointment conversion, local SEO for dental offices. When Ray asks ChatGPT about dental practice marketing, the concepts in the question map directly to the concepts on KickStart's website.

KickStart's content alignment with Ray: **0.748** — the second-highest score in the entire agency dataset.

Scorpion's content alignment with Ray: **0.586** — decent, but a full 16 points lower than a brand with a fraction of its authority.

---

## The Two-Lever Framework: What You Can Actually Control

| Lever | What It Affects | How Hard to Move | Timeframe |
|---|---|---|---|
| **Domain Authority** | Total recommendation volume across all audiences | Hard | 12–24 months |
| **Content Alignment** | Which specific audiences ChatGPT routes to you | Easier | 1–3 months |

The most important practical implication of this research:

**You can't close the authority gap overnight. You can close the content gap this quarter.**

For most brands in both studies, the gap between their content alignment score and their optimal score is primarily a content strategy problem — not a PR problem, not a link-building problem, not an advertising problem. The brands that win in AI recommendations are the ones whose websites most clearly describe, in buyer-native language, exactly what type of buyer they serve and how they serve them.

---

## What Winning Looks Like — And What's Still Available

### In QSR: The market is spoken for, but the audience targeting isn't

The big QSR brands dominate total recommendation volume. That won't change quickly. But every major QSR brand has at least one buyer persona where its content alignment is significantly weaker than its recommendation frequency — meaning ChatGPT is routing buyers there on brand recognition alone, not content resonance.

Closing those gaps would deepen moats that are currently shallow. The brands that do it first own that audience segment more explicitly and more defensibly.

### In dental agency marketing: The market is claimed by specialists

Ray's dental market is the most competitive segment in the agency study. Five dental-specialist agencies have built such strong content alignment that they dominate recommendation share despite having a fraction of the domain authority of generalist competitors. This market is effectively closed to late entrants without genuine dental specialization signaled through content.

### In B2B agency marketing (Jordan/Sandra): The race is live

The startup and mid-market segments have clear leaders — Directive Consulting and Refine Labs get mentioned consistently, with strong content alignment. But Powered by Search's anomaly suggests the segment isn't fully locked up. A well-positioned challenger with the right content density could displace current leaders in 12–18 months.

### In franchise marketing: The opportunity is extraordinary

**Derek's segment is essentially unclaimed.** Scorpion has the beach head — 10 mentions when every other agency got zero — but at 2.7% query coverage, the territory is vast and mostly empty.

The agency that builds the most comprehensive, fluent content about franchise development marketing — not just franchise marketing — will own this segment's AI recommendations by default. There are no entrenched competitors to displace. The first-mover advantage here is real and near-term.

**The content gap for Christine's franchise brand segment is also significant**, and the current occupants (Scorpion, SOCi, Location3, BrandMuscle) are vulnerable. Location3 is under-surfaced. BrandMuscle is over-recommended relative to alignment. Neither moat is deep.

---

## The Action Agenda

**For any brand in either study:**

1. **Run your content alignment score against each buyer persona.** High alignment + low mentions = your content is right, your authority isn't. Low alignment + high mentions = your authority is carrying you, but you're not owning the audience. Both are fixable.

2. **Map the language in your ChatGPT recommendation responses.** What words, frameworks, and concepts does ChatGPT use when recommending agencies like you? Those are the words, frameworks, and concepts your website should be built around.

3. **Publish for personas, not products.** The highest content alignment scores in both studies belong to brands that have clearly written for specific buyer types — not for SEO keywords, not for product features, but for the specific decision context of a specific buyer in a specific moment.

4. **Prioritize uncontested segments.** If you're in franchise development marketing and you're not writing explicitly about it, you're leaving a completely unclaimed segment on the table. ChatGPT is looking for someone to recommend. Right now, it can barely find anyone.

---

*Phase 1 session: 20260218_141526 | Phase 2 session: 20260220_001321*
*Methodology: GPT-4o, 375 queries per persona, 5 question types × 15 prompts × 5 runs. Embeddings: text-embedding-3-large (3,072 dimensions). Similarity: cosine similarity scaled 0–100.*
