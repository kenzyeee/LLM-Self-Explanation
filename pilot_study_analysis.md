# Pilot Study Analysis: LLM Explanation Agreement Study

## 1. Executive Summary

Your pilot ran **11 instances** (4 SST-2, 3 MNLI, 4 AG News) through **Llama-3.3-70b** at temperature 0, eliciting 4 self-explanation strategies per instance: Highlighting (H), Rationale (R), Counterfactual (CF), and Rank Ordering (RO). The core metric — **Explanation Consensus Score (ECS)** — averages pairwise Jaccard similarities across cross-paradigm strategy pairs to measure how much these methods agree on which tokens matter.

### Key Pilot Findings

| Finding | Significance |
|---------|-------------|
| **Mean ECS = 0.177** (95% CI: [0.131, 0.227]) | Very low agreement across strategies |
| **Only 4/11 (36%) "complete cases"** (all 4 strategies valid) | CF is the bottleneck — 55% valid |
| **CF flip verified: only 6/11** (3 failed to actually flip the label) | Self-reported flips are unreliable without verification |
| **Primary ECS (H–CF, CF–RO pairs) = 0.087** | Cross-paradigm agreement is near-zero |
| **H–RO pair: highest Jaccard (0.299 mean)** | Same-paradigm methods agree most — expected but uninformative |
| **No long inputs (>50 words) sampled** | Length stratification is untestable at this scale |

---

## 2. Positioning Against Related Work

Your study sits at the intersection of several active research threads. Here's how it maps to the key papers:

### 2.1 Madsen et al. (2024) — *"Are self-explanations from Large Language Models faithful?"* (ACL Findings)

**What they did:** Tested faithfulness of LLM self-explanations (counterfactual, feature attribution, redaction) via self-consistency checks across Llama-2, Mistral, and Falcon-40B. Used logit-based measurements where available.

**Overlap with your work:**
- Both use counterfactual and feature-attribution style explanations
- Both evaluate faithfulness through behavioral tests (flip rate, consistency)
- Both find faithfulness is model/task/explanation-dependent

**Critical difference:**
- Madsen et al. test **faithfulness** (does the explanation match the model's internal behavior?) by perturbing inputs and checking prediction changes
- Your study tests **agreement** (do different explanation methods agree with each other?)
- These are **different constructs**: methods can agree with each other while all being unfaithful, or disagree while some are faithful

> [!IMPORTANT]
> **Your study measures inter-method agreement, not faithfulness.** Calling ECS a "signal for faithfulness" (as your title implies) requires bridging this gap with validity evidence. The consensus-core flip test is a step in that direction but needs to be the central finding, not a side metric.

### 2.2 Matton et al. (2025) — *"Walk the Talk?"* (ICLR 2025 Spotlight)

**What they did:** Defined faithfulness formally as the discrepancy between concepts the LLM *claims* are influential vs. concepts that are *causally* influential. Used auxiliary LLMs for counterfactual generation and hierarchical Bayesian models for causal quantification.

**Implication for your work:**
- This is the current gold standard for causal faithfulness evaluation
- Your ECS metric operates at a much simpler level (token-set overlap)
- You could position your work as a **lightweight, API-only alternative** to their resource-intensive causal approach
- If ECS correlates with their causal faithfulness scores, that would be a strong validation result

### 2.3 Parcalabescu & Frank (2024) — *CC-SHAP* (ACL 2024)

**What they did:** Measured self-consistency by comparing how input tokens contribute to the LLM's prediction vs. its explanation using Shapley values. Crucially, they made the distinction that this is **self-consistency, not faithfulness**.

**Implication for your work:**
- CC-SHAP is a direct competitor to ECS as a consistency metric
- CC-SHAP has a major advantage: it doesn't require input editing (avoiding OOD issues)
- CC-SHAP provides a **continuous** per-instance score; your ECS also does, but Jaccard is cruder
- You should cite this and be honest about what ECS measures (consistency, not faithfulness)

### 2.4 Krishna et al. (2022) — *"The Disagreement Problem in XAI"*

**What they did:** Formalized the observation that different XAI methods (LIME, SHAP, gradient methods) produce conflicting explanations. Introduced a quantitative framework for measuring disagreement.

**This is your closest conceptual ancestor.** Your study essentially applies the disagreement-problem framework to LLM **self-explanations** rather than post-hoc attribution methods. This is a **genuinely novel framing** — most disagreement studies compare externally-applied methods, while you compare methods that the model itself generates.

> [!TIP]
> **Positioning recommendation:** Frame your study as "the disagreement problem, revisited for self-explanations." This is the strongest narrative angle for a paper.

### 2.5 Lyu et al. (2024) — *"Towards Faithful Model Explanation in NLP: A Survey"* (Computational Linguistics)

This comprehensive survey of 110+ explanation methods distinguishes plausibility from faithfulness. Your four strategies map onto their taxonomy:
- **H and RO** → extraction-based feature attribution
- **R** → free-text rationalization
- **CF** → perturbation-based counterfactual

The survey identifies a gap in **cross-paradigm comparison**, which your study directly addresses.

### 2.6 NeuroFaith (2024-2025) — Internal Representation Alignment

Frameworks that use mechanistic interpretability to test if explanations align with neural activations. Your approach is **complementary** — you don't need model internals, which makes ECS applicable to **closed-source API models** (a clear advantage worth emphasizing).

---

## 3. What's Right About Your Approach

### ✅ Strengths

1. **Novel research question.** Cross-strategy agreement of *self-explanations* is genuinely under-studied. Most work compares external XAI methods or evaluates a single explanation type.

2. **Good experimental hygiene.** Temperature=0, prompt hashing, response caching, seed=42, CF flip verification — these are all best practices.

3. **Thoughtful paradigm taxonomy.** Recognizing that H and RO are the same paradigm (extraction-based) and excluding H–RO from ECS is methodologically sound.

4. **Validity testing infrastructure.** The consensus-core masking test and random baseline comparison are the right idea for bridging agreement → faithfulness.

5. **Coverage analysis is itself a finding.** Counterfactual validity at only 55% is a meaningful result about the reliability of different explanation modalities.

6. **Multi-task evaluation.** SST-2, MNLI, and AG News cover sentiment, NLI, and topic classification — reasonable diversity.

---

## 4. What Needs Improvement

### 🔴 Critical Issues

#### 4.1 Sample Size (N=11 is Unusable)

This is a pilot, so N=11 is fine for debugging the pipeline. But even the 95% CI on ECS ([0.131, 0.227]) is uninformatively wide. For a paper:

- **Minimum:** 100 instances per dataset (300 total)
- **Recommended:** 200 per dataset (600 total)
- **Power analysis:** With ECS std ≈ 0.08 and the effect sizes you're likely to see, you need ~150 instances per condition to achieve 80% power on within-task comparisons

#### 4.2 Single Model

Using only Llama-3.3-70b makes all findings model-specific. Madsen et al. showed faithfulness is heavily model-dependent. Your config already lists 3 models — **you must run all of them**.

Recommended minimum:
- 1 open-source large (Llama-3.3-70b ✓)
- 1 open-source small (Llama-3.1-8b)
- 1 closed-source (GPT-4o or Claude) — tests whether the API-only advantage holds

#### 4.3 ECS Metric Design Flaws

The ECS metric (mean Jaccard over cross-paradigm pairs) has several issues:

**Problem 1: Jaccard ignores token importance magnitude.**
Highlighting assigns scores 1–10, but you threshold to top-5 and then compute Jaccard on the resulting sets. This discards the rich information in the salience distribution. Two tokens with scores 10 and 6 are treated identically after thresholding.

*Fix:* Add **weighted Jaccard** or **cosine similarity** between the full score vectors as a secondary metric.

**Problem 2: Set sizes are artificially fixed.**
H selects top-5 tokens, RO selects 3–5 tokens, CF selects 1–2 tokens. When you compute Jaccard between a 5-token set and a 1-token set, the maximum possible Jaccard is 1/5 = 0.20. This **ceiling effect** means low ECS may be an artifact of mismatched set sizes, not true disagreement.

*Fix:* Normalize for set size. Use the **overlap coefficient** (intersection/min(|A|, |B|)) alongside Jaccard.

**Problem 3: Rationale token extraction is fragile.**
R produces free-text, and your parser extracts tokens that co-occur with the input. This is semantically different from H/RO/CF, which produce explicit token sets. A rationale saying "the negative tone is driven by the word 'terrible'" should credit "terrible", but your parser might also extract "negative", "tone", "driven" if they appear in the input.

*Fix:* Document the rationale parser's precision/recall against human annotations on a small subset.

**Problem 4: ECS conflates parsability failures with low agreement.**
When CF fails to parse (45% of instances!), those pairwise scores become missing, and ECS is computed over fewer pairs. An instance with only H and RO valid gets ECS from just H–R and R–RO, which may have different statistical properties than the full 5-pair ECS.

*Fix:* Report ECS **only for complete cases** as the primary metric (you already compute this). Analyze the relationship between validity rate and ECS separately.

#### 4.4 Highlighting Prompt Leaks Task Instructions

Looking at your MNLI highlighting results ([lines 743-760](file:///c:/OpenSource/Research/outputs/20260616_131003_46f2b811/report.md#L743-L760)), the model assigned high importance to instruction words like "premise" (9), "hypothesis" (9), "contradict" (8), "entail" (8). These are **not input tokens** — they're part of the classification prompt that was re-presented in the explanation prompt.

This is a significant confound. The model is highlighting the *task framing* rather than the *input content*. For MNLI instance 0000, the "top tokens" are `hypothesis` and `premise` — which are instruction words, not evidence.

**Code-level investigation:** The parser's anchoring check in [parser.py:96](file:///c:/OpenSource/Research/src/parsing/parser.py#L96) (`normalizer.is_anchored(word, input_text)`) *should* filter these out — but it depends on what `input_text` is passed. If the full classification prompt (including "premise", "hypothesis", "entail", "contradict") is passed as `input_text`, then those instruction words **pass the anchor check** because they do appear in the prompt. Additionally, `DISCOURSE_WORDS` in [normalizer.py:17-30](file:///c:/OpenSource/Research/src/normalization/normalizer.py#L17-L30) includes `"entailment"` and `"contradiction"` but **not** `"premise"`, `"hypothesis"`, `"entail"`, or `"contradict"` — so these leak through normalization too.

> [!CAUTION]
> **For MNLI and potentially AG News, your highlighting prompt includes the full classification prompt, causing the model to assign high salience to instruction-framing words.** The parser's anchor check doesn't catch this because the instruction words appear in the prompt text passed as `input_text`.

*Fix (two-part):*
1. The explanation prompts should present **only the input text** (premise + hypothesis for MNLI), not the full classification prompt
2. Add `"premise"`, `"hypothesis"`, `"entail"`, `"contradict"`, `"neutral"` to `DISCOURSE_WORDS` in [normalizer.py](file:///c:/OpenSource/Research/src/normalization/normalizer.py#L17-L30) as a safety net

#### 4.5 Counterfactual Flip Verification Failures

In 5/11 instances, the counterfactual either failed to parse or the reconstructed text was re-classified with the **same label** (e.g., `ag_news_test_0001`: CF predicted flip to "World" but re-classification stayed "Sports"). This is concerning:

- CF flip failure could mean the model's self-identified "important" words aren't actually decisive
- Or it could mean single-word substitutions are insufficient for the task
- Either way, **unverified CF outputs should not be included in ECS computation**

Your report already tracks this — good. But the analysis should make this a central finding.

---

### 🟡 Moderate Issues

#### 4.6 No Semantic Similarity Metrics

Jaccard is purely lexical. Two explanations could identify the same *concept* using different surface tokens (e.g., "homer" vs. "hit" in the baseball example both point to the scoring event). Consider adding:
- **BERTScore** between rationale texts
- **Embedding cosine similarity** between highlighted token sets
- **Soft Jaccard** using word embedding distances

#### 4.7 Missing Inter-Run Stability

At temperature=0, you should get deterministic outputs. But API-level non-determinism can occur. Running each instance 3 times and measuring self-agreement would strengthen claims about what ECS captures.

#### 4.8 No Human Baseline

Without human annotations of which tokens are important, you can't tell if low ECS means "strategies disagree" or "strategies agree on the wrong things." Even a small human annotation (20 instances × 3 annotators) would provide a reference point.

#### 4.9 Stopword Removal — Partially Mitigated but Verify

For NLI, function words like "not", "every", "some" can be semantically critical. **Good news:** Your code already handles this — [normalizer.py:83-84](file:///c:/OpenSource/Research/src/normalization/normalizer.py#L83-L84) explicitly preserves `POLARITY_WORDS` (`"no"`, `"not"`, `"never"`, `"nor"`, `"neither"`, etc.) before the stopword filter. Similarly, [parser.py:227-228](file:///c:/OpenSource/Research/src/parsing/parser.py#L227-L228) keeps polarity words in rank ordering.

However, note that `"every"`, `"some"`, `"any"`, `"all"` are **not** in `POLARITY_WORDS` and **will** be removed by stopword filtering. For NLI, these quantifiers are often critical. Consider expanding the preservation list for the MNLI dataset.

---

## 5. Is This a Good Study for a Research Paper?

### Verdict: **Yes, with significant revisions.**

The core research question — *do LLM self-explanation methods agree with each other, and does their agreement predict faithfulness?* — is timely and fills a genuine gap. Here's the assessment:

### What Makes It Publishable

| Factor | Assessment |
|--------|------------|
| **Novelty** | ✅ Strong. Krishna et al.'s disagreement problem applied to self-explanations is new. |
| **Timeliness** | ✅ Hot topic. Madsen (ACL '24), Matton (ICLR '25), CC-SHAP (ACL '24) show intense interest. |
| **Infrastructure** | ✅ The pipeline is well-engineered (415 tests, ablation studies, validity testing). |
| **Practical value** | ✅ An API-only, lightweight agreement metric would be useful for practitioners. |

### What Blocks Publication Right Now

| Factor | Assessment |
|--------|------------|
| **Sample size** | 🔴 N=11 is a pilot, not a study. Need 300–600 instances. |
| **Model diversity** | 🔴 Single model. Need ≥3 models (ideally including one closed-source). |
| **Metric validation** | 🔴 No evidence that ECS correlates with actual faithfulness. The flip test is the bridge, but it's not reported in the pilot. |
| **Prompt confound** | 🔴 Instruction tokens in highlighting need fixing before full run. |
| **Comparison to baselines** | 🟡 No comparison to existing metrics (CC-SHAP, feature necessity/sufficiency). |

### Recommended Paper Structure

1. **Introduction:** The disagreement problem for LLM self-explanations
2. **Related Work:** Madsen, Matton, CC-SHAP, Krishna, Lyu survey
3. **Method:** Four strategies, ECS metric, consensus core, validity test
4. **Experiments:** 3 models × 3 datasets × 200 instances, ablations
5. **Results:**
   - RQ1: Do self-explanation strategies agree? (ECS analysis)
   - RQ2: Does agreement vary by model/task/input properties? (Stratified analysis)
   - RQ3: Does agreement predict faithfulness? (Flip rate correlation)
6. **Discussion:** Implications for practitioners, limitations of ECS

### Target Venues

| Venue | Fit | Notes |
|-------|-----|-------|
| **ACL / EMNLP Findings** | Best fit | Interpretability track, short or long paper |
| **NAACL** | Good | If timing works |
| **AAAI** | Possible | Broader AI audience |
| **TACL** | Stretch | Would need very strong validation results |

---

## 6. Prioritized Action Items

### Before Full Run
1. **Fix the highlighting prompt** — strip instruction/task tokens from the input presented for salience scoring
2. **Add overlap coefficient** alongside Jaccard to control for set-size mismatch
3. **Verify stopword list** — ensure negation words are preserved for MNLI
4. **Add a rationale parser accuracy check** on 20 manually annotated instances

### For the Full Experiment
5. **Run 3 models × 3 datasets × 200 instances** (1,800 total explanation sessions)
6. **Add the consensus-core flip test** as a primary analysis (correlate ECS with flip rate)
7. **Add embedding-based similarity** (BERTScore or cosine similarity) as a secondary agreement metric
8. **Run ablations** already configured (prompt variants, normalization variants, k-values)

### For the Paper
9. **Frame the contribution** as "the disagreement problem for self-explanations"
10. **Compare ECS to CC-SHAP** if model internals are accessible (even on a subset)
11. **Add a small human annotation study** (20 instances, 3 annotators) for ground-truth calibration
12. **Report effect sizes** (Cohen's d) for all comparisons, not just p-values

---

## 7. Code-Level Findings

After reviewing the implementation, here are specific code-level observations:

### ✅ Well-Implemented

| Component | File | Assessment |
|-----------|------|------------|
| Polarity word preservation | [normalizer.py:83-84](file:///c:/OpenSource/Research/src/normalization/normalizer.py#L83-L84) | Negation words bypass stopword removal — correct for sentiment |
| Discourse word filtering | [normalizer.py:17-30](file:///c:/OpenSource/Research/src/normalization/normalizer.py#L17-L30) | Filters meta-words like "classification", "positive", "negative" |
| Anchoring check | [parser.py:96](file:///c:/OpenSource/Research/src/parsing/parser.py#L96) | Only tokens appearing in input text are kept |
| CF validation pipeline | [parser.py:151-196](file:///c:/OpenSource/Research/src/parsing/parser.py#L151-L196) | Thorough: checks anchoring, edit ratio, label flip, single-word constraint |
| H–RO exclusion from ECS | [metrics_calculator.py:54](file:///c:/OpenSource/Research/src/metrics/metrics_calculator.py#L54) | Correctly excludes same-paradigm pair |
| Rationale dep-parse extraction | [parser.py:122-149](file:///c:/OpenSource/Research/src/parsing/parser.py#L122-L149) | Uses spaCy dependency labels to extract meaningful content words |

### ⚠️ Issues Found

| Issue | File | Impact |
|-------|------|--------|
| `DISCOURSE_WORDS` missing MNLI terms | [normalizer.py:17-30](file:///c:/OpenSource/Research/src/normalization/normalizer.py#L17-L30) | `"premise"`, `"hypothesis"`, `"entail"`, `"contradict"` leak through |
| `POLARITY_WORDS` too narrow for NLI | [normalizer.py:35](file:///c:/OpenSource/Research/src/normalization/normalizer.py#L35) | Quantifiers (`"every"`, `"some"`, `"any"`) are filtered as stopwords |
| Highlighting top-k is hardcoded to 5 | [parser.py:105](file:///c:/OpenSource/Research/src/parsing/parser.py#L105) | Creates set-size asymmetry with CF (1-2 tokens) — inflates ceiling effect |
| Rationale fallback is crude | [parser.py:120-121](file:///c:/OpenSource/Research/src/parsing/parser.py#L120-L121) | Without spaCy, falls back to raw word split, which would extract all words |
| `classify_with_mask` returns empty label | [inference_engine.py:149-151](file:///c:/OpenSource/Research/src/inference/inference_engine.py#L149-L151) | Returns `predicted_label=""` — flip detection in validity checker may misfire |

### Ceiling Effect Quantification

With the current set sizes:
- **H** produces 5 tokens (top-5 of salience scores)
- **RO** produces 3–5 tokens (ranked list, typically 5)
- **CF** produces 1–2 tokens (edited words)
- **R** produces variable tokens (dep-parsed content words anchored in input)

The **maximum possible Jaccard** for each pair under typical set sizes:

| Pair | Typical |A| | Typical |B| | Max Jaccard |
|------|---------|---------|-------------|
| H–CF | 5 | 1 | 0.20 |
| CF–RO | 1 | 5 | 0.20 |
| H–R | 5 | 2-3 | 0.40-0.60 |
| R–CF | 2-3 | 1 | 0.33-0.50 |
| R–RO | 2-3 | 5 | 0.40-0.60 |

The **primary ECS** (H–CF, CF–RO) has a hard ceiling of ~0.20, which explains why your pilot's primary ECS is 0.087 — it's working within a very compressed range. This is **not** evidence of genuine disagreement; it's an artifact of CF producing far fewer tokens.
