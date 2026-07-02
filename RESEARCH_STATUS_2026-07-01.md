# LLM Explanation Agreement Study — Research Status & Roadmap

**Compiled:** 2026-07-01
**Repository:** `C:\OpenSource\Research`
**Model under study:** `llama-3.3-70b-versatile` (Groq API, temperature 0)
**Current phase:** Instrument validation + qualitative pilot (N=10/dataset, single model — deliberate)
**Latest run:** [`outputs/20260701_155518_c2494797`](outputs/20260701_155518_c2494797/report.md) — SST-2 N=7, MNLI N=7, AG News N=0 (Groq daily quota halted)

> **One-line status.** The measurement instrument (data-collection + metric pipeline) is well-engineered, literature-grounded, and — after the 2026-07-01 audit fixes — largely correct. The *scientific deliverable does not yet exist*: the evidence is a ~14-instance single-model pilot, the two headline analyses have no usable data, no significance testing runs, no paper has been generated, and the spec documents describe a larger study than the code implements. **This is a well-built instrument in search of an experiment.**

---

## 1. Executive Summary

This project studies **cross-strategy agreement among an LLM's own self-explanations**. For each input, the same model is asked to explain its classification four different ways — token **Highlighting (H)**, free-text **Rationale (R)**, minimal-edit **Counterfactual (CF)**, and **Rank-Ordering (RO)** — and we measure how much these methods agree on *which tokens matter*. The central metric is the **Explanation Consensus Score (ECS)**, reported as **lift over a random baseline**, not as a faithfulness score.

The intellectual framing is *"the [Disagreement Problem](https://arxiv.org/abs/2202.01602) (Krishna et al. 2022), revisited for self-explanations."* Where prior work compares externally-applied XAI methods (LIME/SHAP/gradients), this study compares methods the model **generates about itself**, on API-only models where internals are inaccessible.

**Where things stand:**

| Dimension | State |
|---|---|
| **Pipeline / engineering** | ✅ Strong. Deterministic, checkpointed, prompt-hashed, curated frozen datasets, dual erasure operators, honest failure handling, 490 passing tests. |
| **Metric correctness** | ✅ Mostly fixed (2026-07-01 audit). Headline ECS/ECS-lift computed correctly; a real pair-key bug in the ECS-composite decomposition was fixed. |
| **Evidence / data** | 🔴 Pilot only. ~14 usable instances, one model. The two intended headline results have **no data**. |
| **Inferential statistics** | 🔴 None actually run. Config declares bootstrap/permutation/Bonferroni; only a bootstrap CI on mean ECS is produced. |
| **Deliverables (paper, specs)** | 🔴 No `paper/draft_paper.tex`; `design.md`/`requirements.md` describe a different, larger study. |
| **Reproducibility of committed numbers** | 🟡 Improving. Latest run's prompts match disk, but the run is uncommitted, AG News is empty, and the config snapshot mislabels prompt files. |

---

## 2. Research Question, Motivation & Positioning

### 2.1 The question

> Do different self-explanation *strategies* elicited from a single LLM agree with each other on which input tokens are important — and if they do, does that consensus localise tokens that are actually causally influential on the model's prediction?

Two sub-questions define the intended contribution:

1. **Does consensus predict causal importance?** When explanation strategies converge on the same tokens (a "Consensus Core"), does erasing those tokens flip the prediction more often than erasing random tokens — and does that gap *grow* with the degree of agreement (ECS-lift tier)?
2. **When is cross-strategy agreement structurally possible at all?** On some tasks (notably AG News topic classification), different strategies draw from *disjoint vocabularies*, so low agreement may be a structural artifact rather than genuine disagreement.

### 2.2 Why it matters

Self-explanations are increasingly used to build trust in LLM decisions, but a stated explanation need not track the computation that produced the label (Turpin et al. 2023; Lanham et al. 2023). If cheap, API-only *agreement* among self-explanation methods were a reliable signal of which tokens matter, practitioners could use it without model internals. The honest finding may instead be that agreement is low, ceiling-bound, and confounded — which is itself worth documenting.

### 2.3 Literature anchors (all web-verified in the 2026-07-01 audit)

| Work | Relationship to this study |
|---|---|
| **Krishna et al. 2022** — *The Disagreement Problem in XAI* | Closest ancestor. This study applies the disagreement framework to *self*-explanations. Defines feature/rank/sign agreement metrics — **which ECS reinvents and should benchmark against.** |
| **Madsen et al. 2024** (ACL Findings) — *Are self-explanations faithful?* | Self-consistency tests; shows faithfulness is model-dependent → motivates multi-model design. CF prompt adopts their "make as few edits as possible" phrasing. |
| **Matton et al. 2025** (ICLR) — *Walk the Talk?* | Gold-standard causal faithfulness. ECS positioned as a lightweight, API-only alternative. |
| **Parcalabescu & Frank 2024** — *CC-SHAP* | Direct competitor consistency metric; needs no input editing (avoids the OOD problem in erasure). **A paper introducing ECS must show what it buys over CC-SHAP.** |
| **Ross et al. 2021** — *MiCE* | Minimal-edit counterfactual definition; CF canonical = minimal flip-verified edit. |
| **Huang et al. 2023** (2310.11207) + Dynamic Top-k (2310.05619) | Self-generated feature attribution; source of the `dynamic_k` set sizing. |
| **DeYoung et al. 2020** — *ERASER* | Comprehensiveness/sufficiency; the `1−k/n` erasure metric is an acknowledged *proxy*, not true comprehensiveness. |
| **ICE 2026** (2603.18579) | Operator choice (mask vs delete) can flip erasure conclusions → both operators are run. |
| **Bastings & Filippova 2020; Hooker et al. 2019 (ROAR)** | The "elephant in the room": erasure pushes inputs OOD; a flip may be distribution shift, not importance. |

**Novelty verdict (from review):** the bare "agreement ≠ faithfulness" point is semi-obvious. The genuine delta rests entirely on the two headline analyses (§2.1) — **which currently have no data.**

---

## 3. Methodology & Pipeline

### 3.1 Pipeline stages

```
config ─► load/curate ─► inference ─► parse ─► normalize ─► metrics ─► [erasure] ─► statistics ─► plots ─► paper
                             │                        │
                          Groq API              NLTK / spaCy
```

1. **Load / curate** — datasets are **frozen, hand-vetted "clean-gold" sets** (200/dataset target), curated with *no model in the loop* (`src/load/curator.py`, decisions in `scripts/curation_vetting.py`). SST-2 (sentiment), MNLI (NLI), AG News (topic).
2. **Inference** — Groq API, temperature 0, top_p 1. Label-only classification (no CoT), then 4 explanation strategies elicited as **follow-up turns after the label** (maximally post-hoc by construction).
3. **Parse** — extract structured evidence per strategy (`src/parsing/parser.py`).
4. **Normalize** — lowercase, strip punctuation, remove stopwords; **lemmatization is OFF** in the live config (`use_lemmatization: false`), though the *anchoring* lemmatizer is always on (an intentional asymmetry). Polarity words (`not`, `never`…) and discourse/task words (`premise`, `hypothesis`, `entail`…) are handled specially.
5. **Metrics** — pairwise Jaccard, Overlap Coefficient, Kendall τ, RBO, ECS, ECS-lift, consensus cores (`src/metrics/metrics_calculator.py`).
6. **Erasure / validity** — a *separate* post-hoc pass (`scripts/run_validity_tests.py`) consumes `instance_results.jsonl` and masks/deletes Consensus-Core vs. a same-size random control, bucketed by ECS-lift tier. **A second consistency axis, explicitly not ground truth.**
7. **Statistics** — bootstrap CI on mean ECS. (Permutation/Wilcoxon/t-test machinery exists but is **not wired into any analysis** — see §6.)
8. **Plots / Paper** — generators exist (`src/plots/`, `src/paper/`) but **produce no committed output**.

### 3.2 The four strategies and the paradigm taxonomy

| ID | Strategy | Elicitation | Evidence extracted |
|----|----------|-------------|--------------------|
| **H** | Highlighting | Graded per-word 1–10 salience (Huang phrasing) | Top-`k` content words, `k = max(3, round(content/5))` |
| **R** | Rationale | Abstractive one-sentence "explain why" | Open-class POS content-word lemmas (NOUN/PROPN/VERB/ADJ/ADV) anchored to input |
| **CF** | Counterfactual | Minimal edit to flip the label (MiCE); MNLI edits **hypothesis only** | Changed tokens (replace/delete opcodes); flip-verified |
| **RO** | Rank-Ordering | Top-`k` most important words, ranked | Ranked content-word list |

**Paradigms:** extraction (H, RO), rationalization (R), perturbation (CF). Because H and RO are the *same paradigm* asked two ways, **the H–RO pair is excluded from ECS** (its high agreement is closer to test–retest reliability than independent corroboration). ECS averages the 5 remaining cross-paradigm Jaccard pairs; a "primary ECS" reports two composites: extraction–rationale `mean(H–R, R–RO)` and extraction–perturbation `mean(H–CF, CF–RO)`.

### 3.3 Key deliberate design decisions (settled — do not re-litigate)

- **ECS = lift over chance.** Raw overlap is confounded by set size and vocabulary, so the reported quantity is `ECS − ECS_random`, where `ECS_random` is a seeded Monte-Carlo expectation conditioned on the actual set sizes and content-vocabulary size.
- **Erasure is a separate consistency axis**, not a faithfulness ground truth (the earlier "erasure = ground-truth anchor" framing was an internal contradiction and was corrected).
- **Confidence axis dropped.** Groq returns 400 "logprobs not supported" for this model; the pre-registered confidence–ECS correlation RQ was removed (not re-attempted via verbalised confidence).
- **Scale is intentionally small now.** N=10/dataset, one model = instrument validation, *built to scale* (config-driven N and model list).
- **Curation uses human judgment, not a model** — Claude hand-vetted 900 candidates, dropping mislabeled/ambiguous/garbled items; decisions are frozen and reproducible.

---

## 4. Current Progress

### 4.1 What is built and working (genuine strengths)

- **Reproducible harness:** temperature 0, seed 42, prompt hashing, response caching, checkpointing, `environment_snapshot.json` + `config_snapshot.yaml` per run, graceful daily-quota halt with partial-output checkpoint.
- **Frozen curated datasets** with datasheets (`data/processed/{ds}_{candidates,decisions,curated}.jsonl` + `_datasheet.json`), model-independent and seeded.
- **Correct headline metric:** ECS/ECS-lift computed from Jaccard over cross-paradigm pairs, with a properly de-confounding Monte-Carlo null.
- **Dual erasure operators** (mask AND delete) per ICE 2026.
- **Honest failure handling:** unparseable re-classifications become `None` (unknown), not silently counted as no-flip; insertion-only CFs treated as non-attributable rather than mislabeled.
- **Careful framing in code/reports:** ECS labelled consistency-not-faithfulness; erasure labelled a second axis.
- **490 passing tests** after the audit fixes.

### 4.2 Latest run — the actual evidence ([`20260701_155518_c2494797`](outputs/20260701_155518_c2494797/report.md))

**Coverage:** SST-2 N=7 (100% accuracy), MNLI N=7 (71%), **AG News N=0** — Groq daily rate limit hit at AG News instance 1/7. 14 usable instances total.

| Metric | Value |
|---|---|
| Mean ECS (all, N=14) | **0.270** |
| Mean ECS (complete cases, N=8) | 0.325 |
| **Mean ECS lift over chance** | **+0.156** (random baseline 0.114) |
| Std / Median ECS | 0.189 / 0.261 |
| Complete cases (all 4 valid) | 8/14 (57%) |
| ECS — correct (N=11) vs incorrect (N=2) | 0.280 vs 0.217 *(both incorrect are MNLI → confounded)* |
| CF canonical (minimal) validity / minimality | 86% / 0.127 edits-per-token |
| CF contrast (free) validity / minimality | 86% / 0.155 |
| Introduced-concept rate (R) | 0.626 |
| Mean CC3 / CC4 size | 1.36 / 0.64 tokens |
| % instances with CC3 / CC4 | 57.1% / 42.9% |

**Per-strategy coverage:** H 93%, R 100%, CF 86%, RO 71%. **Pairwise Jaccard:** H–R 0.296, H–CF 0.330, H–RO 0.465, R–CF 0.193, R–RO 0.255, CF–RO 0.361. **Rank agreement:** RBO(H,RO) 0.213, Kendall τ(H,RO) 0.000. **ECS by length:** short ≤20 words 0.419 (N=4), medium 21–50 words 0.204 (N=9), long >50 words none sampled.

**What is NOT in this run:** no `aggregate_erasure.json` (the headline erasure analysis was not executed), no AG News (headline #2 mechanism), no significance tests, no generated paper.

### 4.3 Study evolution (why so many prior outputs are stale)

The design has been sharpened across several revisions, each invalidating earlier `outputs/`:

- **2026-06-16** — normalization began actually lemmatizing (code made to match config).
- **2026-06-17** — reframed: ECS = lift-over-chance consistency; erasure = separate axis; confidence dropped; misclassified instances included.
- **2026-06-23** — literature realignment: CF = minimal edit (MiCE); dynamic top-k (Huang); POS-based rationale tokens; MNLI hypothesis-only CF. **Curated clean-gold datasets frozen** (reversing the deliberate-misclassified inclusion).
- **2026-06-24** — prompts rewritten to terse Madsen/Huang forms; 4 parser bugs fixed (lemma anchoring, JSON quote-repair, H top-k ordering, CF label quoting). Live re-run left pending (quota exhausted).
- **2026-07-01** — full robustness audit; all engineering fixes applied (§5); a clean re-run executed (this is `20260701_155518_c2494797`).

---

## 5. Fixes Already Applied (2026-07-01 audit — in working tree, uncommitted)

The `git diff` against HEAD reflects a completed robustness pass. All run-blockers and analysis-time engineering fixes landed; the suite is green (490 passed).

| ID | Fix | File |
|----|-----|------|
| **M0** | ECS-composite pair-key bug — `compute_ecs_primary` looked up reversed tuples `("RO","R")`/`("RO","CF")` that never match stored keys, silently collapsing each composite to a single pair. Now uses `("R","RO")`/`("CF","RO")`; regression tests added. *(Headline ECS was unaffected; the decomposition was.)* | `metrics_calculator.py` |
| **M2** | Paired tests (`wilcoxon`, `paired_ttest`) sliced unequal groups to `min_len`, pairing unrelated points → invalid p-values. Now raise `ValidationError`; `permutation_test` seeded (reproducible). | `statistical_tests.py` |
| **M3** | MNLI CF edit-ratio cap of 0.8 (an 80%-edit "counterfactual" is not minimal, breaks cross-dataset comparability) removed → shares the 0.3 default. | `experiment.yaml` |
| **M4** | Random-erasure control was drawn from **all** surface words (incl. stopwords), understating random flip-rate and **inflating the headline CC-vs-random gap**. Now drawn from content words only, count-matched. | `run_validity_tests.py` |
| **M5** | Surface-only erasure vs. lemma-based anchoring → lemma-anchored tokens weren't found in inflected input → silent under-erasure. Both erasure paths now match via WordNet lemma sets. | `redaction_test.py`, `run_validity_tests.py` |
| **Med1** | Report prose contradicted the computation ("Primary ECS averages H–CF and CF–RO"; "Overlap Coefficient is the primary metric"). Aligned narrative to code: Jaccard feeds the headline. | `data_models.py` |
| **Med2** | First-occurrence-only masking left repeated evidence words un-erased. Now erases every occurrence. | `redaction_test.py` |
| **Med3** | Docstring now states `1−k/n` is an ad-hoc first-flip-depth proxy, not ERASER comprehensiveness. | `redaction_test.py` |
| **Med4** | Explicit caveat added above the ECS-lift tier breakdown (data-dependent tertiles, no test, descriptive only). | `show_results.py` |
| **Med5** | `ensure_spacy_available()` fail-fast at startup — prevents silent fallback from POS-lemma to whitespace-split R-token extraction (which would make the R set environment-dependent). | `parser.py` |
| **M1** | Explicit "no significance testing applied" caveat added to the report. *(Full NHST wiring intentionally NOT added — a research-design decision.)* | report generator |
| Minor | `std_ecs` now sample std (ddof=1); deleted dead `validity_checker.py` + test; deleted orphaned `config/normalization.yaml`. | various |

---

## 6. Open Issues & Fixes Still Needed

Ordered by severity. The first block is what a strict NeurIPS/ACL reviewer flagged as **blocking**; these are why the current deliverables "cannot be reviewed as a result."

### 6.1 Blocking (science, not engineering)

1. **Statistical power is ~zero.** Usable N ≈ 14 this run (≤23 historically), single model; complete cases 5–8. The study's own power estimate is ~150 instances/condition for 80% power. Every N≤5 comparison (e.g. correct-vs-incorrect ECS) must be reported as anecdote or dropped.
2. **Single model confounds every finding.** One model (`llama-3.3-70b-versatile`) cannot separate *method* effects from *this model's* idiosyncrasies. Faithfulness/consistency is known to be strongly model-dependent (Madsen, Matton).
3. **The two headline analyses have no data.**
   - Headline #1 (Consensus-Core erasure vs. random by ECS-lift tier): **no `aggregate_erasure.json` for any current-pipeline run.** Only stale erasure output exists, from a pre-literature-alignment run with different prompts.
   - Headline #2 (AG News disjoint-vocabulary mechanism): AG News has run **N=0–3** every time (quota dies first).
4. **No inferential statistics run.** Despite `permutation_tests: 10000` and `bonferroni_correction: true` in config, no significance test touches any comparative claim. The full NHST wiring (which comparisons, what correction) is an open **researcher decision**, not a bug.
5. **Provenance not yet frozen.** The latest run's prompts match disk (good), **but** the run is uncommitted, AG News is empty, and `config_snapshot.yaml` still names `prompts/highlighting.txt` etc. as `prompt_file` while the executed prompts come from the `*_explain.txt` files — a naming/provenance discrepancy to fix before any committed "camera-ready" run.
6. **Spec documents describe a different study.** `.kiro/specs/…/design.md` and `requirements.md` still say: ECS = mean of **6** pairs (code uses 5, excludes H–RO); confidence + Spearman with bootstrap/permutation (removed); `Validity_Checker` masking CC3/CC4 with paired t-test (component **deleted**); **3 models × 3 datasets × ~200** (runs 1×3×10); "ECS differs from 0.5 baseline" (now Monte-Carlo lift). These are read as the contract — they must be rewritten to the *actual* study.

### 6.2 Methodology (major)

- **ECS estimand is unstable under missingness.** ECS averages *whichever* cross-paradigm pairs survive; with CF missing ~14–48% of the time, an H,R,RO instance contributes an extraction↔rationalisation number while an H,CF,RO instance contributes an extraction↔perturbation number — different constructs pooled into one "Mean ECS." **Complete-case ECS (N=8) is the only internally consistent estimand**, but is unusable at this N.
- **Jaccard set-size ceiling.** CF yields 1–2 tokens; H/RO yield ~3–5. `Jaccard(|1|,|5|) ≤ 0.20`. The extraction–perturbation composite sits against a structurally lower maximum than extraction–rationale. Lift-over-chance recentres but does not remove the ceiling; a set-size-invariant metric (overlap coefficient / feature-agreement@k) would make composites comparable.
- **Never pool CIs across datasets.** SST-2 / MNLI / AG News means differ 3–10×; a pooled bootstrap CI describes a mixture, not a population. Report per-dataset or use a mixed model with dataset as a factor.
- **Self-correction loops inflate coverage.** CF gets up to two flip-verification attempts with a "pick a more impactful word" re-prompt; RO gets a hallucination self-correction pass. Reported CF/RO rates are for a *multi-shot search*, not single-shot elicitation → not comparable to Madsen/MiCE. Report single-shot rates separately; treat corrected instances as a stratum.
- **Introduced-concept rate (0.63) is confounded with abstractiveness.** The rationale prompt *demands* prose ("do not list individual words"), so unanchored connective vocabulary is an artifact of the instruction, not evidence of post-hoc rationalisation.
- **CF insertion-only flips are dropped** (difflib captures replace/delete only), biasing the CF set toward substitution and under-representing negation/scope — exactly the phenomena that matter for MNLI.
- **Curation selection bias.** Human-dropping "mislabeled/ambiguous" items can preferentially remove the *hard* stratum, inflating accuracy (SST-2 100%) and plausibly agreement (easy items → one obvious cue → high consensus).
- **"Higher agreement on wrong predictions" is fully confounded** — all incorrect instances are MNLI, and MNLI has higher ECS overall. Stratify within-dataset or drop.
- **Erasure is coarse and OOD-prone.** `[MASK]`/`delete` push inputs off-distribution; with no logprobs there is only a binary flip at 1–5 tokens (no sufficiency/comprehensiveness curve).

### 6.3 Study-design decisions still open (not bugs — researcher calls)

- Full significance-testing design (which comparisons, what correction, whether N=10 is worth testing pre-scale).
- The **circular self-generated + self-verified CF loop** (same model generates and checks the flip) — MiCE/Polyjuice use a held-out predictor. Induces survivorship (only easy-to-flip instances yield a valid CF).
- **No within-method baseline.** Single-sample ECS at T=0 can't separate cross-method disagreement from the model's own generation noise. A same-method resample (run H twice) would floor generation noise for ECS-lift to be read against.
- **No external baseline or human anchor.** ECS is not benchmarked against Krishna et al.'s agreement suite or CC-SHAP, and there is no human token-importance reference (e-SNLI is free for MNLI) to distinguish "methods disagree" from "methods agree on the wrong tokens."

---

## 7. Constraints & Risks

- **Groq free-tier TPD = 100,000 tokens/day.** Each instance costs ~3,400 tokens (classification + 4 strategies + CF flip re-classifications + free contrast + redaction), so a single-model run caps at **~29 instances/day**. The advertised 3 models × 3 datasets × 200 ≈ 6.1M tokens ≈ **~61 days on free tier**. Scaling requires a paid tier, multi-day additive runs, or per-instance pruning — and this must be reconciled with the plan explicitly.
- **Benchmark contamination.** Llama-3.3 has likely seen SST-2/MNLI/AG News; classification accuracy is not a clean measurement.
- **Post-hoc by construction.** Classification is single-shot label-only (no reasoning trace); explanations are elicited *after* the committed label. None of these methods can be process-faithful — there is no process — so ECS is at best consistency among post-hoc reconstructions. The write-up must hold this line everywhere (some metric names — "faithfulness", "validity" — still leak the stronger reading).

---

## 8. Prioritized Next Steps

### 8.1 Immediate (before spending more API budget)
1. **Commit a frozen run.** Freeze code, run once end-to-end on committed prompts, commit matching artifacts. Fix the `config_snapshot.yaml` `prompt_file` naming so the snapshot names the `*_explain.txt` files actually executed.
2. **Finish AG News** additively after quota reset (`python scripts/run_experiment.py --datasets ag_news`) so at least one complete 3-dataset pilot exists.
3. **Run the erasure pass** (`scripts/run_validity_tests.py`) on the current-pipeline `instance_results.jsonl` so Headline #1 has *some* current-code data, even at pilot N.
4. **Rewrite `design.md` + `requirements.md`** to the actual study (5-pair ECS, no confidence RQ, no `Validity_Checker`, 1 model × N=10 pilot, lift-over-chance baseline). Stop calling consistency "faithfulness/validity."

### 8.2 The full experiment (the actual science)
5. **Scale:** ≥150–200 instances/dataset and **≥3 models including one closed-source** (tests the API-only pitch). Budget the ~6M tokens up front (paid tier or multi-day plan).
6. **Compute the headline erasure analysis properly:** defensible null (salience-weighted, not flat uniform), both operators, report with uncertainty. This *is* the contribution.
7. **Fix the estimand:** report complete-case ECS as primary, or switch ECS to a set-size-invariant metric so pooled means are meaningful; never pool CIs across tasks.
8. **Add a within-method resample** (e.g. run H twice) to establish the generation-noise floor.

### 8.3 For the paper
9. **Benchmark ECS** against Krishna et al.'s feature/rank agreement and against CC-SHAP; add a small human-rationale anchor (e-SNLI, ~20 items × 3 raters).
10. **Report single-shot elicitation rates** separately from the coached loops; treat corrected CFs as a stratum.
11. **Wire and run inferential statistics** (after deciding the comparison set + correction), or state clearly that pilot N precludes NHST.
12. **Generate the LaTeX paper** — `src/paper/` exists but has produced no `draft_paper.tex`. Frame as *"the disagreement problem for self-explanations."*
13. **Drop or condition** the confounded claims (wrong-pred agreement, introduced-concept rate) until they can be stratified.

---

## 9. Reference Map

**Key documents**
- [`README.md`](README.md) — pipeline overview, usage. *(Note: claims "415 tests / 21 files" — stale; now 490 tests, `test_validity_checker.py` deleted.)*
- [`REVIEW_strict_reviewer_2026-07.md`](REVIEW_strict_reviewer_2026-07.md) — adversarial NeurIPS/ACL-style review (Reject-in-current-form; blocking issues enumerated).
- [`pilot_study_analysis.md`](pilot_study_analysis.md) — earlier literature positioning + ECS ceiling-effect analysis.
- [`claude_context/audit_findings.md`](claude_context/audit_findings.md) — the 2026-07-01 engineering audit with the applied-fixes log.
- [`.kiro/specs/llm-explanation-agreement-study/design.md`](.kiro/specs/llm-explanation-agreement-study/design.md) — original design (desynchronised — see §6.1.6).

**Code**
- [`scripts/run_experiment.py`](scripts/run_experiment.py) — collection pipeline (classification + 4 strategies, CF/RO self-correction, checkpointing).
- [`scripts/run_validity_tests.py`](scripts/run_validity_tests.py) — separate erasure pass.
- [`src/metrics/metrics_calculator.py`](src/metrics/metrics_calculator.py) — Jaccard/overlap/τ/RBO/ECS/CC + Monte-Carlo null.
- [`src/metrics/redaction_test.py`](src/metrics/redaction_test.py) — `1−k/n` first-flip-depth proxy.
- [`src/parsing/parser.py`](src/parsing/parser.py) — per-strategy evidence extraction, anchoring, CF diff.
- [`src/normalization/normalizer.py`](src/normalization/normalizer.py) — normalization + polarity/discourse handling + lemma anchoring.
- [`src/statistics/statistical_tests.py`](src/statistics/statistical_tests.py) — **not wired into any analysis.**
- [`config/experiment.yaml`](config/experiment.yaml) — live config (N=10, single model, lemmatization off).

**Metric glossary**
| Metric | Meaning |
|---|---|
| **ECS** | Mean Jaccard over 5 cross-paradigm strategy pairs (excludes H–RO). Reported as **lift over a Monte-Carlo random baseline**. |
| **ECS-lift** | `ECS − ECS_random`; the de-confounded headline number. |
| **CC3 / CC4** | Consensus Core: tokens appearing in ≥3 / all 4 strategies. |
| **Overlap Coefficient** | `|A∩B| / min(|A|,|B|)` — size-robust per-pair complement (not the headline). |
| **Kendall τ / RBO** | Rank agreement between H's salience order and RO's ranking (τ needs ≥4 overlapping tokens; often `None` at k≈3). |
| **Flip rate / `1−k/n`** | Erasure proxy: does masking/deleting Consensus-Core tokens change the prediction, vs. a same-size content-word random control? |

---

*Bottom line: the hardest part — an honest, literature-grounded, reproducible instrument — is largely built and, post-audit, largely correct. What remains is to actually run the experiment it was built for: at scale, on multiple models, with the erasure bridge computed and the specifications telling the truth.*
