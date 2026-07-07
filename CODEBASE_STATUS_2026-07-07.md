# Codebase Status & Audit — 2026-07-07

> ## ✅ RESOLUTION UPDATE (2026-07-07, applied)
> The fixes below have since been **implemented and verified** (suite green at 589 tests):
> - **P0.1** vocab_size now counted in the normalized lemma space (`run_experiment.py`).
> - **P0.2** prompt-paraphrase ablation repaired — `highlighting_alt.txt`/`counterfactual_alt.txt`
>   rewritten to the canonical `salience`/`rewritten` schemas, R now extracts evidence
>   (`skip_validation=False`), ablation reads the frozen curated sets, H gets a length-proportional
>   budget, results JSON is written before the (fixed) plot.
> - **P0.3** erasure `erase()` now matches multi-step fixed-point lemmas (`grounds`→`grind`, `pass`→`pa`).
> - **P0.4** free-CF sensitivity normalizes contrast tokens before Jaccarding against H/R/RO.
> - **P1**: top-k RO evidence set is persisted (`rank_ordering_set`) and used by the erasure/cross-model/
>   free-CF consumers; no-spaCy rationale fallback keeps polarity/negation; `analyze_results.py` reads
>   `aggregate_erasure.json` instead of the retired `validity_tests.jsonl`.
> - **Legacy ECS deprecated (Decision D1):** ECS-adj is now the PRIMARY estimand and its sign-flip test is
>   pre-registered family (a) in `report.md` and `show_results.py`; legacy ECS/lift is labeled DEPRECATED.
> - **Cleanup/deletions:** `.kiro/` specs, `scripts/check_failures.py`, `scripts/generate_paper.py`,
>   `src/paper/` (+ its test) removed; `tiktoken` and an orphaned `import string` removed; README setup
>   (spaCy model, boto3) and test count corrected. The plan/review markdown docs are KEPT — code comments
>   cite them by name (e.g. `ECS_ROBUSTNESS_PLAN_2026-07-05.md §3.5`).
>
> Still deliberately NOT changed (researcher launch decisions, not bugs): **D2** config flip
> (`sample_size: 200`, drop `-pilot`) and running the smokes/main/erasure passes. The sections below are
> retained as the original audit record.

**Scope:** full audit of `src/`, `scripts/`, `prompts/`, `config/`, tests, and data artifacts at commit `465cb3f` ("Changes to ECS"), cross-checked against the review trail (`REVIEW_strict_reviewer_2026-07.md`, `FIX_PLAN_2026-07-02.md`, `IMPROVEMENT_PLAN_2026-07-04.md`, `ECS_ROBUSTNESS_PLAN_2026-07-05.md`, `ECS_ADJ_PILOT_RESCORE_2026-07-06.md`) and verified empirically against the pilot run `outputs/20260703_124843_013dd120` (N=90).

**Verdict in one line:** the collection pipeline, metrics, erasure pass, statistics, and provenance machinery are complete, previously-reported defects are genuinely fixed, and the suite is green (590/590) — but **four data-corrupting defects remain (P0 below), the one pre-registered ablation is broken in 3 of its 4 arms, and the ECS-adj adoption decision is unrecorded**. Fix the P0 list and record the two decisions, and the codebase is ready for the 200×3×3 production run.

---

## 1. Completion status by component

| Component | File(s) | Status | Notes |
|---|---|---|---|
| Dataset curation (frozen clean-gold) | `src/load/curator.py`, `scripts/curate_dataset.py`, `data/processed/` | ✅ Complete | 200/dataset curated + datasheets + committed decisions; deterministic, model-free |
| Dataset loading / balanced sampling | `src/load/dataset_loader.py` | ✅ Complete | Curated sets preferred; live sampling fallback works |
| Inference engine (Bedrock Converse) | `src/inference/inference_engine.py` | ✅ Complete | Retry/backoff, throttling vs retryable split, truncation auto-recovery, real token+request accounting |
| Collection pipeline | `scripts/run_experiment.py` | ✅ Complete (1 bug, P0.1) | Concurrent 3-model orchestration, checkpoint/resume, prompt manifest, D5 wrong-pred stratum, verbalized confidence, CF minimal+free dual elicitation, single-shot/coached strata |
| Parsing (5 formats) | `src/parsing/parser.py` | ✅ Complete | Canonical-tokenization CF diff (P0.1 of 07-04 plan), span-restricted MNLI CF, JSON repair, H list-of-pairs schema. One latent bug (P1.6) |
| Normalization v3.0 (shared lemma space) | `src/normalization/normalizer.py` | ✅ Complete | Fixed-point WordNet lemmas, polarity/contraction whitelist shared by all 4 paths |
| Metrics incl. ECS-adj | `src/metrics/metrics_calculator.py` | ✅ Complete | Exact hypergeometric null, AJ with degeneracy guard, paradigm-balanced aggregation, weighted null, cross-model agreement w/ paired contrast |
| Statistics (pre-registered families) | `src/statistics/statistical_tests.py` | ✅ Complete | Sign-flip permutation, Holm, cluster bootstrap, Spearman+τ-b |
| Erasure pass (Headline #1 instrument) | `scripts/run_validity_tests.py` | ✅ Complete (1 bug, P0.3) | Per-model engines, both operators, content-word random control, held-out CF judge, family (b) test |
| Checkpoint / resume | `src/utils/checkpoint_manager.py`, `scripts/resume_experiment.py` | ✅ Complete | Resume drill covered by `tests/test_resume_drill.py` |
| Report generator | `src/utils/data_models.py` | ✅ Complete | All 11 P1.2 + 4 P1.3 report bugs verifiably fixed; ECS-adj tables present |
| ECS-adj validation gate | `scripts/simulate_planted_agreement.py`, `ECS_ADJ_SIMULATION_2026-07-06.json`, `tests/test_scientific_invariants.py` | ✅ All 3 legs passed | Simulation `all_pass: true` (properties a–d); pilot rescore done; property tests in suite. **Adoption decision not yet recorded** (Decision D1) |
| Ablation (prompt paraphrase) | `scripts/run_ablations.py`, `prompts/*_alt.txt` | 🔴 **Broken** | 3 of 4 arms are no-ops (P0.2); crashes at plot step; ignores curated data |
| Visualization | `src/plots/visualization_generator.py` | 🟡 Minimal | 5 generic plots; none of the paper's actual figures (ECS-adj, erasure, CF trade-off, cross-model). See `PAPER_DATA_VIZ_PLAN_2026-07-07.md` |
| Paper generation | `src/paper/paper_generator.py`, `scripts/generate_paper.py` | 🔴 Stub | `generate_paper.py` crashes (no `paper/` dir), writes a 15-line hardcoded skeleton, never uses `PaperGenerator`; `PaperGenerator` has no tables/figures/related work |
| Specs / docs | `.kiro/specs/*`, `README.md` | 🟡 Stale under corrigenda | Bodies still describe the Groq study; README test counts wrong (says 415, actual 590) |
| Test suite | `tests/` (28 files) | ✅ **590 passed, 0 failed** | Verified 2026-07-07 after environment fix (see §4) |

Historic defects from the strict review (§8.1–8.6) and both fix plans were **individually re-verified in the current code** during this audit: all are genuinely fixed (per-model erasure engines, shared lemma space, CF evidence normalization, contracted negations, wrong-pred rendering, prompt manifest, H/RO rank normalization, MNLI span ratio, cluster bootstrap, real API accounting, CF tokenization canonicalization, length-proportional H budget, confidence persistence, resume machinery).

---

## 2. P0 — Bugs that corrupt data or pre-registered results. Fix BEFORE the 200-run.

### P0.1 `vocab_size` is computed in the wrong token space → every chance-corrected number is mis-centered, dataset-differentially

- **Where:** `scripts/run_experiment.py:637–642`.
- **What:** the instance vocabulary is unique **surface** words of the lowercased input (structural labels and pure punctuation removed) — stopwords are **kept** and words are **not lemmatized**. But every evidence set lives in the normalized space (stopwords removed, fixed-point lemmas), and both nulls model "random draws from the vocabulary the strategies select from."
- **Measured impact (pilot, N=90):** stored V is inflated ×1.70 (SST-2), ×1.89 (MNLI), ×1.43 (AG News) vs. the normalized content vocabulary. For a typical (3,4) pair on a short SST-2 input, E[J] = 0.101 at stored V=21 vs 0.205 at corrected V=11 → **ECS-lift overstated by ~0.10 per pair**, and by different amounts per dataset (a dataset-confounded mis-centering — the same class of defect as review §8.2/§8.3).
- **What it contaminates:** `ecs_random`/`ecs_lift` (pre-registered test family (a) becomes anti-conservative), `ecs_adj`/AJ and its degeneracy guard (denominator uses E[J]), and the erasure pass's ECS-lift tier assignment. Raw ECS/Jaccard/overlap are unaffected. The salience-weighted secondary null is *already correct* (its vocabulary is H's normalized salience vector).
- **Fix:** compute `vocab_size = len({normalizer.normalize(w) for w in input words} - {None})` (structural labels excluded). ~5 lines. Then: re-run the pilot rescore (`ECS_ADJ_PILOT_RESCORE`) offline with corrected V — zero API cost, since text is stored per instance — and confirm the degeneracy-guard rate stays sane (E[J] rises, so slightly more pairs will be guarded).

### P0.2 The one pre-registered ablation is broken in 3 of 4 arms (and crashes at the end)

- **Where:** `scripts/run_ablations.py` + `prompts/*_alt.txt`.
- Four independent defects:
  1. **H arm no-op:** `highlighting_alt.txt` returns `{"highlights": [...]}` but `parse_highlighting` only accepts `salience` → every H-alt parse fails → the "H_alt delta" actually measures *removing H from ECS*, not paraphrase sensitivity.
  2. **CF arm no-op:** `counterfactual_alt.txt` returns `{"counterfactual_text": ...}` but `parse_counterfactual` reads `rewritten` → every CF-alt parse fails the same way.
  3. **R arm double no-op:** `parse_rationale(..., skip_validation=True)` returns an **empty evidence list** (`parser.py:244–245`), and `run_ablations.parse_raw_tokens` calls it with `skip_validation=True` for both baseline and variant → R evidence is empty on both sides, R-alt deltas are structurally 0.0 and R never enters the ablation ECS at all.
  4. **Crash after all API spend:** `plot_robustness_analysis(plot_df)` receives a dataframe with column `ECS_delta`, but the plot function hardcodes `y="ECS"` → seaborn `ValueError` → the combined `ablation_results.json` write (which comes *after* the plot) never happens.
- Additionally: the ablation samples **live from HuggingFace** (`sample_balanced`) instead of subsetting the frozen curated sets, so ablation instances ≠ study instances; and `run_explain` uses a flat `max_tokens=512` for H (the P0.2-of-07-04 length-proportional budget was not applied here).
- **Fix:** rewrite the two alt prompts to the canonical schemas (`salience` list-of-pairs; `rewritten`); parse R with `skip_validation=False` (or thread the anchored evidence out); slice instances from `data/processed/{ds}_curated.jsonl` with the same seed; pass the H budget; plot `y="ECS_delta"` (or rename the column); write the JSON before plotting.

### P0.3 Erasure cannot match evidence tokens produced by multi-step fixed-point lemmatization → silent under-erasure in the headline instrument

- **Where:** `scripts/run_validity_tests.py::erase()` (lemma matching via `Normalizer._anchor_lemmas`, single WordNet pass per POS).
- **What:** normalization v3.0 lemmatizes to a **fixed point**, which can take ≥2 steps: real pilot tokens include `grounds → grind` and `pass → pa`. The erasure matcher compares single-pass anchor-lemma sets, and `_anchor_lemmas("grind") = {grind}` does not intersect `_anchor_lemmas("grounds") = {grounds, ground}` — **verified: `erase("the movie grounds itself…", {"grind"}, …)` leaves the text unchanged**, same for `pa` vs "pass".
- **Impact:** CC/strategy erasure silently skips exactly these tokens → flip rates understated → the CC-vs-random gap (pre-registered family (b)) is biased — conservative in direction, but the measurement is wrong, and the random control (which erases surface words it sampled, always matching) is not affected → asymmetric bias between the two arms of the paired test.
- **Fix:** in `erase()`, additionally match an input word when `normalizer.normalize(clean_word) == token` (fixed-point on the input side, mirrors how the evidence token was produced). ~3 lines + a regression test pinning `grind`→"grounds".

### P0.4 Free-CF sensitivity ECS compares unnormalized tokens against normalized sets (review §8.3, resurfaced in the sensitivity path)

- **Where:** `scripts/run_experiment.py` — `cf_contrast_tokens = cf_free_from` stores the raw difflib surface tokens; `compute_free_cf_sensitivity_ecs` then Jaccards them directly against normalized H/R/RO sets.
- **Verified in pilot data:** contrast sets contain `you, in, re, ve, to, scenes, moved…`; normalization would reduce e.g. 14 raw tokens → 8 canonical tokens. Every free-CF pair's union is inflated by unmatchable tokens → **the sensitivity ECS is structurally deflated**, which would falsely read as "conclusions do not survive without the minimal-edit gate" — the opposite of the analysis's purpose.
- **Fix (zero API cost):** normalize inside `compute_free_cf_sensitivity_ecs` (pass the normalizer, keep the stored raw field for minimality accounting), or store a parallel normalized field at collection time. Also note `cf_contrast_minimality` should keep using the raw token count (it does).

---

## 3. P1 — Real but smaller: fix or consciously accept before the frozen run

1. **RO evidence-set construct inconsistency.** ECS uses the top-`dynamic_k` RO subset (`RO_set`), which is **never persisted**; the erasure pass, cross-model agreement, and free-CF sensitivity all rebuild RO from `rank_ordering_tokens` = the **full** ranked list. On short inputs (k=3, model returns 5) the constructs diverge. Fix: persist `RO_set` on `InstanceResult` (or re-apply the `dynamic_k` cap in the three consumers).
2. **Latent negation drop in the no-spaCy rationale fallback** (`parser.py:248–262`): the fallback checks `STOPWORDS` without the `POLARITY_WORDS` carve-out, so "not" is discarded. Latent (production fail-fasts via `ensure_spacy_available`), but it is exactly the review-§8.4 asymmetry and two invariant tests fail whenever the spaCy model is absent. 2-line fix.
3. **`parse_confidence` scale ambiguity:** a numeric reply of `1` (meaning 1/100) is interpreted as probability 1.0 (`conf > 1.0` is the only rescale trigger). Rare at T=0 with the current prompt, but a one-line guard (treat integers ≤ 1 with no decimal point as 0-100 scale, or demand 0–100) removes the tail risk.
4. **`compute_aggregate_metrics` empty-ECS early return** zeroes/defaults every ECS-adj, CF-validity, confidence field for a cell with no legacy-ECS instance even when `ecs_adj` values exist (ECS-adj is deliberately computable below the ≥3-valid gate). Rare at N=200; either compute the full body regardless or accept.
5. **`generate_paper.py` crashes out of the box** (`paper/` does not exist; `open("paper/draft_paper.tex","w")` → `FileNotFoundError`) and ignores `PaperGenerator`. Decide: implement the real asset generator (recommended — see the companion plan) or delete the script and the Req-21 claim.
6. **`analyze_results.py` dead branch:** reads `validity_tests.jsonl`, a filename retired with the deleted `Validity_Checker`; the current erasure pass writes `erasure_instances.jsonl`/`aggregate_erasure.json`. Update or remove.
7. **Model-level aggregates keyed by raw Bedrock id** (`run_experiment.py` "model" level uses `r.model`), inconsistent with `model_dataset` labels ("nova-pro_sst2") — cosmetic, JSON only, report tables unaffected.
8. **`ExecutionSummary.api_failures` never populated** (always 0 next to a real `api_requests_failed`) — cosmetic; remove the line or wire it.
9. **Kendall τ with duplicate normalized RO ranks:** `RO_ranked` may contain the same normalized token at two ranks; `{t: r}` keeps the *last* (worst) rank, while H's `assign_implicit_ranks` keeps the first. Rare; a `setdefault` makes them symmetric.
10. **Stale files:** `scripts/check_failures.py` hardcodes a June run dir (delete); `requirements.txt` lists unused `tiktoken` and has no version pins (Req 22.2 asks for exact versions — pin at freeze time via `pip freeze` into the run's environment snapshot, which already exists, or a `requirements.lock`).
11. **Docs:** `.kiro` spec bodies still describe the Groq study under corrigendum banners; README says "415 tests (100% coverage)" and "500+" in different places (actual: 590; coverage unverified). Rewrite before the camera-ready freeze (FIX_PLAN P5, still open).

**Verified non-issues** (checked, no action): per-model erasure engines; MNLI `secondary_text_field` merge (span-restricted CF active); prompt manifest vs disk; Holm implementation (monotonic step-down, None-safe); cluster bootstrap for pooled CI and confidence correlation; checkpoint dedup/resume idempotence; CF correction-loop bookkeeping; salience-weighted null vocabulary; report SKIPPED-vs-D5 rendering; curated-set determinism.

---

## 4. Environment & reproducibility state

- **Tests:** `590 passed, 0 failed` (2026-07-07). Two prerequisites were missing on this machine and are now installed: `boto3` (in requirements but not installed) and the spaCy model `en_core_web_sm` (a post-`pip` step: `python -m spacy download en_core_web_sm`). Both must be in the README setup block — without spaCy the suite fails 2 invariant tests and the pipeline refuses to start (by design).
- **Git:** working tree clean except one untracked old run dir (`outputs/20260701_101716_690abdc7/`). Commit or delete it before the frozen run so `git_dirty=false` at launch (the snapshot now records this flag).
- **Data:** three frozen curated sets of exactly 200 with datasheets and committed decisions.
- **No committed Bedrock-era production run yet** — the largest current-code evidence remains the 90-cell pilot. This is the known execution gap, not a code gap.

## 5. Open decisions that gate the launch (researcher, not engineering)

| # | Decision | State | Recommendation |
|---|---|---|---|
| D1 | **Adopt ECS-adj as pre-registered primary estimand** | All 3 gate legs passed (property tests green; simulation `all_pass: true` 2026-07-06; pilot rescore 2026-07-06). Code still labels it "candidate"; report headline is still legacy ECS | Adopt. Swap the report headline + promote the ECS-adj sign-flip test to family (a′), demote legacy ECS/lift to "as previously reported" rows, add the spec corrigendum note — all before launch, per the plan's own pre-registration window. Re-run the pilot rescore once after P0.1 (corrected V) so the recorded adoption evidence uses the fixed null |
| D2 | Launch config flip | Not flipped (by design) | `sample_size: 200` ×3 + drop `-pilot` from `experiment.name`; nothing else changes |
| D3 | Pending quota-blocked smokes | H long-input live smoke (206-word MNLI × 3 models), erasure smoke (`--max-instances 6 --trials 3`), fixed-ablation smoke | Run all three the day quota clears, before the 200-run (sequence in the companion plan) |

## 6. Cost/scale reality check (unchanged from the 07-04 forecast)

Main collection ≈ 16k calls / ~4 h; erasure at 5 trials ≈ 30–35k calls / ~9 h; ablation ≈ 2–4k calls / ~1 h. Total ≈ **$10–25** at eu-north-1 on-demand rates. Cost is not a constraint; the P0 list above is.
