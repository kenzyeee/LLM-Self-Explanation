# Pre-200-Run Fix Plan — 2026-07-08

**Scope:** independent audit of pilot run `outputs/20260707_223054_6c9bce68` (225 instances, 25 per model×dataset cell, commit `f780e4f`, code identical to HEAD `8c305f0` — the post-run commit adds only output files, so the pilot reflects exactly the current code). Every number below was **recomputed from the raw per-instance evidence sets** in `instance_results.jsonl`, not taken from the report.

**Verdict in one line:** the pipeline is sound and the pilot numbers are exactly reproducible (zero recompute mismatches; 598/598 tests green; prompt manifest verifies 27/27), but there are **two pre-registration/analysis gaps that a reviewer would exploit (P0.1, P0.2), one missing pre-registered analysis (P0.3), and three never-executed instruments (P0.4)** — all fixable at zero-to-trivial API cost before the 200-run. Nothing found invalidates the pilot's conclusions.

---

## 0. What was independently verified as CORRECT (do not touch)

| Check | Result |
|---|---|
| `vocab_size` recomputed from text via Normalizer (P0.1 of 07-07 audit) | 225/225 match stored values |
| `ecs_adj` recomputed from raw H/R/CF/RO sets via `compute_ecs_adjusted` | 225/225 match (incl. None/complete flags) |
| All 9 cell means (available + complete) and Holm-corrected sign-flip p-values | reproduce the report exactly |
| Test suite | **598 passed, 0 failed** (2026-07-08) |
| Prompt manifest (`prompt_manifest.json`) | 27/27 sha256 verify against disk (text-mode hashing — raw-byte checks will "fail" on CRLF; that is expected, not a defect) |
| Evidence-space wiring | H/R/CF/RO all re-normalized into the shared fixed-point lemma space; RO top-k persisted (`rank_ordering_set`); free-CF normalizes before Jaccard — the four 07-07 P0 fixes are genuinely in |
| Truncations / refusals | 0 truncated strategies, 0 refusals, 100% instance success |
| Wrong-pred stratum sensitivity | excluding wrong predictions leaves all 9 cells significant (p ≤ 0.0048 raw); complete-case mean moves 0.4413 → 0.4599 |
| Statistics machinery | sign-flip has Phipson–Smyth +1 correction; Holm is monotone step-down, None-safe; pooled bootstrap is cluster-aware |
| Provenance | `git_dirty: true` in the snapshot is explained: the dirty state was the in-flight output files themselves, committed unchanged as `8c305f0` |

Also verified as **good news for the metric**: both robustness strata **reverse direction** under ECS-adj relative to legacy ECS, exactly as a working chance/ceiling correction should — see P1.2.

---

## P0 — Fix BEFORE the 200-run (reviewer-facing gaps)

### P0.1 Resolve the primary-estimand / primary-test mismatch, and record it as a pre-registration amendment

- **What:** `ECS_ROBUSTNESS_PLAN_2026-07-05.md` §3.4 declares **complete-case ECS-adj** the primary estimand; §3.5 registers the test as "mean(ECS_adj) > 0" **without naming the population**; the report runs test family (a) on **available-component** ECS-adj. So the headline test certifies a different population than the headline estimand. Measured composition: of the 211 available-component instances, **126 (60%) have exactly one component, and 118 are E-R-only** — the significant available-component result is, for more than half its N, a statement about extraction↔rationale agreement only.
- **Why it matters:** "your significance test is not on your estimand" is a one-sentence rejection at a serious venue, and it is discoverable from the report alone.
- **Fix (zero API cost):**
  1. Add an amendment section to `ECS_ROBUSTNESS_PLAN_2026-07-05.md` (dated, BEFORE the 200-run launches): family (a) = one-sided sign-flip on **complete-case** per-instance ECS-adj per cell, `min_n_for_test: 6` gate as configured; family (a2, sensitivity) = the same test on available-component (larger N; framed as "above-chance agreement across whichever paradigm pairs were elicitable").
  2. Implement (a) alongside (a2) in `scripts/run_experiment.py` (the aggregation block around the existing `ecs_adj_p_value`) and render both in `src/utils/data_models.py`'s report tables, complete-case first.
  3. In the same amendment, fix §3.4's incorrect parenthetical: code-complete means "all 3 components defined", which does NOT imply "no degenerate pairs" — **7/72 pilot complete-case instances contain ≥1 degenerate pair** (a component can rest on one of its two pairs). Keep the code semantics; correct the plan text and state the count in the report.
- **Pilot evidence it will survive:** complete-case pooled mean +0.4413 (n=72), pooled sign-flip p ≈ 1e-4; per-cell complete-case projections at 200/cell from pilot completeness rates: sst2 88–128, mnli 16–56, ag_news 16–104 — every cell clears the N=6 gate, three cells (qwen ag_news ~16, nova mnli ~16, nova ag_news ~24) will still be wide-CI and should be presented as such.

### P0.2 Move the cross-model contrast onto the AJ scale (the current version does not survive its own adjustment)

- **What:** the report's strongest inference ("cross-model same-strategy agreement exceeds within-model cross-paradigm ECS in every dataset — Δ CI entirely above 0 → generic task prior") compares **raw Jaccard** (same-strategy pairs: similar set sizes, same elicitation format, high J ceiling) against **legacy ECS** (cross-paradigm pairs: dissimilar set sizes, structurally capped J). The Δ is partly a set-size-geometry artifact, and the comparator is the *deprecated* metric.
- **Measured on the pilot (both sides recomputed as adjusted Jaccard, same instance vocab):**

  | Dataset | raw Δ (report) | AJ Δ | AJ 95% CI |
  |---|---|---|---|
  | ag_news | +0.2315 | +0.1064 | [+0.0082, +0.1976] |
  | mnli | +0.2482 | +0.2010 | [+0.0745, +0.3456] |
  | sst2 | +0.2297 | +0.1536 | **[−0.0130, +0.3561]** |

  The direction survives everywhere, but the effect roughly halves and **sst2's CI crosses zero** — the report's "CI entirely above 0 in every dataset" sentence is false on the fair scale. Any competent reviewer who recomputes this kills the current framing; conversely, the adjusted version is still a positive, publishable result and will likely tighten decisively at N=200 (~2.8× narrower CIs).
- **Fix (zero API cost):** add an AJ variant inside `MetricsCalculator.compute_cross_model_agreement` (per-pair `adjusted_jaccard` with the instance's V for the cross-model side; per-instance mean `ecs_adj` for the within side), render it as the headline contrast table with the raw version demoted to descriptive companion, and soften the direction sentence to what the adjusted CIs support. Update the paired-contrast wording in `src/utils/data_models.py`.

### P0.3 Implement the free-CF sensitivity in AJ form (a pre-registered analysis that is currently missing)

- **What:** plan §3.4 explicitly requires the free-CF (minimal-edit-gate-free) sensitivity "also … computed in AJ form". The report only has the legacy-ECS form (N=176, 0.3547). Since ECS-adj is now primary, the MNAR robustness check must be on the primary scale.
- **Measured on the pilot (so this is pure wiring, the result is already known):** free-CF ECS-adj complete-case n=114, mean **+0.4047**, positive in **all 9 cells** (+0.29 to +0.50), pooled sign-flip p ≈ 1e-4 — the primary conclusion survives removing the minimal-edit gate. This *strengthens* the paper; it just needs to exist in the pipeline.
- **Fix (zero API cost):** extend the existing `compute_free_cf_sensitivity_ecs` block in `scripts/run_experiment.py` to also compute `compute_ecs_adjusted` with the free-CF set substituted, aggregate per cell, and add a report row/table beside the legacy sensitivity.

### P0.4 Execute the three deferred smokes (D3) — two pre-registered instruments have NEVER run on fixed code

- **Erasure smoke** (`run_validity_tests.py --max-instances 6 --trials 3`): the only erasure artifacts in `outputs/` are from `20260617_141418` — **before** the P0.3 lemma-matching fix. Pre-registered test family (b)'s instrument has zero post-fix execution evidence. The 200-run erasure pass (~30–35k calls, ~9h) must not be its own first test.
- **Ablation smoke** (`run_ablations.py` on a few instances): the 07-07 audit fixed four defects here; no `ablation_results.json` exists anywhere — the fixed script has never completed end-to-end.
- **H long-input live smoke:** the pilot's longest input was **70 words**, but curated MNLI contains up to **206 words** (p95=58). The length-proportional H budget is in the code (12×206+200 = 2672 tokens) and truncation auto-recovery exists, but no model has actually been asked for a 206-item salience list in this study. Run the 206-word MNLI instance × 3 models once before launch; the failure mode being probed is malformed/truncated JSON on long salience lists, which would create length-correlated MNAR missingness at scale.
- **Cost:** all three together ≈ a few hundred calls, minutes of wall-clock.

---

## P1 — Cheap hardening (do with P0; none blocks launch by itself)

### P1.1 Close the null's support gap: `vocab_tokens |= (H ∪ R ∪ CF ∪ RO)` (~1 line)

- **Measured:** 64 evidence tokens across 44 instance-strategy combos live **outside** the vocabulary the hypergeometric null draws from (R:41, CF:8, H:7, RO:8; overwhelmingly ag_news). Mechanism: vocab is built from whitespace tokens with edge-punctuation stripping, so AG News' glued ellipses ("Senate...Supercomputer" = one vocab token) and possessives ("turkey's") keep the atoms the strategies actually select ("senate", "supercomputer", "turkey") out of the urn, and V is occasionally undercounted.
- **Measured impact of fixing:** re-scoring the whole pilot under an aligned tokenization moves 6 of 9 cells by ≤0.001 ECS-adj; worst case ±0.04–0.09 on N≤7 MNLI complete-case cells; all 9 conclusions (sign + significance) invariant. So this is a **wart, not a corruption** — but unioning the evidence sets into the vocab guarantees the support assumption *by construction*, costs one line in `scripts/run_experiment.py` (vocab block, ~line 654), and removes an entire reviewer attack ("your null's urn does not contain the tokens the strategies picked").
- Record it in the P0.1 amendment; re-run the offline pilot rescore once to log the deltas (zero API cost).

### P1.2 Re-do the robustness strata on the primary metric (the current tables support the deprecated one and their story is stale)

- **Measured — the gradients reverse under ECS-adj:**

  | Stratum | legacy ECS (report) | ECS-adj (recomputed) |
  |---|---|---|
  | short inputs (≤20 w, n=60) | 0.4233 (highest) | 0.4047 (lowest) |
  | medium (n=143/144) | 0.3649 | 0.4943 |
  | long (>50 w, n=8) | 0.3758 | 0.6374 (highest, tiny N) |
  | short-vocab (n≈148) | 0.4032 (higher) | 0.4575 (lower) |
  | normal-vocab (n≈64) | 0.3323 | 0.5119 |

  The report's caveats ("ECS may partly reflect brevity"; "short-vocab yields inflated/trivial ECS") describe the deprecated metric and are **evidence FOR ECS-adj** (the adjustment removes exactly those confounds). Add ECS-adj strata tables to `compute_aggregate_metrics` + the report; rewrite the two caveat sentences to say the confound exists in raw ECS and disappears under adjustment.
- **Retire or re-calibrate the `short_vocab` flag as a "filter":** post-P0.1 it flags 72/75 sst2 and 63/75 mnli instances (median normalized vocab: sst2=9, mnli=11 — the threshold of 20 was calibrated for surface vocab). As a filter it is useless on 2 of 3 datasets and the "conservative estimate" framing is now effectively an AG-News-only estimate wearing a robustness costume. Recommended: keep the flag for provenance, drop the "conservative estimate" claim, and report the continuous vocab_size↔ECS-adj association instead (or per-dataset thresholds).

### P1.3 Refresh the ECS-adj adoption evidence on corrected-V data

- `ECS_ADJ_PILOT_RESCORE_2026-07-06.md` (cited as Decision D1's gate evidence) was computed on the old N=90 pilot with the **inflated pre-P0.1 vocab** — its headline (0.61–0.63) is systematically high versus the corrected-V reality (0.44–0.47 in this pilot, exactly the direction the V correction predicts). Re-run the rescore offline against `20260707_223054_6c9bce68` (or annotate the old doc and cite this pilot as the adoption evidence). Zero API cost; keeps the D1 decision trail honest.

### P1.4 Document the AJ floor and the test's conservativeness

- **Measured:** AJ is not bounded below by −1: pilot minimum pair value **−1.5**, 7 pair values < −1, most negative per-instance ECS-adj = −1.13; 106/629 scored pairs (16.9%) are negative. The floor is −E[J]/(J_max−E[J]), asymmetric by design.
- Under H0 the AJ distribution is left-skewed (bounded above at 1, long negative tail), which makes the one-sided sign-flip test **conservative** for "mean > 0" — worth one paragraph in the plan/report so the asymmetry is a documented property, not a discovered one. Negative *cell means* (nova-pro mnli complete-case −0.10 at N=2) must never be interpreted as "below-chance magnitude" without noting the floor asymmetry. Optional: report a clamped-at-−1 companion mean as a secondary presentation (pre-register the clamp if adopted).

---

## P2 — Paper-writing hygiene (during/after the 200-run; not launch-gating)

1. **Confidence ceiling:** 109/225 instances answered exactly 0.95; observed range 0.7–1.0. The confidence↔ECS association is a restricted-range/heavy-ties estimate and should be framed as an expected-null descriptive (τ-b already reported — good); do not let it drift into a "calibration finding".
2. **Non-nested Ns:** n_ecs=212 vs n_ecs_adj=211 are overlapping but non-nested populations (ECS-adj is computable below the ≥3-valid gate; all-degenerate instances lose ECS-adj while keeping legacy ECS). One clarifying sentence in the report prevents a "your Ns don't reconcile" note.
3. **Component-subset caveat:** available-component E-P (0.5633) vs E-R (0.4710) means are computed over different instance subsets (E-P exists only where a minimal CF was found — 43%, heavily selected). Add an explicit sentence; the complete-case table is the clean within-instance comparison.
4. **Docs at freeze:** README test count (now 598), `.kiro` corrigenda bodies, `requirements.txt` pinning via the run's environment snapshot (already captured per-run — snapshot is the authority).

---

## Launch sequence (after P0/P1 land)

1. Commit fixes; suite green; working tree clean at launch (snapshot `git_dirty: false`).
2. Run the three smokes (P0.4). Gate: all pass.
3. Re-run the offline pilot rescore (P1.1/P1.3 evidence) and commit the amendment + rescore refresh together — this is the close of the pre-registration window.
4. Launch main collection (200×3×3 ≈ 16k calls / ~4h; pilot retry rate was 16.8% — throttling headroom exists, expect wall-clock closer to 5–6h).
5. Erasure pass (~30–35k calls / ~9h), then the fixed ablation on its 50-instance subset.
6. Regenerate report; verify the new tables (complete-case test primary, AJ cross-model contrast, free-CF AJ row, ECS-adj strata) render from real data.

## What this plan does NOT claim

- It does not claim the 200-run results are pre-ordained: three complete-case cells will land at N≈16–24 and stay wide; MNAR from CF validity (8%–80% per cell) remains the study's structural limitation, mitigated — not eliminated — by the free-CF sensitivity and the availability-vs-complete dual reporting.
- It does not claim acceptance anywhere. It claims: after P0.1–P0.4, every number in the report is reproducible from raw artifacts, every headline test matches its estimand, every pre-registered analysis exists, and the known attack surfaces (null support, metric floor, set-size confounds, selection effects, format confound in the cross-model contrast) are measured and disclosed rather than discoverable.
