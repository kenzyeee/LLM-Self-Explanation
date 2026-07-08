# ECS-adj Pilot Rescore — 2026-07-06

> **ANNOTATION 2026-07-08 (P1.3, corrected-V):** the ECS-adj headline figures in this
> document (≈0.61–0.63) were computed on the **old N=90 pilot with the pre-P0.1 INFLATED
> vocabulary** (V counted over raw surface words, ~1.4–1.9× too large). A larger V lowers
> E[J], which mechanically inflates AJ — so these numbers are systematically **high**. The
> corrected-V reality, measured on the independently-audited N=225 pilot
> `20260707_223054_6c9bce68` (V over normalized content lemmas, plus the P1.1 support-closure
> union), is **complete-case ECS-adj ≈ +0.4413 (pooled) / +0.29–+0.69 per cell** and
> available-component ≈ +0.47 — exactly the downward direction the V correction predicts.
> **For the paper, cite the 20260707 pilot (or the frozen 200-run) as the ECS-adj adoption
> evidence, not the 0.61–0.63 figures below.** This document is retained as the Phase B.3
> methodology record (estimator-vs-estimator on identical raw sets); its *relative* finding
> (ECS-adj behaves as designed vs legacy ECS) stands, only its absolute level is stale.

**Status:** Phase B.3 validation evidence (`ECS_ROBUSTNESS_PLAN_2026-07-05.md` §6.3),
not adoption. **Zero API calls**: every number below is recomputed offline from the
evidence token sets already persisted in pilot run `20260703_124843_013dd120`
(`outputs/20260703_124843_013dd120/instance_results.jsonl`, N=90: 3 models ×
3 datasets × 10 instances, normalization v3.0 — the shared lemma space ECS-adj
requires was already active in this run, so its tokens are directly reusable).

**Important caveat on methodology.** The pilot's *stored* `ecs` field predates
several parser/metric fixes made after 2026-07-03 (see `IMPROVEMENT_PLAN_2026-07-04.md`,
the 2026-07-05 fix pass) and is stale — comparing against it would confound
"old estimator" with "old code," which is not the question this document
answers. Instead, both the legacy ECS and the new ECS-adj below were
**recomputed fresh from the same raw H/R/CF/RO token sets, under the same
current codebase**, so the only thing that differs between the two columns
in every table is the estimator itself.

---

## 1. Headline comparison

| Metric | Value | N |
|---|---|---|
| Legacy ECS (flat 5-pair mean, ≥3 valid) | 0.4125 | 86 |
| Legacy ECS-lift (over uniform-random null) | 0.2985 | 86 |
| Legacy ECS-overlap (size-robust secondary) | 0.7357 | 86 |
| **ECS-adj (available-component)** | **0.6084** | 88 |
| **ECS-adj (complete-case, all 3 paradigm contrasts defined) — candidate primary** | **0.6261** | 24 |
| ECS-adj component: E–R (extraction–rationalization) | 0.5634 | — |
| ECS-adj component: E–P (extraction–perturbation) | 0.8006 | — |
| ECS-adj component: R–P (rationalization–perturbation) | 0.6068 | — |
| Degenerate pairs flagged (`J_max − E[J] < 0.10`) | 1 pair / 1 instance | of 90 |
| Legacy complete-case (`n_valid_strategies == 4`) | 22 instances | — |
| ECS-adj complete-case (`ecs_adj_complete == True`) | 24 instances | — |

Two things jump out immediately:

1. **ECS-adj sits much closer to ECS-overlap than to legacy ECS/lift.** This is
   exactly the mechanism the plan predicted (§2, W2): legacy ECS is dragged down
   by the Jaccard ceiling on size-mismatched pairs (CF is often 1–2 tokens
   against H/RO's 3–9), and ECS-overlap was the ad-hoc patch for that same
   problem. ECS-adj recovers a similar magnitude *without* discarding chance
   correction the way overlap does — it is the single estimator the plan set
   out to build instead of running two incompatible patches side by side.
2. **The complete-case N barely moves** (22 → 24): the paradigm-balanced
   definition (all 3 of E-R/E-P/R-P defined) is close to, but not identical
   with, the old "n_valid_strategies == 4" gate — two instances have all 4
   strategies valid but a degenerate/missing underlying pair, or vice versa.

## 2. Component breakdown — what the paradigm split reveals

| Component | Mean | Reads as |
|---|---|---|
| E–R (H,RO vs R) | 0.563 | Extraction methods and free-text rationale agree moderately above chance |
| E–P (H,RO vs CF) | 0.801 | Once size-adjusted, perturbation evidence tracks extraction evidence closely |
| R–P (R vs CF) | 0.607 | Rationale and perturbation agree moderately |

Under the old flat mean, CF touched 3 of 5 pairs (60% of the weight) and pulled
the composite toward whatever CF's reliability was that instance. Under
ECS-adj, CF's signal is isolated into E-P and R-P (each 1/3 weight) — and the
data shows E-P is actually the *strongest* paradigm contrast (0.80), not the
weakest, once the ceiling stops punishing CF for being a small set. That is a
substantively different empirical story than the legacy composite told, and it
is only visible because the aggregation no longer conflates "CF disagrees"
with "CF's evidence set is just smaller."

## 3. Per-dataset

| Dataset | N | Legacy ECS | Legacy lift | ECS-adj (avail., N) | ECS-adj (complete, N) | E–R | E–P | R–P | Degenerate |
|---|---|---|---|---|---|---|---|---|---|
| ag_news | 30 | 0.4018 | 0.3087 | 0.5412 (30) | 0.5783 (6) | 0.528 | 0.699 | 0.523 | 0 |
| mnli | 30 | 0.4372 | 0.3198 | 0.6633 (30) | 0.6263 (7) | 0.643 | 0.802 | 0.540 | 1 |
| sst2 | 30 | 0.3991 | 0.2663 | 0.6217 (28) | 0.6520 (11) | 0.515 | 0.851 | 0.695 | 0 |

sst2 has the most complete cases (11/30) — its shorter inputs make all four
strategies converge on parseable, valid evidence more often. mnli's single
degenerate pair (below) comes from a singleton rationale set, a known short-R
failure mode independent of dataset.

## 4. Per-model

| Model | N | Legacy ECS | Legacy lift | ECS-adj (avail.) | ECS-adj (complete, N) | Degenerate |
|---|---|---|---|---|---|---|
| deepseek-v3 | 30 | 0.3776 | 0.2757 | 0.5873 | 0.6001 (16) | 0 |
| nova-pro | 30 | 0.4142 | 0.2937 | 0.6283 | 0.8510 (3) | 1 |
| qwen3-235b | 30 | 0.4482 | 0.3277 | 0.6105 | 0.5744 (5) | 0 |

nova-pro's complete-case ECS-adj (0.851) is the highest of any cell in this
report, but N=3 — a single-digit complete-case count at this sample size is
not evidence of a real model effect yet; it is exactly the kind of read the
200-instance run is needed to disambiguate from noise.

## 5. Per model × dataset cell

| Model | Dataset | N | Legacy ECS | Legacy lift | ECS-adj (avail., N) | ECS-adj (complete, N) | Degenerate |
|---|---|---|---|---|---|---|---|
| deepseek-v3 | ag_news | 10 | 0.3634 | 0.2815 | 0.5425 (10) | 0.6450 (4) | 0 |
| deepseek-v3 | mnli | 10 | 0.3681 | 0.2580 | 0.5543 (10) | 0.5072 (4) | 0 |
| deepseek-v3 | sst2 | 10 | 0.4012 | 0.2877 | 0.6650 (10) | 0.6241 (8) | 0 |
| nova-pro | ag_news | 10 | 0.4320 | 0.3316 | 0.5594 (10) | 0.8364 (1) | 0 |
| nova-pro | mnli | 10 | 0.4186 | 0.3139 | 0.7419 (10) | 0.8583 (2) | 1 |
| nova-pro | sst2 | 10 | 0.3899 | 0.2315 | 0.5787 (9) | — (0) | 0 |
| qwen3-235b | ag_news | 10 | 0.4101 | 0.3131 | 0.5218 (10) | 0.0537 (1) | 0 |
| qwen3-235b | mnli | 10 | 0.5325 | 0.3943 | 0.6936 (10) | 0.6389 (1) | 0 |
| qwen3-235b | sst2 | 10 | 0.4061 | 0.2774 | 0.6166 (9) | 0.7265 (3) | 0 |

The qwen3-235b/ag_news complete-case mean (0.054, N=1) is a visible landmine:
a **single** complete-case instance in that cell happens to score near zero,
and with N=1 it swings the cell mean from "high 0.6s available-component" to
"near zero complete-case." This is precisely why the plan's §3.4 missing-data
policy insists the complete-case number is reported *with its N*, never
silently treated as comparable across cells with wildly different complete
counts — at pilot scale, several cells do not yet have enough complete cases
to say anything about the primary estimand on their own.

## 6. Rank stability — does ECS-adj preserve the pilot's ordering?

| Comparison | N (paired) | Spearman ρ | p |
|---|---|---|---|
| Legacy ECS vs. ECS-adj | 86 | **0.768** | 6.0e-18 |
| Legacy ECS-lift vs. ECS-adj | 86 | **0.819** | 6.1e-22 |

This matches the plan's expected outcome (§6.3): "instance ordering broadly
preserved" — strong, highly significant correlation, but *not* ρ≈1. The
disagreement is not noise; it is concentrated in a specific, explainable
subset: instances with a small/singleton evidence set on one strategy, where
the Jaccard ceiling most distorts the legacy number. The eight largest rank
movers (out of 86 scored instances) below are all exactly that pattern.

| Instance | Model / Dataset | Legacy ECS (rank) | ECS-adj (rank) | Δrank | n_valid |
|---|---|---|---|---|---|
| mnli_validation_matched_006801 | nova-pro / mnli | 0.188 (76) | 1.000 (9) | +67 | 3 |
| ag_news_test_003283 | nova-pro / ag_news | 0.194 (75) | 0.670 (38) | +37 | 3 |
| ag_news_test_004983 | deepseek-v3 / ag_news | 0.229 (69) | 0.772 (32) | +37 | 4 |
| sst2_validation_000665 | nova-pro / sst2 | 0.450 (36) | 1.000 (0) | +36 | 3 |
| sst2_validation_000394 | nova-pro / sst2 | 0.476 (33) | 1.000 (1) | +32 | 3 |
| mnli_validation_matched_009361 | nova-pro / mnli | 0.233 (67) | 0.717 (36) | +31 | 4 |
| sst2_validation_000394 | deepseek-v3 / sst2 | 0.348 (50) | 0.911 (20) | +30 | 4 |
| ag_news_test_003454 | qwen3-235b / ag_news | 0.583 (18) | 0.638 (45) | −27 | 3 |

Every upward mover has `n_valid ∈ {3, 4}` with at least one small evidence set
(rationale is almost always the culprit — R is frequently 1–2 tokens). None is
an instance where ECS-adj *invents* agreement that isn't there; each is a case
where a small set is a genuine (near-)subset of a larger one and the ceiling
was hiding that. Case studies below make this concrete.

## 7. Case studies

### 7.1 The single biggest mover: singleton rationale, ceiling masking real agreement

`mnli_validation_matched_006801` (nova-pro): H has 8 tokens, R has exactly
one — `{sell}` — and RO has 4 tokens.

```
H  = {any, blanket, clothe, don't, herdwick, ram's-horn, sell, souveineers}
R  = {sell}
CF = {}                                    (invalid — excluded)
RO = {souveineers, don't, item, sell}
```

- Legacy: `J(H,R) = 1/8 = 0.125`, `J(R,RO) = 1/4 = 0.25` → mean = **0.188**.
  R is a strict subset of both H and RO — perfect agreement given its size —
  yet the flat Jaccard mean reports this as one of the worst-agreeing
  instances in the whole pilot (rank 76/86).
- ECS-adj: `AJ(RO,R) = 1.0` (ceiling attained, chance-corrected).
  `AJ(H,R)` is **flagged degenerate** (`J_max=1/8=0.125`, `E[J]≈0.029`,
  denominator `0.096 < ε=0.10`) — the guard correctly refuses to report a
  ratio for a pair whose ceiling is only marginally above chance. With CF
  missing and one sub-pair guarded out, `E-R = AJ(RO,R) = 1.0` is the only
  surviving component, so `ecs_adj = 1.0` (rank 9/86, `ecs_adj_complete=False`
  since E-P and R-P have no CF to compute).

  This single instance demonstrates **both** halves of the plan's design in
  one example: the *ceiling adjustment* recovers a real signal the flat mean
  destroyed, and the *degeneracy guard* stops that same recovery from being
  applied to a pair too close to its own chance floor to trust.

### 7.2 A near-miss ceiling case: partial, not perfect, nesting

`ag_news_test_003283` (nova-pro): `R={contract, offer}` is fully nested in
`H` (9 tokens) but only partially covered by `RO` (`offer` is missing from
RO's 5 tokens).

```
H  = {ailton, contract, footballer, german, japan, offer, schalke, sportbild, striker}
R  = {contract, offer}
CF = {}                                    (invalid — excluded)
RO = {striker, contract, japan, footballer, reveal}
```

- Legacy: `J(H,R)=2/9=0.222`, `J(R,RO)=1/6=0.167` → mean = **0.194**.
- ECS-adj: `AJ(H,R) = 1.0` (R nests exactly in H — full ceiling), but
  `AJ(RO,R) = 0.341` (R is *not* a subset of RO, so the ceiling isn't fully
  attained — AJ correctly reports partial, not perfect, agreement instead of
  saturating to 1). `E-R = mean(1.0, 0.341) = 0.670`.

  This is the contrast case to 7.1: ECS-adj does not simply push every
  small-set pair to 1 — it only does so when the smaller set is *actually*
  nested. `RO`'s miss on "offer" is real disagreement and stays visible.

### 7.3 Where the paradigm split earns its keep: masked post-hoc rationalization signal

`sst2_validation_000228` (qwen3-235b): four small sets, each strategy telling
a different story.

```
H  = {anchor, honest, strive}
R  = {community, new}
CF = {honest, strive}
RO = {honest, strive, anchor, community}
```

- Legacy ECS = **0.273** — a middling, uninformative number that blends
  three very different underlying relationships into one blur.
- ECS-adj decomposes it: `E-R = 0.020` (R shares almost nothing with
  extraction methods — at chance), `E-P = 1.000` (CF is a perfect nested
  subset of both H and RO — perturbation evidence fully agrees with
  extraction once size is accounted for), `R-P = −0.101` (R and CF are
  *disjoint* — R actively disagrees with the perturbation evidence, scoring
  **below chance**). `ecs_adj = mean(0.020, 1.000, −0.101) = 0.306`.

  The flat legacy mean and the paradigm-balanced composite land at similar
  final numbers here (0.27 vs 0.31) — but only the paradigm split exposes
  *why*: this instance's rationale looks like it may be a post-hoc
  confabulation (R correlates with neither extraction nor perturbation
  evidence), which the single blended number cannot distinguish from
  "rationale is just moderately noisy." This is the diagnostic value the
  plan's §3.3 paradigm-balanced aggregation was designed to unlock, not
  something legacy ECS can express even in principle.

## 8. Degeneracy guard: how often does it fire?

**1 pair out of 90 × 5 = up to 450 attempted pairs** (and only 1 of 90
instances) was flagged degenerate at the pre-registered ε=0.10. The guard is
not a blunt instrument that silently discards a large fraction of the data —
at this sample size it fires exactly where the theory predicts (a singleton
set whose geometric ceiling, `1/|larger set|`, sits within 0.10 of the chance
expectation), and case 7.1 shows the discarded pair really was too fragile to
trust. This is reassuring for the 200-instance run: the guard is a scalpel,
not a filter that will quietly gut the sample.

## 9. Against the plan's decision gate (§6)

The plan pre-registers that ECS-adj is adopted **only** on simulation and
pilot-rescore evidence, never because it "flatters" the data. This document
covers the pilot-rescore leg (§6.3) only — the planted-agreement simulation
(§6.1) and property tests (§6.2, landed in
[`tests/test_scientific_invariants.py`](tests/test_scientific_invariants.py))
are the other two legs of the same gate. Reading this rescore against the
plan's own expected outcomes:

| Plan's expected outcome (§6.3) | Observed here |
|---|---|
| "Instance ordering broadly preserved" | ρ = 0.77–0.82 vs. legacy ECS/lift — strong, not identical |
| "CF-pair contribution variance visibly reduced" | CF's effective composite weight drops from 60% (3/5 flat pairs) to 33% (1/3 paradigm contrasts), and CF's own paradigm contrast (E-P=0.80) is no longer conflated with rationale agreement |
| Degeneracy guard behaves sanely | 1/90 instances flagged, and the flagged pair is a genuine boundary case (§7.1) |
| No silent estimand drift | Complete-case N (24) and available-component N (88) both reported separately, per §3.4 |

**This document alone is not sufficient to adopt ECS-adj as the pre-registered
primary estimand** — the plan requires the simulation study (§6.1) as well,
which has not been run. What this rescore does establish: ECS-adj is
computable end-to-end on real pilot data at zero API cost, behaves as the
plan predicted on every qualitative axis above, and surfaces at least one
concrete case (§7.3) where it is diagnostically richer than the metric it
would replace.

## 10. Recommendation

1. Run the planted-agreement simulation (plan §6.1) before the 200-instance
   launch — this document does not substitute for it.
2. If the simulation passes its four pre-registered properties, adopt
   ECS-adj as the pre-registered primary estimand and demote legacy ECS/lift
   to a secondary "as previously reported" row, per the plan's §7 report
   changes.
3. Regardless of the adoption decision, the per-cell complete-case counts in
   §5 (several cells at N≤3) are a sample-size warning independent of which
   estimator wins — the 200-instance run is needed before any complete-case
   cell mean is treated as a stable estimate.
