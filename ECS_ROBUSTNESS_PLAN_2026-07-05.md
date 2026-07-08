# ECS Robustness Plan — 2026-07-05

**Status:** proposal (nothing implemented). **Window:** the 200-instance production run has not
launched yet (config still `sample_size: 10` + `-pilot` suffix), so this is the last cheap
opportunity to change the primary estimand. Everything below is API-cost-free: it recomputes
from evidence sets already persisted per instance.

**Scope:** the ECS metric itself — how per-pair agreement is scored, chance-corrected,
aggregated, and reported. Out of scope: prompts, parsing, normalization, erasure.

---

## 1. Current design (baseline being improved)

- Per instance, 4 strategies (H, R, CF, RO) yield normalized token sets in one shared
  lemma space (normalization v3.0).
- `compute_ecs` ([metrics_calculator.py:143](src/metrics/metrics_calculator.py:143)) = arithmetic
  mean of raw Jaccard over the **5 cross-paradigm pairs** (H–R, H–CF, R–CF, R–RO, CF–RO;
  H–RO excluded as same-paradigm). Computed only when ≥3 strategies valid
  ([run_experiment.py:661](scripts/run_experiment.py:661)); missing pairs are silently dropped
  from the mean.
- Headline = **lift**: `ecs − ecs_random`, where `ecs_random` is a Monte-Carlo expectation
  (2000 sims, seed 42, lru-cached) of Jaccard between uniform random subsets of the instance
  content vocabulary at the observed set sizes ([metrics_calculator.py:7](src/metrics/metrics_calculator.py:7)).
- Secondaries: overlap-coefficient ECS (size-robust), salience-weighted MC null, free-CF
  sensitivity ECS, extraction–rationale / extraction–perturbation composites, complete-case
  ECS as primary estimand.

## 2. Weaknesses to fix

| # | Weakness | Consequence |
|---|----------|-------------|
| W1 | Null is Monte-Carlo (2000 sims, fixed seed) when an **exact closed form exists** | Avoidable noise (~±0.005 per pair), seed dependence, cache complexity; a reviewer can ask "why simulate a hypergeometric expectation?" |
| W2 | Lift corrects for chance but **not for the Jaccard ceiling**. With \|CF\|≈1–2 vs \|H\|≈3–5, J_max = min/max ≈ 0.2–0.33, so a *perfect* CF pair scores lift ≈ 0.15–0.30 while a perfect H–R pair scores ≈ 0.9 | Pairs are incommensurable; the flat mean is dominated by geometry, not agreement. The overlap-coefficient secondary fixes the ceiling but throws away chance correction (tiny sets have high chance overlap). Two half-fixes instead of one estimator |
| W3 | **CF appears in 3 of 5 pairs** (60% of the composite weight) and is the least reliable strategy (0–90% validity across cells in the pilot) | The composite mostly measures the noisiest paradigm; per-cell ECS differences partly re-express CF reliability differences |
| W4 | **Available-pair averaging** silently changes the estimand per instance (2–5 pairs depending on which strategy dropped) | Instances are not measuring the same quantity; complete-case gating helps but the partial-case number that still gets reported is a moving target |
| W5 | **Hard top-k membership** (dynamic k = max(3, round(L/5))) discards H's graded 1–10 salience and RO's ranks; a token at rank k+1 counts exactly 0 | ECS is sensitive to the arbitrary k formula; two strategies agreeing on ranks 1–3 but differing at the k boundary look as disagreeing as genuinely divergent ones |
| W6 | **Exact-match token identity**: R (free-text rationale) can express the same evidence with a synonym; after lemmatization "terrible"≠"awful" | ECS conflates lexical variation with evidential disagreement; deflates R-pairs specifically |
| W7 | Short-vocab instances (V ≤ 20) are **flagged but not handled** in the estimator; chance and ceiling nearly coincide there | Near-degenerate pairs enter the mean with huge leverage |
| W8 | Per-instance ECS is a mean of ≤5 numbers with **no per-instance sensitivity indicator** | One quantized CF pair (J ∈ {0, 1/(a+b−1)} when \|CF\|=1) can swing an instance from "low" to "high" example bins |
| W9 | Weighted null (`expected_random_overlap_weighted`) is MC over 2000 sims, returns None silently on degenerate weights | Same MC objections as W1, plus silent missingness in a pre-registered secondary |

---

## 3. Proposed metric: chance- and ceiling-adjusted ECS (ECS-adj)

### 3.1 Exact hypergeometric null (fixes W1)

For sets of sizes `a`, `b` drawn uniformly without replacement from vocab `V`, the
intersection size K ~ Hypergeom(V, a, b), and Jaccard at K=k is `k/(a+b−k)`. So:

```
E[J] = Σ_{k=max(0,a+b−V)}^{min(a,b)}  hypergeom.pmf(k; V, a, b) · k/(a+b−k)
E[Ovl] = E[K]/min(a,b) = a·b / (V·min(a,b))          # one-liner, exact
```

k ranges over ≤ min(a,b) ≤ ~10 terms — cheaper than one MC draw. Replace
`_expected_random_overlap` with this; delete the seed/cache machinery; keep the MC version
only inside a property test asserting MC ≈ exact (validates both).

### 3.2 Adjusted per-pair agreement (fixes W2, W7)

Kappa-style normalization using both the chance floor and the geometric ceiling:

```
J_max(a,b) = min(a,b) / max(a,b)                      # attained when one set nests in the other
AJ(A,B)    = (J − E[J]) / (J_max − E[J])
```

Properties: **0 = chance**, **1 = maximum agreement achievable given the sizes**, negative =
below chance (floor is −E[J]/(J_max−E[J]), not −1 — document the asymmetry). AJ is
comparable across pairs, instances, datasets, and models regardless of set-size geometry —
it **unifies** the current lift (chance) and overlap-ECS (ceiling) patches into one estimator.

Degeneracy guard: if `J_max − E[J] < ε` (pre-register **ε = 0.10**), the pair returns None
and sets a `degenerate_pair` flag; count flagged pairs per instance and per cell. This turns
the passive `short_vocab` flag into an actual estimator guard.

### 3.3 Paradigm-balanced aggregation (fixes W3)

Aggregate the 5 pair scores into the 3 paradigm-level contrasts first, then average:

```
E–R (extraction vs rationalization)  = mean( AJ(H,R),  AJ(RO,R) )
E–P (extraction vs perturbation)     = mean( AJ(H,CF), AJ(RO,CF) )
R–P (rationalization vs perturbation)=       AJ(R,CF)
ECS_adj = mean( E–R, E–P, R–P )
```

Each paradigm contrast contributes exactly 1/3; CF's effective weight drops from 60% to
33% (and within E–P it is averaged over two extraction partners). This is also cleaner
conceptually: ECS claims to measure *cross-paradigm* consistency, so the unit of aggregation
should be the paradigm pair, not the strategy pair. The existing
`ecs_extraction_rationale`/`ecs_extraction_perturbation` composites become two of the three
components (R–P is the missing third).

### 3.4 Explicit missing-data policy (fixes W4)

- **Primary estimand: ECS_adj_complete** — defined only when all 3 components are defined.
  **[CORRECTED 2026-07-08, see amendment §A below]** The original parenthetical here
  ("requires all 4 strategies valid and no degenerate pairs") was wrong on both counts:
  (i) completeness requires R, CF, and at least one of H/RO valid — NOT necessarily all
  four (E-R and E-P each survive on either of their two extraction partners); (ii)
  code-complete does NOT imply degeneracy-free — a two-pair component (E-R, E-P) stays
  defined when one of its pairs is degeneracy-guarded, so a complete-case instance can
  still contain ≥1 degenerate pair (7/72 in the pilot). Keep the code semantics; this is
  a text correction. This is the current complete-case philosophy, stated at the
  component level.
- **Secondary: available-component ECS_adj** with `n_components` stored — never silently
  renormalized into the headline; reported with its N in its own row.
- Keep the existing MNAR framing: minimal-CF gating drives missingness; the free-CF
  sensitivity ECS (already implemented) is the robustness check and should also be computed
  in AJ form.

### 3.5 Inference target (simplifies the test)

The pre-registered sign-flip permutation test currently tests `mean(ecs_lift) > 0`. With
AJ the null value is **0 by construction**, so the test becomes `mean(ECS_adj) > 0`
directly — same machinery ([statistical_tests.py:111](src/statistics/statistical_tests.py:111)),
same Holm correction, cleaner statement. The cluster-aware bootstrap for pooled CIs applies
unchanged.

---

## 4. Weighted ECS variant (secondary; fixes W5, W9)

Use the graded information instead of hard top-k sets:

- **Weight vectors** over the shared vocab per strategy:
  H = salience/10 over *all* scored tokens (no k cutoff at all);
  RO = geometric decay `p^(rank−1)` with p = 0.9 (tied to the existing RBO parameter);
  R, CF = binary (they are inherently set-valued).
- **Pair score**: Ruzicka similarity (weighted Jaccard) `W = Σ_t min(w_A,t, w_B,t) / Σ_t max(w_A,t, w_B,t)`.
- **Exact permutation null** (no MC): under independent random placement of the weight
  vectors on the vocab,
  `E[Σ min] = (1/V)·Σ_i Σ_j min(a_i, b_j)` and `E[Σ max] = (1/V)·Σ_i Σ_j max(a_i, b_j)`;
  use the plug-in ratio `E[Σmin]/E[Σmax]` (state it is a ratio-of-expectations approximation).
- **Ceiling**: by the rearrangement inequality, max Σ min over alignments is attained by
  sorting both weight vectors descending and aligning — exact, no search.
- Then apply the same AJ normalization → **ECS_adj_weighted**.

This makes the metric *smooth* in the k threshold (a rank-k+1 token contributes slightly
less than rank-k instead of 0) and eliminates the weighted-MC machinery of W9. The existing
salience-∝-sampling null stays as-is as the conservative secondary for the binary metric —
it answers a different question (are all strategies just drawn to the obvious token?) and
should be kept, but raise `n_sims` to 10000 and report its MC standard error.

## 5. Semantic soft-matching sensitivity (optional; addresses W6)

A sensitivity analysis (never the headline, to avoid introducing an embedding-model
confound): soft-Jaccard where unmatched tokens across sets may partially match via a local,
pinned, offline embedding model (e.g. fastText or MiniLM), maximum-weight bipartite matching,
credit only above pre-registered cosine τ = 0.8. Chance correction via label-permutation MC
(exact form is intractable for soft intersections; MC with reported SE is acceptable for a
sensitivity analysis). Purpose: bound how much measured "disagreement" is mere lexical
variation in R. Gate on need: if the pilot rescore shows R-pair AJ ≈ other pairs, skip
this entirely.

---

## 6. Validation before adoption (decision gate)

Adopt ECS_adj **only on simulation evidence, never on which metric flatters the pilot** —
pre-register this rule to close the forking-paths objection.

1. **Planted-agreement simulation** (pure NumPy, no API): generate synthetic instances over
   a (V, a, b) grid with a planted common core of size c. Verify:
   (a) E[AJ] ≈ 0 when c = 0 for **every** geometry (the current lift passes this too);
   (b) AJ is monotone in c at fixed geometry;
   (c) mean AJ at fixed *relative* agreement is **stable across geometries** — this is the
   property raw-Jaccard lift demonstrably fails (a perfect 1-vs-5 pair caps at lift ≈ 0.3),
   and the figure this produces is the justification paragraph for the paper;
   (d) AJ is invariant to padding the vocab with never-selected tokens once E[J] adjusts.
2. **Property tests** in [test_scientific_invariants.py](tests/test_scientific_invariants.py):
   exact null equals MC null within tolerance; AJ = 1 for nested sets; AJ ∈ [floor, 1];
   degenerate guard triggers exactly when `J_max − E[J] < ε`; each paradigm contributes 1/3
   to ECS_adj; ECS_adj is None unless all 3 components defined (complete variant).
3. **Pilot rescore** (run 20260703_124843, zero API): compute old and new per-instance
   scores side by side; report Spearman between them, cell-ranking stability, and how many
   pairs the degeneracy guard removes. Expected outcome: instance ordering broadly preserved,
   CF-pair contribution variance visibly reduced.
4. Document the change in the design.md corrigendum block and in the pre-registration text
   *before* the production run.

## 7. Code touchpoints

| File | Change |
|------|--------|
| [metrics_calculator.py](src/metrics/metrics_calculator.py) | Add `expected_jaccard_exact(a,b,V)`, `expected_overlap_exact`, `adjusted_jaccard(set1,set2,V, eps)`, `compute_ecs_adjusted(explanations, V)` (returns components E–R/E–P/R–P + composite + degenerate flags), `compute_ecs_adjusted_weighted(...)` (Ruzicka + exact nulls). Keep all legacy functions untouched |
| [run_experiment.py:656–715](scripts/run_experiment.py:656) | Compute new fields alongside legacy ones (legacy `ecs`, `ecs_lift`, `ecs_overlap` still populated for pilot comparability) |
| [data_models.py](src/utils/data_models.py) | `InstanceResult`: `ecs_adj`, `ecs_adj_er/ep/rp`, `ecs_adj_n_components`, `ecs_adj_complete` (bool), `n_degenerate_pairs`, `ecs_adj_weighted`; serialization + `from_dict` defaults (back-compat with old JSON). `AggregateMetrics`: matching means/CIs; report headline swapped to ECS_adj with the legacy table retained below it |
| [statistical_tests.py](src/statistics/statistical_tests.py) | Point the pre-registered sign-flip test at `ecs_adj` (H0: mean = 0); Holm unchanged |
| [config/experiment.yaml](config/experiment.yaml) | Pre-registered constants: `ecs_adj.epsilon: 0.10`, `ecs_adj.ro_decay_p: 0.9`, (optional) `semantic_match.tau: 0.8` |
| specs | Corrigendum note: ECS_adj definition, adoption rule, legacy metrics demoted to secondary |

## 8. Phasing

- **Phase A** — core estimator: exact null, AJ, paradigm-balanced aggregation, missing-data
  policy, test-target swap, property tests. (~1 focused day; blocks nothing else.)
- **Phase B** — validation: simulation study + pilot rescore + spec corrigendum. **Gate:**
  production run should not launch with the old headline if Phase A/B land in time — this is
  the last pre-registration window.
- **Phase C** — weighted ECS_adj variant (secondary metric).
- **Phase D** — semantic soft-matching sensitivity, only if the pilot rescore shows R-pairs
  systematically depressed relative to other pairs.

## 9. Explicitly rejected alternatives (for the paper's "design decisions" note)

- **Replace Jaccard with overlap coefficient as primary** — fixes the ceiling but inflates
  chance agreement for tiny sets and loses the union denominator's symmetry; AJ dominates it.
- **Embedding-based ECS as primary** — introduces an unmeasured third model into the
  measurement instrument; acceptable only as a bounded sensitivity analysis.
- **Dropping CF from ECS** — perturbation is one of the three paradigms the metric exists to
  compare; down-weighting via paradigm balancing is the principled version of this instinct.
- **Model-based aggregation (mixed-effects over pairs)** — statistically elegant but opaque
  as a headline metric; the cluster-aware bootstrap already handles the pooled-inference
  concern.

---

# Amendment — 2026-07-08 (pre-registration, BEFORE the 200-run launches)

This amendment closes the estimand/test and support-support gaps found by the independent
pre-200-run audit (`PRE_200RUN_FIX_PLAN_2026-07-08.md`). It is dated and recorded **before**
the frozen production run launches, so it is a pre-registration amendment, not a
post-hoc rationalization. Every number below was recomputed from the pilot's raw
per-instance evidence sets (`outputs/20260707_223054_6c9bce68/instance_results.jsonl`),
not read from a report.

## §A. Primary estimand ↔ primary test alignment (P0.1)

- **Family (a), PRIMARY:** one-sided sign-flip permutation on **complete-case** per-instance
  ECS-adj (all three paradigm components defined), per model×dataset cell, `min_n_for_test`
  gate as configured, Holm-corrected across the run's cells. This certifies the SAME
  population as the primary estimand (§3.4). Prior to this amendment the headline test ran
  on the available-component pool — of the pilot's 211 available-component instances,
  **126 (60%) have exactly one component and 118 are E-R-only**, so that test was, for
  more than half its N, a statement about extraction↔rationalization agreement alone.
- **Family (a2), SENSITIVITY:** the same test on **available-component** ECS-adj (wider N;
  framed as "above-chance agreement across whichever paradigm pairs were elicitable").
  Reported beside (a), never as the headline. Its own Holm family.
- **§3.4 text correction:** see the corrected parenthetical inline above — "complete" means
  all 3 components defined; it does NOT require all 4 strategies valid, nor does it imply
  degeneracy-free (**7/72** pilot complete-case instances contain ≥1 degeneracy-guarded
  pair). The report states this count per run.
- **Pilot evidence it survives:** complete-case pooled mean **+0.4413** (n=72), pooled
  sign-flip **p ≈ 1e-4**; per-cell complete-case at 200/cell projects to sst2 88–128, mnli
  16–56, ag_news 16–104 — every cell clears N=6, three cells stay wide-CI and are flagged.

## §B. Cross-model contrast moved onto the AJ scale (P0.2)

The headline cross-model contrast is now **adjusted-Jaccard** (cross-model side) vs
**per-instance mean ECS-adj** (within side). The prior raw-Jaccard-vs-legacy-ECS version
compared same-strategy pairs (similar sizes, high ceiling) against cross-paradigm pairs
(dissimilar sizes, capped ceiling), so part of the gap was set-size geometry. Recomputed
paired deltas: ag_news **+0.1064** [−0.001, +0.199], mnli **+0.2010** [+0.081, +0.342],
sst2 **+0.1536** [−0.013, +0.347]. Direction survives but the effect ~halves and **sst2's
(and ag_news's) paired CI includes 0** — the "CI entirely above 0 in every dataset"
framing is retired. The raw table is retained as a set-size-confounded descriptive
companion only.

## §C. Free-CF sensitivity in AJ form (P0.3)

The plan §3.4-required "also computed in AJ form" free-CF MNAR robustness check now exists
on the primary scale: complete-case free-CF ECS-adj **n=114, mean +0.4047**, positive in
all 9 cells (available-component n=176, +0.3723), pooled sign-flip p ≈ 1e-4. The primary
conclusion survives removing the minimal-edit gate.

## §D. Null support-closure (P1.1) and its HONEST measured impact

`vocab_tokens |= (H ∪ R ∪ CF ∪ RO)` guarantees the hypergeometric urn contains every token
any strategy selected (64 evidence tokens across 42 instance-strategy combos were previously
outside it; R:41, CF:8, H:7, RO:8). **Measured impact of the literal union-closure (not the
plan's original "6 of 9 cells ≤0.001" estimate, which described a different tokenization
remediation):** the POOLED estimand barely moves (complete-case +0.4413 → **+0.4392**;
available +0.4742 → **+0.4823**, both <0.01) and each population gains exactly one instance
as a larger V resolves a borderline degenerate pair. **Per-cell, however, 6/9 complete-case
cell means move by >0.001** — worst ≈ **−0.055** (deepseek/sst2, N=17) and **+0.062**
(nova/mnli, N=2) — and the largest single-instance ECS-adj swing is ≈ **0.87** on a small-V
instance whose pair crosses the degeneracy guard. **No cell changes sign and no tested cell
changes significance**, so all conclusions are invariant; but this is a genuine
support-closure change on small-vocab instances, not a cosmetic nudge, and is disclosed as
such.

## §E. AJ floor & one-sided-test conservativeness (P1.4)

AJ is not bounded below by −1; the floor is −E[J]/(J_max−E[J]), asymmetric by design (pilot
min pair value ≈ −1.5). The AJ null is therefore left-skewed (bounded above at +1, long
negative tail), making the one-sided sign-flip test for `mean > 0` **conservative** — a
documented property. Negative cell means (possible at tiny N) are reported as "not above
chance at this N", never as a signed below-chance effect size.

## §F. Strata & short-vocab flag re-scoped (P1.2)

Length/vocab strata are now reported on the primary (ECS-adj) scale beside legacy ECS; the
brevity/short-vocab confounds visible in raw ECS are a property of the deprecated metric and
are evidence FOR the adjustment. The `short_vocab` flag is retained for provenance only —
post-P0.1 it flags the majority of SST-2/MNLI instances (median normalized vocab ~9–11), so
the "conservative-estimate filter" framing is dropped.
