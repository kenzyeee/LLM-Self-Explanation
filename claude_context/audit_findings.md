# Pipeline Robustness Audit — Findings

**STATUS (2026-07-01): All fixes below have been applied and the full test suite passes
(490 passed, 0 failed).** See the "Applied fixes" section at the end for what changed and
what remains a judgment call for the researcher rather than an engineering fix.

**Scope:** Verify the collect+compute pipeline of the LLM cross-strategy explanation-agreement
(ECS) study is reliable for a research paper; flag wrong methodology / dead-ends; ground in
literature. **Method:** performed inline (the prior background workflow did not persist across
sessions and produced no output). Every code claim below was read directly; every cited paper was
web-verified.

**One-line verdict:** The collection pipeline is well-built and the *headline* ECS / ECS-lift are
computed correctly and properly de-confounded. But one reported decomposition is bugged, no
inferential statistics actually run, the erasure headline is biased by a mis-specified control and a
surface-vs-lemma asymmetry, and N=10/1-model is pilot-scale — so several comparative claims are not
yet defensible. Most fixes are analysis-time; a few must land before the full collection run.

---

## Confirmed by direct code read (the 3 pre-flagged items — all true)
- **N = 10 / dataset live.** `experiment.yaml` lists `sample_size: 10`; `config_loader.py:48` merges
  `datasets.yaml` (200) with `setdefault`, which is a no-op when the key already exists → live N=10.
- **Lemmatization OFF live.** `normalization.yaml` has no top-level `normalization:` key (only
  `default`/`no_lemmatization`/…), so the merge at `config_loader.py:63` never fires and
  `experiment.yaml`'s `use_lemmatization: false` wins. The **anchor** lemmatizer is always on
  regardless (`normalizer.py:99-106`) — an intentional asymmetry, but see M5.
- **Statistics module is dead + `spearman_rho` hardcoded.** The test functions in
  `statistical_tests.py` have zero production callers (only `tests/` + a `CorrelationResult` dataclass
  import); `spearman_rho`/`spearman_p_value` are hardcoded `0.0`/`1.0` at `run_experiment.py:744,850`.

---

## MAJOR (a reviewer would demand these)

### M0 — `compute_ecs_primary` silently drops the RO pairs (tuple-key ordering bug)
`src/metrics/metrics_calculator.py:154-161`. It looks up `("RO","R")` and `("RO","CF")`, but
`compute_pairwise_agreements` stores keys ordered by the fixed `["H","R","CF","RO"]` index, i.e.
`("R","RO")` and `("CF","RO")`. The mis-ordered lookups return `None` and are filtered out. Result:
- `ecs_extraction_rationale` = Jaccard(H,R) **only** (never averaged with R–RO)
- `ecs_extraction_perturbation` = Jaccard(H,CF) **only** (never averaged with CF–RO)
- `ecs_primary_pairs` caps at **2**, never 4 → the "reduced primary (<2 pairs)" diagnostic at
  `run_experiment.py:627` is miscalibrated.
The reported `mean_ecs_extraction_rationale/_perturbation` are therefore computed on half the intended
pairs, with all RO-involving cross-paradigm agreement absent. **Headline `ecs`/`ecs_lift` are NOT
affected** (they iterate `.items()` on real keys). **Fix:** use order-insensitive keys
(`frozenset`/sorted tuple) or write `("R","RO")`,`("CF","RO")`. **Recompute at analysis time** from the
per-pair Jaccards already stored in `InstanceResult` — no re-collection needed.

### M1 — No inferential statistics are actually computed
`compute_aggregate_metrics` produces point estimates + a bootstrap CI for mean ECS only. There is **no**
significance test on any comparative claim (ECS-lift>0, CC-minus-random, tier trend, correct-vs-incorrect
ECS), and **no multiple-comparison correction**, despite `experiment.yaml` declaring
`permutation_tests: 10000` and `bonferroni_correction: true`. **Fix:** at pilot N, report effect sizes +
bootstrap CIs and explicitly state "no NHST (pilot)"; for the full run, wire the tests in — after M2.

### M2 — Paired tests destroy pairing (latent, in the dead module)
`statistical_tests.py:74-77,103-105`: `wilcoxon_signed_rank_test` / `paired_ttest` slice unequal groups
to `min_len` and pair `group1[:m]` with `group2[:m]` — arbitrary, unrelated pairs → invalid p-values.
Must fix before the module is ever wired in (M1).

### M3 — MNLI counterfactual cap (0.8) breaks minimality & cross-dataset comparability
`run_experiment.py:343` (`cf_max_edit_ratio`, default 0.3; MNLI 0.8 in `experiment.yaml`). An 80%-edit
"counterfactual" is not a minimal edit (MiCE's defining property, arXiv:2012.13985). CF evidence = changed
tokens, so an 0.8 cap yields far larger CF sets for MNLI → mechanically different overlap with H/R/RO →
MNLI ECS is not comparable to SST2/AG News. **Fix:** lower MNLI's cap, or report MNLI CF separately and
don't pool ECS across the 0.3/0.8 split.

### M4 — Random-erasure control is not token-type-matched
`run_validity_tests.py:104` samples the random control from **all** surface words (incl. stopwords
"the","a"); the CC tokens erased are normalized **content** words. Stopword erasure rarely flips → random
flip-rate under-estimated → **CC-minus-random gap (the headline erasure result) is inflated**. **Fix:**
draw the random control from content words only, matched in count. (ICE 2026, arXiv:2603.18579, explicitly
calls for proper random baselines — the control exists but is mis-specified.)

### M5 — Surface-only erasure vs lemma-based anchoring → silent under-erasure
`is_anchored` (`normalizer.py:223-240`) matches evidence to input via WordNet lemma sets (anchor
lemmatizer always on), so "scene" anchors to input "scenes". But `erase()` (`run_validity_tests.py:57`)
and `RedactionTest._redact_tokens` (`redaction_test.py:80`) match **surface** words only → a lemma-anchored
token isn't found in the inflected input → **not erased** → all erasure/faithfulness flip rates are
deflated/biased. **Fix:** erase using the same anchoring logic, or store & erase the matched input surface
forms. *This corrupts the collected erasure evidence — fix before the full run.*

---

## MEDIUM

- **Med1 — Report prose contradicts the computation (integrity).** `data_models.py:709` says "Primary ECS
  averages H–CF and CF–RO"; the code intends H-R,RO-R / H-CF,RO-CF; and (per M0) actually computes H-R /
  H-CF — three different definitions. `data_models.py:680` calls Overlap Coefficient "the primary pairwise
  metric," but the headline ECS/lift use **Jaccard** (`run_experiment.py:490,531`). Align narrative to code.
- **Med2 — First-occurrence-only masking.** `redaction_test.py:83-90` (`remaining.discard`) masks only the
  first occurrence of each ranked token → repeated evidence words survive → comprehensiveness/`1−k/n`
  understated on texts with repeats (AG News/MNLI). Fix: erase all occurrences. *Before full run.*
- **Med3 — `faithfulness = 1 − k/n` is ad-hoc, and only H/RO get it.** `redaction_test.py:64`. Defensible as
  a deletion-rank proxy but not ERASER comprehensiveness/sufficiency (DeYoung 2020); name/justify it as such
  or switch to comprehensiveness AUC. R and CF have no ranking → cross-strategy faithfulness comparison is
  H-vs-RO only.
- **Med4 — Tier trend has no statistical support.** `run_validity_tests.py:203-220`: data-dependent tertiles
  over this run's own lifts, N~10-30 → 1-3 instances/tier. "Gap grows with agreement" is not supportable at
  pilot N. Report descriptively or pre-register thresholds.
- **Med5 — spaCy path is environment-dependent & silent.** `parser.py:156-172` + `_get_spacy` swallows all
  exceptions → R-token extraction silently switches between POS-lemma and whitespace-split paths → the R set
  (hence ECS) depends on the environment. Pin spaCy+model as a hard dependency; fail loudly; log the path.
  *Before full run.*

---

## MINOR / polish
- `normalization.yaml` merge is a broken/misleading no-op (Med-level mechanism, minor effect while the
  inline value already = intended). Either fix the loader to read a named variant or delete the file.
- `std_ecs` uses population std (`np.std`, ddof=0), not sample std.
- `ValidityChecker` (`validity_checker.py`) is **dead** and diverges from canonical `erase()` (it's
  case-sensitive, `_mask_tokens` doesn't lowercase). Remove to prevent accidental use.
- `permutation_test` uses unseeded `np.random.shuffle` (non-reproducible) if ever wired in.
- `compute_jaccard_similarity(∅,∅)=1.0` (`metrics_calculator.py:48`) — guarded out of the ECS path, but a
  latent trap; assert non-empty at call sites.
- Duplicate text cleaners (`pre_clean_text` in `run_experiment` vs `clean_text` in `dataset_loader`) —
  verify they can't diverge (curated text vs prompted text).

---

## STRENGTHS (verified, balanced)
- **Citations check out.** MiCE (arXiv:2012.13985), Dynamic Top-k / Kamp (arXiv:2310.05619), Huang
  self-explanations (arXiv:2310.11207), Rationalization survey (arXiv:2301.08912), ICE 2026
  (arXiv:2603.18579) — all real, correctly attributed, on-point. No fabrications.
- **ECS as lift-over-chance** with a Monte-Carlo null conditioned on set sizes + vocab
  (`metrics_calculator.py:7-45`), seeded and cached — the right way to de-confound raw overlap.
- **Dual erasure operators (mask AND delete)** — directly follows ICE 2026's finding that operator choice
  can flip conclusions. Genuine strength.
- **Honest failure handling** in the erasure pass: unparseable re-classifications → `None` (unknown), not
  silently counted as no-flip (`run_validity_tests.py:96`; `_rate` drops `None`).
- **Careful framing:** ECS labeled consistency-not-faithfulness; erasure labeled a second consistency axis,
  not ground truth (`data_models.py:653`).
- **Insertion-only CF treated as non-attributable** rather than mislabeled (`parser.py:248`) — principled.
- **Bootstrap CI is seeded** (reproducible); consensus-core + complete-case analyses are sensible.

---

## DEAD-ENDS / methodological liabilities (per literature)
- **Self-generated + self-verified CF with an iterative correction loop** (`run_experiment.py:370-419`) is
  circular and induces **survivorship** — only easy-to-flip instances yield a valid CF, so the CF-bearing
  subset is non-random. MiCE/Polyjuice use a separate/fixed predictor. Minimum: report CF-valid rate and
  state the subset is non-random; better: verify flips with a held-out classifier.
- **Single-model, temperature=0, single-sample ECS cannot separate cross-method disagreement from the
  model's own generation noise** (no self-consistency floor; cf. Wang et al. 2022). Add a same-method
  resample (e.g., run H twice) to establish a within-method agreement baseline that cross-method ECS-lift is
  read against.

---

## Priority actions

**Run-blockers — fix BEFORE spending API budget on the full 200/dataset run** (they corrupt data at
collection time):
1. **M5** surface-vs-lemma erasure asymmetry (erase anchoring-consistently).
2. **M4** random-control token-type match.
3. **M3** decide MNLI CF cap / pooling policy.
4. **Med5** pin spaCy + model; fail loudly.
5. **Med2** erase all occurrences (comprehensiveness curves).

**Analysis-time — no re-collection needed:**
6. **M0** recompute `ecs_extraction_*` from stored per-pair Jaccards (and fix the code).
7. **M1/M2** add effect sizes + CIs now; wire tests (after fixing pairing) for the full run.
8. **Med1** reconcile report prose with the computed metric; **Med3/Med4** rename/pre-register; remove dead
   `spearman`/`ValidityChecker`.

**Publication-readiness:** headline ECS/ECS-lift are defensible as a *pilot* descriptive result with CIs.
Erasure "consensus localizes causal tokens better than chance" is **not** defensible until M4+M5 are fixed
and a test is applied. Any "trend across agreement tiers" claim is premature at N=10. The framing is honest
and literature-grounded — the hardest part is already right.

---

## Applied fixes (2026-07-01)

All run-blockers and analysis-time fixes were applied; full suite green (490 passed).

- **M0** `metrics_calculator.py::compute_ecs_primary` — corrected pair keys (`("R","RO")`/`("CF","RO")`,
  not the reversed tuples); added regression tests (there was previously zero coverage of this function).
- **M2** `statistical_tests.py::wilcoxon_signed_rank_test`/`paired_ttest` — now raise `ValidationError` on
  unequal-length inputs instead of silently truncating-and-mispairing; `permutation_test` takes a `seed`
  and uses `np.random.default_rng` (reproducible).
- **M3** MNLI `cf_max_edit_ratio` override removed from `experiment.yaml` — now shares the 0.3 default with
  SST2/AG News. *(User's explicit choice among the two options presented; accepts the risk that MNLI's
  CF-valid rate may drop, since entailment/contradiction flips may need larger edits than sentiment flips
  — watch `cf_canonical_validity_rate` for MNLI on the next run.)*
- **M4** `run_validity_tests.py::random_flip_rate` — random-erasure control now drawn from content words
  only (via `normalizer.normalize`), matching what CC tokens actually are, instead of all surface words
  including stopwords.
- **M5** `redaction_test.py`/`run_validity_tests.py::erase` — both now accept an optional `Normalizer` and
  match erasure targets via WordNet lemma sets (`_anchor_lemmas`), so an evidence lemma ("movie") now
  erases inflected input occurrences ("movies") instead of silently surviving. Wired the live `Normalizer`
  into both `RedactionTest` (run_experiment.py:703) and the erasure pass (run_validity_tests.py).
- **Med1** `data_models.py` report prose — fixed "Primary ECS averages H–CF and CF–RO" (was wrong/
  incomplete even before M0) to correctly describe both composites; fixed "Overlap Coefficient is the
  primary pairwise metric" to state that Jaccard, not Overlap, feeds the headline ECS/lift.
- **Med2** `redaction_test.py::_redact_tokens` — now erases every occurrence of a token, not just the
  first (the `remaining.discard()` once-only bug).
- **Med3** `redaction_test.py::RedactionTest.run` — docstring now states `1-k/n` is an ad-hoc first-flip-
  depth proxy, not ERASER comprehensiveness (no class probabilities available to compute the real thing).
- **Med4** `show_results.py` — added an explicit caveat above the ECS-lift tier breakdown: tertiles are
  data-dependent, no significance test applied, descriptive only.
- **Med5** `parser.py::ensure_spacy_available()` (new) — called at the start of `run_experiment()`;
  raises immediately with install instructions if spaCy/`en_core_web_sm` is missing, instead of letting
  every instance silently fall back to a different (whitespace-split) R-token extraction path.
- **M1** — added an explicit "no significance testing applied" caveat to the generated report next to the
  ECS-lift/gap numbers (full NHST wiring was NOT added — that's a research-design decision about which
  comparisons and correction scheme, left to the researcher; `permutation_test`'s reproducibility bug was
  fixed regardless since it's latent).
- **Minor** — `std_ecs` now uses `ddof=1` (sample std); deleted dead `src/metrics/validity_checker.py` +
  its test (no production caller, diverged from canonical `erase()`); deleted orphaned
  `config/normalization.yaml` (unused by both the config-loader merge and `run_ablations.py`'s own
  hardcoded variants — kept the loader's merge mechanism itself, since a test exercises it directly and
  restructuring the real file risked silently overriding the live `use_lemmatization: false` setting).

**Verified, no action needed:** `pre_clean_text` (run_experiment.py) and `clean_text` (dataset_loader.py)
are byte-for-byte identical — no divergence risk despite the duplication. `compute_jaccard_similarity`'s
`(∅,∅)→1.0` branch is unreachable from production code (`compute_pairwise_agreements` only calls it under
`if set1 and set2`) — left as-is rather than adding a defensive check for a case that can't occur.

**Not fixed — still open, needs a researcher decision, not an engineering fix:**
- Full significance-testing wiring (M1): which comparisons to test, what correction scheme, and whether
  N=10/dataset is even worth testing vs. waiting for the full run.
- The circular self-generated/self-verified CF loop and the lack of a same-method resample to floor
  generation noise (both under "Dead-ends" above) are study-design changes, not bugs.
