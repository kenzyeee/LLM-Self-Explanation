# Adversarial ML-Reviewer Review — Prompts & Method (2026-07-08)

**Scope:** an ACL/EMNLP-caliber reviewer's read of the *prompts* and *measurement design*
(not the statistics — those were audited separately in `PRE_200RUN_FIX_PLAN_2026-07-08.md`
and are strong). The question asked: what would a tunnel-visioned team, deep in the metric
mechanics, MISS that gets the paper rejected or the results dismissed? The statistical
plumbing here is unusually careful; the exposure is almost entirely **construct validity and
missing baselines** — the layer the ECS-adj work does not touch. Findings are ordered by
rejection risk. Each is tagged **[cheap]** (prompt/analysis change, little/no new API) or
**[scope]** (adds an experimental arm).

Numbers below are recomputed from pilot `20260707_223054_6c9bce68` (N=225).

---

## R1 — No within-strategy test–retest baseline: the ECS-adj number has no denominator. [scope, CRITICAL]

Every strategy is elicited **once, at temperature 0**. The study then reports how much
*different* strategies agree (ECS-adj ≈ +0.44 complete-case). But there is no measurement of
how much the *same* strategy agrees **with itself** on a re-elicitation. Without that ceiling,
ECS-adj is uninterpretable:

- If same-strategy test–retest AJ ≈ 0.75, then cross-method 0.44 is a real, large divergence
  between paradigms — an interesting result.
- If same-strategy test–retest AJ ≈ 0.45, then cross-method 0.44 is **just the model's own
  elicitation noise** — the paradigms don't disagree at all; explanations are simply unstable,
  and the entire "cross-paradigm consistency" framing collapses.

Temperature 0 makes this worse, not better: it *hides* the instability behind a single
deterministic draw instead of characterizing it, so the reported consistency is a
single-sample artifact. The cross-model same-strategy analysis (P0.2) is the closest thing in
the study, but it varies the *model*, not the *draw* — it cannot serve as the within-model
self-consistency floor.

**A reviewer will ask for this in the first round and it is close to a required baseline for
any "consistency" claim.** Fix: for one model, re-elicit each strategy a second time (either
temperature > 0, e.g. T=0.7 with a handful of samples, or a paraphrased prompt — the ablation
`*_alt.txt` machinery already exists and is nearly this) and report same-strategy AJ as the
ceiling against which cross-strategy ECS-adj is read. This is the single highest-value
addition before or alongside the 200-run. It reframes ECS-adj from an absolute number into
"fraction of the achievable self-consistency ceiling," which is both defensible and novel.

## R2 — The predicted label is revealed in every explanation prompt → you are measuring consistency of *post-hoc rationalizations*, and the anchor can inflate agreement. [cheap framing + optional control]

All four explanation prompts, plus confidence, open with `The text was classified as:
{predicted_label}`. So the model is never introspecting a computation — it is **justifying a
label it is handed**. Two consequences:

1. **Claim ceiling.** This is consistency of *post-hoc rationalization conditioned on a given
   answer*, which the self-explanation literature treats as confabulation-prone. The paper
   must not drift into language implying the model reveals its own reasoning; it reveals how
   it rationalizes a supplied label. (The repo is disciplined about "not faithfulness" — good —
   but the label-anchor point is a distinct, sharper caveat and should be stated explicitly.)
2. **Agreement inflation.** Handing every strategy the same label can make them all anchor to
   the same obvious cue token(s), manufacturing agreement that would not survive if each
   strategy had to commit to a label itself. This biases ECS-adj **upward** and is a confound
   the chance/ceiling adjustment does **not** remove (adjustment corrects set-size geometry,
   not shared-anchor priming).

**Cheap:** state the post-hoc/anchor framing as a first-class limitation. **Optional control
[scope]:** a no-label ("explain what drives the classification, then also give the label")
condition on a subset to bound the anchoring effect.

## R3 — Counterfactual evidence is a different KIND of object than H/R/RO; low E-P / R-P agreement may be *definitional*, not an inconsistency. [cheap framing, important]

H/RO/R identify tokens that **support** the current label. CF identifies tokens that, when
**changed**, **flip** it. These diverge for principled reasons: in "not good," the flip hinges
on "not," but the salient sentiment token is "good"; the minimal edit and the top attribution
are routinely different features. Forcing "words you'd change to flip" and "words that
support" into the same token-set Jaccard and calling a low value "inconsistency" is a category
issue. Paradigm-balancing (E-P, R-P weighting) redistributes weight but does not make the two
constructs commensurable — the E-P/R-P components may be measuring an *expected* divergence.

Pilot symptom: CF median evidence-set size is **2**, and **36% of valid CFs have ≤1 token** —
so CF-pair Jaccard is heavily quantized (|CF|=1 ⇒ Jaccard ∈ {0, 1/(a+b−1)}), and CF is the
paradigm driving most of the "disagreement." The paper needs an explicit argument that
minimal-edit sets and attribution sets *should* coincide for these tasks, or it must frame the
E-P/R-P gap as "attribution vs. intervention diverge," not as the model being inconsistent.

## R4 — No anchor to gold human rationales, though e-SNLI provides them for NLI. Agreement ≠ correctness. [cheap–moderate, external-validity]

ECS measures inter-method agreement and nothing else. High agreement is fully consistent with
**all methods converging on the same spurious shortcut** (sentiment/topic keyword, negation,
dataset artifact). The study disclaims faithfulness, which is honest, but a reviewer will note
that a cheap sanity anchor is *available and unused*: **e-SNLI** ships human highlight
rationales for SNLI/NLI, directly comparable to the H/RO token sets on the MNLI arm. Even a
small overlap-with-gold number ("H rationales recover X% of e-SNLI human highlights") would
convert "the methods agree with each other" into "the methods agree, and they agree *with
humans* to degree X" — a far stronger, harder-to-dismiss result. Not using it reads as
avoiding the one available ground truth.

## R5 — Prompt design shapes the metric: one-sentence R and score-every-word H. [cheap]

- **R is capped at one sentence** ("In one sentence, explain why…"). This structurally forces
  a small R evidence set (pilot median 4 content words) biased toward the single most salient
  concept, then that set is compared under a size-geometry-correcting metric — the prompt is
  pre-determining one input to the correction. Worse, R's set is produced by an *unexamined
  extraction step* (spaCy content-word pull from free prose), so the R token set is a
  heavily-processed, synonym-fragile object. Combined with **exact-token identity after
  lemmatization** (plan W6, deferred): "awful" ≠ "terrible," "great" ≠ "excellent," so R pairs
  are deflated for lexical, not evidential, reasons. This is a real R-specific bias, and W6 is
  currently gated on "if R looks depressed" — it should be checked, not assumed benign.
- **H scores every word 1–10**, then a threshold/top-k turns the graded vector into a set; the
  cutoff choice does hidden work and is the exact brittle long-output failure mode the 206-word
  smoke probes. Consider reporting ECS-adj sensitivity to the H selection threshold.

**Cheap fixes:** allow R up to ~2–3 sentences (or don't cap), and *run* the W6 semantic-match
sensitivity on the pilot rather than gating it — if R-pair AJ is systematically below the
others, the deflation is real and must be disclosed or corrected.

## R6 — The verbalized-confidence RQ is effectively dead at temperature 0. [cheap — cut or reframe]

Pilot: **109/225 answers are exactly 0.95**; the top three values (0.95/0.98/0.85) cover most
of the sample; range 0.70–1.00. At T=0 on near-solved tasks the models emit a near-constant
high confidence, so any confidence↔ECS association is a restricted-range/heavy-ties estimate
with almost no signal (P2.1 already flags the ceiling — this goes further: the RQ cannot be
answered by this design). Reviewers dislike a reported RQ that the data cannot address.
Recommend: **drop it from the headline** or reframe explicitly as "verbalized confidence is
non-informative under greedy decoding on these tasks" (itself a small finding), rather than
presenting a correlation table that invites "your confidence variable is a constant."

## R7 — Datasets are near-solved and single-cue-dominated; the interesting regime is under-sampled. [scope / framing]

SST-2, AG News, and (largely) MNLI are high-accuracy for these models, and many instances have
one obvious decisive token (sentiment word, topic keyword). On such instances all methods
trivially point to the same token, so ECS-adj is inflated by easy single-cue cases (the
`short_vocab` mass is a symptom — post-P0.1 the ≤20-token flag catches the majority of SST-2
and MNLI). The genuinely interesting regime — multi-feature reasoning where methods *could*
meaningfully diverge — is thin. At minimum, report ECS-adj stratified by instance difficulty
(model confidence proxy is dead per R6; use input length / number of content cues / a
difficulty tag), and consider a harder task (e.g., a reasoning or long-document classification
set) so the headline isn't dominated by "everyone agrees the movie word is 'terrible'."

## R8 — MNAR from CF validity makes the complete-case estimand a ~30% selected subsample. [known, but the biggest external-validity threat]

CF validity is **43% overall** (97/225) and far lower on multiclass/MNLI, so the complete-case
primary estimand rests on a heavily selected minority where a minimal flip was *easy to find* —
i.e., the easy, high-agreement instances. The free-CF AJ sensitivity (P0.3, +0.4047 complete)
mitigates but does not eliminate this: free-CF is still gated on the free rewrite flipping.
This is acknowledged in the plan, but it is the dominant threat to any general claim, and the
paper should lead with availability-vs-complete dual reporting rather than the complete-case
number alone. (Handled structurally by the dual reporting added in P0.1 — this is a
framing/emphasis note, not a defect.)

---

## Priority summary

| # | Issue | Rejection risk | Cost | Recommendation |
|---|-------|----------------|------|----------------|
| R1 | No within-strategy test–retest ceiling | **Highest** | scope (1 model, re-elicit) | Add before/with the 200-run; reframe ECS-adj vs the self-consistency ceiling |
| R2 | Label leak → post-hoc rationalization + agreement inflation | High | cheap framing (+optional control) | State plainly; optional no-label control |
| R3 | CF (flip-set) ≠ attribution (support-set) | High | cheap framing | Argue commensurability or reframe E-P/R-P as attribution-vs-intervention |
| R4 | No gold-rationale anchor though e-SNLI exists | Med-High | cheap–moderate | Add H-vs-e-SNLI overlap on the NLI arm |
| R5 | One-sentence R + exact-match deflation | Medium | cheap | Uncap R; actually run the W6 semantic-match sensitivity |
| R6 | Confidence RQ dead at T=0 | Medium | cheap | Cut or reframe as a null finding |
| R7 | Easy, single-cue datasets | Medium | scope/framing | Difficulty-stratify; consider a harder task |
| R8 | CF-MNAR selects ~30% subsample | Structural | none (framing) | Lead with dual availability/complete reporting |

**One-line verdict:** the metric and statistics are publication-grade after the P0/P1 fixes;
the *design* has one likely-required missing baseline (R1, self-consistency ceiling) and two
framing exposures (R2 label anchor, R3 CF-construct mismatch) that a good reviewer will hit
first. None of R1–R3 is expensive to address, and addressing them is what turns "the methods
disagree by 0.44" into a claim that survives questioning.
