Verify that the whole pipeline to collect and compute results is robust. It should be such that the results are reliable for research paper. Search the web for scientific literature to verify and improve research accuracy, such that nothing is the wrong methodology or dead end approach.

I'll start by understanding the project deeply before orchestrating any verification. Let me scout the codebase and read the memory context in parallel.

**Searched****/*.py

**Recalled** 6 memories

Good context. This is an LLM self-explanation cross-strategy agreement (ECS/consistency) study. Now let me read the core pipeline source to understand what's actually implemented.

**Read** 3 files

Let me read the rest of the core pipeline: parser, normalizer, inference, redaction/validity, and the orchestration scripts.

**Read** 4 files

Now the orchestration layer: inference engine, the main experiment runner, validity tests, and config.

**Read** 2 files

Let me read the validity-test (erasure) pass, config, dataset loading/curation, and the analysis scripts, plus locate the config and prompts.

**Read** 2 files, **found** files

Let me read the configs and the active elicitation prompts to complete my picture before designing the verification.

**Read** 4 files

Now the active elicitation prompts — these directly determine methodological validity.

**Read** 8 files

I now have a complete picture of the pipeline. Before designing the verification, let me quickly confirm two things that determine which config actually drives runs and whether the Bonferroni/correlation machinery is live.

**Read** a file, **searched** code

I have a thorough, verified picture of the pipeline now — including a confirmed signal that the entire `statistics` module appears to have no production caller (only tests reference it) and that `experiment.yaml`'s `sample_size: 10` overrides the `200` in `datasets.yaml` via `setdefault`.

The user explicitly asked for a web-literature-grounded methodology audit, and ultracode is on. This is exactly what a multi-agent workflow is for: fan out literature-grounded reviews across every methodological dimension, adversarially verify each finding (including checking that cited papers are real), then synthesize. Let me mark a chapter and launch it.

Pipeline robustness audit

Now I'll author and launch the verification workflow.

**Used Workflow**

The apostrophe escaping broke the parse (`\\'` inside single-quoted strings closes the string). Let me fix those and resend.

**Used Workflow**

Let me write the script to a file so I can inspect the exact failing line and iterate cleanly, then invoke via `scriptPath`.

**Created**.audit_workflow.js**+394**-0

I rewrote the script using string concatenation (no nested template literals or escaped apostrophes). Now invoking via `scriptPath`:

**Ran workflow**.audit_workflow

I've launched a comprehensive literature-grounded audit as a background multi-agent workflow (`/workflows` to watch it live). It fans out 7 methodological reviewers — each reads the actual code *and* searches scientific literature (WebSearch/WebFetch) — then adversarially verifies every finding (re-checking the code **and** confirming each cited paper is real, to guard against fabricated citations), runs a completeness critic to catch what the dimensions missed, and synthesizes a prioritized publication-readiness report.

The seven dimensions under review:

1. **Construct & framing** — is "agreement among self-explanations of one LLM" a valid, novel construct, or a dead-end? Is the "consistency ≠ faithfulness" framing airtight? (Turpin 2023, Madsen 2024, Krishna disagreement-problem, Parcalabescu & Frank…)
2. **ECS metric** — Jaccard-vs-Overlap inconsistency, the Monte-Carlo random-lift null model, intersection-only Kendall τ, RBO.
3. **Counterfactual** — MiCE minimality, the 0.3-vs-0.8 edit-ratio split, insertion-only exclusion bias, self-verification circularity.
4. **Erasure / faithfulness** — `[MASK]` OOD on an autoregressive model, mismatched random control, `1−k/n` vs ERASER comprehensiveness, two divergent erasure implementations.
5. **Statistics & design** — N=10 power, and whether the significance-testing machinery is even wired in.
6. **Data & sampling** — benchmark contamination of Llama-3.3, single-annotator curation, the N=10 slice's balance.
7. **Parsing & normalization** — input-anchoring bias on rationales, the lemmatization config resolution, `dynamic_k`-driven set sizes.

While scouting to build this, I already independently confirmed three things the workflow is now stress-testing:

* **The `statistics` module looks dead** — `wilcoxon`, `paired_ttest`, `permutation_test`, `bonferroni`, the Spearman correlation are referenced *only* by `tests/`, never by any analysis script, and `spearman_rho` is hard-coded to `0.0` in `run_experiment.py:850`. So despite the config declaring Bonferroni/permutation tests,  **no significance testing appears to actually run** .
* **The live N is 10, not 200** — `experiment.yaml` `sample_size: 10` overrides the `200` in `datasets.yaml` via `setdefault` in `config_loader.py:48`.
* **Lemmatization is OFF in the live run** — `use_lemmatization: false` in experiment.yaml; the `normalization.yaml` merge is a no-op (it has no top-level `normalization:` key). The anchoring lemmatizer stays on regardless, which is a subtle asymmetry.

I'll let the workflow run (it's doing the real literature verification) and report the synthesized findings when it completes — no need for you to do anything in the meantime.
