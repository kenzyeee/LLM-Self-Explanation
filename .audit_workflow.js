export const meta = {
  name: 'pipeline-research-audit',
  description: 'Literature-grounded methodology + code-correctness audit of the LLM explanation-agreement (ECS) research pipeline',
  whenToUse: 'Verify a research pipeline is methodologically sound and reliable for publication, grounded in web literature',
  phases: [
    { title: 'Review', detail: 'One reviewer per methodological dimension: read code + web-search literature' },
    { title: 'Verify', detail: 'Adversarially verify each finding: re-check code, confirm cited papers are real' },
    { title: 'Critic', detail: 'Completeness critic hunts for missed failure modes' },
    { title: 'VerifyGaps', detail: 'Verify the critic-surfaced findings' },
    { title: 'Synthesize', detail: 'Prioritized publication-readiness report' },
  ],
}

const CONTEXT = [
  'PROJECT: An empirical study of CROSS-STRATEGY AGREEMENT among LLM self-explanations.',
  'A single LLM (llama-3.3-70b-versatile, via Groq, temperature=0, ONE sample per call)',
  'classifies a text, then is asked to explain that SAME prediction four ways:',
  '  H  = Highlighting (per-word salience 1-10, top-k by dynamic_k)',
  '  R  = Rationale (one free-text sentence -> content-word lemmas anchored to input)',
  '  CF = Counterfactual (minimal edit that flips the label, MiCE-style; changed input tokens = evidence)',
  '  RO = Rank Ordering (3-5 ranked important words)',
  'The headline construct is ECS = mean pairwise token-set agreement across the cross-paradigm',
  'pairs (H-RO excluded as same-paradigm), reported as LIFT over a Monte-Carlo random-selection',
  'baseline. A SEPARATE post-hoc erasure pass (run_validity_tests.py) erases each strategy tokens',
  '/ the consensus-core (CC3/CC4) tokens and a random control, under mask AND delete operators,',
  'and reports flip-rate, bucketed by ECS-lift tier. Datasets: SST2, MNLI, AG News.',
  'Live config runs N=10 per dataset (pilot); datasets.yaml mentions 200 (curated). Framing',
  '(per the authors): ECS is an inter-method CONSISTENCY measure, explicitly NOT a faithfulness',
  'score; erasure is a second consistency axis, not ground truth.',
  '',
  'GOAL OF THIS AUDIT: determine whether the pipeline that COLLECTS and COMPUTES results is robust',
  'enough that the numbers would be reliable in a research paper, and flag anything that is WRONG',
  'METHODOLOGY or a DEAD-END APPROACH. Ground every judgement in the actual published literature.',
  '',
  'Working dir is C:/OpenSource/Research. Key files:',
  '  scripts/run_experiment.py        (collection orchestration, ECS + lift + CF + CC + redaction)',
  '  scripts/run_validity_tests.py    (erasure pass)',
  '  src/metrics/metrics_calculator.py (Jaccard/overlap/Kendall/RBO/ECS/random-baseline/consensus-core)',
  '  src/metrics/redaction_test.py     (progressive comprehensiveness erasure: faithfulness=1-k/n)',
  '  src/metrics/validity_checker.py   (CC removal vs random removal)',
  '  src/parsing/parser.py             (token extraction for H/R/CF/RO, dynamic_k, CF difflib)',
  '  src/normalization/normalizer.py   (lemmatization, is_anchored)',
  '  src/statistics/statistical_tests.py (bootstrap, permutation, wilcoxon, paired-t, bonferroni)',
  '  src/utils/data_models.py          (InstanceResult/AggregateMetrics, generate_md_report prose)',
  '  src/load/dataset_loader.py, src/load/curator.py (sampling/curation)',
  '  config/experiment.yaml, config/datasets.yaml, config/models.yaml, config/normalization.yaml',
  '  prompts/*.txt (elicitation prompts)',
  '',
  'HOW TO SEARCH THE WEB: you have WebSearch and WebFetch but their schemas are deferred.',
  'First call ToolSearch with query "select:WebSearch,WebFetch" to load them, then use them.',
  'Cite REAL sources only: give an arXiv id / DOI / URL you actually retrieved. If you cannot',
  'verify a paper exists, say so — do NOT invent citations. Honesty about uncertainty is required.',
].join('\n')

const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['dimension', 'findings', 'strengths', 'searches_performed'],
  properties: {
    dimension: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'title', 'severity', 'category', 'location', 'description', 'code_evidence', 'literature', 'recommendation', 'confidence'],
        properties: {
          id: { type: 'string', description: 'stable slug, e.g. ecs-jaccard-vs-overlap' },
          title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
          category: { type: 'string', enum: ['wrong-methodology', 'dead-end', 'bug', 'weakness', 'unsupported-claim'] },
          location: { type: 'string', description: 'file:line or design' },
          description: { type: 'string', description: 'what is wrong and WHY it threatens result validity' },
          code_evidence: { type: 'string', description: 'concrete quote/paraphrase of the offending code or config' },
          literature: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['citation', 'identifier', 'claim_from_source', 'relation'],
              properties: {
                citation: { type: 'string' },
                identifier: { type: 'string', description: 'arXiv id / DOI / URL actually retrieved, or UNVERIFIED' },
                claim_from_source: { type: 'string' },
                relation: { type: 'string', enum: ['supports-our-method', 'contradicts-our-method', 'suggests-alternative', 'context'] },
              },
            },
          },
          recommendation: { type: 'string' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
    strengths: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['aspect', 'why_sound', 'supporting_citation'],
        properties: {
          aspect: { type: 'string' },
          why_sound: { type: 'string' },
          supporting_citation: { type: 'string' },
        },
      },
    },
    searches_performed: { type: 'array', items: { type: 'string' } },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['finding_id', 'code_verdict', 'code_recheck_notes', 'citation_verdict', 'citation_notes', 'final_severity', 'verdict_summary'],
  properties: {
    finding_id: { type: 'string' },
    code_verdict: { type: 'string', enum: ['confirmed', 'partly-confirmed', 'refuted', 'could-not-check'] },
    code_recheck_notes: { type: 'string', description: 'what you found when you re-read the cited code yourself' },
    citation_verdict: { type: 'string', enum: ['all-real-and-relevant', 'some-questionable', 'fabricated-or-wrong', 'no-citations-to-check'] },
    citation_notes: { type: 'string', description: 'result of independently web-verifying each cited paper' },
    final_severity: { type: 'string', enum: ['critical', 'major', 'minor', 'drop'] },
    verdict_summary: { type: 'string' },
  },
}

const CRITIC_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['missed_findings', 'coverage_assessment'],
  properties: {
    coverage_assessment: { type: 'string' },
    missed_findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'title', 'severity', 'category', 'location', 'description', 'why_missed', 'recommendation'],
        properties: {
          id: { type: 'string' },
          title: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
          category: { type: 'string', enum: ['wrong-methodology', 'dead-end', 'bug', 'weakness', 'unsupported-claim'] },
          location: { type: 'string' },
          description: { type: 'string' },
          why_missed: { type: 'string' },
          recommendation: { type: 'string' },
        },
      },
    },
  },
}

const DIMENSIONS = [
  {
    key: 'construct-and-framing',
    label: 'construct/framing',
    focus: 'The CORE PREMISE. Is measuring AGREEMENT among four self-explanations of a single LLM own prediction a meaningful, publishable construct, or a dead-end? Stress-test the consistency-NOT-faithfulness framing: is it airtight, or does the paper implicitly smuggle faithfulness claims (e.g. via the erasure pass, the consensus-core causally-important-tokens language in run_validity_tests.py docstring)? Is the contribution already covered by the explanation-disagreement literature (Krishna et al.)? Does temperature=0 single-sample mean ECS conflates method-disagreement with the model own generation noise?',
    files: 'scripts/run_experiment.py (process_instance, ECS), src/utils/data_models.py (generate_md_report framing prose, ~lines 650-710), scripts/run_validity_tests.py (docstring claims)',
    seeds: [
      'Is agreement-not-equal-faithfulness a known/semi-obvious result (Krishna et al. 2022 disagreement problem; Atanasova et al.)? Is there novel headroom left?',
      'Do LLM self-explanations even reflect the computation? (Turpin et al. 2023; Madsen et al. 2024; Lanham et al. 2023; Parcalabescu & Frank 2024). If self-explanations are known-unfaithful, what does cross-method AGREEMENT actually measure?',
      'With temperature=0 and a single generation per strategy, ECS has no measure of the LLM own stochastic consistency. Is cross-method disagreement separable from sampling noise? (self-consistency: Wang et al. 2022).',
      'The consensus-core erasure section uses causal language (localize causally important tokens); does that contradict the non-faithfulness framing?',
    ],
  },
  {
    key: 'ecs-metric-design',
    label: 'ECS metric',
    focus: 'The ECS metric machinery and its random-baseline lift. Are the set-overlap and rank-agreement metrics computed and aggregated in a statistically defensible way that supports the paper headline numbers?',
    files: 'src/metrics/metrics_calculator.py (all of it), scripts/run_experiment.py (~lines 483-555 ECS + ecs_random/lift), src/utils/data_models.py (report prose about Overlap Coefficient being primary)',
    seeds: [
      'compute_ecs averages JACCARD over cross-paradigm pairs, but generate_md_report prose declares the Overlap Coefficient the primary pairwise metric. Which feeds the headline ECS/lift? Is there an internal inconsistency between the computed quantity and the narrated one?',
      'Averaging Jaccard across heterogeneous method pairs (H-CF, R-CF, R-RO, CF-RO) into one scalar ECS: is a mean of pair-similarities a sound construct, or does it hide structure? Compare to disagreement-metric practice (Krishna et al.).',
      'Monte-Carlo expected_random_overlap draws subsets from a content vocab of size vocab_size with the observed set sizes. Is the null model right (independence, uniform, without-replacement)? n_sims=2000, seed fixed. Does ECS-lift correctly de-confound set size and vocab size? Any analytic check (expected Jaccard of random sets)?',
      'Kendall tau requires >=4 common tokens else None; RBO p=0.9 hardcoded; tau computed only on the INTERSECTION of ranked tokens (not full lists). Is intersection-only Kendall a valid rank-agreement measure or biased? (RBO: Webber et al. 2010).',
      'Is excluding H-RO from ECS principled, and does the resulting variable pair-count per instance (ecs_primary_pairs) bias the mean?',
    ],
  },
  {
    key: 'counterfactual-method',
    label: 'counterfactual',
    focus: 'Counterfactual generation as an explanation/attribution method: minimality, flip verification, edit-ratio caps, and the insertion-only exclusion. Does the CF pipeline yield valid token attributions for ECS?',
    files: 'src/parsing/parser.py (parse_counterfactual, _extract_changed_tokens, _word_edit_ratio), scripts/run_experiment.py (~lines 273-422 CF stages + correction loop, ~557-597 CF-free), prompts/counterfactual_explain*.txt, config/datasets.yaml (cf_max_edit_ratio)',
    seeds: [
      'MNLI uses cf_max_edit_ratio=0.8 while SST2/AG News use 0.3. Does an 0.8 cap still constitute a minimal edit (MiCE; Ross et al. 2021)? Does this make MNLI CF attributions non-comparable to the others?',
      'Insertion-only flips (e.g. adding not) produce no replaced/deleted original token, so the instance is dropped as CF-invalid. Does this systematically bias which instances/labels get CF evidence (esp. MNLI), and thus bias ECS?',
      'Flip-verification re-classifies the edited text and runs a 1-shot change-a-different-word correction loop. Does selecting only verified flips + iterative correction bias toward easy-to-flip instances (survivorship)? Is asking the model to both generate AND verify its own counterfactual circular?',
      'Using the model itself as the oracle to verify the flip (vs a held-out classifier): is self-verification standard in CF-explanation work (Polyjuice Wu et al. 2021; Madsen et al. 2024; Mayne et al. 2025)?',
      'minimality = changed-tokens / orig-words via difflib opcodes: does difflib word-level diff correctly capture the edit, and is normalizing by original length the MiCE convention?',
    ],
  },
  {
    key: 'erasure-faithfulness',
    label: 'erasure',
    focus: 'The erasure / faithfulness pass and consensus-core causal test. Are the erasure operators, baselines and metrics methodologically valid, or do known OOD/baseline pitfalls invalidate the flip-rate comparisons?',
    files: 'src/metrics/redaction_test.py, src/metrics/validity_checker.py, scripts/run_validity_tests.py (erase, random_flip_rate, aggregate, tiers)',
    seeds: [
      'Erasure inserts literal [MASK] strings into prompts for an AUTOREGRESSIVE instruct model (Llama) that never saw [MASK] in pretraining: is mask-infilling OOD here, inflating flips? (the code cites ICE 2026; verify; compare ROAR Hooker et al. 2019; Jacovi & Goldberg 2020; Bastings et al. on OOD perturbation).',
      'random_flip_rate samples random tokens from ALL surface words (incl. stopwords), but CC tokens are normalized CONTENT words: is the random control matched in token type/count? An unmatched control invalidates the CC-minus-random gap.',
      'faithfulness = 1 - k/n (first-flip depth): how does this relate to ERASER comprehensiveness/sufficiency (DeYoung et al. 2020)? Is 1-k/n a recognized faithfulness metric or ad-hoc?',
      'There are TWO separate erasure implementations (redaction_test.py used inline for H/RO at collection; run_validity_tests.erase() post-hoc). Do they mask consistently (punctuation handling, whole-word, case)? Divergence = irreproducible erasure numbers.',
      'Progressive erasure only runs for H and RO (ordered); R and CF have no ranking so no comprehensiveness curve. Is the faithfulness comparison across strategies therefore apples-to-oranges?',
      'ECS-lift tiers are tertiles computed on this run own lifts (data-dependent thresholds) over N as small as ~10-30. Are tier comparisons statistically meaningful at this N? No significance test is applied to the gap.',
    ],
  },
  {
    key: 'statistics-and-design',
    label: 'stats/design',
    focus: 'Statistical validity and experimental design: sample size, replication, significance testing, and whether the declared statistical machinery is actually used.',
    files: 'src/statistics/statistical_tests.py, scripts/run_experiment.py (compute_aggregate_metrics, bootstrap CI), scripts/analyze_results.py, scripts/run_ablations.py, scripts/show_results.py, config/experiment.yaml',
    seeds: [
      'GREP-VERIFY: statistical_tests.py functions (wilcoxon_signed_rank_test, paired_ttest, permutation_test, apply_bonferroni_correction, are_significant, compute_confidence_ecs_correlation) appear to be referenced ONLY by tests/, never by any analysis script. If so, NO significance testing or multiple-comparison correction is actually applied to reported results despite config declaring bonferroni_correction/permutation_tests. Confirm by reading the analysis scripts.',
      'wilcoxon_signed_rank_test and paired_ttest truncate unequal-length groups via slicing to min length: this destroys instance pairing. If ever used, results are invalid. Confirm the bug; check callers.',
      'spearman_rho / spearman_p_value are hard-coded to 0.0 in compute_aggregate_metrics (confidence axis dropped) but still emitted in AggregateMetrics: dead/misleading fields.',
      'Live config: experiment.yaml sample_size=10 and config_loader uses setdefault so 10 OVERRIDES datasets.yaml 200, so actual run is N=10/dataset (30 total), single model, temp=0 single sample. Quantify the statistical power for the headline claims (ECS-lift>0, CC-vs-random gap, tier trend). (Card et al. 2020 With Little Power; Dror et al. 2018 hitchhiker guide to testing).',
      'bootstrap CI for mean ECS is computed; is a percentile bootstrap at N~10-30 trustworthy? permutation_test uses unseeded np.random.shuffle (non-reproducible) if used at all.',
      'No correction for the many pairwise/tier/operator comparisons reported in the markdown report: multiple-comparisons risk.',
    ],
  },
  {
    key: 'data-and-sampling',
    label: 'data/sampling',
    focus: 'Data provenance, curation, balance, dataset choice, and contamination. Are the instances a sound, unbiased basis for the claims, and do benchmark-contamination risks undermine explanation interpretation?',
    files: 'src/load/dataset_loader.py (sample_balanced, clean_text, load_curated), src/load/curator.py, scripts/curate_dataset.py, scripts/curation_vetting.py, config/datasets.yaml',
    seeds: [
      'SST2, MNLI, AG News are classic public benchmarks almost certainly in Llama-3.3 pretraining. Label memorization/contamination would mean the model recalls labels rather than reasons, so self-explanations explain a recalled answer. Does this undermine the study? (data contamination: Sainz et al. 2023; Balloccu et al. 2024; Golchin & Surdeanu 2023).',
      'Curation hand-vetting was done by a single annotator (Claude) with no inter-annotator agreement / human check: researcher-as-annotator bias in which instances are clean-gold. Is that defensible? (datasheets: Gebru et al.).',
      'sample_balanced balances by label via min available count; with the curated path it shuffles then slices to sample_size=10: does the N=10 slice preserve label balance and stratification, or can it skew? Check the slice logic in run_experiment.',
      'Only 3 English classification tasks, 1 model: what is the generalization envelope, and is it honestly bounded in the framing?',
      'clean_text vs pre_clean_text are duplicated cleaners (dataset_loader vs run_experiment): divergence would mean curated text != prompted text.',
    ],
  },
  {
    key: 'parsing-normalization',
    label: 'parsing/norm',
    focus: 'Token extraction & normalization, the layer that DEFINES the sets ECS compares. Do preprocessing choices systematically inflate or deflate measured agreement?',
    files: 'src/parsing/parser.py (parse_highlighting, parse_rationale, parse_rank_ordering, dynamic_k, is_anchored usage), src/normalization/normalizer.py (normalize, is_anchored, lemmatization), config/experiment.yaml + config/normalization.yaml (use_lemmatization mismatch)',
    seeds: [
      'is_anchored REQUIRES every evidence token to appear in the input (surface or WordNet-lemma match). Rationale introduced concepts (not in input) are dropped from R set. Since R is abstractive, anchoring may systematically shrink/distort R set and deflate (or inflate) R-pair agreement. Is input-anchoring a justified constraint for cross-method comparison?',
      'CONFIG MISMATCH: experiment.yaml normalization.use_lemmatization=false but normalization.yaml default=true and Normalizer defaults true. Determine what actually wins at runtime: config_loader replaces exp_data normalization wholesale with norm_data normalization key, but normalization.yaml has top-level keys default/no_lemmatization/etc and NO normalization key. Determine the ACTUAL lemmatization setting of the live run. Memory says lemmatization changes every downstream metric.',
      'dynamic_k = max(3, round(content_words/5)) sets H and RO set sizes; equal k for H and RO mechanically inflates their overlap. Is length-proportional k (Huang et al. 2023; arXiv:2310.05619) applied consistently, and does forcing equal k bias the H-RO rank metrics?',
      'Rationale tokens = open-class POS lemmas via spaCy; if spaCy missing, a different fallback path runs (whitespace split). Two code paths -> different token sets depending on environment. Reproducibility risk.',
      'Masking under-coverage: consensus-core tokens are lemmatized (e.g. movie) but erasure matches SURFACE words (movies) -> CC erasure may silently fail to remove inflected words, weakening the causal test (noted in project memory).',
      'normalize() drops a DISCOURSE_WORDS list (positive/negative/premise/hypothesis/etc) and stopwords; could legitimately-evidential words be removed, biasing sets?',
    ],
  },
]

for (const d of DIMENSIONS) {
  const lines = []
  for (let i = 0; i < d.seeds.length; i++) {
    lines.push('  (' + (i + 1) + ') ' + d.seeds[i])
  }
  d.seedsText = lines.join('\n')
}

function reviewPrompt(dim) {
  return CONTEXT + '\n\n' +
    'YOU ARE THE REVIEWER FOR DIMENSION: ' + dim.key + '\n' +
    dim.focus + '\n\n' +
    'Read these files FIRST (use Read/Grep yourself, do not trust my summary):\n  ' + dim.files + '\n\n' +
    'Then investigate these specific HYPOTHESES. They are NOT assumed true. Confirm or REFUTE each ' +
    'against the actual code and against published literature. Go beyond them if you find more:\n' +
    dim.seedsText + '\n\n' +
    'For each real problem, produce a finding with: precise location (file:line), concrete code ' +
    'evidence, a clear statement of HOW it threatens the reliability of the reported results, at ' +
    'least one REAL literature citation (arXiv id/DOI/URL you actually retrieved via WebSearch/' +
    'WebFetch) marked supporting/contradicting/suggesting-alternative, a concrete recommendation, ' +
    'and your confidence. Also record genuine STRENGTHS (things this dimension does correctly, with ' +
    'a citation) so the final report is balanced. Severity: critical = would invalidate or seriously ' +
    'mislead a headline result; major = a reviewer would demand it be fixed; minor = polish. Do NOT ' +
    'pad with trivia. Be skeptical and concrete. Cite only papers you verified exist.'
}

function verifyPrompt(dimKey, f) {
  return CONTEXT + '\n\n' +
    'You are an ADVERSARIAL VERIFIER. Another agent produced this finding about the research ' +
    'pipeline. Your job is to try to BREAK it: independently re-check the code AND independently ' +
    'verify the literature. Default to skepticism. If you cannot confirm, say so.\n\n' +
    'FINDING (dimension ' + dimKey + '):\n' + JSON.stringify(f, null, 2) + '\n\n' +
    'Do BOTH:\n' +
    '1) CODE: open the cited file/location yourself with Read/Grep and decide whether the code ' +
    'actually does what the finding claims. Quote what you see. Verdict: confirmed / partly-confirmed ' +
    '/ refuted / could-not-check.\n' +
    '2) CITATIONS: for EACH cited paper, use WebSearch/WebFetch (load via ToolSearch ' +
    '"select:WebSearch,WebFetch") to confirm it EXISTS and actually makes the claimed point. Flag any ' +
    'fabricated, mis-attributed, or irrelevant citation. Verdict: all-real-and-relevant / ' +
    'some-questionable / fabricated-or-wrong / no-citations-to-check.\n\n' +
    'Then assign a FINAL severity (critical/major/minor, or drop if the finding is wrong/non-issue). ' +
    'Be concrete; this gates what reaches the paper authors.'
}

phase('Review')
const reviewed = await pipeline(
  DIMENSIONS,
  (dim) => agent(reviewPrompt(dim), { label: 'review:' + dim.label, phase: 'Review', schema: REVIEW_SCHEMA, effort: 'high' }),
  (review, dim) => {
    if (!review || !review.findings || review.findings.length === 0) return []
    return parallel(review.findings.map((f) => () =>
      agent(verifyPrompt(dim.key, f), { label: 'verify:' + f.id, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'medium' })
        .then((v) => ({ dimension: dim.key, finding: f, verdict: v }))
    ))
  }
)

const verifiedFindings = reviewed.flat().filter(Boolean)
const survived = verifiedFindings.filter((x) => x.verdict && x.verdict.final_severity !== 'drop')
log('Round 1: ' + verifiedFindings.length + ' findings verified, ' + survived.length + ' survived adversarial check')

phase('Critic')
const survivedTitles = survived.map((x) => '  - [' + x.verdict.final_severity + '] ' + x.dimension + ': ' + x.finding.title).join('\n')
const criticPrompt = CONTEXT + '\n\n' +
  'You are the COMPLETENESS CRITIC. Seven dimension reviewers already audited the pipeline ' +
  '(construct/framing, ECS metric, counterfactual, erasure, statistics/design, data/sampling, ' +
  'parsing/normalization). Here are the issues they ALREADY found (do not repeat these):\n' +
  survivedTitles + '\n\n' +
  'Your job: find what they MISSED. Read broadly across the pipeline (collection -> compute -> ' +
  'report) and hunt for failure modes that fall BETWEEN the dimensions or were overlooked, e.g.: ' +
  'reproducibility/determinism gaps (seeds, ordering, environment, spaCy/NLTK availability); ' +
  'silent error-swallowing that turns failed API calls into no-flip/empty sets; leakage of the ' +
  'committed label into the explanation prompts (conversation history) and whether that confounds ' +
  'agreement; token-budget truncation / retry logic corrupting outputs; checkpoint/resume or ' +
  'aggregation bugs that double-count or drop instances; report numbers that do not match what was ' +
  'computed; any place a result is actually a hard-coded constant or placeholder. Verify against ' +
  'code with Read/Grep, and cite literature where relevant (verify it exists). Return only ' +
  'genuinely missed, real issues with concrete locations.'
const critic = await agent(criticPrompt, { label: 'completeness-critic', phase: 'Critic', schema: CRITIC_SCHEMA, effort: 'high' })

phase('VerifyGaps')
const criticFindings = (critic && critic.missed_findings) || []
const criticVerified = await parallel(criticFindings.map((f) => () =>
  agent(verifyPrompt('completeness', f), { label: 'verify-gap:' + f.id, phase: 'VerifyGaps', schema: VERIFY_SCHEMA, effort: 'medium' })
    .then((v) => ({ dimension: 'completeness', finding: f, verdict: v }))
))

const criticSurvived = criticVerified.filter(Boolean).filter((x) => x.verdict && x.verdict.final_severity !== 'drop')
const allSurvived = survived.concat(criticSurvived)
log('After critic: ' + allSurvived.length + ' total confirmed issues going to synthesis')

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['executive_summary', 'publication_readiness', 'critical', 'major', 'minor', 'dead_ends', 'strengths', 'top_recommendations', 'suggested_run_blockers'],
  properties: {
    executive_summary: { type: 'string', description: '4-8 sentence verdict for the researcher' },
    publication_readiness: { type: 'string', description: 'honest assessment: what claims are currently defensible vs not, given N=10 single-model pilot' },
    critical: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['title', 'location', 'why', 'fix'], properties: { title: { type: 'string' }, location: { type: 'string' }, why: { type: 'string' }, fix: { type: 'string' } } } },
    major: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['title', 'location', 'why', 'fix'], properties: { title: { type: 'string' }, location: { type: 'string' }, why: { type: 'string' }, fix: { type: 'string' } } } },
    minor: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['title', 'location', 'fix'], properties: { title: { type: 'string' }, location: { type: 'string' }, fix: { type: 'string' } } } },
    dead_ends: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['approach', 'why_dead_end', 'alternative'], properties: { approach: { type: 'string' }, why_dead_end: { type: 'string' }, alternative: { type: 'string' } } }, description: 'approaches that are methodological dead-ends per the literature' },
    strengths: { type: 'array', items: { type: 'string' } },
    top_recommendations: { type: 'array', items: { type: 'string' }, description: 'ordered, highest-leverage actions to make results paper-reliable' },
    suggested_run_blockers: { type: 'array', items: { type: 'string' }, description: 'things that MUST be fixed BEFORE spending API budget on the full data-collection run' },
  },
}

const synthPrompt = CONTEXT + '\n\n' +
  'You are the SYNTHESIS LEAD writing the final audit for the researcher. Below are ALL the ' +
  'adversarially-verified findings (each with a code verdict, citation verdict, and final severity). ' +
  'De-duplicate overlapping issues, weigh them by verified severity, and produce a single coherent, ' +
  'prioritized, HONEST report. Distinguish: (a) outright bugs that corrupt numbers, (b) wrong/' +
  'unsupported methodology, (c) genuine methodological DEAD-ENDS (with the literature-backed ' +
  'alternative), and (d) defensible strengths. Be specific about file locations and concrete fixes. ' +
  'Crucially, separate issues that MUST be fixed BEFORE the (API-budget-limited) full collection run ' +
  'from issues that can be fixed at analysis time. Calibrate the publication-readiness verdict to the ' +
  'reality of an N=10, single-model, temperature=0 pilot.\n\n' +
  'VERIFIED FINDINGS (JSON):\n' +
  JSON.stringify(allSurvived.map((x) => ({ dimension: x.dimension, finding: { id: x.finding.id, title: x.finding.title, severity: x.finding.severity, category: x.finding.category, location: x.finding.location, description: x.finding.description, recommendation: x.finding.recommendation }, verdict: x.verdict })), null, 2) + '\n\n' +
  'COMPLETENESS-CRITIC COVERAGE NOTE:\n' + ((critic && critic.coverage_assessment) || 'n/a')

const synthesis = await agent(synthPrompt, { label: 'synthesis', phase: 'Synthesize', schema: SYNTH_SCHEMA, effort: 'xhigh' })

return {
  counts: {
    raw_findings: verifiedFindings.length,
    survived_round1: survived.length,
    critic_added: criticSurvived.length,
    total_confirmed: allSurvived.length,
  },
  synthesis,
  all_findings: allSurvived.map((x) => ({
    dimension: x.dimension,
    id: x.finding.id,
    title: x.finding.title,
    final_severity: x.verdict.final_severity,
    code_verdict: x.verdict.code_verdict,
    citation_verdict: x.verdict.citation_verdict,
    location: x.finding.location,
    description: x.finding.description,
    recommendation: x.finding.recommendation,
    verdict_summary: x.verdict.verdict_summary,
  })),
}
