# LLM Explanation Agreement Study

Investigates cross-strategy agreement among LLM self-explanations across four explanation strategies: **Highlighting (H)**, **Rationale (R)**, **Counterfactual (CF)**, and **Rank Ordering (RO)**.

## Pipeline

```
config/ ──► load ──► inference ──► parsing ──► normalization ──► metrics ──► validity ──► statistics ──► plots ──► paper
                              ▲                      ▲
                       AWS Bedrock             NLTK / spaCy
```

1. **Load** -- Download & balance datasets (SST-2, MNLI, AG News)
2. **Inference** -- Query LLMs via AWS Bedrock (Converse API) for classification + 4 explanations; the 3 configured models run concurrently per dataset
3. **Parse** -- Extract structured outputs from raw LLM responses
4. **Normalize** -- Lowercase, strip punctuation, remove stopwords, lemmatize
5. **Metrics** -- Compute Jaccard similarity, Kendall's tau, ECS, consensus cores
6. **Validity** -- Mask consensus-core tokens, test flip rate vs random baseline
7. **Statistics** -- Bootstrap correlation, permutation tests, Wilcoxon, paired t-test
8. **Plots** -- Heatmaps, ECS distributions, scatter plots, flip rate comparisons
9. **Paper** -- Generate LaTeX draft with methodology, results, and figures

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # required: rationale (R) evidence extraction
```

The spaCy English model is a required post-install step — `run_experiment.py` fails fast
at startup without it (rationale extraction would otherwise silently degrade).

Authenticate to Bedrock with a **Bedrock API key** (bearer token) — create one in the
Bedrock console, then:

```bash
set AWS_BEARER_TOKEN_BEDROCK=ABSK...     # Windows
set AWS_REGION=us-east-1

export AWS_BEARER_TOKEN_BEDROCK=ABSK...   # Linux/Mac
export AWS_REGION=us-east-1
```

Alternatively use standard AWS SigV4 credentials (`AWS_ACCESS_KEY_ID` +
`AWS_SECRET_ACCESS_KEY`, `~/.aws/credentials`, or an IAM role). Either way, enable
access to the three configured models (Amazon Nova Pro, Qwen3-235B, DeepSeek V3) for your
account and region in the Bedrock console first. The default config targets `eu-north-1`,
where Nova Pro uses the `eu.*` cross-region inference profile (requires an EU region) and
Qwen3 / DeepSeek V3 are on-demand.

The config in `config/experiment.yaml` is pre-configured for deterministic (T=0) inference across 3 models, 3 datasets, and 4 strategies.

## Usage

```bash
# Run the full experiment pipeline
python scripts/run_experiment.py

# Run with overrides (--models selects a subset of the configured models by name)
python scripts/run_experiment.py --models nova-pro deepseek-v3 --sample-size 50 --force-restart

# Prompt-paraphrase ablation (the one pre-registered robustness ablation)
python scripts/run_ablations.py

# Erasure pass (second consistency axis; run after a completed experiment)
python scripts/run_validity_tests.py

# Analyze results
python scripts/analyze_results.py
```

See `python scripts/run_experiment.py --help` for all CLI options.

## Project Structure

```
├── config/              YAML configuration files
│   ├── experiment.yaml  Main experiment config (normalization lives inline here)
│   ├── datasets.yaml    Dataset-specific overrides
│   └── models.yaml      Model-specific overrides
├── prompts/             9 prompt templates (*.txt)
├── src/
│   ├── load/            Dataset loading & balanced sampling
│   ├── inference/       AWS Bedrock inference engine
│   ├── parsing/         Response parsers (5 formats)
│   ├── normalization/   Token normalization
│   ├── metrics/         ECS, Jaccard, Kendall, consensus cores + validity
│   ├── statistics/      Statistical tests (bootstrap, permutation, t-test)
│   ├── plots/           Visualization generation
│   └── utils/           Config, logging, checkpointing, exceptions, data models
├── scripts/             Entry-point scripts
│   ├── run_experiment.py      Full pipeline
│   ├── run_ablations.py       Prompt-paraphrase ablation
│   ├── run_validity_tests.py  Erasure pass (second consistency axis)
│   ├── analyze_results.py     Result analysis
│   └── show_results.py        Quick summary of the latest run
├── tests/               Automated test suite (see Development)
├── outputs/             Experiment outputs & checkpoints
├── paper/               Generated LaTeX + figures + tables
├── data/                Dataset cache
└── logs/                Experiment logs
```

## Key Metrics

| Metric                  | Description                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| **ECS**           | Explanation Consensus Score -- mean of the **5 cross-paradigm** pairwise Jaccard similarities (H–RO excluded as same-paradigm), reported as **lift over a Monte-Carlo random baseline** |
| **ECS-overlap**   | Size-robust secondary composite -- mean Overlap Coefficient over the same 5 pairs          |
| **CC3**           | Consensus Core 3 -- tokens appearing in >=3 of 4 strategies                                |
| **CC4**           | Consensus Core 4 -- tokens appearing in all 4 strategies                                   |
| **Jaccard**       | Token-set overlap between any two strategies (evidence sets share one lemmatized token space) |
| **Kendall's tau** | Rank correlation between ordered token lists (H, RO)                                       |
| **Flip rate**     | Erasure pass: rate at which masking/deleting consensus-core tokens changes the model's prediction, vs. a same-size content-word random control |

## Development

```bash
# Run tests
python -m pytest

# With coverage
python -m pytest --cov=src --cov-report=term

# Run a specific test file
python -m pytest tests/test_metrics_calculator.py -v
```

The test suite covers the collection, metric, statistics, and erasure paths (589 tests, including a `test_scientific_invariants.py` fixture suite that pins ECS on synthetic all-agree / all-disagree / inflection-equivalent inputs).

## Citation

If using this pipeline in research, cite as:

```
@software{llm_explanation_agreement,
  title = {LLM Explanation Agreement Study},
  description = {Research pipeline for investigating cross-strategy agreement among LLM self-explanations},
  year = {2026},
}
```
