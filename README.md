# LLM Explanation Agreement Study

Investigates cross-strategy agreement among LLM self-explanations across four explanation strategies: **Highlighting (H)**, **Rationale (R)**, **Counterfactual (CF)**, and **Rank Ordering (RO)**.

## Pipeline

```
config/ ──► load ──► inference ──► parsing ──► normalization ──► metrics ──► validity ──► statistics ──► plots ──► paper
                              ▲                      ▲
                         Groq API              NLTK / spaCy
```

1. **Load** -- Download & balance datasets (SST-2, MNLI, AG News)
2. **Inference** -- Query LLMs via Groq API for classification + 4 explanations
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
```

Set your Groq API key:

```bash
set GROQ_API_KEY=gsk_your_key_here    # Windows
export GROQ_API_KEY=gsk_your_key_here  # Linux/Mac
```

The config in `config/experiment.yaml` is pre-configured for deterministic (T=0) inference across 3 models, 3 datasets, and 4 strategies.

## Usage

```bash
# Run the full experiment pipeline
python scripts/run_experiment.py

# Run with overrides
python scripts/run_experiment.py --models llama-3.1-8b --sample-size 50 --force-restart

# Ablation studies
python scripts/run_ablations.py --variants prompt normalization --k-values 2 3

# Validity tests
python scripts/run_validity_tests.py

# Analyze results & generate paper
python scripts/analyze_results.py
python scripts/generate_paper.py
```

See `python scripts/run_experiment.py --help` for all CLI options.

## Project Structure

```
├── config/              YAML configuration files
│   ├── experiment.yaml  Main experiment config
│   ├── datasets.yaml    Dataset-specific overrides
│   ├── models.yaml      Model-specific overrides
│   └── normalization.yaml  Normalization variants
├── prompts/             9 prompt templates (*.txt)
├── src/
│   ├── load/            Dataset loading & balanced sampling
│   ├── inference/       Groq API inference engine
│   ├── parsing/         Response parsers (5 formats)
│   ├── normalization/   Token normalization
│   ├── metrics/         ECS, Jaccard, Kendall, consensus cores + validity
│   ├── statistics/      Statistical tests (bootstrap, permutation, t-test)
│   ├── plots/           Visualization generation
│   ├── paper/           LaTeX paper generator
│   └── utils/           Config, logging, checkpointing, exceptions, data models
├── scripts/             Entry-point scripts
│   ├── run_experiment.py      Full pipeline
│   ├── run_ablations.py       Ablation studies
│   ├── run_validity_tests.py  Validity testing
│   ├── analyze_results.py     Result analysis
│   └── generate_paper.py      Paper generation
├── tests/               415 tests (100% coverage)
├── outputs/             Experiment outputs & checkpoints
├── paper/               Generated LaTeX + figures + tables
├── data/                Dataset cache
└── logs/                Experiment logs
```

## Key Metrics

| Metric                  | Description                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| **ECS**           | Explanation Consensus Score -- mean of 6 pairwise Jaccard similarities across 4 strategies |
| **CC3**           | Consensus Core 3 -- tokens appearing in >=3 of 4 strategies                                |
| **CC4**           | Consensus Core 4 -- tokens appearing in all 4 strategies                                   |
| **Jaccard**       | Token-set overlap between any two strategies                                               |
| **Kendall's tau** | Rank correlation between ordered token lists (H, RO)                                       |
| **Flip rate**     | Rate at which masking consensus-core tokens changes the model's prediction                 |

## Development

```bash
# Run tests
python -m pytest

# With coverage
python -m pytest --cov=src --cov-report=term

# Run a specific test file
python -m pytest tests/test_metrics_calculator.py -v
```

All source files under `src/` have 100% test coverage (415 tests across 21 test files).

## Citation

If using this pipeline in research, cite as:

```
@software{llm_explanation_agreement,
  title = {LLM Explanation Agreement Study},
  description = {Research pipeline for investigating cross-strategy agreement among LLM self-explanations},
  year = {2026},
}
```
