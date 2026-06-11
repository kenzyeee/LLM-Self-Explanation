# Requirements Document

## Introduction

This document specifies the requirements for an empirical NLP research project that investigates cross-strategy agreement among LLM self-explanations. The system will conduct controlled experiments across multiple datasets, models, and explanation strategies to determine whether agreement between different explanation methods can serve as a reliability signal for model predictions. The project produces a complete, reproducible research pipeline with datasets, metrics, visualizations, and a publication-ready paper.

## Glossary

- **System**: The complete research pipeline including data loading, model inference, explanation extraction, metric computation, and paper generation
- **Explanation_Strategy**: A specific prompting method to elicit explanations from language models (H=highlighting, R=rationale, CF=counterfactual, RO=rank-ordering)
- **ECS**: Explanation Consensus Score - the mean pairwise agreement across all explanation strategy pairs for a single instance
- **Consensus_Core**: The set of tokens that appear in k out of 4 explanation strategies (CCk notation)
- **Groq_API**: The API service used for all model inference with deterministic settings
- **Dataset_Loader**: Component responsible for loading and sampling from SST-2, MNLI, and AG News datasets
- **Inference_Engine**: Component that executes classification and explanation requests via Groq API
- **Parser**: Component that extracts structured evidence from raw model outputs
- **Normalizer**: Component that standardizes extracted evidence for comparison
- **Metrics_Calculator**: Component that computes agreement metrics between explanation strategies
- **Validity_Checker**: Component that performs consensus-core removal tests
- **Visualization_Generator**: Component that produces publication-quality figures
- **Paper_Generator**: Component that creates the first-draft research paper
- **Test_Suite**: Automated tests verifying correctness of all components

## Requirements

### Requirement 1: Dataset Preparation

**User Story:** As a researcher, I want to load and prepare three standard NLP datasets with balanced sampling, so that I can conduct controlled experiments across diverse task types.

#### Acceptance Criteria

1. THE Dataset_Loader SHALL load SST-2, MNLI, and AG News datasets from Hugging Face or local cache
2. WHEN a dataset is loaded, THE Dataset_Loader SHALL sample approximately 200 examples with balanced label distribution
3. FOR ALL sampled datasets, THE Dataset_Loader SHALL preserve the original text and ground-truth labels
4. THE Dataset_Loader SHALL export cleaned datasets to structured output files in the data directory
5. THE Dataset_Loader SHALL log dataset statistics including total count, label distribution, and average text length

### Requirement 2: Model Inference Configuration

**User Story:** As a researcher, I want to configure deterministic model inference via Groq API, so that all experiments are reproducible.

#### Acceptance Criteria

1. THE Inference_Engine SHALL use the GROQ_API_KEY environment variable for authentication
2. THE Inference_Engine SHALL set temperature to 0 and top_p to 1 for all requests
3. THE Inference_Engine SHALL execute requests in stateless mode without conversation context
4. THE Inference_Engine SHALL support three Groq-compatible models for all experiments
5. WHEN an API request fails, THE Inference_Engine SHALL log the error and retry up to 3 times with exponential backoff
6. IF a request fails after all retries, THEN THE Inference_Engine SHALL log the failure and continue with remaining instances

### Requirement 3: Classification and Confidence Extraction

**User Story:** As a researcher, I want to obtain model predictions with confidence scores, so that I can correlate confidence with explanation agreement.

#### Acceptance Criteria

1. WHEN an instance is processed, THE Inference_Engine SHALL request a classification prediction and confidence score from 0 to 100
2. THE Parser SHALL extract the predicted label and confidence value from the model response
3. IF the model response does not contain a valid confidence score, THEN THE Parser SHALL log the parsing failure and assign a null confidence value
4. THE System SHALL store classification results alongside explanation data for correlation analysis

### Requirement 4: Explanation Strategy Elicitation

**User Story:** As a researcher, I want to collect four different types of explanations for each instance, so that I can measure cross-strategy agreement.

#### Acceptance Criteria

1. THE Inference_Engine SHALL execute four explanation prompts per instance: token highlighting (H), one-sentence rationale (R), minimal-edit counterfactual (CF), and explicit rank-ordering (RO)
2. WHEN collecting explanations for an instance, THE Inference_Engine SHALL randomize the execution order of the four strategies
3. THE Inference_Engine SHALL use separate, independent requests for each explanation strategy
4. FOR ALL explanation requests, THE System SHALL use identical model settings (temperature=0, top_p=1)
5. THE System SHALL log the timestamp and strategy order for each instance

### Requirement 5: Token Highlighting Extraction

**User Story:** As a researcher, I want to extract the 3 most important tokens from highlighting explanations, so that I can compute agreement with other strategies.

#### Acceptance Criteria

1. WHEN processing a highlighting explanation, THE Parser SHALL extract exactly 3 tokens marked as most important
2. THE Parser SHALL handle multiple output formats including quoted tokens, numbered lists, and inline markers
3. IF the model returns more than 3 tokens, THEN THE Parser SHALL select the first 3 tokens in order
4. IF the model returns fewer than 3 tokens, THEN THE Parser SHALL log the incomplete response and use all available tokens
5. THE Normalizer SHALL convert extracted tokens to lowercase and remove punctuation

### Requirement 6: Rationale Evidence Extraction

**User Story:** As a researcher, I want to extract key content words from rationale explanations, so that I can compare evidence across strategies.

#### Acceptance Criteria

1. WHEN processing a rationale explanation, THE Parser SHALL extract the complete one-sentence rationale
2. THE Normalizer SHALL tokenize the rationale and extract content words (nouns, verbs, adjectives, adverbs)
3. THE Normalizer SHALL remove stopwords from the extracted content words
4. THE Normalizer SHALL convert all tokens to lowercase
5. WHERE lemmatization is enabled, THE Normalizer SHALL apply lemmatization to extracted tokens

### Requirement 7: Counterfactual Evidence Extraction

**User Story:** As a researcher, I want to identify tokens that differ between the original and counterfactual text, so that I can determine which words the model considers critical.

#### Acceptance Criteria

1. WHEN processing a counterfactual explanation, THE Parser SHALL extract the minimal-edit counterfactual text
2. THE Parser SHALL perform token-level alignment between the original input and counterfactual text
3. THE Normalizer SHALL identify tokens that appear in the original but not in the counterfactual (deleted tokens)
4. THE Normalizer SHALL identify tokens that appear in the counterfactual but not in the original (added tokens)
5. THE System SHALL combine deleted and added tokens as the evidence set for counterfactual explanations

### Requirement 8: Rank-Ordering Evidence Extraction

**User Story:** As a researcher, I want to extract explicitly ranked tokens with their positions, so that I can compute rank-based agreement metrics.

#### Acceptance Criteria

1. WHEN processing a rank-ordering explanation, THE Parser SHALL extract exactly 5 ranked tokens with their ordinal positions
2. THE Parser SHALL handle multiple output formats including numbered lists and natural language descriptions
3. IF the model provides fewer than 5 tokens, THEN THE Parser SHALL use all available tokens with their provided ranks
4. THE Normalizer SHALL convert extracted tokens to lowercase and remove punctuation
5. THE System SHALL preserve rank information for Kendall's tau computation

### Requirement 9: Pairwise Agreement Computation

**User Story:** As a researcher, I want to compute agreement between all pairs of explanation strategies, so that I can measure cross-strategy consistency.

#### Acceptance Criteria

1. THE Metrics_Calculator SHALL compute Jaccard similarity for all six pairs of explanation strategies (H-R, H-CF, H-RO, R-CF, R-RO, CF-RO)
2. WHEN computing Jaccard similarity, THE Metrics_Calculator SHALL use normalized token sets for both strategies
3. THE Metrics_Calculator SHALL compute Kendall's tau between rank-ordering (RO) and token highlighting (H) when both provide rank information
4. THE Metrics_Calculator SHALL compute Kendall's tau between rank-ordering (RO) and other strategies by assigning implicit ranks based on token order
5. THE System SHALL store all pairwise agreement scores for statistical analysis

### Requirement 10: Explanation Consensus Score Calculation

**User Story:** As a researcher, I want to compute a single consensus score per instance, so that I can measure overall explanation agreement.

#### Acceptance Criteria

1. WHEN all pairwise agreements are computed for an instance, THE Metrics_Calculator SHALL compute ECS as the mean of all six pairwise Jaccard similarities
2. THE Metrics_Calculator SHALL compute ECS only for instances with valid explanations from all four strategies
3. IF any explanation strategy fails for an instance, THEN THE Metrics_Calculator SHALL exclude that instance from ECS calculation
4. THE System SHALL compute aggregate ECS statistics per dataset, per model, and across all experiments

### Requirement 11: Consensus Core Identification

**User Story:** As a researcher, I want to identify tokens that appear across multiple explanation strategies, so that I can test whether high-consensus evidence is more reliable.

#### Acceptance Criteria

1. THE Metrics_Calculator SHALL compute Consensus Core CC3 as the set of tokens appearing in at least 3 of 4 explanation strategies
2. THE Metrics_Calculator SHALL compute Consensus Core CC4 as the set of tokens appearing in all 4 explanation strategies
3. WHEN computing consensus cores, THE Metrics_Calculator SHALL use normalized token sets from all strategies
4. THE System SHALL compute the size distribution of CC3 and CC4 across all instances
5. THE System SHALL identify instances with empty consensus cores for error analysis

### Requirement 12: Confidence-ECS Correlation Analysis

**User Story:** As a researcher, I want to measure correlation between model confidence and explanation consensus, so that I can determine whether confidence predicts explanation agreement.

#### Acceptance Criteria

1. THE Metrics_Calculator SHALL compute Spearman's rank correlation between confidence scores and ECS values
2. THE Metrics_Calculator SHALL compute correlation per dataset and per model
3. THE Metrics_Calculator SHALL compute 95% confidence intervals for correlation coefficients using bootstrap resampling with 1000 iterations
4. THE System SHALL test the null hypothesis that correlation equals zero using permutation testing with 10000 permutations
5. THE System SHALL export correlation results to structured tables for paper inclusion

### Requirement 13: Consensus Core Validity Testing

**User Story:** As a researcher, I want to test whether masking consensus-core tokens reduces prediction accuracy, so that I can validate that high-consensus evidence is causally important.

#### Acceptance Criteria

1. THE Validity_Checker SHALL mask all CC3 tokens in each input and re-run classification
2. THE Validity_Checker SHALL compute the prediction flip rate (percentage of instances where the prediction changes after masking)
3. THE Validity_Checker SHALL create a random-removal baseline by masking the same number of randomly selected tokens
4. THE Validity_Checker SHALL compare CC3 flip rate against random baseline flip rate using paired t-test
5. THE Validity_Checker SHALL repeat the validity test for CC4 tokens
6. IF CC3 flip rate is significantly higher than baseline, THEN THE System SHALL report consensus cores as causally validated

### Requirement 14: Prompt Wording Robustness Testing

**User Story:** As a researcher, I want to test whether results are robust to prompt variations, so that I can ensure findings generalize beyond specific wording choices.

#### Acceptance Criteria

1. THE System SHALL implement at least 2 alternative prompt wordings for each explanation strategy
2. WHEN executing robustness tests, THE System SHALL run a subset of 50 instances per dataset with alternative prompts
3. THE Metrics_Calculator SHALL compute ECS for alternative prompts and compare against baseline ECS
4. THE System SHALL compute the mean absolute difference in ECS between prompt variants
5. IF the mean ECS difference exceeds 0.15, THEN THE System SHALL log a robustness warning

### Requirement 15: Counterfactual Target Wording Ablation

**User Story:** As a researcher, I want to test whether counterfactual instructions affect agreement, so that I can assess robustness to task framing.

#### Acceptance Criteria

1. THE System SHALL implement alternative counterfactual target wordings (e.g., "flip prediction" vs "change to neutral")
2. WHEN executing counterfactual ablations, THE System SHALL run 50 instances per dataset with alternative targets
3. THE Metrics_Calculator SHALL compute agreement between counterfactual variants and other strategies
4. THE System SHALL report the variance in CF-based agreement across target wordings

### Requirement 16: Highlight Top-K Variation Testing

**User Story:** As a researcher, I want to test whether varying the number of highlighted tokens affects agreement, so that I can assess sensitivity to extraction parameters.

#### Acceptance Criteria

1. THE System SHALL execute highlighting with k=2, k=3, and k=5 tokens on a subset of 50 instances per dataset
2. THE Metrics_Calculator SHALL compute pairwise agreement between different k values
3. THE Metrics_Calculator SHALL compute correlation between k value and average agreement with other strategies
4. THE System SHALL report whether agreement is monotonic with respect to k

### Requirement 17: Normalization Variation Testing

**User Story:** As a researcher, I want to test whether normalization choices affect agreement metrics, so that I can justify preprocessing decisions.

#### Acceptance Criteria

1. THE System SHALL compute agreement metrics with and without lemmatization
2. THE System SHALL compute agreement metrics with and without stopword removal
3. THE Metrics_Calculator SHALL compare ECS distributions across normalization variants using Wilcoxon signed-rank test
4. THE System SHALL report the normalization configuration that maximizes mean ECS
5. THE System SHALL document normalization choices in the methodology section

### Requirement 18: Publication-Quality Visualization Generation

**User Story:** As a researcher, I want to generate camera-ready figures for publication, so that I can communicate findings effectively.

#### Acceptance Criteria

1. THE Visualization_Generator SHALL create a 6x6 heatmap showing mean Jaccard similarity for all strategy pairs across datasets and models
2. THE Visualization_Generator SHALL create distribution plots showing ECS distributions per dataset and per model
3. THE Visualization_Generator SHALL create scatterplots showing confidence vs ECS with regression lines and confidence bands
4. THE Visualization_Generator SHALL create bar charts comparing CC3 flip rate against random baseline flip rate
5. THE Visualization_Generator SHALL create robustness plots showing ECS variance across ablation conditions
6. FOR ALL figures, THE Visualization_Generator SHALL use publication-quality settings with 300 DPI resolution, readable fonts (size >= 10pt), and color-blind friendly palettes
7. THE Visualization_Generator SHALL export all figures to both PNG and PDF formats

### Requirement 19: Statistical Analysis and Significance Testing

**User Story:** As a researcher, I want to compute statistical significance for all key findings, so that I can make valid inferences from experimental results.

#### Acceptance Criteria

1. THE Metrics_Calculator SHALL compute 95% confidence intervals for mean ECS using bootstrap resampling with 1000 iterations
2. THE Metrics_Calculator SHALL perform pairwise comparisons between datasets using Wilcoxon signed-rank test with Bonferroni correction
3. THE Metrics_Calculator SHALL perform pairwise comparisons between models using Wilcoxon signed-rank test with Bonferroni correction
4. THE Metrics_Calculator SHALL test whether mean ECS differs significantly from 0.5 (random baseline) using one-sample t-test
5. THE System SHALL report all p-values, confidence intervals, and effect sizes in structured tables

### Requirement 20: Execution Logging and Error Tracking

**User Story:** As a researcher, I want comprehensive logs of all inference requests and parsing operations, so that I can diagnose failures and ensure reproducibility.

#### Acceptance Criteria

1. THE System SHALL log every API request with timestamp, model name, prompt hash, and response status
2. THE System SHALL log parsing failures with the raw model output and attempted extraction strategy
3. THE System SHALL log all refusals or invalid responses from models
4. THE System SHALL compute failure rates per model and per explanation strategy
5. THE System SHALL export execution logs to timestamped files in the outputs directory
6. THE System SHALL create a summary report showing total instances processed, success rate, and failure breakdown

### Requirement 21: Paper Structure Generation

**User Story:** As a researcher, I want to generate a first-draft paper with all standard sections, so that I can accelerate the writing process.

#### Acceptance Criteria

1. THE Paper_Generator SHALL create a paper with Title, Abstract, Introduction, Related Work, Methodology, Experiments, Results, Discussion, Conclusion, and Appendix sections
2. THE Paper_Generator SHALL populate the Methodology section with precise descriptions of datasets, models, explanation strategies, and metrics
3. THE Paper_Generator SHALL populate the Experiments section with experimental design, ablation configurations, and validity tests
4. THE Paper_Generator SHALL populate the Results section with metric tables, correlation coefficients, and significance tests
5. THE Paper_Generator SHALL populate the Discussion section with interpretation of findings and limitations
6. THE Paper_Generator SHALL populate the Appendix with complete prompt templates and supplementary tables
7. THE Paper_Generator SHALL generate the paper in LaTeX format suitable for NLP conference submission

### Requirement 22: Reproducibility Documentation

**User Story:** As a researcher, I want complete documentation for reproducing all experiments, so that other researchers can verify and extend the work.

#### Acceptance Criteria

1. THE System SHALL create a README file with installation instructions, environment setup, and execution commands
2. THE README SHALL document all required dependencies with exact version numbers
3. THE README SHALL provide step-by-step instructions for running each experiment stage (data preparation, inference, analysis, paper generation)
4. THE README SHALL document the expected runtime for each stage on standard hardware
5. THE README SHALL specify the required GROQ_API_KEY environment variable and Groq API rate limits
6. THE System SHALL create a requirements.txt file for Python dependencies
7. THE System SHALL create example configuration files showing all tunable parameters

### Requirement 23: Automated Testing Suite

**User Story:** As a researcher, I want automated tests verifying all components, so that I can ensure correctness and catch regressions.

#### Acceptance Criteria

1. THE Test_Suite SHALL test Dataset_Loader with mock data to verify balanced sampling and label preservation
2. THE Test_Suite SHALL test Parser with example model outputs for all four explanation strategies
3. THE Test_Suite SHALL test Normalizer with edge cases including empty strings, special characters, and non-ASCII text
4. THE Test_Suite SHALL test Metrics_Calculator with known input-output pairs for Jaccard similarity and Kendall's tau
5. THE Test_Suite SHALL test round-trip properties: FOR ALL valid explanation data, parsing then normalizing then metric computation SHALL produce valid numerical outputs
6. THE Test_Suite SHALL test Visualization_Generator to ensure all figures are created without errors
7. THE Test_Suite SHALL achieve at least 80% code coverage across all source modules
8. WHEN any test fails, THE Test_Suite SHALL report the specific assertion failure with input data and expected output

### Requirement 24: Repository Structure Organization

**User Story:** As a researcher, I want a well-organized repository structure, so that all artifacts are easy to locate and the project is maintainable.

#### Acceptance Criteria

1. THE System SHALL organize code into src/ subdirectories: load/, inference/, parsing/, normalization/, metrics/, statistics/, plots/, utils/
2. THE System SHALL store all prompt templates in the prompts/ directory organized by explanation strategy
3. THE System SHALL store raw and processed datasets in the data/ directory
4. THE System SHALL store experimental results in the outputs/ directory organized by timestamp
5. THE System SHALL store configuration files in the config/ directory
6. THE System SHALL store the generated paper and figures in the paper/ directory
7. THE System SHALL store all tests in the tests/ directory mirroring the src/ structure

### Requirement 25: Configuration Management

**User Story:** As a researcher, I want centralized configuration for all experimental parameters, so that I can easily modify settings without changing code.

#### Acceptance Criteria

1. THE System SHALL define all configurable parameters in a central configuration file including dataset names, sample sizes, model names, explanation strategies, metric thresholds, and output paths
2. THE System SHALL validate all configuration parameters at startup
3. IF any required configuration parameter is missing, THEN THE System SHALL report the specific missing parameter and exit
4. THE System SHALL support configuration overrides via command-line arguments
5. THE System SHALL log the complete configuration used for each experiment run

### Requirement 26: Incremental Execution and Checkpointing

**User Story:** As a researcher, I want to checkpoint progress during long-running experiments, so that I can resume after interruptions without restarting from scratch.

#### Acceptance Criteria

1. WHEN processing instances, THE System SHALL save results after each batch of 20 instances
2. THE System SHALL detect existing checkpoint files and skip already-processed instances
3. THE System SHALL validate checkpoint files to ensure data integrity before resuming
4. IF a checkpoint file is corrupted, THEN THE System SHALL log the corruption and restart processing from the last valid checkpoint
5. THE System SHALL provide a command-line flag to force reprocessing from scratch, ignoring checkpoints

### Requirement 27: Pretty Printer for Explanation Data

**User Story:** As a researcher, I want to format explanation data into human-readable text, so that I can inspect and debug individual instances.

#### Acceptance Criteria

1. THE Pretty_Printer SHALL format each instance with input text, ground truth label, predicted label, confidence, and all four explanations
2. THE Pretty_Printer SHALL format normalized token sets for all strategies with visual alignment
3. THE Pretty_Printer SHALL format pairwise agreement scores and ECS in a summary table
4. THE Pretty_Printer SHALL highlight consensus core tokens in the original input text
5. WHERE an explanation strategy failed, THE Pretty_Printer SHALL display an error indicator with the failure reason

### Requirement 28: Round-Trip Validation for Data Pipeline

**User Story:** As a researcher, I want to ensure data integrity throughout the pipeline, so that I can trust that metrics are computed on correct data.

#### Acceptance Criteria

1. FOR ALL instances, parsing raw model output then pretty-printing then re-parsing SHALL preserve the explanation token sets
2. FOR ALL instances, normalizing token sets then denormalizing then re-normalizing SHALL produce equivalent token sets
3. THE Test_Suite SHALL verify round-trip properties with property-based testing using at least 100 randomly generated inputs
4. IF any round-trip test fails, THEN THE Test_Suite SHALL report the specific input that violated the property

### Requirement 29: Model Output Validation

**User Story:** As a researcher, I want to validate that model outputs conform to expected formats, so that I can detect and handle malformed responses.

#### Acceptance Criteria

1. WHEN the Inference_Engine receives a classification response, THE Parser SHALL validate that it contains a label and confidence score
2. WHEN the Inference_Engine receives an explanation response, THE Parser SHALL validate that it contains extractable evidence
3. IF a response is malformed, THEN THE Parser SHALL attempt fuzzy matching to extract partial information
4. IF fuzzy matching fails, THEN THE Parser SHALL log the complete raw response and mark the instance as failed
5. THE System SHALL report validation failure rates per model and per strategy in the execution summary

### Requirement 30: Performance Optimization

**User Story:** As a researcher, I want the system to execute efficiently, so that I can complete experiments within reasonable time constraints.

#### Acceptance Criteria

1. THE Inference_Engine SHALL support concurrent API requests up to the Groq API rate limit
2. THE Inference_Engine SHALL implement request batching to minimize API overhead
3. THE Metrics_Calculator SHALL cache intermediate results to avoid redundant computation
4. THE System SHALL complete processing of 200 instances for a single dataset-model pair in under 30 minutes on standard hardware
5. THE System SHALL log timing information for each pipeline stage to identify bottlenecks
