# Implementation Plan: LLM Explanation Agreement Study

## Overview

This implementation plan breaks down the LLM Explanation Agreement Study into concrete coding tasks. The system is a research pipeline that investigates cross-strategy agreement among LLM self-explanations across three datasets (SST-2, MNLI, AG News), three Groq-compatible models, and four explanation strategies (highlighting, rationale, counterfactual, rank-ordering).

The pipeline includes: dataset loading with balanced sampling, Groq API integration with retry logic, parsing and normalization of explanations, metrics computation (Jaccard similarity, Kendall's tau, ECS, consensus cores), validity testing with consensus-core masking, statistical analysis, publication-quality visualization, and automated paper generation.

Implementation follows a bottom-up approach: first build core data structures and utilities, then implement data loading and API integration, then parsing and normalization, followed by metrics and analysis, and finally visualization and paper generation.

## Tasks

- [x] 1. Set up project structure and configuration management
  - Create directory structure: src/, config/, prompts/, data/, outputs/, paper/, tests/, scripts/
  - Create src/ subdirectories: load/, inference/, parsing/, normalization/, metrics/, statistics/, plots/, paper/, utils/
  - Create __init__.py files for all Python packages
  - Create README.md with installation instructions and execution overview
  - Create requirements.txt with all dependencies: datasets, groq, nltk, spacy, numpy, scipy, matplotlib, seaborn, pytest, hypothesis, pyyaml
  - Create .env.example file with GROQ_API_KEY placeholder
  - Create setup.py for package installation
  - _Requirements: 22.1, 22.2, 22.6, 22.7, 24.1_

- [x] 2. Implement configuration management system
  - [x] 2.1 Create configuration data models and YAML schemas
    - Create config/experiment.yaml with all experimental parameters (datasets, models, inference settings, normalization, metrics, ablations, output)
    - Create config/datasets.yaml for dataset-specific configurations
    - Create config/models.yaml for model-specific configurations
    - Create config/normalization.yaml for normalization variants
    - Define Python dataclasses for all configuration sections (ExperimentConfig, DatasetConfig, ModelConfig, InferenceConfig, NormalizationConfig)
    - _Requirements: 25.1, 25.4_
  
  - [x] 2.2 Implement configuration validation and loading
    - Create src/utils/config_loader.py with YAML loading functions
    - Implement ConfigValidator class with validation rules for all required fields and value ranges
    - Implement validation for: required fields, positive sample sizes, temperature bounds, strategy count, valid file paths
    - Add command-line argument parsing for configuration overrides using argparse
    - Implement configuration logging to output directory
    - _Requirements: 25.2, 25.3, 25.5_
  
  - [x] 2.3 Write property tests for configuration validation
    - **Property 15: Configuration Validation Completeness**
    - **Validates: Requirements 25.1, 25.2, 25.3**
    - Test that validator detects all missing required fields
    - Test that validator detects all invalid value ranges
    - Use Hypothesis to generate invalid configurations
    - _Requirements: 25.1, 25.2, 25.3_


- [x] 3. Create prompt templates for all explanation strategies
  - Create prompts/classification.txt with classification prompt template using {label_set} and {input_text} placeholders
  - Create prompts/highlighting.txt for token highlighting (H) strategy requesting 3 most important tokens
  - Create prompts/rationale.txt for one-sentence rationale (R) strategy
  - Create prompts/counterfactual.txt for minimal-edit counterfactual (CF) strategy
  - Create prompts/rank_ordering.txt for explicit rank-ordering (RO) strategy requesting 5 ranked tokens
  - Create alternative prompt variants: highlighting_alt.txt, rationale_alt.txt, counterfactual_alt.txt, rank_ordering_alt.txt for robustness testing
  - _Requirements: 4.1, 14.1, 15.1_

- [x] 4. Implement logging and error handling infrastructure
  - [x] 4.1 Create logging configuration module
    - Create src/utils/logging_config.py with setup_logging() function
    - Implement rotating file handler with 10MB max size and 5 backups
    - Implement console handler with configurable log level
    - Create detailed formatter for file logs and simple formatter for console
    - Add structured logging support with extra fields (model, strategy, instance_id, timestamp)
    - _Requirements: 20.1, 20.2, 20.5_
  
  - [x] 4.2 Create custom exception hierarchy
    - Create src/utils/exceptions.py with custom exception classes
    - Implement ExplanationStudyError as base exception
    - Implement DataLoadError, APIError, ParsingError, ValidationError, ConfigurationError
    - Add error codes and descriptive messages for each exception type
    - _Requirements: 20.2, 20.3_

- [x] 5. Implement dataset loading and sampling
  - [x] 5.1 Create dataset loader with balanced sampling
    - Create src/load/dataset_loader.py with DatasetLoader class
    - Implement load_dataset() method using Hugging Face datasets library
    - Implement sample_balanced() method using stratified sampling to ensure proportional label representation
    - Implement export_to_file() method to save as JSONL with fields: instance_id, text, label, dataset, split
    - Implement compute_statistics() method to calculate label distribution and average text length
    - Add logging for dataset loading progress and statistics
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [x] 5.2 Write unit tests for dataset loader
    - Test balanced sampling with mock datasets of various label distributions
    - Test handling of datasets with uneven label counts
    - Test export format correctness (JSONL structure, required fields)
    - Test statistics computation accuracy
    - _Requirements: 23.1_
  
  - [x] 5.3 Write property tests for dataset sampling
    - **Property 2: Balanced Sampling Preserves Label Distribution**
    - **Property 3: Sampling Preserves Data Integrity**
    - **Validates: Requirements 1.2, 1.3**
    - Test that sampled label proportions are within ±10% of original distribution
    - Test that sampling preserves all fields without modification
    - Use Hypothesis to generate synthetic datasets with various label distributions
    - _Requirements: 1.2, 1.3_


- [x] 6. Checkpoint - Ensure configuration and data loading tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement Groq API inference engine
  - [x] 7.1 Create inference engine with retry logic
    - Create src/inference/inference_engine.py with InferenceEngine class
    - Implement __init__() with API key loading from environment variable GROQ_API_KEY
    - Implement _make_request() method with exponential backoff retry logic (base 2 seconds, max 3 retries)
    - Implement classify() method for classification with confidence score extraction
    - Implement explain() method for explanation generation with strategy parameter
    - Implement classify_with_mask() method for validity testing with token masking
    - Add support for concurrent requests using asyncio with configurable concurrency limit
    - Set deterministic parameters: temperature=0, top_p=1, max_tokens=512
    - Add comprehensive logging for all API requests with timestamp, model, prompt hash, response status
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 20.1, 30.1_
  
  - [x] 7.2 Write unit tests for inference engine
    - Test retry logic with mock API failures (use monkeypatch for API calls)
    - Test exponential backoff timing
    - Test graceful failure after max retries
    - Test API key validation
    - Test request parameter configuration
    - _Requirements: 23.1_
  
  - [x] 7.3 Write property test for retry logic
    - **Property 4: Retry Logic Respects Maximum Attempts**
    - **Validates: Requirements 2.5**
    - Test that retry logic attempts at most 3 retries then fails gracefully
    - Test exponential backoff timing (2s, 4s, 8s delays)
    - Use Hypothesis to generate sequences of failures
    - _Requirements: 2.5_

- [x] 8. Implement parser for all explanation strategies
  - [x] 8.1 Create core parser with classification parsing
    - Create src/parsing/parser.py with Parser class
    - Implement parse_classification() method with regex patterns for "Prediction: X, Confidence: Y" format
    - Add fuzzy matching for label names (handle case variations, typos)
    - Add confidence extraction for both integer (0-100) and float (0.0-1.0) formats
    - Implement _fuzzy_extract() helper method for fallback matching
    - Add logging for parsing failures with raw response
    - _Requirements: 3.1, 3.2, 3.3, 29.1, 29.2_
  
  - [x] 8.2 Implement highlighting and rank-ordering parsers
    - Implement parse_highlighting() method to extract exactly 3 tokens
    - Handle multiple formats: numbered lists, quoted tokens, comma-separated tokens
    - Take first 3 tokens if more provided, log warning if fewer than 3
    - Implement parse_rank_ordering() method to extract 5 tokens with ranks
    - Handle formats: numbered lists "1. X", natural language "Most important: X, Second: Y"
    - Return list of (token, rank) tuples
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 8.1, 8.2, 8.3_

  
  - [x] 8.3 Implement rationale and counterfactual parsers
    - Implement parse_rationale() method to extract one-sentence rationale
    - Extract first complete sentence after prompt, handle explanation before/after rationale
    - Implement parse_counterfactual() method to extract counterfactual text
    - Handle formats with "Original:" and "Counterfactual:" sections
    - Handle inline counterfactuals in natural language responses
    - Add validation for extracted content (non-empty, reasonable length)
    - _Requirements: 6.1, 7.1_
  
  - [x] 8.4 Write unit tests for all parser methods
    - Create tests/fixtures/sample_responses.json with example model outputs for all strategies
    - Test parse_highlighting() with various formats (numbered, quoted, comma-separated)
    - Test parse_rationale() with rationales in different positions
    - Test parse_counterfactual() with different section markers
    - Test parse_rank_ordering() with numbered and natural language formats
    - Test fuzzy matching fallback logic
    - Test handling of malformed responses
    - _Requirements: 23.2_
  
  - [x] 8.5 Write property test for parser extraction completeness
    - **Property 5: Parser Extraction Completeness**
    - **Validates: Requirements 5.1, 5.2, 6.1, 7.1, 8.1, 8.2**
    - Test that parser successfully extracts structured evidence from valid formats without data loss
    - Use Hypothesis to generate valid response formats for all strategies
    - Verify extracted data matches input data after parsing
    - _Requirements: 5.1, 5.2, 6.1, 7.1, 8.1, 8.2_

- [x] 9. Implement normalization pipeline
  - [x] 9.1 Create normalizer with configurable pipeline
    - Create src/normalization/normalizer.py with Normalizer class
    - Implement __init__() with NormalizationConfig parameter
    - Implement normalize() method applying pipeline: lowercase → remove punctuation → stopword removal → lemmatization
    - Implement normalize_set() method to normalize list of tokens and return as set
    - Implement _lowercase(), _remove_punctuation(), _lemmatize() helper methods
    - Add WordNetLemmatizer and spaCy lemmatizer support (configurable)
    - Load NLTK stopwords for configurable language
    - _Requirements: 5.5, 6.4, 6.5, 8.4, 17.1_
  
  - [x] 9.2 Implement specialized extraction methods
    - Implement extract_content_words_from_rationale() method
    - Tokenize rationale using NLTK or spaCy
    - Extract content words (nouns, verbs, adjectives, adverbs) using POS tagging
    - Remove stopwords and apply normalization pipeline
    - Implement extract_counterfactual_diff() method
    - Tokenize both original and counterfactual texts
    - Compute set difference: deleted = original - counterfactual, added = counterfactual - original
    - Return union of deleted and added tokens after normalization
    - _Requirements: 6.2, 6.3, 7.2, 7.3, 7.4, 7.5_

  
  - [x] 9.3 Write unit tests for normalization pipeline
    - Test normalization with edge cases: empty strings, special characters, non-ASCII text, numbers
    - Test stopword removal with various stopword sets
    - Test lemmatization with both WordNet and spaCy
    - Test content word extraction from various rationale formats
    - Test counterfactual diff with minimal edits, substantial edits, no edits
    - _Requirements: 23.3_
  
  - [x] 9.4 Write property tests for normalization
    - **Property 1: Normalization Pipeline Idempotence**
    - **Property 6: Counterfactual Diff Extraction Correctness**
    - **Property 17: Normalization Idempotence**
    - **Validates: Requirements 5.5, 6.3, 6.4, 6.5, 7.2, 7.3, 7.4, 7.5, 8.4, 28.2**
    - Test that normalizing twice gives same result as normalizing once
    - Test counterfactual diff correctly identifies union of deleted and added tokens
    - Test normalize(normalize(T)) = normalize(T) for any token set T
    - Use Hypothesis to generate random tokens and text pairs
    - _Requirements: 5.5, 6.4, 7.2, 7.3, 7.4, 7.5, 8.4, 28.2_

- [x] 10. Checkpoint - Ensure parsing and normalization tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement metrics calculator for agreement and consensus
  - [x] 11.1 Create core metrics calculator with Jaccard and Kendall's tau
    - Create src/metrics/metrics_calculator.py with MetricsCalculator class
    - Implement compute_jaccard_similarity() method: intersection / union
    - Handle edge case: empty sets return 1.0 (identical)
    - Implement compute_kendalls_tau() method using scipy.stats.kendalltau
    - Implement assign_implicit_ranks() to assign ranks based on token order for non-ranked strategies
    - Find common tokens between rankings before computing tau
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [x] 11.2 Implement pairwise agreement and ECS computation
    - Implement compute_pairwise_agreements() method
    - Compute Jaccard for all 6 strategy pairs: H-R, H-CF, H-RO, R-CF, R-RO, CF-RO
    - Compute Kendall's tau between RO and other strategies with implicit ranks
    - Return dictionary mapping (strategy1, strategy2) tuples to agreement scores
    - Implement compute_ecs() method computing arithmetic mean of 6 pairwise Jaccard similarities
    - Handle instances with failed strategies by excluding from ECS computation
    - _Requirements: 9.5, 10.1, 10.2, 10.3_
  
  - [x] 11.3 Implement consensus core identification
    - Implement compute_consensus_core() method with parameter k
    - Count token occurrences across all explanation strategies using defaultdict
    - Select tokens appearing in at least k strategies
    - Return CC3 (k=3) and CC4 (k=4) token sets
    - Compute size distribution of consensus cores across instances
    - Identify instances with empty consensus cores for error analysis
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  
  - [x] 11.4 Write unit tests for metrics calculator
    - Test Jaccard similarity with known input-output pairs (including edge cases: empty sets, identical sets, disjoint sets)
    - Test Kendall's tau with known rankings (perfect agreement, perfect disagreement, partial overlap)
    - Test ECS computation with all six pairwise agreements
    - Test consensus core identification with various token distributions
    - Test handling of missing strategies (instances with failed explanations)
    - _Requirements: 23.4_
  
  - [x] 11.5 Write property tests for metrics
    - **Property 7: Rank Information Preservation**
    - **Property 8: Jaccard Similarity Mathematical Properties**
    - **Property 9: Kendall's Tau Properties**
    - **Property 11: ECS Mean Computation**
    - **Property 12: Consensus Core Set Operations**
    - **Validates: Requirements 8.5, 9.1, 9.2, 9.3, 9.4, 10.1, 11.1, 11.2**
    - Test Jaccard symmetry: J(A,B) = J(B,A)
    - Test Jaccard bounds: 0 ≤ J(A,B) ≤ 1
    - Test Jaccard identity: J(A,A) = 1 for non-empty A
    - Test Kendall's tau symmetry, bounds, perfect agreement
    - Test ECS equals mean of 6 pairwise agreements
    - Test CC4 ⊆ CC3
    - Test CC4 equals intersection of all 4 strategy sets
    - Use Hypothesis to generate random token sets and rankings
    - _Requirements: 8.5, 9.1, 9.2, 9.3, 9.4, 10.1, 11.1, 11.2_

- [x] 12. Implement statistical analysis module
  - [x] 12.1 Create correlation analysis with bootstrap confidence intervals
    - Create src/statistics/statistical_tests.py with statistical analysis functions
    - Implement compute_confidence_ecs_correlation() function using scipy.stats.spearmanr
    - Implement bootstrap resampling with 1000 iterations to compute 95% confidence intervals
    - Store bootstrap correlation coefficients and compute 2.5th and 97.5th percentiles
    - Return CorrelationResult dataclass with rho, p-value, ci_lower, ci_upper
    - _Requirements: 12.1, 12.2, 12.3_
  
  - [x] 12.2 Implement significance testing functions
    - Implement permutation_test() function with 10000 permutations for null hypothesis testing
    - Implement wilcoxon_signed_rank_test() with Bonferroni correction for multiple comparisons
    - Implement one_sample_ttest() to test if mean ECS differs from 0.5 baseline
    - Implement paired_ttest() for comparing flip rates in validity testing
    - Return structured results with t-statistic/z-statistic, p-value, effect size (Cohen's d)
    - _Requirements: 12.4, 12.5, 19.1, 19.2, 19.3, 19.4, 19.5_

- [x] 13. Implement validity checker with consensus-core masking
  - [x] 13.1 Create validity checker for consensus-core removal tests
    - Create src/metrics/validity_checker.py with ValidityChecker class
    - Implement test_consensus_core_removal() method that masks CC3/CC4 tokens with [MASK]
    - Use inference_engine.classify_with_mask() to get prediction with masked tokens
    - Compare original prediction vs masked prediction to determine flip
    - Return FlipResult dataclass with original_prediction, masked_prediction, flipped boolean, masked_tokens
    - _Requirements: 13.1, 13.2_

  
  - [x] 13.2 Implement random baseline and statistical comparison
    - Implement test_random_removal_baseline() method that randomly selects n tokens and masks them
    - Use uniform random selection for token masking (same count as consensus core)
    - Implement compute_flip_rate() method computing percentage of instances where prediction changed
    - Implement compare_flip_rates() method using paired t-test (scipy.stats.ttest_rel)
    - Compare CC3 flip rate vs random flip rate across all instances
    - Return StatisticalTest dataclass with t_statistic, p_value, mean_diff
    - _Requirements: 13.3, 13.4, 13.5, 13.6_
  
  - [x] 13.3 Write unit tests for validity checker
    - Test masking logic with various token sets
    - Test flip detection (prediction changed vs unchanged)
    - Test random token selection uniformity
    - Test flip rate computation with known boolean arrays
    - Test statistical comparison methods
    - _Requirements: 23.5_
  
  - [x] 13.4 Write property tests for validity checker
    - **Property 13: Flip Rate Computation**
    - **Property 14: Random Token Selection Fairness**
    - **Validates: Requirements 13.2, 13.3**
    - Test that flip rate equals proportion of True values, bounded [0.0, 1.0]
    - Test that random selection produces exactly k tokens with uniform probability
    - Use Hypothesis to generate flip result lists and token sets
    - _Requirements: 13.2, 13.3_

- [x] 14. Implement checkpoint manager for incremental execution
  - Create src/utils/checkpoint_manager.py with CheckpointManager class
  - Implement save_checkpoint() method saving results after each batch of 20 instances to JSONL
  - Implement load_checkpoint() method detecting existing checkpoint files
  - Implement validate_checkpoint() method checking data integrity (valid JSON, required fields)
  - Implement skip_processed_instances() method filtering already-processed instances
  - Add command-line flag --force-restart to ignore checkpoints and reprocess from scratch
  - Handle corrupted checkpoints by logging error and restarting from last valid checkpoint
  - _Requirements: 26.1, 26.2, 26.3, 26.4, 26.5_

- [x] 15. Checkpoint - Ensure metrics and validity tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Implement pretty printer for explanation data
  - Create src/utils/pretty_printer.py with PrettyPrinter class
  - Implement format_instance() method displaying input text, ground truth, prediction, confidence, and all 4 explanations
  - Implement format_normalized_tokens() method showing normalized token sets with visual alignment
  - Implement format_pairwise_agreements() method showing 6 pairwise scores and ECS in table format
  - Implement highlight_consensus_core() method highlighting CC3/CC4 tokens in original input text using ANSI colors
  - Display error indicators with failure reason for failed explanation strategies
  - _Requirements: 27.1, 27.2, 27.3, 27.4, 27.5_


- [x] 17. Implement data models and result storage
  - Create src/utils/data_models.py with all dataclasses
  - Define InstanceResult dataclass with 30+ fields: identifiers, input, classification, raw explanations, normalized explanations, parsing status, pairwise agreements, consensus metrics
  - Define AggregateMetrics dataclass with aggregation level, ECS statistics, pairwise agreement means, consensus core statistics, correlation results, parsing success rates
  - Define ValidityTestResult dataclass with CC3/CC4 removal tests and random baseline
  - Define AggregateValidityResults dataclass with flip rates and statistical comparisons
  - Define CorrelationResult, StatisticalTest, FlipResult, ExecutionSummary dataclasses
  - Implement serialization methods (to_dict, from_dict) for all dataclasses for JSON export
  - _Requirements: Design Section: Data Models_

- [x] 18. Implement main experiment orchestration pipeline
  - [x] 18.1 Create main experiment execution script
    - Create scripts/run_experiment.py with main experiment orchestration
    - Load configuration and validate all parameters
    - Initialize all components: dataset_loader, inference_engine, parser, normalizer, metrics_calculator, validity_checker
    - Setup logging to timestamped output directory
    - Load and sample all datasets (SST-2, MNLI, AG News)
    - _Requirements: 25.5, 20.5, 20.6_
  
  - [x] 18.2 Implement per-instance processing loop
    - For each instance: run classification, extract confidence
    - Randomize order of 4 explanation strategies (H, R, CF, RO)
    - For each strategy: run explanation request, parse response, normalize tokens
    - Log strategy execution order and timestamps
    - Handle API failures gracefully with logging and continue
    - Save checkpoint every 20 instances
    - _Requirements: 3.1, 3.2, 3.3, 4.2, 4.3, 4.5, 20.1, 20.2_
  
  - [x] 18.3 Implement metrics computation and aggregation
    - Compute pairwise agreements (Jaccard for all 6 pairs, Kendall's tau for RO pairs)
    - Compute ECS for instances with all 4 valid explanations
    - Compute consensus cores (CC3, CC4) for all instances
    - Store per-instance results to outputs/{timestamp}/instance_results.jsonl
    - Aggregate metrics by dataset, by model, and overall
    - Store aggregate metrics to outputs/{timestamp}/aggregate_metrics.json
    - _Requirements: 9.5, 10.4, 11.4_
  
  - [x] 18.4 Write property test for strategy randomization
    - **Property 10: Strategy Order Randomization**
    - **Validates: Requirements 4.2**
    - Test that execution order is a valid permutation of [H, R, CF, RO]
    - Test randomness across multiple executions with different seeds
    - Use Hypothesis to verify randomization distribution
    - _Requirements: 4.2_


- [x] 19. Implement validity testing pipeline
  - Create scripts/run_validity_tests.py for consensus-core masking experiments
  - Load instance results from previous experiment run
  - For each instance with non-empty CC3: run test_consensus_core_removal() for CC3 and CC4
  - For each instance: run test_random_removal_baseline() with same token count as CC3
  - Store validity test results to outputs/{timestamp}/validity_tests.jsonl
  - Compute aggregate flip rates for CC3, CC4, and random baseline
  - Perform paired t-test comparing CC3 vs random flip rates
  - Store aggregate validity results with statistical tests
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6_

- [x] 20. Checkpoint - Ensure main pipeline executes successfully
  - Ensure all tests pass, ask the user if questions arise.

- [x] 21. Implement ablation studies for robustness testing
  - [x] 21.1 Create prompt wording robustness tests
    - Create scripts/run_ablations.py for all ablation experiments
    - Implement run_prompt_ablation() function loading alternative prompts
    - Run subset of 50 instances per dataset with alternative prompts for each strategy
    - Compute ECS for alternative prompts and compare against baseline ECS
    - Compute mean absolute difference in ECS between prompt variants
    - Log robustness warning if mean ECS difference exceeds 0.15
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_
  
  - [x] 21.2 Create normalization variation tests
    - Implement run_normalization_ablation() function testing variants: with/without lemmatization, with/without stopwords
    - Compute agreement metrics for each normalization variant
    - Compare ECS distributions using Wilcoxon signed-rank test
    - Report normalization configuration that maximizes mean ECS
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5_
  
  - [x] 21.3 Create highlighting k-variation and counterfactual target tests
    - Implement run_highlighting_k_ablation() testing k=2, k=3, k=5 tokens on 50 instances per dataset
    - Compute pairwise agreement between different k values
    - Test if agreement is monotonic with k
    - Implement run_counterfactual_target_ablation() with alternative target wordings
    - Compute agreement variance across counterfactual targets
    - Store all ablation results to outputs/{timestamp}/robustness_tests.json
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 16.1, 16.2, 16.3, 16.4_

- [x] 22. Implement visualization generator for publication figures
  - [x] 22.1 Create agreement heatmap visualization
    - Create src/plots/visualization_generator.py with VisualizationGenerator class
    - Implement plot_agreement_heatmap() method creating 6x6 heatmap using seaborn
    - Show mean Jaccard similarity for all strategy pairs across datasets and models
    - Use 'RdYlGn' colormap with vmin=0, vmax=1
    - Set figure size to 8x6 inches, font size ≥ 10pt
    - Export to both PDF and PNG at 300 DPI
    - _Requirements: 18.1, 18.6, 18.7_

  
  - [x] 22.2 Create ECS distribution and confidence-ECS scatter plots
    - Implement plot_ecs_distributions() method creating distribution plots using matplotlib histograms or seaborn distplot
    - Show ECS distributions per dataset and per model with overlays
    - Implement plot_confidence_ecs_scatter() method creating scatterplot with regression line
    - Add confidence bands using seaborn regplot
    - Annotate with Spearman's rho and p-value
    - Use color-blind friendly palette (seaborn 'colorblind')
    - _Requirements: 18.2, 18.3, 18.6, 18.7_
  
  - [x] 22.3 Create flip rate comparison and robustness analysis plots
    - Implement plot_flip_rate_comparison() method creating bar chart comparing CC3 vs random baseline
    - Add error bars showing 95% confidence intervals
    - Annotate with statistical significance indicators (*, **, ***)
    - Implement plot_robustness_analysis() method creating plots showing ECS variance across ablation conditions
    - Create faceted plots for prompt variants, normalization variants, and k-value variants
    - Export all figures to paper/figures/ directory
    - _Requirements: 18.4, 18.5, 18.6, 18.7_
  
  - [x] 22.4 Write unit tests for visualization generator
    - Test that all figure generation functions complete without errors (smoke tests)
    - Test figure file creation (PDF and PNG exist)
    - Verify figure format specifications (check DPI, file size)
    - Use mock data to test plotting functions
    - _Requirements: 23.6_

- [x] 23. Implement paper generator for LaTeX output
  - [x] 23.1 Create paper structure generator
    - Create src/paper/paper_generator.py with PaperGenerator class
    - Implement generate_paper() method creating complete LaTeX document
    - Implement _generate_title_abstract() creating title and 150-200 word abstract
    - Implement _generate_introduction() with background, motivation, research questions (placeholder for manual completion)
    - Create LaTeX document structure with standard NLP conference template
    - _Requirements: 21.1_
  
  - [x] 23.2 Create methodology and experiments sections
    - Implement _generate_methodology() populating complete descriptions of datasets, models, strategies, and metrics
    - Include dataset statistics tables, model configurations, prompt templates (referenced from appendix)
    - Implement _generate_experiments() describing experimental design, ablation configurations, validity tests
    - Include detailed protocol for each experiment type
    - _Requirements: 21.2, 21.3_
  
  - [x] 23.3 Create results, discussion, and appendix sections
    - Implement _generate_results() populating metric tables, correlation coefficients, significance tests
    - Generate LaTeX tables from aggregate_metrics.json and validity_tests results
    - Include references to all figures (heatmap, scatter, distributions, flip rates)
    - Implement _generate_discussion() with interpretation of findings and limitations
    - Implement _generate_appendix() with complete prompt templates and supplementary tables
    - Implement _generate_conclusion() summarizing contributions
    - Export to paper/draft_paper.tex
    - _Requirements: 21.4, 21.5, 21.6, 21.7_


- [x] 24. Create execution scripts and CLI
  - Create scripts/generate_paper.py script that loads experimental results and generates paper
  - Load figures from paper/figures/, metrics from outputs/latest/, validity tests from outputs/latest/
  - Call paper_generator.generate_paper() with all data
  - Create scripts/analyze_results.py for interactive result exploration (optional: Jupyter notebook or CLI tool)
  - Add argument parsing to all scripts for dataset selection, model selection, output directory
  - Add --help documentation for all CLI arguments
  - _Requirements: 22.3, 25.4_

- [x] 25. Checkpoint - Ensure end-to-end pipeline runs successfully
  - Ensure all tests pass, ask the user if questions arise.

- [x] 26. Implement comprehensive testing suite
  - [x] 26.1 Create round-trip property tests
    - Create tests/test_round_trip_properties.py
    - **Property 16: Pretty Printer Round-Trip**
    - **Validates: Requirements 28.1**
    - Test parse → normalize → pretty_print → parse → normalize preserves token sets
    - Use Hypothesis to generate instance results with explanation data
    - Test round-trip for all explanation strategies
    - _Requirements: 28.1_
  
  - [x] 26.2 Create response validation property tests
    - **Property 18: Response Format Validation**
    - **Validates: Requirements 29.1, 29.2**
    - Test validator correctly classifies valid/invalid responses for each strategy
    - Use Hypothesis to generate valid and invalid response formats
    - Test fuzzy matching boundaries
    - _Requirements: 29.1, 29.2_
  
  - [x] 26.3 Create integration test for full pipeline
    - Create tests/test_integration.py with end-to-end integration test
    - Use small test dataset (10 instances) with mocked Groq API responses
    - Test complete pipeline: load → infer → parse → normalize → metrics → validity → visualize
    - Verify all output files created correctly
    - Test checkpoint save/load functionality
    - _Requirements: 23.7_
  
  - [x] 26.4 Run full test suite and verify coverage
    - Run pytest with coverage: pytest tests/ --cov=src --cov-report=html
    - Verify minimum 80% code coverage across all source modules
    - Generate coverage report and identify untested code paths
    - Add additional tests for uncovered branches and edge cases
    - _Requirements: 23.7, 23.8_


- [x] 27. Create comprehensive documentation
  - Update README.md with complete installation instructions (Python 3.9+, dependencies, NLTK data downloads, spaCy model downloads)
  - Add environment setup instructions (GROQ_API_KEY configuration, virtual environment)
  - Add step-by-step execution commands for each stage: data preparation, main experiment, validity tests, ablations, paper generation
  - Document expected runtime for each stage on standard hardware
  - Document Groq API rate limits and recommended concurrency settings
  - Add example outputs showing expected file structure after each stage
  - Add troubleshooting section for common issues (API errors, parsing failures, missing dependencies)
  - Document how to customize configuration (datasets, models, normalization, ablations)
  - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.5_

- [x] 28. Implement execution summary and reporting
  - Create src/utils/execution_summary.py with ExecutionSummary dataclass
  - Track start_time, end_time, duration_seconds
  - Track processing statistics: total_instances, successful_instances, failed_instances
  - Track failure breakdown: parsing_failures by strategy, api_failures, normalization_failures
  - Track performance statistics: avg_time_per_instance, api_requests_total, api_requests_failed
  - Implement generate_report() method creating human-readable summary report
  - Save summary report to outputs/{timestamp}/execution_summary.txt
  - Display summary report to console at end of experiment
  - _Requirements: 20.4, 20.6_

- [x] 29. Implement performance optimizations
  - Implement concurrent API requests using asyncio with configurable concurrency limit (default: 5)
  - Implement request batching to minimize API overhead where possible
  - Implement caching for metrics_calculator intermediate results (pairwise agreements) to avoid redundant computation
  - Add timing instrumentation for each pipeline stage (data loading, inference, parsing, normalization, metrics, validity, visualization)
  - Log timing information to identify bottlenecks
  - Add performance benchmarks to documentation (expected time for 200 instances)
  - _Requirements: 30.1, 30.2, 30.3, 30.4, 30.5_

- [x] 30. Final integration testing and validation
  - Run complete pipeline on small test dataset (30 instances across 3 datasets)
  - Verify all output files created: instance_results.jsonl, aggregate_metrics.json, validity_tests.jsonl, robustness_tests.json, all figures, draft_paper.tex
  - Verify all figures render correctly and match specifications (300 DPI, correct dimensions, readable fonts)
  - Verify LaTeX paper compiles successfully
  - Verify execution logs contain all required information
  - Verify checkpoint/resume functionality works correctly
  - Test with different configuration variants (different models, normalization settings, ablation parameters)
  - Test error handling with intentional failures (invalid API key, malformed responses, network errors)
  - _Requirements: All requirements_

- [x] 31. Final checkpoint - Complete system validation
  - Ensure all tests pass, ask the user if questions arise.


## Notes

- Tasks marked with `*` are optional testing tasks and can be skipped for faster MVP delivery
- Each task references specific requirements from requirements.md for traceability
- Checkpoint tasks ensure incremental validation at key milestones
- Property-based tests validate 18 universal correctness properties from the design document
- Unit tests and integration tests complement property-based tests for comprehensive coverage
- Testing tasks are sub-tasks under their parent implementation tasks to ensure testing happens immediately after implementation
- All code uses Python 3.9+ with type hints for clarity and maintainability
- The implementation follows a bottom-up approach: utilities and data structures first, then core pipeline components, then analysis and output generation
- Alternative prompt templates support robustness testing without modifying core prompts
- Configuration management enables easy experimentation with different parameters
- Checkpointing enables resumption of long-running experiments after interruptions
- All figures and paper content are generated programmatically for reproducibility
- The system is designed for extensibility: new datasets, models, strategies, or metrics can be added with minimal changes


## Task Dependency Graph

```json
{
  "waves": [
    {
      "id": 0,
      "tasks": ["1", "3"]
    },
    {
      "id": 1,
      "tasks": ["2.1", "4.1", "4.2"]
    },
    {
      "id": 2,
      "tasks": ["2.2", "5.1", "17"]
    },
    {
      "id": 3,
      "tasks": ["2.3", "5.2", "5.3", "7.1"]
    },
    {
      "id": 4,
      "tasks": ["7.2", "7.3", "8.1"]
    },
    {
      "id": 5,
      "tasks": ["8.2", "8.3"]
    },
    {
      "id": 6,
      "tasks": ["8.4", "8.5", "9.1"]
    },
    {
      "id": 7,
      "tasks": ["9.2"]
    },
    {
      "id": 8,
      "tasks": ["9.3", "9.4", "11.1"]
    },
    {
      "id": 9,
      "tasks": ["11.2", "11.3"]
    },
    {
      "id": 10,
      "tasks": ["11.4", "11.5", "12.1", "12.2", "14", "16"]
    },
    {
      "id": 11,
      "tasks": ["13.1"]
    },
    {
      "id": 12,
      "tasks": ["13.2"]
    },
    {
      "id": 13,
      "tasks": ["13.3", "13.4", "18.1"]
    },
    {
      "id": 14,
      "tasks": ["18.2"]
    },
    {
      "id": 15,
      "tasks": ["18.3"]
    },
    {
      "id": 16,
      "tasks": ["18.4", "19"]
    },
    {
      "id": 17,
      "tasks": ["21.1", "21.2", "21.3"]
    },
    {
      "id": 18,
      "tasks": ["22.1", "22.2"]
    },
    {
      "id": 19,
      "tasks": ["22.3"]
    },
    {
      "id": 20,
      "tasks": ["22.4", "23.1"]
    },
    {
      "id": 21,
      "tasks": ["23.2"]
    },
    {
      "id": 22,
      "tasks": ["23.3", "24"]
    },
    {
      "id": 23,
      "tasks": ["26.1", "26.2", "26.3"]
    },
    {
      "id": 24,
      "tasks": ["26.4", "27", "28", "29"]
    },
    {
      "id": 25,
      "tasks": ["30"]
    }
  ]
}
```
