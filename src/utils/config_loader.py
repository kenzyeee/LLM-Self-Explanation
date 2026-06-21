"""Configuration loading and validation."""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import (
    AblationsConfig, Config, DatasetConfig, ExperimentConfig,
    ExplanationStrategyConfig, InferenceConfig, MetricsConfig,
    ModelConfig, NormalizationConfig, OutputConfig,
    ReproducibilityConfig, ValidityConfig,
)
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def load_yaml(filepath: Path) -> Dict[str, Any]:
    if not filepath.exists():
        raise ConfigurationError(f"Configuration file not found: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if data is None:
            raise ConfigurationError(f"Empty configuration file: {filepath}")
        return data
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {filepath}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading {filepath}: {e}")


def load_experiment_config(config_dir: Path = Path("config")) -> Config:
    experiment_path = config_dir / "experiment.yaml"
    exp_data = load_yaml(experiment_path)

    datasets_path = config_dir / "datasets.yaml"
    if datasets_path.exists():
        datasets_data = load_yaml(datasets_path)
        if "datasets" in datasets_data:
            dataset_details = {d["name"]: d for d in datasets_data["datasets"].values()}
            for ds in exp_data.get("datasets", []):
                if ds["name"] in dataset_details:
                    for key, value in dataset_details[ds["name"]].items():
                        ds.setdefault(key, value)

    models_path = config_dir / "models.yaml"
    if models_path.exists():
        models_data = load_yaml(models_path)
        if "models" in models_data:
            model_details = {m["name"]: m for m in models_data["models"].values()}
            for model in exp_data.get("models", []):
                if model["name"] in model_details:
                    for key, value in model_details[model["name"]].items():
                        model.setdefault(key, value)

    normalization_path = config_dir / "normalization.yaml"
    if normalization_path.exists():
        norm_data = load_yaml(normalization_path)
        if "normalization" in norm_data:
            exp_data["normalization"] = norm_data["normalization"]

    try:
        config = _parse_config(exp_data)
        logger.info(f"Successfully loaded configuration from {config_dir}")
        return config
    except Exception as e:
        raise ConfigurationError(f"Error parsing configuration: {e}")


def _parse_config(data: Dict[str, Any]) -> Config:
    experiment = ExperimentConfig(**data.get("experiment", {}))
    datasets = [DatasetConfig(**ds) for ds in data.get("datasets", [])]
    models = [ModelConfig(**m) for m in data.get("models", [])]
    inference = InferenceConfig(**data.get("inference", {}))
    strategies = [ExplanationStrategyConfig(**s) for s in data.get("explanation_strategies", [])]
    normalization = NormalizationConfig(**data.get("normalization", {}))
    metrics = MetricsConfig(**data.get("metrics", {}))
    validity = ValidityConfig(**data.get("validity", {}))
    ablations = AblationsConfig(**data.get("ablations", {}))
    output = OutputConfig(**data.get("output", {}))
    reproducibility = ReproducibilityConfig(**data.get("reproducibility", {}))

    return Config(
        experiment=experiment, datasets=datasets, models=models,
        inference=inference, explanation_strategies=strategies,
        normalization=normalization, metrics=metrics, validity=validity,
        ablations=ablations, output=output, reproducibility=reproducibility,
    )


class ConfigValidator:
    def validate(self, config: Config) -> None:
        logger.info("Validating configuration...")
        self._validate_experiment(config.experiment)
        self._validate_datasets(config.datasets)
        self._validate_models(config.models)
        self._validate_inference(config.inference)
        self._validate_explanation_strategies(config.explanation_strategies)
        self._validate_normalization(config.normalization)
        self._validate_metrics(config.metrics)
        self._validate_validity(config.validity)
        self._validate_ablations(config.ablations)
        self._validate_output(config.output)
        self._validate_reproducibility(config.reproducibility)
        logger.info("Configuration validation passed")

    def _validate_experiment(self, exp: ExperimentConfig) -> None:
        if not exp.name:
            raise ConfigurationError("Experiment name is required")
        if not exp.version:
            raise ConfigurationError("Experiment version is required")
        if exp.seed < 0:
            raise ConfigurationError("Random seed must be non-negative")

    def _validate_datasets(self, datasets: List[DatasetConfig]) -> None:
        if not datasets:
            raise ConfigurationError("At least one dataset must be configured")
        for ds in datasets:
            if not ds.name:
                raise ConfigurationError("Dataset name is required")
            if not ds.huggingface_id:
                raise ConfigurationError(f"Dataset {ds.name}: huggingface_id is required")
            if ds.sample_size <= 0:
                raise ConfigurationError(f"Dataset {ds.name}: sample_size must be positive, got {ds.sample_size}")
            if not ds.labels:
                raise ConfigurationError(f"Dataset {ds.name}: labels list cannot be empty")
            if not ds.text_field:
                raise ConfigurationError(f"Dataset {ds.name}: text_field is required")
            if not ds.label_field:
                raise ConfigurationError(f"Dataset {ds.name}: label_field is required")

    def _validate_models(self, models: List[ModelConfig]) -> None:
        if not models:
            raise ConfigurationError("At least one model must be configured")
        for model in models:
            if not model.name:
                raise ConfigurationError("Model name is required")
            if not model.model_id:
                raise ConfigurationError(f"Model {model.name}: model_id is required")
            if model.context_window <= 0:
                raise ConfigurationError(f"Model {model.name}: context_window must be positive")

    def _validate_inference(self, inference: InferenceConfig) -> None:
        if not (0.0 <= inference.temperature <= 2.0):
            raise ConfigurationError(f"Temperature must be in range [0.0, 2.0], got {inference.temperature}")
        if not (0.0 <= inference.top_p <= 1.0):
            raise ConfigurationError(f"top_p must be in range [0.0, 1.0], got {inference.top_p}")
        if inference.max_tokens <= 0:
            raise ConfigurationError("max_tokens must be positive")
        if inference.max_retries < 0:
            raise ConfigurationError("max_retries must be non-negative")
        if inference.retry_delay_base <= 0:
            raise ConfigurationError("retry_delay_base must be positive")
        if inference.concurrent_requests <= 0:
            raise ConfigurationError("concurrent_requests must be positive")
        if inference.request_timeout <= 0:
            raise ConfigurationError("request_timeout must be positive")

    def _validate_explanation_strategies(self, strategies: List[ExplanationStrategyConfig]) -> None:
        if not strategies:
            raise ConfigurationError("At least one explanation strategy must be configured")
        if len(strategies) != 4:
            raise ConfigurationError(f"Exactly 4 explanation strategies required, got {len(strategies)}")
        required_ids = {"H", "R", "CF", "RO"}
        actual_ids = {s.id for s in strategies}
        if actual_ids != required_ids:
            raise ConfigurationError(f"Strategy IDs must be {required_ids}, got {actual_ids}")
        for strategy in strategies:
            if not strategy.name:
                raise ConfigurationError(f"Strategy {strategy.id}: name is required")
            if not strategy.prompt_file:
                raise ConfigurationError(f"Strategy {strategy.id}: prompt_file is required")
            prompt_path = Path(strategy.prompt_file)
            if not prompt_path.exists():
                raise ConfigurationError(f"Strategy {strategy.id}: prompt file not found: {prompt_path}")
            if strategy.id in ("H", "RO") and (strategy.n_tokens is None or strategy.n_tokens <= 0):
                raise ConfigurationError(f"Strategy {strategy.id}: n_tokens must be positive")

    def _validate_normalization(self, norm: NormalizationConfig) -> None:
        if not norm.version:
            raise ConfigurationError("Normalization version is required")
        if norm.lemmatizer not in ("wordnet", "spacy"):
            raise ConfigurationError(f"Lemmatizer must be 'wordnet' or 'spacy', got '{norm.lemmatizer}'")

    def _validate_metrics(self, metrics: MetricsConfig) -> None:
        if metrics.bootstrap_iterations <= 0:
            raise ConfigurationError("bootstrap_iterations must be positive")
        if metrics.permutation_tests <= 0:
            raise ConfigurationError("permutation_tests must be positive")
        if not (0.0 < metrics.confidence_level < 1.0):
            raise ConfigurationError(f"confidence_level must be in range (0.0, 1.0), got {metrics.confidence_level}")

    def _validate_validity(self, validity: ValidityConfig) -> None:
        if not validity.masking_token:
            raise ConfigurationError("masking_token is required")
        if validity.n_random_baseline_trials <= 0:
            raise ConfigurationError("n_random_baseline_trials must be positive")

    def _validate_ablations(self, ablations: AblationsConfig) -> None:
        if ablations.subset_size <= 0:
            raise ConfigurationError("Ablation subset_size must be positive")
        if not ablations.highlighting_k_values:
            raise ConfigurationError("highlighting_k_values cannot be empty")
        for k in ablations.highlighting_k_values:
            if k <= 0:
                raise ConfigurationError(f"All k values must be positive, got {k}")

    def _validate_output(self, output: OutputConfig) -> None:
        if not output.base_dir:
            raise ConfigurationError("Output base_dir is required")
        if output.checkpoint_frequency <= 0:
            raise ConfigurationError("checkpoint_frequency must be positive")
        if output.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ConfigurationError(f"Invalid log_level: {output.log_level}")
        if output.figure_dpi <= 0:
            raise ConfigurationError("figure_dpi must be positive")
        if not output.figure_formats:
            raise ConfigurationError("At least one figure format must be specified")
        valid_formats = {"pdf", "png", "svg", "jpg", "jpeg"}
        for fmt in output.figure_formats:
            if fmt not in valid_formats:
                raise ConfigurationError(f"Invalid figure format '{fmt}'. Must be one of: {valid_formats}")

    def _validate_reproducibility(self, repro: ReproducibilityConfig) -> None:
        pass


def parse_command_line_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Explanation Agreement Study")
    parser.add_argument("--config-dir", type=str, default="config", help="Directory containing configuration files")
    parser.add_argument("--experiment-name", type=str, help="Override experiment name")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--datasets", type=str, nargs="+", help="Override dataset selection")
    parser.add_argument("--sample-size", type=int, help="Override sample size for all datasets")
    parser.add_argument("--models", type=str, nargs="+", help="Override model selection")
    parser.add_argument("--temperature", type=float, help="Override inference temperature")
    parser.add_argument("--max-retries", type=int, help="Override max API retries")
    parser.add_argument("--concurrent-requests", type=int, help="Override concurrent request limit")
    parser.add_argument("--output-dir", type=str, help="Override output base directory")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Override logging level")
    parser.add_argument("--force-restart", action="store_true", help="Ignore checkpoints and restart from scratch")
    parser.add_argument("--skip-validation", action="store_true", help="Skip configuration validation")
    return parser.parse_args(args)


def apply_command_line_overrides(config: Config, args: argparse.Namespace) -> Config:
    if args.experiment_name:
        config.experiment.name = args.experiment_name
    if args.seed is not None:
        config.experiment.seed = args.seed
    if args.datasets:
        dataset_names = set(args.datasets)
        config.datasets = [d for d in config.datasets if d.name in dataset_names]
        if not config.datasets:
            raise ConfigurationError(f"No valid datasets found matching: {args.datasets}")
    if args.sample_size is not None:
        for dataset in config.datasets:
            dataset.sample_size = args.sample_size
    if args.models:
        model_names = set(args.models)
        config.models = [m for m in config.models if m.name in model_names]
        if not config.models:
            raise ConfigurationError(f"No valid models found matching: {args.models}")
    if args.temperature is not None:
        config.inference.temperature = args.temperature
    if args.max_retries is not None:
        config.inference.max_retries = args.max_retries
    if args.concurrent_requests is not None:
        config.inference.concurrent_requests = args.concurrent_requests
    if args.output_dir:
        config.output.base_dir = args.output_dir
    if args.log_level:
        config.output.log_level = args.log_level
    return config


def save_config_to_file(config: Config, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = config.to_dict()
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise ConfigurationError(f"Failed to save configuration: {e}")


def load_and_validate_config(config_dir: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> Config:
    if config_dir is None:
        config_dir = args.config_dir if args else "config"
    config_path = Path(config_dir)
    logger.info(f"Loading configuration from {config_path}")
    config = load_experiment_config(config_path)
    if args:
        logger.info("Applying command-line overrides")
        config = apply_command_line_overrides(config, args)
    if not args or not args.skip_validation:
        validator = ConfigValidator()
        validator.validate(config)
    else:
        logger.warning("Configuration validation skipped (not recommended)")
    return config
