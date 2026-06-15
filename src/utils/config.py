"""Configuration data models for the LLM Explanation Agreement Study."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ExperimentConfig:
    name: str
    version: str
    seed: int = 42
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetConfig:
    name: str
    huggingface_id: str
    split: str
    sample_size: int
    labels: List[str]
    text_field: str = "text"
    secondary_text_field: Optional[str] = None
    label_field: str = "label"
    cf_max_edit_ratio: float = 0.3
    description: Optional[str] = None
    task_type: Optional[str] = None
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    name: str
    groq_model_id: str
    description: Optional[str] = None
    context_window: int = 8192
    default_temperature: float = 0.0
    default_top_p: float = 1.0
    supports_function_calling: bool = False
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    max_retries: int = 3
    retry_delay_base: float = 2.0
    concurrent_requests: int = 5
    request_timeout: int = 30
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplanationStrategyConfig:
    id: str
    name: str
    prompt_file: str
    n_tokens: Optional[int] = None
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizationConfig:
    version: str = "v1.0"
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    use_lemmatization: bool = True
    stopword_language: str = "english"
    lemmatizer: str = "wordnet"
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricsConfig:
    bootstrap_iterations: int = 1000
    permutation_tests: int = 10000
    confidence_level: float = 0.95
    bonferroni_correction: bool = True
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidityConfig:
    masking_token: str = "[MASK]"
    n_random_baseline_trials: int = 10
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AblationsConfig:
    prompt_variants: bool = True
    normalization_variants: bool = True
    highlighting_k_values: List[int] = field(default_factory=lambda: [2, 3, 5])
    subset_size: int = 50
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OutputConfig:
    base_dir: str = "outputs"
    checkpoint_frequency: int = 20
    log_level: str = "INFO"
    figure_dpi: int = 300
    figure_formats: List[str] = field(default_factory=lambda: ["pdf", "png"])
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReproducibilityConfig:
    log_git_commit: bool = True
    log_package_versions: bool = True
    save_config_with_results: bool = True
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Config:
    experiment: ExperimentConfig
    datasets: List[DatasetConfig]
    models: List[ModelConfig]
    inference: InferenceConfig
    explanation_strategies: List[ExplanationStrategyConfig]
    normalization: NormalizationConfig
    metrics: MetricsConfig
    validity: ValidityConfig
    ablations: AblationsConfig
    output: OutputConfig
    reproducibility: ReproducibilityConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment': self.experiment.to_dict(),
            'datasets': [d.to_dict() for d in self.datasets],
            'models': [m.to_dict() for m in self.models],
            'inference': self.inference.to_dict(),
            'explanation_strategies': [s.to_dict() for s in self.explanation_strategies],
            'normalization': self.normalization.to_dict(),
            'metrics': self.metrics.to_dict(),
            'validity': self.validity.to_dict(),
            'ablations': self.ablations.to_dict(),
            'output': self.output.to_dict(),
            'reproducibility': self.reproducibility.to_dict(),
        }

    def get_dataset_by_name(self, name: str) -> Optional[DatasetConfig]:
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        return None

    def get_model_by_name(self, name: str) -> Optional[ModelConfig]:
        for model in self.models:
            if model.name == name:
                return model
        return None

    def get_strategy_by_id(self, strategy_id: str) -> Optional[ExplanationStrategyConfig]:
        for strategy in self.explanation_strategies:
            if strategy.id == strategy_id:
                return strategy
        return None
