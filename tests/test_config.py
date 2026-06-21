from src.utils.config import (
    ExperimentConfig, DatasetConfig, ModelConfig, InferenceConfig,
    ExplanationStrategyConfig, NormalizationConfig, MetricsConfig,
    ValidityConfig, AblationsConfig, OutputConfig, ReproducibilityConfig, Config
)
from pathlib import Path


class TestExperimentConfig:
    def test_default_seed(self):
        cfg = ExperimentConfig(name="test", version="1.0")
        assert cfg.seed == 42
        assert cfg.name == "test"
        assert cfg.version == "1.0"

    def test_custom_seed(self):
        cfg = ExperimentConfig(name="test", version="1.0", seed=123)
        assert cfg.seed == 123

    def test_to_dict(self):
        cfg = ExperimentConfig(name="test", version="1.0")
        d = cfg.to_dict()
        assert d == {"name": "test", "version": "1.0", "seed": 42}


class TestDatasetConfig:
    def test_minimal(self):
        cfg = DatasetConfig(name="sst2", huggingface_id="stanfordnlp/sst2", split="train", sample_size=100, labels=["pos", "neg"])
        assert cfg.name == "sst2"
        assert cfg.text_field == "text"
        assert cfg.label_field == "label"

    def test_full(self):
        cfg = DatasetConfig(
            name="mnli", huggingface_id="nyu-mll/multi_nli", split="validation_matched",
            sample_size=200, labels=["entailment", "neutral", "contradiction"],
            text_field="premise", secondary_text_field="hypothesis",
            label_field="gold_label", description="MNLI", task_type="nli"
        )
        assert cfg.secondary_text_field == "hypothesis"
        assert cfg.description == "MNLI"
        assert cfg.task_type == "nli"

    def test_to_dict(self):
        cfg = DatasetConfig(name="sst2", huggingface_id="stanfordnlp/sst2", split="train", sample_size=100, labels=["pos", "neg"])
        d = cfg.to_dict()
        assert d["name"] == "sst2"
        assert d["sample_size"] == 100


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(name="llama", model_id="llama-3.3-70b-versatile")
        assert cfg.context_window == 8192
        assert cfg.default_temperature == 0.0
        assert cfg.supports_function_calling is False

    def test_to_dict(self):
        cfg = ModelConfig(name="llama", model_id="llama-3.3-70b-versatile")
        d = cfg.to_dict()
        assert d["model_id"] == "llama-3.3-70b-versatile"


class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 512
        assert cfg.concurrent_requests == 5

    def test_to_dict(self):
        cfg = InferenceConfig(temperature=0.5, max_tokens=256)
        d = cfg.to_dict()
        assert d["temperature"] == 0.5
        assert d["max_tokens"] == 256


class TestExplanationStrategyConfig:
    def test_minimal(self):
        cfg = ExplanationStrategyConfig(id="H", name="highlighting", prompt_file="prompts/highlighting.txt")
        assert cfg.n_tokens is None

    def test_with_ntokens(self):
        cfg = ExplanationStrategyConfig(id="RO", name="rank_ordering", prompt_file="prompts/ro.txt", n_tokens=5)
        assert cfg.n_tokens == 5

    def test_to_dict(self):
        cfg = ExplanationStrategyConfig(id="H", name="highlighting", prompt_file="prompts/h.txt")
        d = cfg.to_dict()
        assert d["id"] == "H"
        assert d["prompt_file"] == "prompts/h.txt"


class TestNormalizationConfig:
    def test_defaults(self):
        cfg = NormalizationConfig()
        assert cfg.lowercase is True
        assert cfg.lemmatizer == "wordnet"

    def test_custom(self):
        cfg = NormalizationConfig(version="v2.0", lowercase=False, remove_stopwords=False, lemmatizer="spacy")
        assert cfg.version == "v2.0"
        assert cfg.lemmatizer == "spacy"

    def test_to_dict(self):
        cfg = NormalizationConfig()
        d = cfg.to_dict()
        assert d["version"] == "v1.0"


class TestMetricsConfig:
    def test_defaults(self):
        cfg = MetricsConfig()
        assert cfg.bootstrap_iterations == 1000
        assert cfg.bonferroni_correction is True

    def test_to_dict(self):
        cfg = MetricsConfig(bootstrap_iterations=500)
        d = cfg.to_dict()
        assert d["bootstrap_iterations"] == 500


class TestValidityConfig:
    def test_defaults(self):
        cfg = ValidityConfig()
        assert cfg.masking_token == "[MASK]"
        assert cfg.n_random_baseline_trials == 10

    def test_to_dict(self):
        cfg = ValidityConfig(masking_token="<MASK>")
        d = cfg.to_dict()
        assert d["masking_token"] == "<MASK>"


class TestAblationsConfig:
    def test_defaults(self):
        cfg = AblationsConfig()
        assert cfg.highlighting_k_values == [2, 3, 5]
        assert cfg.subset_size == 50

    def test_to_dict(self):
        cfg = AblationsConfig(subset_size=100)
        d = cfg.to_dict()
        assert d["subset_size"] == 100


class TestOutputConfig:
    def test_defaults(self):
        cfg = OutputConfig()
        assert cfg.base_dir == "outputs"
        assert cfg.figure_formats == ["pdf", "png"]

    def test_custom(self):
        cfg = OutputConfig(base_dir="results", figure_formats=["svg"])
        assert cfg.base_dir == "results"

    def test_to_dict(self):
        cfg = OutputConfig()
        d = cfg.to_dict()
        assert d["base_dir"] == "outputs"


class TestReproducibilityConfig:
    def test_defaults(self):
        cfg = ReproducibilityConfig()
        assert cfg.log_git_commit is True

    def test_to_dict(self):
        cfg = ReproducibilityConfig(log_git_commit=False)
        d = cfg.to_dict()
        assert d["log_git_commit"] is False


class TestConfig:
    def test_full_config(self):
        cfg = Config(
            experiment=ExperimentConfig(name="test", version="1.0"),
            datasets=[DatasetConfig(name="sst2", huggingface_id="stanfordnlp/sst2", split="train", sample_size=100, labels=["pos", "neg"])],
            models=[ModelConfig(name="llama", model_id="llama-3.3-70b-versatile")],
            inference=InferenceConfig(),
            explanation_strategies=[ExplanationStrategyConfig(id="H", name="h", prompt_file="p.txt")],
            normalization=NormalizationConfig(),
            metrics=MetricsConfig(),
            validity=ValidityConfig(),
            ablations=AblationsConfig(),
            output=OutputConfig(),
            reproducibility=ReproducibilityConfig(),
        )
        assert cfg.experiment.name == "test"
        assert cfg.datasets[0].sample_size == 100

    def test_get_dataset_by_name(self):
        cfg = Config(
            experiment=ExperimentConfig(name="test", version="1.0"),
            datasets=[DatasetConfig(name="sst2", huggingface_id="g", split="s", sample_size=1, labels=["a", "b"])],
            models=[ModelConfig(name="llama", model_id="llama")],
            inference=InferenceConfig(),
            explanation_strategies=[ExplanationStrategyConfig(id="H", name="h", prompt_file="p.txt")],
            normalization=NormalizationConfig(),
            metrics=MetricsConfig(),
            validity=ValidityConfig(),
            ablations=AblationsConfig(),
            output=OutputConfig(),
            reproducibility=ReproducibilityConfig(),
        )
        assert cfg.get_dataset_by_name("sst2") is not None
        assert cfg.get_dataset_by_name("nonexistent") is None

    def test_get_model_by_name(self):
        cfg = Config(
            experiment=ExperimentConfig(name="test", version="1.0"),
            datasets=[DatasetConfig(name="sst2", huggingface_id="g", split="s", sample_size=1, labels=["a", "b"])],
            models=[ModelConfig(name="llama", model_id="llama")],
            inference=InferenceConfig(),
            explanation_strategies=[ExplanationStrategyConfig(id="H", name="h", prompt_file="p.txt")],
            normalization=NormalizationConfig(),
            metrics=MetricsConfig(),
            validity=ValidityConfig(),
            ablations=AblationsConfig(),
            output=OutputConfig(),
            reproducibility=ReproducibilityConfig(),
        )
        assert cfg.get_model_by_name("llama") is not None
        assert cfg.get_model_by_name("nonexistent") is None

    def test_get_strategy_by_id(self):
        cfg = Config(
            experiment=ExperimentConfig(name="test", version="1.0"),
            datasets=[DatasetConfig(name="sst2", huggingface_id="g", split="s", sample_size=1, labels=["a", "b"])],
            models=[ModelConfig(name="llama", model_id="llama")],
            inference=InferenceConfig(),
            explanation_strategies=[ExplanationStrategyConfig(id="H", name="h", prompt_file="p.txt")],
            normalization=NormalizationConfig(),
            metrics=MetricsConfig(),
            validity=ValidityConfig(),
            ablations=AblationsConfig(),
            output=OutputConfig(),
            reproducibility=ReproducibilityConfig(),
        )
        assert cfg.get_strategy_by_id("H") is not None
        assert cfg.get_strategy_by_id("nonexistent") is None

    def test_to_dict(self):
        cfg = Config(
            experiment=ExperimentConfig(name="test", version="1.0"),
            datasets=[DatasetConfig(name="sst2", huggingface_id="g", split="s", sample_size=1, labels=["a", "b"])],
            models=[ModelConfig(name="llama", model_id="llama")],
            inference=InferenceConfig(),
            explanation_strategies=[ExplanationStrategyConfig(id="H", name="h", prompt_file="p.txt")],
            normalization=NormalizationConfig(),
            metrics=MetricsConfig(),
            validity=ValidityConfig(),
            ablations=AblationsConfig(),
            output=OutputConfig(),
            reproducibility=ReproducibilityConfig(),
        )
        d = cfg.to_dict()
        assert d["experiment"]["name"] == "test"
        assert len(d["datasets"]) == 1
        assert len(d["models"]) == 1
