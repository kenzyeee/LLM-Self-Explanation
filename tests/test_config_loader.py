import yaml
import pytest
from pathlib import Path
from unittest.mock import patch
from src.utils.config_loader import (
    load_yaml, _parse_config, ConfigValidator, apply_command_line_overrides,
    parse_command_line_args, save_config_to_file, load_and_validate_config,
    load_experiment_config
)
from src.utils.config import (
    ExperimentConfig, DatasetConfig, ModelConfig, InferenceConfig,
    ExplanationStrategyConfig, NormalizationConfig, MetricsConfig,
    ValidityConfig, AblationsConfig, OutputConfig, ReproducibilityConfig, Config
)
from src.utils.exceptions import ConfigurationError
from argparse import Namespace
import tempfile
import os


class TestLoadYaml:
    def test_load_valid_yaml(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("key: value\nnum: 42")
        data = load_yaml(f)
        assert data == {"key": "value", "num": 42}

    def test_file_not_found(self):
        with pytest.raises(ConfigurationError):
            load_yaml(Path("nonexistent_file.yaml"))

    def test_empty_yaml(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        with pytest.raises(ConfigurationError, match="Empty"):
            load_yaml(f)

    def test_invalid_yaml(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text(": : invalid")
        with pytest.raises(ConfigurationError):
            load_yaml(f)


class TestParseConfig:
    def test_minimal_config(self):
        data = {
            "experiment": {"name": "test", "version": "1.0"},
            "datasets": [{"name": "sst2", "huggingface_id": "g/sst2", "split": "s", "sample_size": 1, "labels": ["a", "b"]}],
            "models": [{"name": "llama", "groq_model_id": "llama-3.3-70b"}],
            "explanation_strategies": [{"id": "H", "name": "h", "prompt_file": "p.txt"}],
        }
        config = _parse_config(data)
        assert config.experiment.name == "test"
        assert len(config.datasets) == 1

    def test_empty_lists(self):
        data = {
            "experiment": {"name": "test", "version": "1.0"},
            "datasets": [],
            "models": [],
            "explanation_strategies": [],
        }
        config = _parse_config(data)
        assert config.datasets == []
        assert config.models == []
        assert config.explanation_strategies == []


class TestConfigValidator:
    def make_minimal_config(self):
        return Config(
            experiment=ExperimentConfig(name="test", version="1.0"),
            datasets=[DatasetConfig(name="sst2", huggingface_id="g/sst2", split="s", sample_size=100, labels=["a", "b"])],
            models=[ModelConfig(name="llama", groq_model_id="llama")],
            inference=InferenceConfig(),
            explanation_strategies=[
                ExplanationStrategyConfig(id="H", name="h", prompt_file="prompts/highlighting.txt", n_tokens=3),
                ExplanationStrategyConfig(id="R", name="r", prompt_file="prompts/rationale.txt"),
                ExplanationStrategyConfig(id="CF", name="cf", prompt_file="prompts/counterfactual.txt"),
                ExplanationStrategyConfig(id="RO", name="ro", prompt_file="prompts/rank_ordering.txt", n_tokens=5),
            ],
            normalization=NormalizationConfig(),
            metrics=MetricsConfig(),
            validity=ValidityConfig(),
            ablations=AblationsConfig(),
            output=OutputConfig(),
            reproducibility=ReproducibilityConfig(),
        )

    def test_valid_config(self, tmp_path):
        for fname in ["highlighting.txt", "rationale.txt", "counterfactual.txt", "rank_ordering.txt"]:
            (tmp_path / "prompts").mkdir(parents=True, exist_ok=True)
            (tmp_path / "prompts" / fname).write_text("prompt content")
        config = self.make_minimal_config()
        config.explanation_strategies[0].prompt_file = str(tmp_path / "prompts" / "highlighting.txt")
        config.explanation_strategies[1].prompt_file = str(tmp_path / "prompts" / "rationale.txt")
        config.explanation_strategies[2].prompt_file = str(tmp_path / "prompts" / "counterfactual.txt")
        config.explanation_strategies[3].prompt_file = str(tmp_path / "prompts" / "rank_ordering.txt")
        validator = ConfigValidator()
        validator.validate(config)

    def test_experiment_name_required(self):
        config = self.make_minimal_config()
        config.experiment.name = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_experiment_version_required(self):
        config = self.make_minimal_config()
        config.experiment.version = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_negative_seed(self):
        config = self.make_minimal_config()
        config.experiment.seed = -1
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_no_datasets(self):
        config = self.make_minimal_config()
        config.datasets = []
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_dataset_no_name(self):
        config = self.make_minimal_config()
        config.datasets[0].name = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_dataset_no_huggingface_id(self):
        config = self.make_minimal_config()
        config.datasets[0].huggingface_id = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_dataset_negative_sample_size(self):
        config = self.make_minimal_config()
        config.datasets[0].sample_size = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_dataset_empty_labels(self):
        config = self.make_minimal_config()
        config.datasets[0].labels = []
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_dataset_no_text_field(self):
        config = self.make_minimal_config()
        config.datasets[0].text_field = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_dataset_no_label_field(self):
        config = self.make_minimal_config()
        config.datasets[0].label_field = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_no_models(self):
        config = self.make_minimal_config()
        config.models = []
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_model_no_name(self):
        config = self.make_minimal_config()
        config.models[0].name = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_model_no_groq_id(self):
        config = self.make_minimal_config()
        config.models[0].groq_model_id = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_model_negative_context_window(self):
        config = self.make_minimal_config()
        config.models[0].context_window = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_inference_temperature_range(self):
        config = self.make_minimal_config()
        config.inference.temperature = -0.1
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_inference_temp_high(self):
        config = self.make_minimal_config()
        config.inference.temperature = 2.1
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_inference_top_p_range(self):
        config = self.make_minimal_config()
        config.inference.top_p = 1.5
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_inference_max_tokens_non_positive(self):
        config = self.make_minimal_config()
        config.inference.max_tokens = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_inference_max_retries_negative(self):
        config = self.make_minimal_config()
        config.inference.max_retries = -1
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_inference_retry_delay_non_positive(self):
        config = self.make_minimal_config()
        config.inference.retry_delay_base = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_inference_concurrent_requests_non_positive(self):
        config = self.make_minimal_config()
        config.inference.concurrent_requests = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_inference_request_timeout_non_positive(self):
        config = self.make_minimal_config()
        config.inference.request_timeout = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_no_strategies(self):
        config = self.make_minimal_config()
        config.explanation_strategies = []
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_wrong_number_of_strategies(self):
        config = self.make_minimal_config()
        config.explanation_strategies = config.explanation_strategies[:3]
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_wrong_strategy_ids(self):
        config = self.make_minimal_config()
        config.explanation_strategies[0].id = "X"
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_strategy_no_id(self):
        config = self.make_minimal_config()
        config.explanation_strategies[0].id = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_strategy_no_name(self):
        config = self.make_minimal_config()
        config.explanation_strategies[0].name = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_strategy_no_prompt_file(self):
        config = self.make_minimal_config()
        config.explanation_strategies[0].prompt_file = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_strategy_prompt_file_not_found(self):
        config = self.make_minimal_config()
        config.explanation_strategies[0].prompt_file = "nonexistent.txt"
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_strategy_n_tokens_missing(self):
        config = self.make_minimal_config()
        config.explanation_strategies[0].n_tokens = None
        # Needs prompt file to exist to get past that check
        with pytest.raises(ConfigurationError, match="n_tokens"):
            ConfigValidator().validate(config)

    def test_normalization_version_empty(self):
        config = self.make_minimal_config()
        config.normalization.version = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_normalization_bad_lemmatizer(self):
        config = self.make_minimal_config()
        config.normalization.lemmatizer = "nonexistent"
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_metrics_bootstrap_non_positive(self):
        config = self.make_minimal_config()
        config.metrics.bootstrap_iterations = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_metrics_permutation_non_positive(self):
        config = self.make_minimal_config()
        config.metrics.permutation_tests = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_metrics_confidence_level_range(self):
        config = self.make_minimal_config()
        config.metrics.confidence_level = 0.0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_metrics_confidence_level_high(self):
        config = self.make_minimal_config()
        config.metrics.confidence_level = 1.0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_validity_empty_masking_token(self):
        config = self.make_minimal_config()
        config.validity.masking_token = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_validity_non_positive_baseline(self):
        config = self.make_minimal_config()
        config.validity.n_random_baseline_trials = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_ablations_subset_size_non_positive(self):
        config = self.make_minimal_config()
        config.ablations.subset_size = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_ablations_empty_k_values(self):
        config = self.make_minimal_config()
        config.ablations.highlighting_k_values = []
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_ablations_negative_k(self):
        config = self.make_minimal_config()
        config.ablations.highlighting_k_values = [-1]
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_output_empty_base_dir(self):
        config = self.make_minimal_config()
        config.output.base_dir = ""
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_output_checkpoint_frequency_non_positive(self):
        config = self.make_minimal_config()
        config.output.checkpoint_frequency = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_output_invalid_log_level(self):
        config = self.make_minimal_config()
        config.output.log_level = "TRACE"
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_output_non_positive_dpi(self):
        config = self.make_minimal_config()
        config.output.figure_dpi = 0
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_output_empty_figure_formats(self):
        config = self.make_minimal_config()
        config.output.figure_formats = []
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)

    def test_output_invalid_figure_format(self):
        config = self.make_minimal_config()
        config.output.figure_formats = ["gif"]
        with pytest.raises(ConfigurationError):
            ConfigValidator().validate(config)


class TestParseCommandLineArgs:
    def test_defaults(self):
        args = parse_command_line_args([])
        assert args.config_dir == "config"
        assert args.force_restart is False
        assert args.skip_validation is False

    def test_custom_config_dir(self):
        args = parse_command_line_args(["--config-dir", "myconfig"])
        assert args.config_dir == "myconfig"

    def test_experiment_name(self):
        args = parse_command_line_args(["--experiment-name", "myexp"])
        assert args.experiment_name == "myexp"

    def test_seed(self):
        args = parse_command_line_args(["--seed", "123"])
        assert args.seed == 123

    def test_datasets(self):
        args = parse_command_line_args(["--datasets", "sst2", "mnli"])
        assert args.datasets == ["sst2", "mnli"]

    def test_sample_size(self):
        args = parse_command_line_args(["--sample-size", "500"])
        assert args.sample_size == 500

    def test_models(self):
        args = parse_command_line_args(["--models", "llama", "mixtral"])
        assert args.models == ["llama", "mixtral"]

    def test_temperature(self):
        args = parse_command_line_args(["--temperature", "0.5"])
        assert args.temperature == 0.5

    def test_max_retries(self):
        args = parse_command_line_args(["--max-retries", "5"])
        assert args.max_retries == 5

    def test_concurrent_requests(self):
        args = parse_command_line_args(["--concurrent-requests", "10"])
        assert args.concurrent_requests == 10

    def test_output_dir(self):
        args = parse_command_line_args(["--output-dir", "results"])
        assert args.output_dir == "results"

    def test_log_level(self):
        args = parse_command_line_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_force_restart(self):
        args = parse_command_line_args(["--force-restart"])
        assert args.force_restart is True

    def test_skip_validation(self):
        args = parse_command_line_args(["--skip-validation"])
        assert args.skip_validation is True


class TestApplyCommandLineOverrides:
    def make_config(self):
        return Config(
            experiment=ExperimentConfig(name="test", version="1.0"),
            datasets=[DatasetConfig(name="sst2", huggingface_id="g/sst2", split="s", sample_size=100, labels=["a", "b"])],
            models=[ModelConfig(name="llama", groq_model_id="llama")],
            inference=InferenceConfig(),
            explanation_strategies=[ExplanationStrategyConfig(id="H", name="h", prompt_file="p.txt")],
            normalization=NormalizationConfig(),
            metrics=MetricsConfig(),
            validity=ValidityConfig(),
            ablations=AblationsConfig(),
            output=OutputConfig(),
            reproducibility=ReproducibilityConfig(),
        )

    def test_experiment_name_override(self):
        config = self.make_config()
        args = Namespace(experiment_name="newname", seed=None, datasets=None, sample_size=None, models=None, temperature=None, max_retries=None, concurrent_requests=None, output_dir=None, log_level=None)
        config = apply_command_line_overrides(config, args)
        assert config.experiment.name == "newname"

    def test_seed_override(self):
        config = self.make_config()
        args = Namespace(experiment_name=None, seed=999, datasets=None, sample_size=None, models=None, temperature=None, max_retries=None, concurrent_requests=None, output_dir=None, log_level=None)
        config = apply_command_line_overrides(config, args)
        assert config.experiment.seed == 999

    def test_datasets_filter(self):
        config = self.make_config()
        config.datasets.append(DatasetConfig(name="mnli", huggingface_id="g/mnli", split="s", sample_size=100, labels=["a", "b", "c"]))
        args = Namespace(datasets=["sst2"], experiment_name=None, seed=None, sample_size=None, models=None, temperature=None, max_retries=None, concurrent_requests=None, output_dir=None, log_level=None)
        config = apply_command_line_overrides(config, args)
        assert len(config.datasets) == 1
        assert config.datasets[0].name == "sst2"

    def test_datasets_filter_no_match(self):
        config = self.make_config()
        args = Namespace(datasets=["nonexistent"], experiment_name=None, seed=None, sample_size=None, models=None, temperature=None, max_retries=None, concurrent_requests=None, output_dir=None, log_level=None)
        with pytest.raises(ConfigurationError):
            apply_command_line_overrides(config, args)

    def test_sample_size_override(self):
        config = self.make_config()
        args = Namespace(sample_size=500, experiment_name=None, seed=None, datasets=None, models=None, temperature=None, max_retries=None, concurrent_requests=None, output_dir=None, log_level=None)
        config = apply_command_line_overrides(config, args)
        assert config.datasets[0].sample_size == 500

    def test_models_filter(self):
        config = self.make_config()
        config.models.append(ModelConfig(name="mixtral", groq_model_id="mixtral"))
        args = Namespace(models=["mixtral"], experiment_name=None, seed=None, datasets=None, sample_size=None, temperature=None, max_retries=None, concurrent_requests=None, output_dir=None, log_level=None)
        config = apply_command_line_overrides(config, args)
        assert len(config.models) == 1
        assert config.models[0].name == "mixtral"

    def test_models_filter_no_match(self):
        config = self.make_config()
        args = Namespace(models=["nonexistent"], experiment_name=None, seed=None, datasets=None, sample_size=None, temperature=None, max_retries=None, concurrent_requests=None, output_dir=None, log_level=None)
        with pytest.raises(ConfigurationError):
            apply_command_line_overrides(config, args)

    def test_temperature_override(self):
        config = self.make_config()
        args = Namespace(temperature=0.7, experiment_name=None, seed=None, datasets=None, sample_size=None, models=None, max_retries=None, concurrent_requests=None, output_dir=None, log_level=None)
        config = apply_command_line_overrides(config, args)
        assert config.inference.temperature == 0.7

    def test_max_retries_override(self):
        config = self.make_config()
        args = Namespace(max_retries=5, experiment_name=None, seed=None, datasets=None, sample_size=None, models=None, temperature=None, concurrent_requests=None, output_dir=None, log_level=None)
        config = apply_command_line_overrides(config, args)
        assert config.inference.max_retries == 5

    def test_concurrent_requests_override(self):
        config = self.make_config()
        args = Namespace(concurrent_requests=10, experiment_name=None, seed=None, datasets=None, sample_size=None, models=None, temperature=None, max_retries=None, output_dir=None, log_level=None)
        config = apply_command_line_overrides(config, args)
        assert config.inference.concurrent_requests == 10

    def test_output_dir_override(self):
        config = self.make_config()
        args = Namespace(output_dir="results", experiment_name=None, seed=None, datasets=None, sample_size=None, models=None, temperature=None, max_retries=None, concurrent_requests=None, log_level=None)
        config = apply_command_line_overrides(config, args)
        assert config.output.base_dir == "results"

    def test_log_level_override(self):
        config = self.make_config()
        args = Namespace(log_level="DEBUG", experiment_name=None, seed=None, datasets=None, sample_size=None, models=None, temperature=None, max_retries=None, concurrent_requests=None, output_dir=None)
        config = apply_command_line_overrides(config, args)
        assert config.output.log_level == "DEBUG"


class TestSaveConfigToFile:
    def test_save_config(self, tmp_path):
        config = Config(
            experiment=ExperimentConfig(name="test", version="1.0"),
            datasets=[DatasetConfig(name="sst2", huggingface_id="g/sst2", split="s", sample_size=100, labels=["a", "b"])],
            models=[ModelConfig(name="llama", groq_model_id="llama")],
            inference=InferenceConfig(),
            explanation_strategies=[ExplanationStrategyConfig(id="H", name="h", prompt_file="p.txt")],
            normalization=NormalizationConfig(),
            metrics=MetricsConfig(),
            validity=ValidityConfig(),
            ablations=AblationsConfig(),
            output=OutputConfig(),
            reproducibility=ReproducibilityConfig(),
        )
        out = tmp_path / "sub" / "saved_config.yaml"
        save_config_to_file(config, out)
        assert out.exists()
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data["experiment"]["name"] == "test"


class TestLoadAndValidateConfig:
    def test_load_and_validate(self, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        for fname in ["highlighting.txt", "rationale.txt", "counterfactual.txt", "rank_ordering.txt"]:
            (prompts_dir / fname).write_text("prompt")

        exp_yaml = {
            "experiment": {"name": "test", "version": "1.0"},
            "datasets": [{"name": "sst2", "huggingface_id": "stanfordnlp/sst2", "split": "validation", "sample_size": 100, "labels": ["negative", "positive"]}],
            "models": [{"name": "llama", "groq_model_id": "llama-3.3-70b-versatile"}],
            "inference": {"temperature": 0, "max_tokens": 512},
            "explanation_strategies": [
                {"id": "H", "name": "highlighting", "prompt_file": str(prompts_dir / "highlighting.txt"), "n_tokens": 3},
                {"id": "R", "name": "rationale", "prompt_file": str(prompts_dir / "rationale.txt")},
                {"id": "CF", "name": "counterfactual", "prompt_file": str(prompts_dir / "counterfactual.txt")},
                {"id": "RO", "name": "rank_ordering", "prompt_file": str(prompts_dir / "rank_ordering.txt"), "n_tokens": 5},
            ],
        }
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        with open(config_dir / "experiment.yaml", "w") as f:
            yaml.dump(exp_yaml, f)

        config = load_and_validate_config(str(config_dir))
        assert config.experiment.name == "test"
        assert len(config.explanation_strategies) == 4

    def test_skip_validation(self, tmp_path):
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        for fname in ["highlighting.txt", "rationale.txt", "counterfactual.txt", "rank_ordering.txt"]:
            (prompts_dir / fname).write_text("prompt")

        exp_yaml = {
            "experiment": {"name": "test", "version": "1.0"},
            "datasets": [{"name": "sst2", "huggingface_id": "stanfordnlp/sst2", "split": "validation", "sample_size": 100, "labels": ["negative", "positive"]}],
            "models": [{"name": "llama", "groq_model_id": "llama-3.3-70b-versatile"}],
            "inference": {"temperature": 0, "max_tokens": 512},
            "explanation_strategies": [
                {"id": "H", "name": "highlighting", "prompt_file": str(prompts_dir / "highlighting.txt"), "n_tokens": 3},
                {"id": "R", "name": "rationale", "prompt_file": str(prompts_dir / "rationale.txt")},
                {"id": "CF", "name": "counterfactual", "prompt_file": str(prompts_dir / "counterfactual.txt")},
                {"id": "RO", "name": "rank_ordering", "prompt_file": str(prompts_dir / "rank_ordering.txt"), "n_tokens": 5},
            ],
        }
        config_dir = tmp_path / "config_skip"
        config_dir.mkdir()
        with open(config_dir / "experiment.yaml", "w") as f:
            yaml.dump(exp_yaml, f)

        args = Namespace(skip_validation=True, config_dir=None, experiment_name=None, seed=None, datasets=None, sample_size=None, models=None, temperature=None, max_retries=None, concurrent_requests=None, output_dir=None, log_level=None, force_restart=False)
        config = load_and_validate_config(str(config_dir), args=args)
        assert config.experiment.name == "test"


class TestLoadYamlEdgeCases:
    def test_yaml_general_exception(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("key: value")
        with patch('yaml.safe_load', side_effect=OSError("IO error")):
            with pytest.raises(ConfigurationError):
                load_yaml(f)


class TestLoadExperimentConfig:
    def test_with_datasets_yaml(self, tmp_path):
        config_dir = tmp_path / "config_with_ds"
        config_dir.mkdir()
        exp_yaml = {
            "experiment": {"name": "test", "version": "1.0"},
            "datasets": [{"name": "sst2"}],
            "models": [{"name": "llama", "groq_model_id": "llama-3.3-70b"}],
            "explanation_strategies": [
                {"id": "H", "name": "h", "prompt_file": str(tmp_path / "prompts/h.txt")},
                {"id": "R", "name": "r", "prompt_file": str(tmp_path / "prompts/r.txt")},
                {"id": "CF", "name": "cf", "prompt_file": str(tmp_path / "prompts/cf.txt")},
                {"id": "RO", "name": "ro", "prompt_file": str(tmp_path / "prompts/ro.txt"), "n_tokens": 3},
            ],
        }
        with open(config_dir / "experiment.yaml", "w") as f:
            yaml.dump(exp_yaml, f)
        ds_yaml = {"datasets": {"sst2": {"name": "sst2", "huggingface_id": "stanfordnlp/sst2", "split": "validation", "sample_size": 100, "labels": ["a", "b"]}}}
        with open(config_dir / "datasets.yaml", "w") as f:
            yaml.dump(ds_yaml, f)

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        for fname in ["h.txt", "r.txt", "cf.txt", "ro.txt"]:
            (prompts_dir / fname).write_text("prompt")

        config = load_experiment_config(config_dir)
        assert config.datasets[0].huggingface_id == "stanfordnlp/sst2"
        assert len(config.explanation_strategies) == 4

    def test_with_models_yaml(self, tmp_path):
        config_dir = tmp_path / "config_with_models"
        config_dir.mkdir()
        exp_yaml = {
            "experiment": {"name": "test", "version": "1.0"},
            "datasets": [{"name": "sst2", "huggingface_id": "stanfordnlp/sst2", "split": "v", "sample_size": 100, "labels": ["a", "b"]}],
            "models": [{"name": "llama"}],
            "explanation_strategies": [
                {"id": "H", "name": "h", "prompt_file": str(tmp_path / "prompts/h.txt")},
                {"id": "R", "name": "r", "prompt_file": str(tmp_path / "prompts/r.txt")},
                {"id": "CF", "name": "cf", "prompt_file": str(tmp_path / "prompts/cf.txt")},
                {"id": "RO", "name": "ro", "prompt_file": str(tmp_path / "prompts/ro.txt"), "n_tokens": 3},
            ],
        }
        with open(config_dir / "experiment.yaml", "w") as f:
            yaml.dump(exp_yaml, f)
        models_yaml = {"models": {"llama": {"name": "llama", "groq_model_id": "llama-3.1-70b"}}}
        with open(config_dir / "models.yaml", "w") as f:
            yaml.dump(models_yaml, f)

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        for fname in ["h.txt", "r.txt", "cf.txt", "ro.txt"]:
            (prompts_dir / fname).write_text("prompt")

        config = load_experiment_config(config_dir)
        assert config.models[0].groq_model_id == "llama-3.1-70b"

    def test_with_normalization_yaml(self, tmp_path):
        config_dir = tmp_path / "config_with_norm"
        config_dir.mkdir()
        exp_yaml = {
            "experiment": {"name": "test", "version": "1.0"},
            "datasets": [{"name": "sst2", "huggingface_id": "stanfordnlp/sst2", "split": "v", "sample_size": 100, "labels": ["a", "b"]}],
            "models": [{"name": "llama", "groq_model_id": "llama-3.3-70b"}],
            "explanation_strategies": [
                {"id": "H", "name": "h", "prompt_file": str(tmp_path / "prompts/h.txt")},
                {"id": "R", "name": "r", "prompt_file": str(tmp_path / "prompts/r.txt")},
                {"id": "CF", "name": "cf", "prompt_file": str(tmp_path / "prompts/cf.txt")},
                {"id": "RO", "name": "ro", "prompt_file": str(tmp_path / "prompts/ro.txt"), "n_tokens": 3},
            ],
        }
        with open(config_dir / "experiment.yaml", "w") as f:
            yaml.dump(exp_yaml, f)
        norm_yaml = {"normalization": {"version": "2.0", "lemmatizer": "spacy"}}
        with open(config_dir / "normalization.yaml", "w") as f:
            yaml.dump(norm_yaml, f)

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        for fname in ["h.txt", "r.txt", "cf.txt", "ro.txt"]:
            (prompts_dir / fname).write_text("prompt")

        config = load_experiment_config(config_dir)
        assert config.normalization.version == "2.0"
        assert config.normalization.lemmatizer == "spacy"

    def test_parse_config_exception(self, tmp_path):
        config_dir = tmp_path / "config_parse_err"
        config_dir.mkdir()
        with open(config_dir / "experiment.yaml", "w") as f:
            yaml.dump({"experiment": {}}, f)
        with pytest.raises(ConfigurationError):
            load_experiment_config(config_dir)


class TestConfigValidatorExtra:
    def test_strategy_name_required(self, tmp_path):
        prompts_dir = tmp_path / "prompts_e"
        prompts_dir.mkdir()
        for fname in ["h.txt", "r.txt", "cf.txt", "ro.txt"]:
            (prompts_dir / fname).write_text("prompt")
        config = Config(
            experiment=ExperimentConfig(name="test", version="1.0"),
            datasets=[DatasetConfig(name="sst2", huggingface_id="g/sst2", split="s", sample_size=100, labels=["a", "b"])],
            models=[ModelConfig(name="llama", groq_model_id="llama")],
            inference=InferenceConfig(),
            explanation_strategies=[
                ExplanationStrategyConfig(id="H", name="h", prompt_file=str(prompts_dir / "h.txt"), n_tokens=3),
                ExplanationStrategyConfig(id="R", name="r", prompt_file=str(prompts_dir / "r.txt")),
                ExplanationStrategyConfig(id="CF", name="cf", prompt_file=str(prompts_dir / "cf.txt")),
                ExplanationStrategyConfig(id="RO", name="", prompt_file=str(prompts_dir / "ro.txt"), n_tokens=5),
            ],
            normalization=NormalizationConfig(),
            metrics=MetricsConfig(),
            validity=ValidityConfig(),
            ablations=AblationsConfig(),
            output=OutputConfig(),
            reproducibility=ReproducibilityConfig(),
        )
        with pytest.raises(ConfigurationError, match="name is required"):
            ConfigValidator().validate(config)


class TestSaveConfigToFileEdgeCases:
    def test_save_config_exception(self, tmp_path):
        config = Config(
            experiment=ExperimentConfig(name="test", version="1.0"),
            datasets=[DatasetConfig(name="sst2", huggingface_id="g/sst2", split="s", sample_size=100, labels=["a", "b"])],
            models=[ModelConfig(name="llama", groq_model_id="llama")],
            inference=InferenceConfig(),
            explanation_strategies=[],
            normalization=NormalizationConfig(),
            metrics=MetricsConfig(),
            validity=ValidityConfig(),
            ablations=AblationsConfig(),
            output=OutputConfig(),
            reproducibility=ReproducibilityConfig(),
        )
        out = tmp_path / "sub" / "saved.yaml"
        with patch('yaml.dump', side_effect=OSError("write error")):
            with pytest.raises(ConfigurationError):
                save_config_to_file(config, out)


class TestLoadAndValidateConfigExtra:
    def test_load_and_validate_with_default_config_dir(self, tmp_path):
        config_dir = tmp_path / "config_default"
        config_dir.mkdir()
        exp_yaml = {
            "experiment": {"name": "test", "version": "1.0"},
            "datasets": [{"name": "sst2", "huggingface_id": "stanfordnlp/sst2", "split": "v", "sample_size": 100, "labels": ["a", "b"]}],
            "models": [{"name": "llama", "groq_model_id": "llama-3.3-70b"}],
            "explanation_strategies": [
                {"id": "H", "name": "h", "prompt_file": str(tmp_path / "prompts/h.txt"), "n_tokens": 3},
                {"id": "R", "name": "r", "prompt_file": str(tmp_path / "prompts/r.txt")},
                {"id": "CF", "name": "cf", "prompt_file": str(tmp_path / "prompts/cf.txt")},
                {"id": "RO", "name": "ro", "prompt_file": str(tmp_path / "prompts/ro.txt"), "n_tokens": 5},
            ],
        }
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        for fname in ["h.txt", "r.txt", "cf.txt", "ro.txt"]:
            (prompts_dir / fname).write_text("prompt")
        with open(config_dir / "experiment.yaml", "w") as f:
            yaml.dump(exp_yaml, f)

        args = Namespace(config_dir=str(config_dir), skip_validation=False, experiment_name=None, seed=None, datasets=None, sample_size=None, models=None, temperature=None, max_retries=None, concurrent_requests=None, output_dir=None, log_level=None, force_restart=False)
        config = load_and_validate_config(args=args)
        assert config.experiment.name == "test"
