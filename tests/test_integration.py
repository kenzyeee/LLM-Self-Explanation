import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock


class TestIntegration:
    @pytest.fixture
    def sample_instance(self):
        from src.load.dataset_loader import Instance
        return Instance(
            instance_id="test_001",
            text="This movie was great and wonderful.",
            label="positive",
            dataset="sst2",
            split="validation"
        )

    def test_dataset_loader_load_and_prepare(self, tmp_path):
        from src.load.dataset_loader import DatasetLoader
        mock_ds = Mock()
        mock_ds.__len__ = Mock(return_value=10)
        mock_data = [{'sentence': f'text {i}', 'label': i % 2} for i in range(10)]
        mock_ds.__iter__ = Mock(return_value=iter(mock_data))
        mock_ds.__getitem__ = Mock(side_effect=lambda idx: mock_data[idx])

        with patch('src.load.dataset_loader.load_dataset', return_value=mock_ds):
            loader = DatasetLoader(seed=42)
            config = {
                'name': 'sst2',
                'huggingface_id': 'stanfordnlp/sst2',
                'split': 'validation',
                'sample_size': 6,
                'text_field': 'sentence',
                'label_field': 'label',
            }
            instances, stats = loader.load_and_prepare_dataset(config, tmp_path)
            assert len(instances) == 6
            assert stats.total_count == 6
            export_file = tmp_path / 'sst2_sampled.jsonl'
            assert export_file.exists()

    def test_data_model_round_trip(self):
        from src.utils.data_models import InstanceResult
        from datetime import datetime

        original = InstanceResult(
            instance_id="test_001",
            dataset="sst2",
            model="llama-3.3-70b",
            timestamp=datetime.now(),
            text="Great movie!",
            ground_truth_label="positive",
            predicted_label="positive",
            confidence=0.85,
            correct=True,
            raw_highlighting="1. great\n2. movie",
            raw_rationale="The text uses positive language.",
            raw_counterfactual="Terrible movie!",
            raw_rank_ordering="1. great\n2. movie\n3. wonderful",
            highlighting_tokens={"great", "movie"},
            rationale_tokens={"positive", "language"},
            counterfactual_tokens={"terrible"},
            rank_ordering_tokens=[("great", 1), ("movie", 2), ("wonderful", 3)],
            highlighting_parsed=True,
            rationale_parsed=True,
            counterfactual_parsed=True,
            rank_ordering_parsed=True,
            jaccard_H_R=0.5,
            jaccard_H_CF=0.0,
            jaccard_H_RO=0.6,
            jaccard_R_CF=0.0,
            jaccard_R_RO=0.3,
            jaccard_CF_RO=0.0,
            kendall_H_RO=0.8,
            ecs=0.233,
            cc3_tokens={"great"},
            cc4_tokens=set(),
            cc3_size=1,
            cc4_size=0,
        )

        d = original.to_dict()
        restored = InstanceResult.from_dict(d)

        assert restored.instance_id == original.instance_id
        assert restored.text == original.text
        assert restored.highlighting_tokens == original.highlighting_tokens
        assert restored.rank_ordering_tokens == original.rank_ordering_tokens
        assert restored.ecs == original.ecs
        assert restored.cc3_tokens == original.cc3_tokens
        assert restored.jaccard_H_R == original.jaccard_H_R

    def test_metrics_pipeline(self):
        from src.metrics.metrics_calculator import MetricsCalculator

        calc = MetricsCalculator()
        explanations = {
            "H": {"great", "movie", "wonderful"},
            "R": {"great", "movie", "acting"},
            "CF": {"great", "terrible"},
            "RO": {"great", "movie", "wonderful", "acting"},
        }

        agreements = calc.compute_pairwise_agreements(explanations)
        assert len(agreements) == 6

        ecs = calc.compute_ecs(agreements)
        assert 0.0 <= ecs <= 1.0

        cc3 = calc.compute_consensus_core(explanations, 3)
        assert "great" in cc3

        cc4 = calc.compute_consensus_core(explanations, 4)
        # great appears in 3 strategies (H, R, CF), not all 4
        # but RO also has "great", so it appears in all 4
        # Let's use a different check
        cc4_2 = calc.compute_consensus_core({"H": {"a","b"}, "R": {"a","c"}, "CF": {"a","d"}, "RO": {"b","c"}}, 4)
        assert cc4_2 == set()  # no token in all 4

    def test_statistical_tests(self):
        from src.statistics.statistical_tests import (
            compute_confidence_ecs_correlation,
            permutation_test,
            sign_flip_permutation_test,
            holm_correction,
        )

        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
        ecs_values = [0.2, 0.3, 0.4, 0.5, 0.6]

        corr = compute_confidence_ecs_correlation(confidences, ecs_values, n_bootstrap=50)
        assert corr.rho > 0  # should be positive correlation

        p_val = permutation_test(confidences, ecs_values, n_permutations=100, seed=1)
        assert 0.0 <= p_val <= 1.0

        p_flip = sign_flip_permutation_test([0.2, 0.3, 0.1, 0.25, 0.15, 0.3], seed=1)
        assert p_flip is not None and 0.0 < p_flip <= 1.0

        adj = holm_correction([0.01, 0.03, None])
        assert adj[2] is None and adj[0] is not None

    def test_checkpoint_manager(self, tmp_path):
        from src.utils.checkpoint_manager import CheckpointManager

        cp_file = tmp_path / "checkpoint.jsonl"
        mgr = CheckpointManager(cp_file)

        mgr.save_checkpoint([{"instance_id": "1"}, {"instance_id": "2"}])
        assert cp_file.exists()

        loaded = mgr.load_checkpoint()
        assert len(loaded) == 2
        assert loaded[0]["instance_id"] == "1"

        assert mgr.validate_checkpoint() is True

        remaining = mgr.skip_processed_instances(
            [Mock(instance_id="1"), Mock(instance_id="2"), Mock(instance_id="3")],
            {"1", "2"}
        )
        assert len(remaining) == 1
        assert remaining[0].instance_id == "3"

    def test_paper_generator(self, tmp_path):
        from src.paper.paper_generator import PaperGenerator

        results = {"mean_ecs": 0.35, "datasets": ["sst2", "mnli", "ag_news"]}
        config = {"experiment": {"name": "test"}}

        gen = PaperGenerator(results=results, config=config)
        output = tmp_path / "draft_paper.tex"
        gen.generate_paper(output)

        assert output.exists()
        content = output.read_text()
        assert "\\begin{document}" in content
        assert "\\section{Methodology}" in content
        assert "\\section{Results}" in content
        assert "\\begin{abstract}" in content
