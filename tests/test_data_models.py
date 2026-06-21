import json
import tempfile
from pathlib import Path
from datetime import datetime

from src.utils.data_models import (
    InstanceResult, AggregateMetrics, ValidityTestResult,
    ExecutionSummary,
    AggregateValidityResults,
    save_instance_results, load_instance_results,
    save_aggregate_metrics, load_aggregate_metrics,
    save_validity_results, load_validity_results,
)


class TestInstanceResult:
    def test_minimal_round_trip(self):
        r = InstanceResult(
            instance_id="1", dataset="sst2", model="llama",
            timestamp=datetime.now(), text="great", ground_truth_label="pos",
            predicted_label="pos", confidence=0.9, correct=True,
            raw_highlighting="", raw_rationale="", raw_counterfactual="", raw_rank_ordering="",
        )
        d = r.to_dict()
        restored = InstanceResult.from_dict(d)
        assert restored.instance_id == "1"

    def test_full_round_trip(self):
        r = InstanceResult(
            instance_id="1", dataset="sst2", model="llama",
            timestamp=datetime.now(), text="great movie", ground_truth_label="pos",
            predicted_label="pos", confidence=0.95, correct=True,
            raw_highlighting="h", raw_rationale="r", raw_counterfactual="cf", raw_rank_ordering="ro",
            highlighting_tokens={"good", "great"}, rationale_tokens={"nice"},
            counterfactual_tokens={"bad"}, rank_ordering_tokens=[("good", 1), ("bad", 2)],
            highlighting_parsed=True, rationale_parsed=True,
            counterfactual_parsed=True, rank_ordering_parsed=True,
            jaccard_H_R=0.5, jaccard_H_CF=0.3, jaccard_H_RO=0.7,
            jaccard_R_CF=0.2, jaccard_R_RO=0.6, jaccard_CF_RO=0.4,
            kendall_H_RO=0.8, ecs=0.5,
            cc3_tokens={"good"}, cc4_tokens=set(), cc3_size=1, cc4_size=0,
        )
        d = r.to_dict()
        restored = InstanceResult.from_dict(d)
        assert restored.ecs == 0.5
        assert "good" in restored.highlighting_tokens
        assert restored.rank_ordering_tokens == [("good", 1), ("bad", 2)]


class TestAggregateMetrics:
    def test_round_trip(self):
        m = AggregateMetrics(
            aggregation_level="dataset", group_name="sst2", n_instances=100,
            mean_ecs=0.5, std_ecs=0.1, median_ecs=0.5,
            ecs_ci_lower=0.45, ecs_ci_upper=0.55,
            mean_jaccard_H_R=0.4, mean_jaccard_H_CF=0.3, mean_jaccard_H_RO=0.6,
            mean_jaccard_R_CF=0.2, mean_jaccard_R_RO=0.5, mean_jaccard_CF_RO=0.4,
            mean_overlap_H_R=0.6, mean_overlap_H_CF=0.5, mean_overlap_H_RO=0.7,
            mean_overlap_R_CF=0.4, mean_overlap_R_RO=0.6, mean_overlap_CF_RO=0.5,
            mean_kendall_H_RO=0.7,
            mean_normalized_kendall_H_RO=0.85,
            mean_rbo_H_RO=0.65,
            mean_cc3_size=3.0, mean_cc4_size=1.0,
            pct_instances_with_cc3=0.8, pct_instances_with_cc4=0.5,
            spearman_rho=0.3, spearman_p_value=0.01,
            correlation_ci_lower=0.1, correlation_ci_upper=0.5,
            highlighting_success_rate=0.9, rationale_success_rate=0.85,
            counterfactual_success_rate=0.8, rank_ordering_success_rate=0.75,
            mean_ecs_extraction_rationale=0.45, mean_ecs_extraction_perturbation=0.35,
        )
        d = m.to_dict()
        restored = AggregateMetrics.from_dict(d)
        assert restored.mean_ecs == 0.5
        assert restored.group_name == "sst2"
        assert restored.mean_ecs_extraction_rationale == 0.45
        assert restored.mean_ecs_extraction_perturbation == 0.35
        assert restored.mean_rbo_H_RO == 0.65


class TestValidityTestResult:
    def test_round_trip(self):
        r = ValidityTestResult(
            instance_id="1", dataset="sst2", model="llama",
            cc3_tokens={"good", "great"}, cc3_original_prediction="pos",
            cc3_masked_prediction="neg", cc3_flipped=True,
            cc4_tokens={"good"}, cc4_original_prediction="pos",
            cc4_masked_prediction="pos", cc4_flipped=False,
            random_tokens={"bad"}, random_original_prediction="pos",
            random_masked_prediction="pos", random_flipped=False,
        )
        d = r.to_dict()
        restored = ValidityTestResult.from_dict(d)
        assert restored.cc3_flipped is True
        assert "good" in restored.cc3_tokens
        assert not restored.random_flipped

    def test_minimal(self):
        r = ValidityTestResult(instance_id="1", dataset="sst2", model="llama")
        d = r.to_dict()
        restored = ValidityTestResult.from_dict(d)
        assert restored.cc3_tokens == set()
        assert not restored.cc3_flipped


class TestExecutionSummary:
    def test_round_trip(self):
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 12, 30, 0)
        s = ExecutionSummary(
            start_time=start, end_time=end, duration_seconds=9000,
            total_instances=100, successful_instances=95, failed_instances=5,
            parsing_failures={"H": 2, "R": 1},
            api_failures=1, normalization_failures=1,
            avg_time_per_instance=90.0, api_requests_total=400, api_requests_failed=5,
        )
        d = s.to_dict()
        restored = ExecutionSummary.from_dict(d)
        assert restored.total_instances == 100
        assert restored.parsing_failures["H"] == 2

    def test_generate_report(self):
        s = ExecutionSummary(
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 5, 0),
            duration_seconds=300, total_instances=50,
            successful_instances=45, failed_instances=5,
            parsing_failures={"H": 3}, api_failures=2, normalization_failures=1,
            avg_time_per_instance=6.0, api_requests_total=200, api_requests_failed=5,
        )
        report = s.generate_report()
        assert "Execution Summary" in report
        assert "50" in report
        assert "90.0" in report

    def test_generate_report_zero_instances(self):
        s = ExecutionSummary(
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 0, 0),
            duration_seconds=0, total_instances=0,
            successful_instances=0, failed_instances=0,
        )
        report = s.generate_report()
        assert "0" in report
        assert "0.0" in report

    def test_generate_report_empty_parsing_failures(self):
        s = ExecutionSummary(
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 5, 0),
            duration_seconds=300, total_instances=50,
            successful_instances=48, failed_instances=2,
            parsing_failures={}, api_requests_total=200,
        )
        report = s.generate_report()
        # Should not crash with empty parsing_failures
        assert "API" in report


class TestAggregateValidityResults:
    def test_round_trip(self):
        r = AggregateValidityResults(
            dataset="sst2", model="llama", n_instances=100,
            cc3_flip_rate=0.25, cc4_flip_rate=0.15, random_flip_rate=0.1,
            t_statistic=3.5, p_value=0.001, effect_size=0.5,
            cc3_flip_ci_lower=0.15, cc3_flip_ci_upper=0.35,
            random_flip_ci_lower=0.05, random_flip_ci_upper=0.15,
        )
        d = r.to_dict()
        restored = AggregateValidityResults.from_dict(d)
        assert restored.t_statistic == 3.5


class TestConvenienceFunctions:
    def test_save_load_instance_results(self, tmp_path):
        r = InstanceResult(
            instance_id="1", dataset="sst2", model="llama",
            timestamp=datetime.now(), text="good", ground_truth_label="pos",
            predicted_label="pos", confidence=0.9, correct=True,
            raw_highlighting="", raw_rationale="", raw_counterfactual="", raw_rank_ordering="",
        )
        f = tmp_path / "results.jsonl"
        save_instance_results([r], str(f))
        loaded = load_instance_results(str(f))
        assert len(loaded) == 1
        assert loaded[0].instance_id == "1"

    def test_save_load_multiple_instance_results(self, tmp_path):
        results = []
        for i in range(3):
            results.append(InstanceResult(
                instance_id=str(i), dataset="sst2", model="llama",
                timestamp=datetime.now(), text=f"text {i}", ground_truth_label="pos",
                predicted_label="pos", confidence=0.9, correct=True,
                raw_highlighting="", raw_rationale="", raw_counterfactual="", raw_rank_ordering="",
            ))
        f = tmp_path / "multi.jsonl"
        save_instance_results(results, str(f))
        loaded = load_instance_results(str(f))
        assert len(loaded) == 3

    def test_save_load_aggregate_metrics(self, tmp_path):
        m = AggregateMetrics(
            aggregation_level="dataset", group_name="sst2", n_instances=100,
            mean_ecs=0.5, std_ecs=0.1, median_ecs=0.5,
            ecs_ci_lower=0.45, ecs_ci_upper=0.55,
            mean_jaccard_H_R=0.4, mean_jaccard_H_CF=0.3, mean_jaccard_H_RO=0.6,
            mean_jaccard_R_CF=0.2, mean_jaccard_R_RO=0.5, mean_jaccard_CF_RO=0.4,
            mean_overlap_H_R=0.6, mean_overlap_H_CF=0.5, mean_overlap_H_RO=0.7,
            mean_overlap_R_CF=0.4, mean_overlap_R_RO=0.6, mean_overlap_CF_RO=0.5,
            mean_kendall_H_RO=0.7,
            mean_normalized_kendall_H_RO=0.85,
            mean_rbo_H_RO=0.65,
            mean_cc3_size=3.0, mean_cc4_size=1.0,
            pct_instances_with_cc3=0.8, pct_instances_with_cc4=0.5,
            spearman_rho=0.3, spearman_p_value=0.01,
            correlation_ci_lower=0.1, correlation_ci_upper=0.5,
            highlighting_success_rate=0.9, rationale_success_rate=0.85,
            counterfactual_success_rate=0.8, rank_ordering_success_rate=0.75,
        )
        f = tmp_path / "agg.json"
        save_aggregate_metrics([m], str(f))
        loaded = load_aggregate_metrics(str(f))
        assert len(loaded) == 1
        assert loaded[0].mean_ecs == 0.5

    def test_save_load_validity_results(self, tmp_path):
        r = ValidityTestResult(instance_id="1", dataset="sst2", model="llama")
        f = tmp_path / "validity.jsonl"
        save_validity_results([r], str(f))
        loaded = load_validity_results(str(f))
        assert len(loaded) == 1
        assert loaded[0].instance_id == "1"
