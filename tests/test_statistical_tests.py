import numpy as np
import pytest
from unittest.mock import patch
from src.statistics.statistical_tests import (
    compute_confidence_ecs_correlation,
    permutation_test,
    wilcoxon_signed_rank_test,
    one_sample_ttest,
    paired_ttest,
    CorrelationResult,
    StatisticalTest,
)
from src.utils.exceptions import ValidationError


class TestCorrelationResult:
    def test_fields(self):
        r = CorrelationResult(rho=0.5, p_value=0.01, ci_lower=0.1, ci_upper=0.8)
        assert r.rho == 0.5
        assert r.p_value == 0.01
        assert r.ci_lower == 0.1
        assert r.ci_upper == 0.8


class TestStatisticalTest:
    def test_fields(self):
        s = StatisticalTest(t_statistic=2.5, p_value=0.02, mean_diff=0.3, effect_size=0.8)
        assert s.t_statistic == 2.5
        assert s.p_value == 0.02
        assert s.mean_diff == 0.3
        assert s.effect_size == 0.8


class TestComputeConfidenceEcsCorrelation:
    def test_normal_case(self):
        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
        ecs_values = [0.2, 0.3, 0.4, 0.5, 0.6]
        result = compute_confidence_ecs_correlation(confidences, ecs_values, n_bootstrap=50)
        assert result.rho > 0
        assert 0 <= result.p_value <= 1
        assert result.ci_lower <= result.ci_upper

    def test_too_few_samples(self):
        result = compute_confidence_ecs_correlation([0.5], [0.2], n_bootstrap=10)
        assert result.rho == 0.0
        assert result.p_value == 1.0
        assert result.ci_lower == 0.0
        assert result.ci_upper == 0.0

    def test_constant_input(self):
        result = compute_confidence_ecs_correlation([0.5, 0.5, 0.5], [0.2, 0.2, 0.2], n_bootstrap=10)
        assert result.rho == 0.0
        assert result.p_value == 1.0

    def test_bootstrap_exception_handling(self):
        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
        ecs_values = [0.2, 0.3, 0.4, 0.5, 0.6]
        call_count = [0]
        original_spearmanr = __import__('scipy.stats').stats.spearmanr
        def mock_spearmanr(x, y):
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.8, 0.01
            raise Exception("bootstrap error")
        with patch('scipy.stats.spearmanr', side_effect=mock_spearmanr):
            result = compute_confidence_ecs_correlation(confidences, ecs_values, n_bootstrap=10)
            assert isinstance(result.rho, float)
            assert call_count[0] > 1


class TestPermutationTest:
    def test_normal_case(self):
        p = permutation_test([1, 2, 3], [4, 5, 6], n_permutations=200)
        assert 0 <= p <= 1

    def test_no_difference(self):
        p = permutation_test([1, 1, 1], [1, 1, 1], n_permutations=200)
        assert p >= 0


class TestWilcoxonSignedRankTest:
    def test_normal_case(self):
        result = wilcoxon_signed_rank_test([1, 2, 3, 4, 5], [1, 2, 3, 4, 6])
        assert isinstance(result.t_statistic, float)
        assert 0 <= result.p_value <= 1

    def test_too_few_samples(self):
        result = wilcoxon_signed_rank_test([1], [2])
        assert result.t_statistic == 0.0
        assert result.p_value == 1.0

    def test_unequal_lengths(self):
        # Paired test: silently truncating would mispair unrelated instances.
        with pytest.raises(ValidationError):
            wilcoxon_signed_rank_test([1, 2, 3, 4, 5], [1, 2, 3])

    def test_identical_groups(self):
        result = wilcoxon_signed_rank_test([1, 1, 1], [1, 1, 1])
        assert result.t_statistic == 0.0
        assert result.p_value == 1.0

    def test_value_error_handling(self):
        with patch('scipy.stats.wilcoxon', side_effect=ValueError("ties")):
            result = wilcoxon_signed_rank_test([1, 2, 3], [2, 3, 4])
            assert result.t_statistic == 0.0
            assert result.p_value == 1.0

    def test_nan_stat_handling(self):
        with patch('scipy.stats.wilcoxon', return_value=(float('nan'), 0.5)):
            result = wilcoxon_signed_rank_test([1, 2, 3], [2, 3, 4])
            assert result.t_statistic == 0.0
            assert result.p_value == 1.0


class TestOneSampleTTest:
    def test_normal_case(self):
        stat, p = one_sample_ttest([0.4, 0.5, 0.6], popmean=0.5)
        assert isinstance(stat, float)
        assert 0 <= p <= 1

    def test_too_few_samples(self):
        stat, p = one_sample_ttest([1.0])
        assert stat == 0.0
        assert p == 1.0

    def test_constant_values(self):
        stat, p = one_sample_ttest([0.5, 0.5, 0.5], popmean=0.5)
        assert stat == 0.0
        assert p == 1.0

    def test_exception_handling(self):
        stat, p = one_sample_ttest([], popmean=0.5)
        assert stat == 0.0
        assert p == 1.0

    def test_scipy_exception_handling(self):
        with patch('scipy.stats.ttest_1samp', side_effect=ValueError("invalid")):
            stat, p = one_sample_ttest([0.5, 0.6, 0.7], popmean=0.5)
            assert stat == 0.0
            assert p == 1.0


class TestPairedTTest:
    def test_normal_case(self):
        result = paired_ttest([1, 2, 3], [2, 3, 4])
        assert isinstance(result.t_statistic, float)
        assert 0 <= result.p_value <= 1

    def test_too_few_samples(self):
        result = paired_ttest([1], [2])
        assert result.t_statistic == 0.0
        assert result.p_value == 1.0

    def test_constant_values(self):
        result = paired_ttest([1, 1, 1], [1, 1, 1])
        assert result.t_statistic == 0.0
        assert result.p_value == 1.0

    def test_unequal_lengths(self):
        # Paired test: silently truncating would mispair unrelated instances.
        with pytest.raises(ValidationError):
            paired_ttest([1, 2, 3, 4], [1, 2, 3])

    def test_scipy_exception_handling(self):
        with patch('scipy.stats.ttest_rel', side_effect=ValueError("invalid")):
            result = paired_ttest([1, 2, 3], [2, 3, 4])
            assert result.t_statistic == 0.0
            assert result.p_value == 1.0
