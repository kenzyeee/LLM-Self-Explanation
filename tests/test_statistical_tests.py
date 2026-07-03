import numpy as np
import pytest
from src.statistics.statistical_tests import (
    compute_confidence_ecs_correlation,
    permutation_test,
    sign_flip_permutation_test,
    holm_correction,
    are_significant,
    CorrelationResult,
)


class TestCorrelationResult:
    def test_fields(self):
        r = CorrelationResult(rho=0.5, p_value=0.01, ci_lower=0.1, ci_upper=0.8, n=10)
        assert r.rho == 0.5
        assert r.p_value == 0.01
        assert r.ci_lower == 0.1
        assert r.ci_upper == 0.8
        assert r.n == 10


class TestConfidenceEcsCorrelation:
    def test_positive_correlation(self):
        confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ecs_values = [0.15, 0.22, 0.35, 0.41, 0.55, 0.58, 0.72, 0.81]
        result = compute_confidence_ecs_correlation(confidences, ecs_values, n_bootstrap=50)
        assert result.rho > 0.9
        assert result.n == 8

    def test_below_min_n_degenerate(self):
        result = compute_confidence_ecs_correlation([0.5], [0.2], n_bootstrap=10)
        assert result.rho == 0.0
        assert result.p_value == 1.0

    def test_constant_input_degenerate(self):
        result = compute_confidence_ecs_correlation([0.5, 0.5, 0.5], [0.2, 0.2, 0.2], n_bootstrap=10)
        assert result.rho == 0.0
        assert result.p_value == 1.0

    def test_unequal_lengths_raise(self):
        # Paired estimate: silently truncating would pair unrelated instances.
        with pytest.raises(ValueError):
            compute_confidence_ecs_correlation([0.1, 0.2, 0.3], [0.1, 0.2], n_bootstrap=10)

    def test_seeded_bootstrap_reproducible(self):
        confidences = [0.1, 0.4, 0.2, 0.9, 0.6, 0.3, 0.8, 0.5]
        ecs_values = [0.2, 0.5, 0.1, 0.8, 0.7, 0.35, 0.75, 0.55]
        r1 = compute_confidence_ecs_correlation(confidences, ecs_values, n_bootstrap=100, seed=7)
        r2 = compute_confidence_ecs_correlation(confidences, ecs_values, n_bootstrap=100, seed=7)
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper


class TestSignFlipPermutationTest:
    def test_clear_positive_effect_small_p(self):
        diffs = [0.5, 0.4, 0.6, 0.5, 0.45, 0.55, 0.5, 0.48]
        p = sign_flip_permutation_test(diffs, n_permutations=2000, seed=1, alternative="greater")
        assert p is not None
        assert p < 0.05

    def test_null_effect_large_p(self):
        diffs = [0.1, -0.1, 0.05, -0.05, 0.02, -0.02, 0.08, -0.08]
        p = sign_flip_permutation_test(diffs, n_permutations=2000, seed=1, alternative="greater")
        assert p is not None
        assert p > 0.2

    def test_negative_effect_with_greater_alternative(self):
        diffs = [-0.5, -0.4, -0.6, -0.5, -0.45, -0.55]
        p = sign_flip_permutation_test(diffs, n_permutations=1000, seed=1, alternative="greater")
        assert p is not None
        assert p > 0.9

    def test_less_alternative(self):
        diffs = [-0.5, -0.4, -0.6, -0.5, -0.45, -0.55]
        p = sign_flip_permutation_test(diffs, n_permutations=1000, seed=1, alternative="less")
        assert p is not None
        assert p < 0.05

    def test_two_sided(self):
        diffs = [0.5, 0.4, 0.6, 0.5, 0.45, 0.55]
        p = sign_flip_permutation_test(diffs, n_permutations=1000, seed=1, alternative="two-sided")
        assert p is not None
        assert p < 0.1

    def test_below_n2_returns_none(self):
        assert sign_flip_permutation_test([0.5], n_permutations=100, seed=1) is None
        assert sign_flip_permutation_test([], n_permutations=100, seed=1) is None

    def test_none_entries_dropped(self):
        diffs = [0.5, None, 0.4, None, 0.6, 0.5, 0.45, 0.55]
        p = sign_flip_permutation_test(diffs, n_permutations=1000, seed=1)
        assert p is not None

    def test_unknown_alternative_raises(self):
        with pytest.raises(ValueError):
            sign_flip_permutation_test([0.1, 0.2], alternative="sideways")

    def test_seeded_reproducible(self):
        diffs = [0.3, -0.1, 0.2, 0.4, -0.05, 0.15]
        p1 = sign_flip_permutation_test(diffs, n_permutations=500, seed=42)
        p2 = sign_flip_permutation_test(diffs, n_permutations=500, seed=42)
        assert p1 == p2

    def test_p_never_zero(self):
        # +1 correction (Phipson & Smyth): a Monte-Carlo p of exactly 0 is invalid.
        diffs = [1.0] * 20
        p = sign_flip_permutation_test(diffs, n_permutations=200, seed=1)
        assert p > 0


class TestPermutationTest:
    def test_different_groups(self):
        p = permutation_test([1, 2, 3], [4, 5, 6], n_permutations=200, seed=3)
        assert p < 0.2

    def test_identical_groups(self):
        p = permutation_test([1, 1, 1], [1, 1, 1], n_permutations=200, seed=3)
        assert p > 0.9


class TestHolmCorrection:
    def test_single_p_unchanged(self):
        assert holm_correction([0.03]) == [0.03]

    def test_orders_and_scales(self):
        # m=3: smallest p × 3; step-down with enforced monotonicity.
        adj = holm_correction([0.01, 0.04, 0.03])
        assert adj[0] == pytest.approx(0.03)   # 0.01 * 3
        assert adj[2] == pytest.approx(0.06)   # 0.03 * 2
        assert adj[1] == pytest.approx(0.06)   # max(0.04 * 1, running max) -> monotone

    def test_caps_at_one(self):
        adj = holm_correction([0.9, 0.8, 0.7])
        assert all(p <= 1.0 for p in adj)

    def test_none_passthrough_not_counted(self):
        # None (test not run) must not inflate the family size m.
        adj = holm_correction([0.02, None, 0.02])
        assert adj[1] is None
        # m=2, so smallest is doubled, not tripled
        assert adj[0] == pytest.approx(0.04)

    def test_empty(self):
        assert holm_correction([]) == []
        assert holm_correction([None, None]) == [None, None]


class TestAreSignificant:
    def test_holm_corrected_flags(self):
        flags = are_significant([0.01, 0.5, None], corrected=True, alpha=0.05)
        assert flags == [True, False, False]

    def test_uncorrected(self):
        flags = are_significant([0.04, 0.06], corrected=False, alpha=0.05)
        assert flags == [True, False]
