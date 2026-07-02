import pytest
from src.metrics.metrics_calculator import MetricsCalculator


class TestMetricsCalculator:
    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    def test_jaccard_identical_sets(self, calc):
        s = {"a", "b", "c"}
        assert calc.compute_jaccard_similarity(s, s) == 1.0

    def test_jaccard_disjoint_sets(self, calc):
        assert calc.compute_jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_partial_overlap(self, calc):
        result = calc.compute_jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert result == 2.0 / 4.0  # intersection {b,c}=2, union {a,b,c,d}=4

    def test_jaccard_empty_sets(self, calc):
        assert calc.compute_jaccard_similarity(set(), set()) == 1.0

    def test_jaccard_one_empty(self, calc):
        assert calc.compute_jaccard_similarity({"a"}, set()) == 0.0

    def test_jaccard_symmetry(self, calc):
        s1 = {"a", "b", "c", "d"}
        s2 = {"c", "d", "e", "f"}
        assert calc.compute_jaccard_similarity(s1, s2) == calc.compute_jaccard_similarity(s2, s1)

    def test_assign_implicit_ranks(self, calc):
        tokens = ["a", "b", "c"]
        result = calc.assign_implicit_ranks(tokens)
        assert result == [("a", 1), ("b", 2), ("c", 3)]

    def test_assign_implicit_ranks_empty(self, calc):
        assert calc.assign_implicit_ranks([]) == []

    def test_kendall_tau_perfect_agreement(self, calc):
        ranks = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
        assert calc.compute_kendalls_tau(ranks, ranks) == 1.0

    def test_kendall_tau_negative(self, calc):
        # a has rank 4 in ranks2 (vs 1 in ranks1), so negative correlation
        ranks1 = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
        ranks2 = [("a", 4), ("b", 3), ("c", 2), ("d", 1)]
        tau = calc.compute_kendalls_tau(ranks1, ranks2)
        assert tau is not None and tau < 0

    def test_kendall_tau_no_common_tokens(self, calc):
        ranks1 = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
        ranks2 = [("e", 1), ("f", 2), ("g", 3), ("h", 4)]
        assert calc.compute_kendalls_tau(ranks1, ranks2) is None

    def test_kendall_tau_few_common_returns_none(self, calc):
        ranks1 = [("a", 1), ("b", 2), ("c", 3)]
        ranks2 = [("a", 2), ("b", 1), ("d", 3)]
        assert calc.compute_kendalls_tau(ranks1, ranks2) is None

    def test_kendall_tau_symmetry(self, calc):
        ranks1 = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
        ranks2 = [("a", 2), ("b", 1), ("c", 3), ("d", 4)]
        tau1 = calc.compute_kendalls_tau(ranks1, ranks2)
        tau2 = calc.compute_kendalls_tau(ranks2, ranks1)
        assert tau1 == tau2

    def test_pairwise_agreements_all_present(self, calc):
        explanations = {
            "H": {"great", "movie"},
            "R": {"great", "acting"},
            "CF": {"movie", "terrible"},
            "RO": {"great", "movie", "acting"},
        }
        agreements = calc.compute_pairwise_agreements(explanations)
        pairs = [("H", "R"), ("H", "CF"), ("H", "RO"), ("R", "CF"), ("R", "RO"), ("CF", "RO")]
        for p in pairs:
            assert p in agreements or (p[1], p[0]) in agreements

    def test_pairwise_agreements_partial(self, calc):
        explanations = {"H": {"a"}, "R": {"b"}}
        agreements = calc.compute_pairwise_agreements(explanations)
        assert ("H", "R") in agreements

    def test_ecs_all_pairs(self, calc):
        agreements = {
            ("H", "R"): 0.5, ("H", "CF"): 0.3, ("H", "RO"): 0.4,
            ("R", "CF"): 0.2, ("R", "RO"): 0.6, ("CF", "RO"): 0.1,
        }
        ecs = calc.compute_ecs(agreements)
        import pytest
        assert ecs == pytest.approx(0.34)  # (0.5+0.3+0.2+0.6+0.1)/5, H-RO excluded

    def test_ecs_missing_pair(self, calc):
        agreements = {("H", "R"): 0.5, ("H", "CF"): 0.3}
        ecs = calc.compute_ecs(agreements)
        assert 0.0 < ecs <= 0.5

    def test_ecs_no_agreements(self, calc):
        assert calc.compute_ecs({}) is None

    def test_ecs_primary_uses_both_pairs_per_composite(self, calc):
        # Pair keys as compute_pairwise_agreements actually stores them: ordered by
        # index within ["H","R","CF","RO"], so RO always appears second — ("R","RO"),
        # ("CF","RO") — never ("RO","R")/("RO","CF"). A regression here (looking up
        # the reversed tuple) silently drops the RO pair from both composites.
        agreements = {
            ("H", "R"): 0.4, ("H", "CF"): 0.6, ("H", "RO"): 0.9,
            ("R", "CF"): 0.5, ("R", "RO"): 0.8, ("CF", "RO"): 0.2,
        }
        er_mean, ep_mean, n_pairs = calc.compute_ecs_primary(agreements)
        assert er_mean == pytest.approx((0.4 + 0.8) / 2)  # (H,R) and (R,RO)
        assert ep_mean == pytest.approx((0.6 + 0.2) / 2)  # (H,CF) and (CF,RO)
        assert n_pairs == 4

    def test_ecs_primary_missing_ro_pairs(self, calc):
        agreements = {("H", "R"): 0.4, ("H", "CF"): 0.6}
        er_mean, ep_mean, n_pairs = calc.compute_ecs_primary(agreements)
        assert er_mean == pytest.approx(0.4)
        assert ep_mean == pytest.approx(0.6)
        assert n_pairs == 2

    def test_ecs_primary_empty(self, calc):
        er_mean, ep_mean, n_pairs = calc.compute_ecs_primary({})
        assert er_mean is None
        assert ep_mean is None
        assert n_pairs == 0

    def test_consensus_core_cc3(self, calc):
        explanations = {
            "H": {"a", "b", "c"},
            "R": {"a", "b", "d"},
            "CF": {"a", "c", "e"},
            "RO": {"b", "c", "f"},
        }
        cc3 = calc.compute_consensus_core(explanations, 3)
        assert "a" in cc3  # appears in H, R, CF
        assert "b" in cc3  # appears in H, R, RO
        assert "c" in cc3  # appears in H, CF, RO
        assert "d" not in cc3  # only in R
        assert "e" not in cc3  # only in CF
        assert "f" not in cc3  # only in RO

    def test_consensus_core_cc4(self, calc):
        explanations = {
            "H": {"a", "b", "c"},
            "R": {"a", "b", "d"},
            "CF": {"a", "c", "e"},
            "RO": {"a", "b", "f"},
        }
        cc4 = calc.compute_consensus_core(explanations, 4)
        assert cc4 == {"a"}

    def test_consensus_core_all_empty(self, calc):
        explanations = {"H": set(), "R": set(), "CF": set(), "RO": set()}
        assert calc.compute_consensus_core(explanations, 3) == set()

    def test_consensus_core_cc4_subset_of_cc3(self, calc):
        explanations = {
            "H": {"a", "b"},
            "R": {"a"},
            "CF": {"a", "b", "c"},
            "RO": {"a", "b"},
        }
        cc3 = calc.compute_consensus_core(explanations, 3)
        cc4 = calc.compute_consensus_core(explanations, 4)
        assert cc4.issubset(cc3)


class TestExpectedRandomOverlapAndTau:
    @pytest.fixture
    def calc(self):
        return MetricsCalculator()

    def test_random_overlap_degenerate_zero(self, calc):
        assert calc.expected_random_overlap(0, 5, 50) == (0.0, 0.0)
        assert calc.expected_random_overlap(5, 5, 0) == (0.0, 0.0)

    def test_random_overlap_in_unit_interval(self, calc):
        ej, eo = calc.expected_random_overlap(5, 5, 50)
        assert 0.0 <= ej <= 1.0 and 0.0 <= eo <= 1.0

    def test_random_overlap_smaller_vocab_more_chance(self, calc):
        # Smaller vocabulary => higher expected overlap by chance (motivates lift).
        ej_large, _ = calc.expected_random_overlap(5, 5, 100)
        ej_small, _ = calc.expected_random_overlap(5, 5, 10)
        assert ej_small > ej_large

    def test_random_overlap_deterministic(self, calc):
        assert calc.expected_random_overlap(4, 6, 40) == calc.expected_random_overlap(4, 6, 40)

    def test_random_overlap_size_exceeds_vocab(self, calc):
        # Sizes clamp to vocab; identical full draws => overlap 1.0.
        ej, eo = calc.expected_random_overlap(10, 10, 5)
        assert ej == pytest.approx(1.0) and eo == pytest.approx(1.0)

    def test_tau_three_common_returns_none(self, calc):
        r1 = [("a", 1), ("b", 2), ("c", 3)]
        r2 = [("a", 1), ("b", 2), ("c", 3)]
        assert calc.compute_kendalls_tau(r1, r2) is None  # 3 common < 4

    def test_tau_four_common_returns_value(self, calc):
        r1 = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
        r2 = [("a", 1), ("b", 2), ("d", 3), ("c", 4)]
        assert calc.compute_kendalls_tau(r1, r2) is not None
