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
        assert ecs == pytest.approx(0.35)  # (0.5+0.3+0.4+0.2+0.6+0.1)/6

    def test_ecs_missing_pair(self, calc):
        agreements = {("H", "R"): 0.5, ("H", "CF"): 0.3}
        ecs = calc.compute_ecs(agreements)
        assert 0.0 < ecs <= 0.5

    def test_ecs_no_agreements(self, calc):
        assert calc.compute_ecs({}) is None

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
