import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from src.metrics.metrics_calculator import MetricsCalculator
from src.normalization.normalizer import Normalizer


calc = MetricsCalculator()
normalizer = Normalizer()

# Strategies for unique token lists and sets
unique_token_list = st.lists(
    st.text(min_size=1, max_size=10), min_size=2, max_size=5
).map(lambda lst: list(dict.fromkeys(lst)))  # deduplicate preserving order

token_set = st.sets(st.text(min_size=1, max_size=10))


@given(tokens=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_normalization_idempotence(tokens):
    first = normalizer.normalize_tokens(tokens)
    second = normalizer.normalize_tokens(list(first))
    assert first == second


@given(s1=token_set, s2=token_set)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_jaccard_symmetry(s1, s2):
    j1 = calc.compute_jaccard_similarity(s1, s2)
    j2 = calc.compute_jaccard_similarity(s2, s1)
    assert j1 == j2


@given(s=token_set.filter(lambda x: len(x) > 0))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_jaccard_identity(s):
    assert calc.compute_jaccard_similarity(s, s) == 1.0


@given(s1=token_set, s2=token_set)
@settings(max_examples=50)
def test_jaccard_bounds(s1, s2):
    j = calc.compute_jaccard_similarity(s1, s2)
    assert 0.0 <= j <= 1.0


@given(tokens1=unique_token_list, tokens2=unique_token_list)
@settings(max_examples=50)
def test_kendall_tau_symmetry(tokens1, tokens2):
    ranks1 = calc.assign_implicit_ranks(tokens1)
    ranks2 = calc.assign_implicit_ranks(tokens2)
    common = set(t for t, _ in ranks1) & set(t for t, _ in ranks2)
    if len(common) >= 2:
        tau1 = calc.compute_kendalls_tau(ranks1, ranks2)
        tau2 = calc.compute_kendalls_tau(ranks2, ranks1)
        assert tau1 == tau2


@given(tokens=unique_token_list.filter(lambda x: len(x) >= 4))
@settings(max_examples=50)
def test_kendall_tau_perfect_agreement(tokens):
    ranks = calc.assign_implicit_ranks(tokens)
    assert calc.compute_kendalls_tau(ranks, ranks) == pytest.approx(1.0)


@given(tokens=unique_token_list.filter(lambda x: len(x) >= 4))
@settings(max_examples=50)
def test_kendall_tau_reverse(tokens):
    ranks1 = calc.assign_implicit_ranks(tokens)
    ranks2 = calc.assign_implicit_ranks(list(reversed(tokens)))
    tau = calc.compute_kendalls_tau(ranks1, ranks2)
    assert tau == pytest.approx(-1.0)


@given(tokens=unique_token_list.filter(lambda x: len(x) == 3))
@settings(max_examples=10)
def test_kendall_tau_few_common_returns_none(tokens):
    ranks = calc.assign_implicit_ranks(tokens)
    assert calc.compute_kendalls_tau(ranks, ranks) is None


@given(s1=token_set, s2=token_set, s3=token_set, s4=token_set)
@settings(max_examples=50)
def test_consensus_core_cc4_subset_of_cc3(s1, s2, s3, s4):
    explanations = {"H": s1, "R": s2, "CF": s3, "RO": s4}
    cc3 = calc.compute_consensus_core(explanations, 3)
    cc4 = calc.compute_consensus_core(explanations, 4)
    assert cc4.issubset(cc3)


@given(a=token_set, b=token_set, c=token_set, d=token_set)
@settings(max_examples=50)
def test_consensus_core_cc4_equals_intersection(a, b, c, d):
    explanations = {"H": a, "R": b, "CF": c, "RO": d}
    cc4 = calc.compute_consensus_core(explanations, 4)
    expected = a & b & c & d
    assert cc4 == expected
