"""Fixture tests pinning SCIENTIFIC invariants, not plumbing.

Review §5 (2026-07-02 re-check): "501 green tests did not notice that morphological
variants of the same word count as disagreement." These tests pin the invariants
that make ECS a meaningful measurement:

  1. ONE token space — evidence sets differing only by inflection agree perfectly.
  2. ECS bounds — all-agree fixture scores 1.0, all-disjoint scores 0.0.
  3. Negation symmetry — contracted negations survive every strategy's evidence
     path (they are the label-critical evidence class on NLI).
  4. CF evidence lives in the shared normalized space (no stopword asymmetry).
  5. Span-restricted CF minimality is judged on the editable span, and edits
     outside the span are rules violations.
  6. The weighted null degrades to the uniform null under uniform weights.
"""
import math
import pytest

from src.normalization.normalizer import Normalizer, POLARITY_WORDS
from src.parsing.parser import Parser
from src.metrics.metrics_calculator import MetricsCalculator
from src.utils.exceptions import ParsingError


@pytest.fixture(scope="module")
def normalizer():
    # v3.0 live settings: lemmatization ON, stopwords removed.
    return Normalizer(use_lemmatization=True, remove_stopwords=True)


@pytest.fixture(scope="module")
def parser():
    return Parser()


@pytest.fixture(scope="module")
def calc():
    return MetricsCalculator()


class TestSharedTokenSpace:
    """Invariant 1 — the review §8.2 regression: inflectional variants must agree."""

    def test_inflection_equivalent_sets_score_jaccard_1(self, normalizer, calc):
        surface = normalizer.normalize_tokens(["moved", "scenes", "happened"])
        lemmas = normalizer.normalize_tokens(["move", "scene", "happen"])
        assert surface == lemmas, (
            "Evidence sets differing only by inflection must normalize to the SAME "
            "tokens — otherwise every R-involving pair is deflated (review §8.2)")
        assert calc.compute_jaccard_similarity(surface, lemmas) == 1.0

    def test_h_r_regression_from_smoke_run(self, normalizer, calc):
        # Exact live failure: H = {ice, moved, scenes, tears}, R = {move, scene}
        # was reported as Jaccard 0.0 despite 2/4 morphological overlap.
        h = normalizer.normalize_tokens(["ice", "moved", "scenes", "tears"])
        r = normalizer.normalize_tokens(["move", "scene"])
        assert calc.compute_jaccard_similarity(h, r) == pytest.approx(2 / 4)

    def test_normalize_is_idempotent_on_evidence(self, normalizer):
        first = normalizer.normalize_tokens(["running", "movies", "happier", "canings"])
        second = normalizer.normalize_tokens(sorted(first))
        assert first == second


class TestEcsBounds:
    """Invariant 2 — synthetic all-agree / all-disjoint fixtures."""

    def test_all_agree_ecs_is_1(self, calc):
        sets = {"H": {"good", "movie"}, "R": {"good", "movie"},
                "CF": {"good", "movie"}, "RO": {"good", "movie"}}
        agreements = calc.compute_pairwise_agreements(sets)
        assert calc.compute_ecs(agreements) == 1.0

    def test_all_disjoint_ecs_is_0(self, calc):
        sets = {"H": {"a1", "a2"}, "R": {"b1", "b2"}, "CF": {"c1"}, "RO": {"d1", "d2"}}
        agreements = calc.compute_pairwise_agreements(sets)
        assert calc.compute_ecs(agreements) == 0.0

    def test_ecs_overlap_excludes_h_ro_and_is_size_robust(self, calc):
        # CF is a 1-token subset of H: Jaccard is ceiling-limited (1/3) but the
        # overlap coefficient must give full credit (1.0).
        sets = {"H": {"a", "b", "c"}, "R": {"a", "b", "c"}, "CF": {"a"}, "RO": {"a", "b", "c"}}
        overlaps = calc.compute_pairwise_overlaps(sets)
        ecs_ov = calc.compute_ecs_overlap(overlaps)
        assert ecs_ov == 1.0
        # H-RO must not contribute: remove it and the value is unchanged.
        overlaps_no_hro = {k: v for k, v in overlaps.items() if k != ("H", "RO")}
        assert calc.compute_ecs_overlap(overlaps_no_hro) == ecs_ov


class TestNegationSymmetry:
    """Invariant 3 — review §8.4: contracted negations are evidence, everywhere."""

    def test_contractions_are_polarity(self):
        for w in ["shouldn't", "don't", "isn't", "won't", "can't", "shouldnt", "shouldn"]:
            assert w in POLARITY_WORDS, f"{w} must be whitelisted as polarity"

    def test_normalize_keeps_contracted_negation(self, normalizer):
        assert normalizer.normalize("shouldn't") is not None
        assert normalizer.normalize("not") == "not"

    def test_rank_ordering_keeps_negation(self, parser, normalizer):
        # The smoke-run failure: RO returned ["Iraq","shouldn't","happened"]; the
        # discarded "shouldn't" left <3 tokens and invalidated the whole strategy.
        text = "Iraq was something that shouldn't have happened"
        result = parser.parse_rank_ordering(
            '{"ranking": ["Iraq", "shouldn\'t", "happened"]}', text, normalizer)
        tokens = [t for t, _ in result]
        assert len(tokens) == 3
        assert "shouldn't" in tokens

    def test_rationale_keeps_negation(self, parser, normalizer):
        text = "the movie is not good"
        _, evidence = parser.parse_rationale(
            '{"rationale": "It was classified this way because the movie is not good."}',
            text, normalizer)
        norm = normalizer.normalize_tokens(evidence)
        assert "not" in norm

    def test_highlighting_keeps_negation(self, parser, normalizer):
        text = "the movie is not good at all"
        result = parser.parse_highlighting(
            '{"salience": [["not", 9], ["good", 8], ["movie", 5], ["the", 1]]}',
            text, normalizer)
        assert "not" in result


class TestCfEvidenceSpace:
    """Invariant 4 — review §8.3: CF evidence must share the normalized space."""

    def test_cf_evidence_normalization_drops_plain_stopwords_keeps_negation(self, normalizer):
        # Raw difflib tokens from the smoke run: {"happened", "have", "shouldn't"}.
        raw = {"happened", "have", "shouldn't"}
        evidence = normalizer.normalize_tokens(sorted(raw))
        assert "have" not in evidence          # plain stopword: gone
        assert normalizer.normalize("happened") in evidence  # content word: kept (lemma)
        assert any("should" in t for t in evidence)          # negation: kept


class TestCfSpanRestriction:
    """Invariant 5 — review §8.6c: minimality is judged on the editable span."""

    PREMISE = ("Premise: the committee met for several hours on Tuesday to discuss the "
               "budget proposal that had been submitted the previous week by the council")
    HYP = "Hypothesis: the committee approved the budget"

    def _input(self):
        return f"{self.PREMISE}\n{self.HYP}"

    def test_full_hypothesis_rewrite_fails_span_ratio(self, parser, normalizer):
        # Long premise dilutes the full-text ratio below 0.3 (5/29 ≈ 0.17), which the
        # old denominator would wrongly ACCEPT; over the editable span it is ~1.0.
        rewritten = f"{self.PREMISE}\nHypothesis: nobody ever discussed anything whatsoever"
        with pytest.raises(ParsingError, match="edit ratio"):
            parser.parse_counterfactual(
                f'{{"rewritten": "{rewritten}", "new_prediction": "contradiction"}}'.replace("\n", "\\n"),
                self._input(), "entailment", ["entailment", "neutral", "contradiction"],
                normalizer, edit_span_marker="Hypothesis:")

    def test_small_hypothesis_edit_passes(self, parser, normalizer):
        rewritten = f"{self.PREMISE}\nHypothesis: the committee rejected the budget"
        text, pred, from_tokens = parser.parse_counterfactual(
            f'{{"rewritten": "{rewritten}", "new_prediction": "contradiction"}}'.replace("\n", "\\n"),
            self._input(), "entailment", ["entailment", "neutral", "contradiction"],
            normalizer, edit_span_marker="Hypothesis:")
        assert pred == "contradiction"
        assert "approved" in from_tokens

    def test_premise_edit_is_rules_violation(self, parser, normalizer):
        rewritten = ("Premise: the committee never met at all\n"
                     "Hypothesis: the committee approved the budget")
        with pytest.raises(ParsingError, match="outside the allowed span"):
            parser.parse_counterfactual(
                f'{{"rewritten": "{rewritten}", "new_prediction": "contradiction"}}'.replace("\n", "\\n"),
                self._input(), "entailment", ["entailment", "neutral", "contradiction"],
                normalizer, edit_span_marker="Hypothesis:")

    def test_dropped_marker_is_rules_violation(self, parser, normalizer):
        with pytest.raises(ParsingError, match="structure"):
            parser.parse_counterfactual(
                '{"rewritten": "the committee rejected the budget", "new_prediction": "contradiction"}',
                self._input(), "entailment", ["entailment", "neutral", "contradiction"],
                normalizer, edit_span_marker="Hypothesis:")


class TestWeightedNull:
    """Invariant 6 — the salience-weighted null is sane."""

    def test_uniform_weights_match_uniform_null(self, calc):
        weights = {f"tok{i}": 5.0 for i in range(12)}
        weighted = MetricsCalculator.expected_random_overlap_weighted(3, 3, weights,
                                                                      n_sims=3000, seed=42)
        uniform_j, _ = MetricsCalculator.expected_random_overlap(3, 3, 12, n_sims=3000, seed=42)
        assert weighted == pytest.approx(uniform_j, abs=0.03)

    def test_concentrated_weights_raise_expected_overlap(self, calc):
        # Two dominant tokens -> both draws almost surely include them -> higher
        # expected agreement than the flat null. This is exactly why the uniform
        # null understates chance agreement (review §2.5).
        weights = {"hot1": 1000.0, "hot2": 1000.0}
        weights.update({f"cold{i}": 0.001 for i in range(10)})
        weighted = MetricsCalculator.expected_random_overlap_weighted(2, 2, weights,
                                                                      n_sims=1500, seed=42)
        uniform_j, _ = MetricsCalculator.expected_random_overlap(2, 2, 12, n_sims=1500, seed=42)
        assert weighted > uniform_j

    def test_degenerate_inputs_return_none(self):
        assert MetricsCalculator.expected_random_overlap_weighted(3, 3, {}) is None
        assert MetricsCalculator.expected_random_overlap_weighted(0, 3, {"a": 1.0}) is None


class TestHighlightingSchema:
    """List-of-pairs salience schema: duplicates preserved, aggregated by max."""

    def test_list_schema_parses(self, parser, normalizer):
        text = "the movie was great and the acting was great too"
        result = parser.parse_highlighting(
            '{"salience": [["movie", 6], ["great", 9], ["acting", 7], ["great", 4], ["the", 1]]}',
            text, normalizer)
        assert "great" in result
        assert "movie" in result

    def test_duplicate_words_aggregate_by_max(self, parser, normalizer):
        text = "good food good service bad music"
        parser.parse_highlighting(
            '{"salience": [["good", 3], ["food", 5], ["good", 9], ["service", 4], ["bad", 8], ["music", 2]]}',
            text, normalizer)
        weights = parser._h_salience_weights
        assert weights["good"] == 9.0  # max over occurrences, not last-wins

    def test_dict_schema_still_accepted(self, parser, normalizer):
        text = "the movie was great and amazing"
        result = parser.parse_highlighting(
            '{"salience": {"great": 10, "amazing": 5}}', text, normalizer)
        assert "great" in result

    def test_h_ordered_is_normalized_space(self, parser, normalizer):
        # Review §8.6b: raw-case H keys vs normalized RO tokens zeroed rank metrics.
        text = "Nokia announces NFC products for phones"
        result = parser.parse_highlighting(
            '{"salience": [["Nokia", 9], ["NFC", 10], ["products", 5], ["phones", 6]]}',
            text, normalizer)
        assert all(t == t.lower() for t in result), "H ranking must be normalized (lowercase lemmas)"
        assert "nfc" in result


class TestParseConfidence:
    def test_percentage_scale(self, parser):
        assert parser.parse_confidence('{"confidence": 85}') == pytest.approx(0.85)

    def test_probability_scale(self, parser):
        assert parser.parse_confidence('{"confidence": 0.7}') == pytest.approx(0.7)

    def test_string_number(self, parser):
        assert parser.parse_confidence('{"confidence": "90%"}') == pytest.approx(0.9)

    def test_clamped(self, parser):
        assert parser.parse_confidence('{"confidence": 150}') == 1.0

    def test_invalid_raises(self, parser):
        with pytest.raises(ParsingError):
            parser.parse_confidence('{"confidence": "very sure"}')
        with pytest.raises(ParsingError):
            parser.parse_confidence('not json at all')
        with pytest.raises(ParsingError):
            parser.parse_confidence('{"confidence": -5}')


class TestCrossModelAgreement:
    def test_same_sets_across_models_score_1(self):
        from src.utils.data_models import InstanceResult
        from datetime import datetime

        def make(model, h):
            return InstanceResult(
                instance_id="i1", dataset="sst2", model=model, timestamp=datetime.now(),
                text="good movie", ground_truth_label="positive", predicted_label="positive",
                correct=True, highlighting_tokens=set(h), highlighting_valid=True,
                rationale_tokens={"good"}, rationale_valid=True,
                cc3_tokens=set(), cc4_tokens=set(), cc3_size=0, cc4_size=0,
                ecs=0.5,
            )

        results = [make("m1", {"good", "movie"}), make("m2", {"good", "movie"})]
        out = MetricsCalculator.compute_cross_model_agreement(results)
        assert out["sst2"]["strategies"]["H"]["mean_jaccard"] == 1.0
        assert out["sst2"]["strategies"]["H"]["n_pairs"] == 1
        assert out["sst2"]["n_instances_multi_model"] == 1
        assert out["sst2"]["within_model_cross_strategy_mean_ecs"] == pytest.approx(0.5)

    def test_single_model_returns_empty(self):
        from src.utils.data_models import InstanceResult
        from datetime import datetime
        r = InstanceResult(
            instance_id="i1", dataset="sst2", model="m1", timestamp=datetime.now(),
            text="t", ground_truth_label="a", predicted_label="a", correct=True,
            cc3_tokens=set(), cc4_tokens=set(), cc3_size=0, cc4_size=0,
        )
        assert MetricsCalculator.compute_cross_model_agreement([r]) == {}
