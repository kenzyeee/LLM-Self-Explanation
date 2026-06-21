import pytest
from src.parsing.parser import Parser
from src.normalization.normalizer import Normalizer


@pytest.fixture
def parser():
    return Parser()


@pytest.fixture
def normalizer():
    return Normalizer()


class TestParseClassification:
    def test_valid_json(self, parser):
        label = parser.parse_classification(
            '{"label":"positive"}', ["positive", "negative"]
        )
        assert label == "positive"

    def test_json_in_code_fence(self, parser):
        resp = '```json\n{"label":"entailment"}\n```'
        label = parser.parse_classification(resp, ["entailment", "neutral", "contradiction"])
        assert label == "entailment"

    def test_json_surrounded_by_text(self, parser):
        resp = 'Here is my answer: {"label":"Sci/Tech"}'
        label = parser.parse_classification(resp, ["World", "Sports", "Business", "Sci/Tech"])
        assert label == "Sci/Tech"

    def test_label_not_in_set_raises(self, parser):
        with pytest.raises(Exception):
            parser.parse_classification('{"label":"invalid"}', ["positive", "negative"])

    def test_empty_response_raises(self, parser):
        with pytest.raises(Exception):
            parser.parse_classification("", ["positive", "negative"])

    def test_non_json_response_raises(self, parser):
        with pytest.raises(Exception):
            parser.parse_classification("Prediction: positive", ["positive", "negative"])


class TestParseHighlighting:
    def test_valid_salience(self, parser, normalizer):
        input_text = "This movie was great and wonderful and amazing."
        tokens = parser.parse_highlighting(
            '{"salience":{"great":10,"wonderful":8,"amazing":5}}', input_text, normalizer
        )
        assert len(tokens) == 3
        assert "great" in tokens

    def test_salience_returns_top5_by_score(self, parser, normalizer):
        input_text = "one two three four five six seven eight nine ten"
        tokens = parser.parse_highlighting(
            '{"salience":{"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}}',
            input_text, normalizer
        )
        assert len(tokens) == 5
        assert tokens == ["ten", "nine", "eight", "seven", "six"]

    def test_salience_unanchored_discarded(self, parser, normalizer):
        result = parser.parse_highlighting(
            '{"salience":{"great":10,"nonexistent":8,"amazing":5}}',
            "This is great and amazing", normalizer
        )
        assert len(result) == 2
        assert "great" in result
        assert "amazing" in result

    def test_too_few_valid_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_highlighting(
                '{"salience":{"great":10}}', "great wonderful", normalizer
            )

    def test_empty_response_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_highlighting("", "some input text", normalizer)


class TestParseRationale:
    def test_valid_rationale(self, parser, normalizer):
        input_text = "The acting was superb and the plot compelling."
        rationale, evidence = parser.parse_rationale(
            '{"rationale":"The acting was superb and the plot compelling."}',
            input_text, normalizer
        )
        assert "superb" in rationale or "acting" in rationale
        assert len(evidence) >= 2

    def test_unanchored_rationale_raises(self, parser, normalizer):
        """Rationale sentence with no dependency tokens that anchor into input text."""
        with pytest.raises(Exception):
            parser.parse_rationale(
                '{"rationale":"This is completely unrelated."}',
                "Great movie", normalizer
            )

    def test_empty_rationale_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_rationale(
                '{"rationale":""}',
                "Great movie", normalizer
            )

    def test_introduced_concepts_tracked(self, parser, normalizer):
        """Rationale concepts absent from the input are surfaced as introduced
        concepts (post-hoc rationalization signal), not silently dropped."""
        input_text = "The movie was great."
        rationale, evidence = parser.parse_rationale(
            '{"rationale":"The great movie felt boring overall."}',
            input_text, normalizer
        )
        introduced = parser._r_introduced
        assert isinstance(introduced, list)
        # Partition invariant: anchored evidence is in the input; introduced is not.
        for tok in evidence:
            assert normalizer.is_anchored(tok, input_text)
        for tok in introduced:
            assert not normalizer.is_anchored(tok, input_text)
        # 'boring' is a salient concept absent from the input → should be introduced.
        assert any(t.startswith("bor") for t in introduced)


class TestParseCounterfactual:
    def test_valid_counterfactual(self, parser, normalizer):
        input_text = "This movie was great."
        cf_text, new_pred, from_tokens = parser.parse_counterfactual(
            '{"rewritten":"This movie was terrible.","new_prediction":"negative"}',
            input_text, "positive", ["positive", "negative"], normalizer
        )
        assert cf_text == "This movie was terrible."
        assert new_pred == "negative"
        assert "great" in from_tokens

    def test_no_flip_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_counterfactual(
                '{"rewritten":"This movie was awesome.","new_prediction":"positive"}',
                "This movie was great.", "positive", ["positive", "negative"], normalizer
            )

    def test_invalid_label_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_counterfactual(
                '{"rewritten":"This movie was terrible.","new_prediction":"invalid"}',
                "This movie was great.", "positive", ["positive", "negative"], normalizer
            )

    def test_edit_ratio_too_high_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_counterfactual(
                '{"rewritten":"This car was terrible.","new_prediction":"negative"}',
                "This movie was great.", "positive", ["positive", "negative"], normalizer,
                max_edit_ratio=0.3
            )

    def test_null_rewritten_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_counterfactual(
                '{"rewritten":null,"new_prediction":null}',
                "This movie was great.", "positive", ["positive", "negative"], normalizer
            )

    def test_identical_text_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_counterfactual(
                '{"rewritten":"This movie was great.","new_prediction":"negative"}',
                "This movie was great.", "positive", ["positive", "negative"], normalizer
            )


class TestParseRankOrdering:
    def test_valid_ranking(self, parser, normalizer):
        input_text = "the quick brown fox jumps over the lazy dog"
        tokens = parser.parse_rank_ordering(
            '{"ranking":["quick","brown","fox","jumps","lazy"]}',
            input_text, normalizer
        )
        assert len(tokens) == 5
        assert tokens[0] == ("quick", 1)
        assert tokens[4] == ("lazy", 5)

    def test_too_few_items_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_rank_ordering(
                '{"ranking":["quick","brown"]}',
                "the quick brown fox", normalizer
            )

    def test_not_anchored_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_rank_ordering(
                '{"ranking":["quick","nonexistent1","nonexistent2"]}',
                "the quick brown fox", normalizer
            )


class TestWordEditRatio:
    def test_identical_texts(self, parser):
        assert parser._word_edit_ratio("hello world", "hello world") == 0.0

    def test_one_word_change(self, parser):
        r = parser._word_edit_ratio("hello world", "hello there")
        assert 0.0 < r < 0.6

    def test_completely_different(self, parser):
        r = parser._word_edit_ratio("hello world", "foo bar baz")
        assert r > 0.5

    def test_empty_both(self, parser):
        assert parser._word_edit_ratio("", "") == 0.0

    def test_one_empty(self, parser):
        assert parser._word_edit_ratio("hello", "") == 1.0


class TestExtractJson:
    def test_direct_json(self, parser):
        assert parser._extract_json('{"a":1}') == {"a": 1}

    def test_code_fence(self, parser):
        result = parser._extract_json('```json\n{"a":1}\n```')
        assert result == {"a": 1}

    def test_surrounding_text(self, parser):
        result = parser._extract_json('Answer: {"a":1} done')
        assert result == {"a": 1}

    def test_invalid_json_returns_none(self, parser):
        assert parser._extract_json("not json") is None
