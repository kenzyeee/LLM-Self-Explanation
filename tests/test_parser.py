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
        label, conf = parser.parse_classification(
            '{"label":"positive","confidence":9}', ["positive", "negative"]
        )
        assert label == "positive"
        assert conf == 0.9

    def test_valid_json_confidence_as_int(self, parser):
        label, conf = parser.parse_classification(
            '{"label":"negative","confidence":8}', ["positive", "negative"]
        )
        assert label == "negative"
        assert conf == 0.8

    def test_json_in_code_fence(self, parser):
        resp = '```json\n{"label":"entailment","confidence":9}\n```'
        label, conf = parser.parse_classification(resp, ["entailment", "neutral", "contradiction"])
        assert label == "entailment"
        assert conf == 0.9

    def test_json_surrounded_by_text(self, parser):
        resp = 'Here is my answer: {"label":"Sci/Tech","confidence":8}'
        label, conf = parser.parse_classification(resp, ["World", "Sports", "Business", "Sci/Tech"])
        assert label == "Sci/Tech"
        assert conf == 0.8

    def test_label_not_in_set_raises(self, parser):
        with pytest.raises(Exception):
            parser.parse_classification('{"label":"invalid","confidence":5}', ["positive", "negative"])

    def test_empty_response_raises(self, parser):
        with pytest.raises(Exception):
            parser.parse_classification("", ["positive", "negative"])

    def test_non_json_response_raises(self, parser):
        with pytest.raises(Exception):
            parser.parse_classification("Prediction: positive", ["positive", "negative"])

    def test_confidence_out_of_range_raises(self, parser):
        with pytest.raises(Exception):
            parser.parse_classification('{"label":"positive","confidence":15}', ["positive", "negative"])


class TestParseHighlighting:
    def test_valid_highlights(self, parser, normalizer):
        input_text = "This movie was great and wonderful and amazing."
        tokens = parser.parse_highlighting(
            '{"highlights":["great","wonderful","amazing"]}', input_text, normalizer
        )
        assert len(tokens) == 3
        assert "great" in tokens

    def test_too_few_highlights_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_highlighting(
                '{"highlights":["great"]}', "great wonderful", normalizer
            )

    def test_highlight_not_anchored_discarded_ok(self, parser, normalizer):
        result = parser.parse_highlighting(
            '{"highlights":["great","nonexistent","amazing"]}',
            "This is great and amazing", normalizer
        )
        assert len(result) == 2
        assert "great" in result
        assert "amazing" in result

    def test_empty_response_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_highlighting("", "some input text", normalizer)


class TestParseRationale:
    def test_valid_rationale(self, parser, normalizer):
        input_text = "The acting was superb and the plot compelling."
        rationale, evidence = parser.parse_rationale(
            '{"rationale":"The acting was superb.","evidence":["acting","superb","plot","compelling"]}',
            input_text, normalizer
        )
        assert "superb" in rationale
        assert len(evidence) == 4

    def test_evidence_not_anchored_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_rationale(
                '{"rationale":"Good movie.","evidence":["nonexistent"]}',
                "Great movie", normalizer
            )

    def test_empty_evidence_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_rationale(
                '{"rationale":"Good movie.","evidence":[]}',
                "Great movie", normalizer
            )


class TestParseCounterfactual:
    def test_valid_counterfactual(self, parser, normalizer):
        input_text = "This movie was great."
        cf_text, new_pred = parser.parse_counterfactual(
            '{"counterfactual_text":"This movie was terrible.","new_prediction":"negative"}',
            input_text, "positive", ["positive", "negative"], normalizer
        )
        assert "terrible" in cf_text
        assert new_pred == "negative"

    def test_no_flip_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_counterfactual(
                '{"counterfactual_text":"This movie was great.","new_prediction":"positive"}',
                "This movie was great.", "positive", ["positive", "negative"], normalizer
            )

    def test_invalid_label_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_counterfactual(
                '{"counterfactual_text":"This movie was terrible.","new_prediction":"invalid"}',
                "This movie was great.", "positive", ["positive", "negative"], normalizer
            )

    def test_edit_ratio_too_high_raises(self, parser, normalizer):
        with pytest.raises(Exception):
            parser.parse_counterfactual(
                '{"counterfactual_text":"This was a completely different sentence with many changes and different words.","new_prediction":"negative"}',
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
