import json
import pytest
from pathlib import Path
from src.parsing.parser import Parser
from src.utils.exceptions import ParsingError


FIXTURES = Path(__file__).parent / "fixtures" / "sample_responses.json"
with open(FIXTURES, 'r') as f:
    SAMPLE_RESPONSES = json.load(f)


class TestParser:
    @pytest.fixture
    def parser(self):
        return Parser()

    def test_parse_classification_valid(self, parser):
        label, conf = parser.parse_classification(
            "Prediction: positive\nConfidence: 85",
            ["positive", "negative"]
        )
        assert label == "positive"
        assert conf == 0.85

    def test_parse_classification_with_percent(self, parser):
        label, conf = parser.parse_classification(
            "Prediction: negative\nConfidence: 72%",
            ["positive", "negative"]
        )
        assert label == "negative"
        assert conf == 0.72

    def test_parse_classification_float_format(self, parser):
        label, conf = parser.parse_classification(
            "Prediction: neutral\nConfidence: 0.65",
            ["entailment", "neutral", "contradiction"]
        )
        assert label == "neutral"
        assert conf == 0.65

    def test_parse_classification_fuzzy_match(self, parser):
        label, conf = parser.parse_classification(
            "Prediction: Positive\nConfidence: 90",
            ["positive", "negative"]
        )
        assert label == "positive"

    def test_parse_classification_missing_confidence(self, parser):
        label, conf = parser.parse_classification(
            "Prediction: positive",
            ["positive", "negative"]
        )
        assert label == "positive"
        assert conf == 0.0

    def test_parse_classification_empty_response(self, parser):
        label, conf = parser.parse_classification("", ["positive", "negative"])
        assert label == ""
        assert conf == 0.0

    def test_parse_highlighting_numbered_list(self, parser):
        tokens = parser.parse_highlighting("1. great\n2. wonderful\n3. amazing")
        assert len(tokens) == 3
        assert tokens == ["great", "wonderful", "amazing"]

    def test_parse_highlighting_quoted_format(self, parser):
        tokens = parser.parse_highlighting('1. "excellent"\n2. "outstanding"\n3. "brilliant"')
        assert len(tokens) == 3

    def test_parse_highlighting_partial(self, parser):
        tokens = parser.parse_highlighting("1. only\n2. two")
        assert len(tokens) == 2

    def test_parse_highlighting_single(self, parser):
        tokens = parser.parse_highlighting("1. justone")
        assert len(tokens) == 1

    def test_parse_highlighting_truncates_to_three(self, parser):
        tokens = parser.parse_highlighting("1. a\n2. b\n3. c\n4. d\n5. e")
        assert len(tokens) == 3

    def test_parse_rationale_valid(self, parser):
        rationale = parser.parse_rationale(
            "Rationale: The text uses positive language which indicates a positive sentiment."
        )
        assert "positive" in rationale

    def test_parse_rationale_with_prediction(self, parser):
        rationale = parser.parse_rationale(
            "Prediction: positive\nRationale: The language is very positive and encouraging."
        )
        assert "positive" in rationale

    def test_parse_rationale_fallback(self, parser):
        rationale = parser.parse_rationale(
            "This is the first sentence about the text. This is more explanation."
        )
        assert "first sentence" in rationale

    def test_parse_counterfactual_valid(self, parser):
        text = parser.parse_counterfactual(
            "Original Prediction: positive\nCounterfactual Text: The movie was terrible.\nCounterfactual Prediction: negative"
        )
        assert "terrible" in text

    def test_parse_counterfactual_no_original(self, parser):
        text = parser.parse_counterfactual(
            "Counterfactual Text: The team lost the game.\nCounterfactual Prediction: negative"
        )
        assert "lost" in text

    def test_parse_counterfactual_fallback(self, parser):
        text = parser.parse_counterfactual(
            "The original text 'good' becomes 'bad' to flip the sentiment."
        )
        assert len(text) > 0

    def test_parse_rank_ordering_valid(self, parser):
        tokens = parser.parse_rank_ordering(
            "1. excellent\n2. wonderful\n3. amazing\n4. great\n5. superb"
        )
        assert len(tokens) == 5
        assert tokens[0] == ("excellent", 1)
        assert tokens[4] == ("superb", 5)

    def test_parse_rank_ordering_partial(self, parser):
        tokens = parser.parse_rank_ordering("1. first\n2. second\n3. third")
        assert len(tokens) == 3

    def test_parse_rank_ordering_truncates_to_five(self, parser):
        tokens = parser.parse_rank_ordering("1. a\n2. b\n3. c\n4. d\n5. e\n6. f")
        assert len(tokens) == 5

    def test_fuzzy_extract_exact_match(self, parser):
        result = parser._fuzzy_extract("positive", ["positive", "negative"])
        assert result == "positive"

    def test_fuzzy_extract_case_insensitive(self, parser):
        result = parser._fuzzy_extract("Positive", ["positive", "negative"])
        assert result == "positive"

    def test_fuzzy_extract_substring(self, parser):
        result = parser._fuzzy_extract("Contradiction!", ["entailment", "neutral", "contradiction"])
        assert result == "contradiction"

    def test_fuzzy_extract_no_match(self, parser):
        result = parser._fuzzy_extract("unknown", ["positive", "negative"])
        assert result is None

    def test_levenshtein_similarity(self, parser):
        sim = parser._levenshtein_similarity("positive", "positive")
        assert sim == 1.0
        sim2 = parser._levenshtein_similarity("positive", "negative")
        assert sim2 < 1.0

    def test_parse_all_formats(self, parser):
        for entry in SAMPLE_RESPONSES["classification"]["valid"]:
            label, conf = parser.parse_classification(
                entry, ["positive", "negative", "entailment", "neutral", "contradiction", "World", "Sports", "Business", "Sci/Tech"]
            )
            assert label in ["positive", "negative", "entailment", "neutral", "contradiction", "World", "Sports", "Business", "Sci/Tech", ""]
            assert 0.0 <= conf <= 1.0

        for entry in SAMPLE_RESPONSES["highlighting"]["valid"]:
            tokens = parser.parse_highlighting(entry)
            assert len(tokens) <= 3

        for entry in SAMPLE_RESPONSES["rationale"]["valid"]:
            text = parser.parse_rationale(entry)
            assert len(text) > 0

        for entry in SAMPLE_RESPONSES["counterfactual"]["valid"]:
            text = parser.parse_counterfactual(entry)
            assert len(text) > 0

        for entry in SAMPLE_RESPONSES["rank_ordering"]["valid"]:
            tokens = parser.parse_rank_ordering(entry)
            assert len(tokens) <= 5
            for token, rank in tokens:
                assert isinstance(rank, int) and rank > 0

    def test_parse_highlighting_quoted_format(self, parser):
        tokens = parser.parse_highlighting('The key words are "important" and "crucial"')
        assert "important" in tokens
        assert "crucial" in tokens

    def test_parse_highlighting_single_quotes(self, parser):
        tokens = parser.parse_highlighting("The key words are 'vital' and 'key'")
        assert "vital" in tokens
        assert "key" in tokens

    def test_parse_highlighting_comma_split(self, parser):
        tokens = parser.parse_highlighting("important, crucial, vital, key")
        assert len(tokens) <= 3
        assert len(tokens) > 0

    def test_parse_rationale_after_prediction(self, parser):
        text = parser.parse_rationale(
            "Prediction: positive\nThis is the rationale text."
        )
        assert "rationale" in text

    def test_parse_counterfactual_modified_text(self, parser):
        text = parser.parse_counterfactual("Modified Text: This is the modified version")
        assert "modified version" in text

    def test_parse_counterfactual_fallback_with_keyword(self, parser):
        text = parser.parse_counterfactual("The counterfactual is: this changed text")
        assert "changed text" in text or "counterfactual" in text

    def test_parse_rank_ordering_important_keyword(self, parser):
        tokens = parser.parse_rank_ordering("most important: token1, token2, token3")
        assert len(tokens) > 0

    def test_parse_rank_ordering_top_keyword(self, parser):
        tokens = parser.parse_rank_ordering("top: word1, word2")
        assert len(tokens) > 0

    def test_levenshtein_swapped_args(self, parser):
        sim = parser._levenshtein_similarity("abc", "abcdef")
        assert 0.0 < sim < 1.0

    def test_parse_counterfactual_keyword_line_detection(self, parser):
        text = parser.parse_counterfactual("some line\ncounterfactual: \nthe actual counterfactual text here")
        assert "counterfactual" in text

    def test_parse_counterfactual_keyword_line_missing_colon(self, parser):
        text = parser.parse_counterfactual("some line\ncounterfactual without colon\nthe text")
        assert len(text) > 0

    def test_levenshtein_empty_args(self, parser):
        sim = parser._levenshtein_similarity("", "")
        assert sim == 0.0
        sim2 = parser._levenshtein_similarity("abc", "")
        assert sim2 == 0.0
        sim3 = parser._levenshtein_similarity("", "abc")
        assert sim3 == 0.0

    def test_fuzzy_extract_levenshtein_match(self, parser):
        result = parser._fuzzy_extract("postive", ["positive", "negative"])
        assert result == "positive"

    def test_levenshtein_identical(self, parser):
        sim = parser._levenshtein_similarity("positive", "positive")
        assert sim == 1.0

    def test_levenshtein_no_match(self, parser):
        sim = parser._levenshtein_similarity("abc", "xyz")
        assert sim == 0.0
