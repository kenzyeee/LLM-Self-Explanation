import pytest
from src.normalization.normalizer import Normalizer, DISCOURSE_WORDS


class TestNormalizer:
    @pytest.fixture
    def normalizer(self):
        return Normalizer()

    @pytest.fixture
    def normalizer_no_stopwords(self):
        return Normalizer(remove_stopwords=False)

    @pytest.fixture
    def normalizer_no_lemmatization(self):
        return Normalizer(use_lemmatization=False)

    def test_normalize_basic(self, normalizer):
        result = normalizer.normalize("running")
        assert result == "running"

    def test_normalize_lowercase(self, normalizer):
        result = normalizer.normalize("GREAT")
        assert result == "great"

    def test_normalize_punctuation(self, normalizer):
        result = normalizer.normalize("great!")
        assert result == "great"

    def test_normalize_empty(self, normalizer):
        assert normalizer.normalize("") is None
        assert normalizer.normalize("  ") is None

    def test_normalize_stopword(self, normalizer):
        result = normalizer.normalize("the")
        assert result is None

    def test_normalize_discourse_word(self, normalizer):
        for word in ["correct", "classification", "text", "indicates", "evidence"]:
            result = normalizer.normalize(word)
            assert result is None, f"Discourse word '{word}' should be filtered"

    def test_normalize_discourse_word_noise(self, normalizer):
        for word in ["important", "positive", "negative", "neutral", "entailment"]:
            result = normalizer.normalize(word)
            assert result is None, f"Discourse word '{word}' should be filtered"

    def test_normalize_html_entity(self, normalizer):
        result = normalizer.normalize("&#39;")
        assert result is None or result == "'"

    def test_normalize_ampersand_entity(self, normalizer):
        result = normalizer.normalize("&amp;")
        assert result is None or result == "&"

    def test_normalize_sep_marker(self, normalizer):
        result = normalizer.normalize("[SEP]")
        assert result is None or result == ""

    def test_normalize_lemmatization(self, normalizer):
        result = normalizer.normalize("running")
        assert result == "running"

    def test_normalize_lemmatization_plural(self, normalizer):
        result = normalizer.normalize("movies")
        assert result == "movie"

    def test_normalize_tokens_single(self, normalizer):
        tokens = ["running"]
        result = normalizer.normalize_tokens(tokens)
        assert result == {"running"}

    def test_normalize_tokens_multi_word(self, normalizer):
        tokens = ["great movie"]
        result = normalizer.normalize_tokens(tokens)
        assert "great" in result
        assert "movie" in result

    def test_normalize_tokens_discourse_filtered(self, normalizer):
        tokens = ["great", "text", "movie", "indicates"]
        result = normalizer.normalize_tokens(tokens)
        assert "great" in result
        assert "movie" in result
        assert "text" not in result
        assert "indicates" not in result

    def test_pre_normalize(self, normalizer):
        result = normalizer.pre_normalize('"Great!"')
        assert result == "great"

    def test_pre_normalize_with_quotes(self, normalizer):
        result = normalizer.pre_normalize("'hello'")
        assert result == "hello"

    def test_is_anchored_exact(self, normalizer):
        assert normalizer.is_anchored("great", "This is a great movie")

    def test_is_anchored_case_insensitive(self, normalizer):
        assert normalizer.is_anchored("Great", "this is a great movie")

    def test_is_anchored_not_found(self, normalizer):
        assert not normalizer.is_anchored("terrible", "This is a great movie")

    def test_is_anchored_multiword(self, normalizer):
        assert normalizer.is_anchored("great movie", "This is a great movie")

    def test_is_anchored_punctuation_robust(self, normalizer):
        assert normalizer.is_anchored("great!", "This is a great movie")
