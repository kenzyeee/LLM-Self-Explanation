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
        assert result == "run"  # default normalizer lemmatizes

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
        assert normalizer.normalize("running") == "run"
        assert normalizer.normalize("cats") == "cat"

    def test_normalize_lemmatization_plural(self, normalizer):
        result = normalizer.normalize("movies")
        assert result == "movie"  # default normalizer lemmatizes plurals -> singular

    def test_normalize_no_lemmatization(self, normalizer_no_lemmatization):
        # With lemmatization disabled, inflected forms are preserved verbatim.
        assert normalizer_no_lemmatization.normalize("running") == "running"
        assert normalizer_no_lemmatization.normalize("movies") == "movies"

    def test_normalize_tokens_single(self, normalizer):
        tokens = ["running"]
        result = normalizer.normalize_tokens(tokens)
        assert result == {"run"}

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

    # ── CF anchor diagnostic: known-failing examples ──────────────────────
    def test_cf_anchor_better(self, normalizer):
        """'better' must anchor in SST-2 positive review."""
        assert normalizer.is_anchored("better", "i think this movie is better than the other")

    def test_cf_anchor_control(self, normalizer):
        """'control' must anchor when it appears as a full word."""
        assert normalizer.is_anchored("control", "A mentally retarded control freak who ...")

    def test_cf_anchor_tennis(self, normalizer):
        """'Tennis' must anchor (case-insensitive)."""
        assert normalizer.is_anchored("Tennis", "Tennis is a great sport")

    def test_cf_anchor_substring_no_false_positive(self, normalizer):
        """'ten' must NOT match 'tend' or 'content' (word-boundary)."""
        assert not normalizer.is_anchored("ten", "These contents tend to be long")

    def test_cf_anchor_substring_no_false_match(self, normalizer):
        """'or' should NOT match 'world' or 'for' (word-boundary)."""
        assert not normalizer.is_anchored("or", "world for thought")

    def test_anchor_inflection_lemma_match(self, normalizer):
        """A lemmatized token must anchor to its inflected input surface form."""
        text = "moved to tears by a couple of scenes in the film"
        assert normalizer.is_anchored("scene", text)   # scene -> scenes
        assert normalizer.is_anchored("move", text)     # move  -> moved
        assert normalizer.is_anchored("tear", text)     # tear  -> tears

    def test_anchor_inflection_independent_of_use_lemmatization(self):
        """Anchoring stays inflection-robust even when evidence lemmatization is OFF."""
        n = Normalizer(use_lemmatization=False, remove_stopwords=True)
        text = "by a couple of scenes"
        assert n.is_anchored("scene", text)
        assert not n.is_anchored("banana", text)

    def test_anchor_adjective_inflection(self, normalizer):
        """Adjective comparative/superlative inflections anchor (multi-POS lemmas)."""
        assert normalizer.is_anchored("happy", "a happier ending")
        assert normalizer.is_anchored("big", "the biggest win")

    def test_anchor_does_not_match_synonyms(self, normalizer):
        """Only morphological variants anchor — true synonyms must NOT (would inflate agreement)."""
        assert not normalizer.is_anchored("excellent", "a good film")
        assert not normalizer.is_anchored("happy", "a sad ending")
