import pytest
from unittest.mock import patch
from src.normalization.normalizer import Normalizer, NormalizationConfig


class TestNormalizer:
    @pytest.fixture
    def normalizer(self):
        return Normalizer()

    @pytest.fixture
    def normalizer_no_stopwords(self):
        config = NormalizationConfig(remove_stopwords=False)
        return Normalizer(config)

    @pytest.fixture
    def normalizer_no_lemmatization(self):
        config = NormalizationConfig(use_lemmatization=False)
        return Normalizer(config)

    @pytest.fixture
    def normalizer_minimal(self):
        config = NormalizationConfig(use_lemmatization=False, remove_stopwords=False)
        return Normalizer(config)

    @pytest.fixture
    def normalizer_punct_only(self):
        config = NormalizationConfig(use_lemmatization=False, remove_stopwords=False, lowercase=False, remove_punctuation=True)
        return Normalizer(config)

    def test_normalize_basic(self, normalizer):
        result = normalizer.normalize("Running")
        assert result == "running"

    def test_normalize_with_punctuation(self, normalizer):
        result = normalizer.normalize("great!")
        assert result == "great"

    def test_normalize_stopword_removal(self, normalizer):
        result = normalizer.normalize("the")
        assert result is None

    def test_normalize_no_stopword_removal(self, normalizer_no_stopwords):
        result = normalizer_no_stopwords.normalize("the")
        assert result == "the"

    def test_normalize_empty_string(self, normalizer):
        result = normalizer.normalize("")
        assert result is None

    def test_normalize_whitespace(self, normalizer):
        # whitespace-only input should be stripped to empty and return None
        result = normalizer.normalize("  ")
        assert result is None

    def test_normalize_numbers(self, normalizer):
        result = normalizer.normalize("123")
        assert result == "123"

    def test_normalize_special_chars(self, normalizer):
        result = normalizer.normalize("hello!!!")
        assert result == "hello"

    def test_normalize_tokens(self, normalizer):
        result = normalizer.normalize_tokens(["Running", "quickly!", "the", "foxes"])
        assert "running" in result
        assert "quickly" in result
        assert "the" not in result  # stopword
        assert "fox" in result  # lemmatized

    def test_normalize_tokens_empty(self, normalizer):
        result = normalizer.normalize_tokens([])
        assert result == set()

    def test_normalize_tokens_all_stopwords(self, normalizer):
        result = normalizer.normalize_tokens(["the", "a", "an", "is"])
        assert result == set()

    def test_extract_content_words_from_rationale(self, normalizer):
        rationale = "The movie was absolutely wonderful and amazing."
        result = normalizer.extract_content_words_from_rationale(rationale)
        assert len(result) > 0

    def test_extract_content_words_empty(self, normalizer):
        result = normalizer.extract_content_words_from_rationale("")
        assert result == set()

    def test_extract_counterfactual_diff_basic(self, normalizer):
        original = "The movie was great and wonderful"
        counterfactual = "The movie was terrible and boring"
        result = normalizer.extract_counterfactual_diff(original, counterfactual)
        assert "great" in result
        assert "wonderful" in result
        assert "terrible" in result
        assert "boring" in result

    def test_extract_counterfactual_diff_identical(self, normalizer):
        result = normalizer.extract_counterfactual_diff("same text", "same text")
        assert result == set()

    def test_extract_counterfactual_diff_empty(self, normalizer):
        result = normalizer.extract_counterfactual_diff("", "")
        assert result == set()

    def test_lemmatization(self, normalizer):
        assert normalizer.normalize("running") == "running"
        assert normalizer.normalize("better") is not None

    def test_no_lemmatization(self, normalizer_no_lemmatization):
        result = normalizer_no_lemmatization.normalize("running")
        assert result == "running"

    def test_idempotence(self, normalizer):
        result1 = normalizer.normalize("Running!")
        result2 = normalizer.normalize(result1) if result1 else None
        assert result1 == result2

    def test_normalize_tokens_phrases(self, normalizer):
        result = normalizer.normalize_tokens(["great movie", "bad acting"])
        assert "great" in result
        assert "movie" in result
        assert "bad" in result
        assert "acting" in result

    def test_normalize_empty_after_punctuation(self, normalizer_punct_only):
        result = normalizer_punct_only.normalize("!!!")
        assert result is None

    def test_normalize_punctuation_only(self, normalizer):
        result = normalizer.normalize("!!!")
        assert result is None

    def test_extract_content_words_handles_exception(self):
        config = NormalizationConfig(use_lemmatization=False, remove_stopwords=False)
        normalizer = Normalizer(config=config)
        words = normalizer.extract_content_words_from_rationale("This is a simple test")
        assert "test" in words or "simple" in words


class TestNormalizerEdgeCases:
    def test_nltk_import_error_fallback(self):
        import sys
        import importlib
        nltk_modules = {k for k in sys.modules if 'nltk' in k}
        for mod in nltk_modules:
            del sys.modules[mod]
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if 'nltk' in name.split('.')[0]:
                raise ImportError(f"No module named {name}")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = mock_import
        try:
            import src.normalization.normalizer as norm_mod
            importlib.reload(norm_mod)
            cfg = norm_mod.NormalizationConfig(use_lemmatization=False, remove_stopwords=False)
            n = norm_mod.Normalizer(cfg)
            result = n.normalize("test")
            assert result == "test"
        finally:
            builtins.__import__ = original_import
            for mod in nltk_modules:
                if mod not in sys.modules:
                    try:
                        importlib.import_module(mod)
                    except ImportError:
                        pass
            importlib.reload(norm_mod)

    def test_stopwords_load_exception_handling(self):
        with patch('nltk.corpus.stopwords.words', side_effect=Exception("stopwords error")):
            config = NormalizationConfig(remove_stopwords=True, use_lemmatization=False)
            normalizer = Normalizer(config=config)
            assert normalizer.stop_words == set()

    def test_extract_content_words_pos_tag_path(self):
        config = NormalizationConfig(use_lemmatization=False, remove_stopwords=False)
        normalizer = Normalizer(config=config)
        words = normalizer.extract_content_words_from_rationale("The movie was absolutely wonderful and amazing")
        assert "wonderful" in words or "amazing" in words

    def test_extract_content_words_pos_tag_exception(self):
        with patch('nltk.pos_tag', side_effect=Exception("pos_tag failed")):
            config = NormalizationConfig(use_lemmatization=False, remove_stopwords=False)
            normalizer = Normalizer(config=config)
            words = normalizer.extract_content_words_from_rationale("The movie was wonderful")
            assert "wonderful" in words
