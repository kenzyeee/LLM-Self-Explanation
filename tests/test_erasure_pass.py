"""Unit tests for the erasure operator in the faithfulness/erasure pass."""
import asyncio
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_validity_tests_mod", ROOT / "scripts" / "run_validity_tests.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
erase = _mod.erase


def test_erase_mask():
    assert erase("a great movie", {"great"}, "mask") == "a [MASK] movie"


def test_erase_delete():
    assert erase("a great movie", {"great"}, "delete") == "a movie"


def test_erase_case_and_punctuation():
    assert erase("This is GREAT!", {"great"}, "mask") == "This is [MASK]"
    assert erase("This is GREAT!", {"great"}, "delete") == "This is"


def test_erase_no_match_unchanged():
    assert erase("nothing matches here", {"absent"}, "mask") == "nothing matches here"


def test_erase_empty_token_set_unchanged():
    assert erase("keep all words", set(), "delete") == "keep all words"


def test_mask_and_delete_differ():
    text = "the plot was thin"
    assert erase(text, {"plot"}, "mask") != erase(text, {"plot"}, "delete")


def test_erase_all_occurrences():
    text = "bad bad food, truly bad"
    assert erase(text, {"bad"}, "mask") == "[MASK] [MASK] food, truly [MASK]"


def test_erase_lemma_aware_matches_inflection():
    from src.normalization.normalizer import Normalizer
    normalizer = Normalizer(use_lemmatization=True)
    # Evidence lemma "movie" should also erase the input's inflected surface "movies".
    assert erase("the movies were great", {"movie"}, "mask", normalizer) == "the [MASK] were great"


def test_erase_without_normalizer_is_surface_only():
    assert erase("the movies were great", {"movie"}, "mask") == "the movies were great"


class _FakeEngine:
    def __init__(self):
        self.seen_prompts = []

    async def _make_request(self, prompt, max_tokens=50):
        self.seen_prompts.append(prompt)
        return '{"label": "pos"}'


def test_random_flip_rate_pool_excludes_stopwords_with_normalizer():
    from src.normalization.normalizer import Normalizer
    from src.parsing.parser import Parser
    normalizer = Normalizer(use_lemmatization=False, remove_stopwords=True)
    parser = Parser()
    engine = _FakeEngine()
    # Only "great" survives the content-word filter; "the"/"a"/"is"/"was" are
    # stopwords. n=5 exceeds the filtered pool size, so every trial must erase
    # exactly the one available content word, never a stopword.
    text = "the a is was great"
    class_prompt = "classify: {input_text} labels: {label_set}"

    rate = asyncio.run(_mod.random_flip_rate(
        engine, parser, class_prompt, text, n=5, operator="mask",
        label_set=["pos", "neg"], original="pos", trials=3, seed=0, normalizer=normalizer))

    assert rate is not None
    assert len(engine.seen_prompts) == 3
    for p in engine.seen_prompts:
        assert p.count("[MASK]") == 1
        for stopword in ("the", "a", "is", "was"):
            assert stopword in p.lower()  # stopwords remain — never erased


def test_random_flip_rate_pool_is_unfiltered_without_normalizer():
    from src.parsing.parser import Parser
    parser = Parser()
    engine = _FakeEngine()
    text = "the a is was great"
    class_prompt = "classify: {input_text} labels: {label_set}"

    asyncio.run(_mod.random_flip_rate(
        engine, parser, class_prompt, text, n=1, operator="mask",
        label_set=["pos", "neg"], original="pos", trials=20, seed=0, normalizer=None))

    stopwords = {"the", "a", "is", "was"}
    # Without a normalizer, the draw pool is all 5 surface words (not just the one
    # content word "great"), so a stopword must get erased in at least one trial.
    erased_a_stopword = any(
        stopwords - set(p.lower().replace("[mask]", "").split())
        for p in engine.seen_prompts
    )
    assert erased_a_stopword
