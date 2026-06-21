"""Unit tests for the erasure operator in the faithfulness/erasure pass."""
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
