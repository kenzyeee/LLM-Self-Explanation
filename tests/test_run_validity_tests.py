"""Unit tests for the erasure pass (scripts/run_validity_tests.py) — the analysis
half only (no API calls): paired-difference construction, aggregate structure with
the pre-registered test family (b), and the erase() operator semantics."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_validity_tests", ROOT / "scripts" / "run_validity_tests.py")
rvt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rvt)

erase = rvt.erase
aggregate = rvt.aggregate
_paired_cc_random_diffs = rvt._paired_cc_random_diffs
_tier = rvt._tier
_rate = rvt._rate
_mean = rvt._mean

from src.normalization.normalizer import Normalizer


def _rec(iid, model, cc3_mask, rand_mask, ecs_lift=0.1, cc3_delete=None, rand_delete=None):
    return {
        "instance_id": iid,
        "dataset": "sst2",
        "model": model,
        "correct": True,
        "ecs": 0.3,
        "ecs_lift": ecs_lift,
        "original_prediction": "positive",
        "strategy_erasure": {"H": {"n": 3, "mask": True, "delete": True}},
        "cc3": {"size": 2, "mask": cc3_mask, "delete": cc3_delete},
        "cc4": {"size": 1, "mask": None, "delete": None},
        "random_cc3": {"n": 2, "mask_rate": rand_mask, "delete_rate": rand_delete},
        "cf_flip_heldout": None,
    }


class TestPairedDiffs:
    def test_pairing_is_within_instance(self):
        records = [
            _rec("a", "m1", True, 0.2),
            _rec("b", "m1", False, 0.5),
            _rec("c", "m1", None, 0.1),   # missing cc side -> excluded
            _rec("d", "m1", True, None),  # missing random side -> excluded
        ]
        diffs = _paired_cc_random_diffs(records, "mask")
        assert diffs == [pytest.approx(0.8), pytest.approx(-0.5)]


class TestAggregate:
    def test_structure_and_test_family(self):
        # 8 instances where CC erasure always flips and random rarely does ->
        # the pre-registered one-sided test should come out small.
        records = [_rec(f"i{k}", "m1", True, 0.1, ecs_lift=0.05 * k,
                        cc3_delete=True, rand_delete=0.2) for k in range(8)]
        agg = aggregate(records, ["mask", "delete"], n_permutations=500,
                        min_n_for_test=6, seed=1)
        o = agg["overall"]
        assert o["cc3_flip_rate"]["mask"] == 1.0
        assert o["random_flip_rate"]["mask"] == pytest.approx(0.1)
        assert o["cc3_minus_random"]["mask"] == pytest.approx(0.9)
        t = o["cc3_vs_random_test"]
        assert t["mask"]["n_paired"] == 8
        assert t["mask"]["p_raw"] is not None and t["mask"]["p_raw"] < 0.05
        assert t["mask"]["p_holm"] is not None
        # Holm within the operator family: adjusted >= raw.
        assert t["mask"]["p_holm"] >= t["mask"]["p_raw"]

    def test_below_min_n_skips_test_but_reports_estimate(self):
        records = [_rec(f"i{k}", "m1", True, 0.0) for k in range(3)]
        agg = aggregate(records, ["mask"], n_permutations=200, min_n_for_test=6, seed=1)
        t = agg["overall"]["cc3_vs_random_test"]["mask"]
        assert t["p_raw"] is None and t["p_holm"] is None
        assert agg["overall"]["cc3_minus_random"]["mask"] is not None

    def test_tier_breakdown_descriptive(self):
        records = [_rec(f"i{k}", "m1", k % 2 == 0, 0.1, ecs_lift=0.1 * k) for k in range(9)]
        agg = aggregate(records, ["mask"], n_permutations=100, min_n_for_test=6, seed=1)
        tiers = agg["by_ecs_lift_tier"]
        assert "_thresholds" in tiers
        assert set(tiers) >= {"low", "high", "_thresholds"}

    def test_heldout_rate_counted(self):
        records = [_rec("a", "m1", True, 0.0), _rec("b", "m1", True, 0.0)]
        records[0]["cf_flip_heldout"] = True
        records[1]["cf_flip_heldout"] = False
        agg = aggregate(records, ["mask"], n_permutations=100, min_n_for_test=6, seed=1)
        assert agg["overall"]["cf_flip_heldout_rate"] == pytest.approx(0.5)
        assert agg["overall"]["n_cf_heldout_checked"] == 2


class TestErase:
    def test_mask_and_delete_all_occurrences(self):
        text = "good food and good service"
        assert erase(text, {"good"}, "mask") == "[MASK] food and [MASK] service"
        assert erase(text, {"good"}, "delete") == "food and service"

    def test_lemma_aware_erasure(self):
        norm = Normalizer(use_lemmatization=True, remove_stopwords=True)
        # Evidence lemma "movie" must erase the inflected occurrence "movies".
        out = erase("great movies here", {"movie"}, "delete", normalizer=norm)
        assert "movies" not in out

    def test_tier_helper(self):
        assert _tier(None, 0.1, 0.2) is None
        assert _tier(0.05, 0.1, 0.2) == "low"
        assert _tier(0.15, 0.1, 0.2) == "mid"
        assert _tier(0.25, 0.1, 0.2) == "high"
