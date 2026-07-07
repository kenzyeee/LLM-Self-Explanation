"""Tests for the concurrent multi-model orchestration in run_experiment.

Verifies that `_run_model_on_dataset` (a) produces a correct per-model bundle,
(b) actually runs concurrently across models when fanned out with asyncio.gather,
(c) counts the three handled failure modes correctly, and (d) resumes correctly
from an existing checkpoint (skips already-done instances, force-restart discards
them). process_instance, InferenceEngine, and CheckpointManager are stubbed so no
API calls or file writes happen — only the orchestration logic is exercised.
"""
import asyncio
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from src.utils.config import InferenceConfig, OutputConfig, ModelConfig, DatasetConfig
from src.utils.exceptions import RateLimitExhausted, PromptValidationError
from src.utils.data_models import InstanceResult

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_experiment_mod", ROOT / "scripts" / "run_experiment.py")
rx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rx)


def _real_instance_result(instance_id, correct=True, dataset="sst2", model="m"):
    """A minimal but REAL InstanceResult (not the DummyResult stub) — needed for
    tests that round-trip through to_dict()/from_dict() (checkpoint save/load)."""
    return InstanceResult(
        instance_id=instance_id, dataset=dataset, model=model, timestamp=datetime(2026, 7, 4),
        text="some text", ground_truth_label="positive",
        predicted_label="positive" if correct else "negative", correct=correct,
    )


class DummyResult:
    """Minimal stand-in for InstanceResult. ecs=None makes compute_aggregate_metrics
    short-circuit, so no other metric fields are needed."""
    def __init__(self, correct=True, prompt_tokens=10, response_tokens=20, model="m", dataset="d"):
        self.ecs = None
        self.correct = correct
        self.prompt_tokens = prompt_tokens
        self.response_tokens = response_tokens
        self.model = model
        self.dataset = dataset

    def to_dict(self):
        return {"stub": True}


def _make_engine(**kwargs):
    """Factory that replaces InferenceEngine — one stub per model, distinct model_name."""
    engine = MagicMock()
    engine.model_name = kwargs.get("model_name", "unknown")
    engine.total_prompt_tokens = 0
    engine.total_completion_tokens = 0
    engine.n_truncated = 0
    engine.total_requests = 0
    engine.total_requests_failed = 0
    engine.requests_by_category = {}
    return engine


def _config(concurrent_requests=4, checkpoint_frequency=20, max_retries=1):
    return SimpleNamespace(
        inference=InferenceConfig(concurrent_requests=concurrent_requests, max_retries=max_retries),
        output=OutputConfig(checkpoint_frequency=checkpoint_frequency),
    )


def _dataset(sample_size, name="sst2"):
    return DatasetConfig(name=name, huggingface_id="hf", split="validation",
                         sample_size=sample_size, labels=["neg", "pos"])


def _instances(n):
    return [SimpleNamespace(instance_id=f"i{k}") for k in range(n)]


async def test_bundle_counts_and_results(tmp_path):
    results_seq = [DummyResult(correct=True), DummyResult(correct=False), DummyResult(correct=True)]
    call = {"n": 0}

    async def fake_pi(instance, engine, *args, **kwargs):
        r = results_seq[call["n"]]
        call["n"] += 1
        return r

    with patch.object(rx, "InferenceEngine", _make_engine), \
         patch.object(rx, "CheckpointManager", MagicMock()), \
         patch.object(rx, "process_instance", fake_pi):
        bundle = await rx._run_model_on_dataset(
            ModelConfig(name="llama", model_id="id", context_window=1000),
            _instances(3), {}, _dataset(3),
            MagicMock(), MagicMock(), MagicMock(), _config(), tmp_path,
        )

    assert len(bundle["results"]) == 3
    assert bundle["successful"] == 3
    assert bundle["failed"] == 0
    assert bundle["wrong_pred_count"] == 1
    # Per-(model,dataset) aggregate is labelled for this model+dataset.
    assert bundle["agg"].group_name == "llama_sst2"


async def test_models_run_concurrently(tmp_path):
    """Fanning three models out with gather must overlap: all three must enter
    process_instance before any is allowed to return. If they ran sequentially the
    shared barrier would never release and wait_for would time out."""
    n_models = 3
    state = {"entered": 0}
    all_entered = asyncio.Event()

    async def barrier_pi(instance, engine, *args, **kwargs):
        state["entered"] += 1
        if state["entered"] >= n_models:
            all_entered.set()
        await asyncio.wait_for(all_entered.wait(), timeout=5)
        return DummyResult(correct=True, model=engine.model_name)

    models = [ModelConfig(name=f"m{k}", model_id=f"id{k}", context_window=1000) for k in range(n_models)]
    ds = _dataset(1)

    with patch.object(rx, "InferenceEngine", _make_engine), \
         patch.object(rx, "CheckpointManager", MagicMock()), \
         patch.object(rx, "process_instance", barrier_pi):
        bundles = await asyncio.wait_for(asyncio.gather(*[
            rx._run_model_on_dataset(m, _instances(1), {}, ds,
                                     MagicMock(), MagicMock(), MagicMock(), _config(), tmp_path)
            for m in models
        ]), timeout=10)

    assert state["entered"] == n_models          # all three overlapped
    assert len(bundles) == n_models
    assert all(b["successful"] == 1 for b in bundles)


async def test_gather_preserves_config_order(tmp_path):
    """gather returns bundles in submission order, so downstream merge/output ordering
    is deterministic regardless of which model finishes first."""
    async def fake_pi(instance, engine, *args, **kwargs):
        return DummyResult(correct=True, model=engine.model_name)

    models = [ModelConfig(name=n, model_id=f"id-{n}", context_window=1000)
              for n in ("nova-pro", "qwen3-235b", "deepseek-v3")]
    ds = _dataset(2)

    with patch.object(rx, "InferenceEngine", _make_engine), \
         patch.object(rx, "CheckpointManager", MagicMock()), \
         patch.object(rx, "process_instance", fake_pi):
        bundles = await asyncio.gather(*[
            rx._run_model_on_dataset(m, _instances(2), {}, ds,
                                     MagicMock(), MagicMock(), MagicMock(), _config(), tmp_path)
            for m in models
        ])

    assert [b["agg"].group_name for b in bundles] == [
        "nova-pro_sst2", "qwen3-235b_sst2", "deepseek-v3_sst2",
    ]


async def test_failure_modes_are_counted(tmp_path):
    # RateLimitExhausted BREAKS the loop (see test_rate_limit_stops_loop_early below),
    # so the 4th item (RuntimeError) is never reached — only 3 of 4 instances are
    # attempted, and only 2 of those 3 count as "failed".
    seq = [
        DummyResult(correct=True),
        PromptValidationError("bad prompt"),
        RateLimitExhausted("throttled"),
        RuntimeError("boom"),
    ]
    call = {"n": 0}

    async def fake_pi(instance, engine, *args, **kwargs):
        item = seq[call["n"]]
        call["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    with patch.object(rx, "InferenceEngine", _make_engine), \
         patch.object(rx, "CheckpointManager", MagicMock()), \
         patch.object(rx, "process_instance", fake_pi):
        bundle = await rx._run_model_on_dataset(
            ModelConfig(name="m", model_id="id", context_window=1000),
            _instances(4), {}, _dataset(4),
            MagicMock(), MagicMock(), MagicMock(), _config(max_retries=1), tmp_path,
        )

    assert call["n"] == 3  # the 4th instance (RuntimeError) was never attempted
    assert bundle["successful"] == 1
    assert bundle["failed"] == 2
    assert bundle["prompt_validation_failures"] == 1
    assert len(bundle["results"]) == 1


async def test_rate_limit_stops_loop_early(tmp_path):
    """A sustained rate/quota exhaustion will not clear up by trying the NEXT
    instance (Bedrock daily-quota exhaustion persists for hours) — the loop must
    stop immediately rather than burning through every remaining instance on
    guaranteed failures, so scripts/resume_experiment.py has something meaningful
    to resume from soon rather than after a long, wasted retry-storm."""
    call = {"n": 0}

    async def fake_pi(instance, engine, *args, **kwargs):
        call["n"] += 1
        if call["n"] == 2:
            raise RateLimitExhausted("quota exhausted")
        return DummyResult(correct=True)

    with patch.object(rx, "InferenceEngine", _make_engine), \
         patch.object(rx, "CheckpointManager", MagicMock()), \
         patch.object(rx, "process_instance", fake_pi):
        bundle = await rx._run_model_on_dataset(
            ModelConfig(name="m", model_id="id", context_window=1000),
            _instances(10), {}, _dataset(10),
            MagicMock(), MagicMock(), MagicMock(), _config(max_retries=1), tmp_path,
        )

    assert call["n"] == 2  # instances 3-10 were never attempted
    assert bundle["successful"] == 1
    assert bundle["failed"] == 1


async def test_bundle_surfaces_engine_request_accounting(tmp_path):
    """The bundle must pass through the engine's OWN authoritative request counters
    (review P0.4) — not recompute or guess them."""
    async def fake_pi(instance, engine, *args, **kwargs):
        return DummyResult(correct=True)

    def make_engine_with_counts(**kwargs):
        engine = _make_engine(**kwargs)
        engine.total_requests = 37
        engine.total_requests_failed = 3
        engine.requests_by_category = {"classification": 10, "H": 10, "CF_verify": 5}
        return engine

    with patch.object(rx, "InferenceEngine", make_engine_with_counts), \
         patch.object(rx, "CheckpointManager", MagicMock()), \
         patch.object(rx, "process_instance", fake_pi):
        bundle = await rx._run_model_on_dataset(
            ModelConfig(name="m", model_id="id", context_window=1000),
            _instances(2), {}, _dataset(2),
            MagicMock(), MagicMock(), MagicMock(), _config(), tmp_path,
        )

    assert bundle["api_requests"] == 37
    assert bundle["api_requests_failed"] == 3
    assert bundle["api_requests_by_category"] == {"classification": 10, "H": 10, "CF_verify": 5}


class TestLoadCheckpointedResults:
    def test_returns_empty_when_no_file(self, tmp_path):
        assert rx._load_checkpointed_results(tmp_path, "sst2", "nova-pro") == []

    def test_roundtrips_real_instance_result(self, tmp_path):
        r = _real_instance_result("sst2_001")
        cp_path = tmp_path / "checkpoint_sst2_nova-pro.jsonl"
        with open(cp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(r.to_dict()) + "\n")

        loaded = rx._load_checkpointed_results(tmp_path, "sst2", "nova-pro")
        assert len(loaded) == 1
        assert loaded[0].instance_id == "sst2_001"
        assert loaded[0].correct is True

    def test_dedupes_by_instance_id_last_wins(self, tmp_path):
        r1 = _real_instance_result("sst2_001", correct=True)
        r2 = _real_instance_result("sst2_001", correct=False)  # same id, rewritten
        cp_path = tmp_path / "checkpoint_sst2_nova-pro.jsonl"
        with open(cp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(r1.to_dict()) + "\n")
            f.write(json.dumps(r2.to_dict()) + "\n")

        loaded = rx._load_checkpointed_results(tmp_path, "sst2", "nova-pro")
        assert len(loaded) == 1
        assert loaded[0].correct is False  # the later line wins


class TestResumeSkipsCompletedInstances:
    async def test_skips_already_checkpointed_instances(self, tmp_path):
        existing = [_real_instance_result("i0", correct=True), _real_instance_result("i1", correct=False)]
        attempted = []

        async def fake_pi(instance, engine, *args, **kwargs):
            attempted.append(instance.instance_id)
            return DummyResult(correct=True)

        with patch.object(rx, "InferenceEngine", _make_engine), \
             patch.object(rx, "CheckpointManager", MagicMock()), \
             patch.object(rx, "process_instance", fake_pi):
            bundle = await rx._run_model_on_dataset(
                ModelConfig(name="m", model_id="id", context_window=1000),
                _instances(4), {}, _dataset(4),
                MagicMock(), MagicMock(), MagicMock(), _config(), tmp_path,
                existing_results=existing,
            )

        # Only the 2 NOT-already-done instances (i2, i3) were attempted.
        assert attempted == ["i2", "i3"]
        # Bundle merges the 2 seeded + 2 newly-processed results.
        assert len(bundle["results"]) == 4
        assert bundle["successful"] == 4
        # wrong_pred_count seeded from the one incorrect existing result.
        assert bundle["wrong_pred_count"] == 1

    async def test_force_restart_ignores_existing_results(self, tmp_path):
        existing = [_real_instance_result("i0"), _real_instance_result("i1")]
        attempted = []

        async def fake_pi(instance, engine, *args, **kwargs):
            attempted.append(instance.instance_id)
            return DummyResult(correct=True)

        with patch.object(rx, "InferenceEngine", _make_engine), \
             patch.object(rx, "CheckpointManager", MagicMock()), \
             patch.object(rx, "process_instance", fake_pi):
            bundle = await rx._run_model_on_dataset(
                ModelConfig(name="m", model_id="id", context_window=1000),
                _instances(4), {}, _dataset(4),
                MagicMock(), MagicMock(), MagicMock(), _config(), tmp_path,
                existing_results=existing, force_restart=True,
            )

        # ALL 4 instances reprocessed, none skipped — the pre-existing results are
        # discarded entirely, not merged in.
        assert attempted == ["i0", "i1", "i2", "i3"]
        assert len(bundle["results"]) == 4

    async def test_no_existing_results_behaves_like_fresh_run(self, tmp_path):
        attempted = []

        async def fake_pi(instance, engine, *args, **kwargs):
            attempted.append(instance.instance_id)
            return DummyResult(correct=True)

        with patch.object(rx, "InferenceEngine", _make_engine), \
             patch.object(rx, "CheckpointManager", MagicMock()), \
             patch.object(rx, "process_instance", fake_pi):
            bundle = await rx._run_model_on_dataset(
                ModelConfig(name="m", model_id="id", context_window=1000),
                _instances(3), {}, _dataset(3),
                MagicMock(), MagicMock(), MagicMock(), _config(), tmp_path,
                existing_results=None,
            )

        assert attempted == ["i0", "i1", "i2"]
        assert len(bundle["results"]) == 3


def _free_cf_result(instance_id, h, r, cf_free, ro, cf_free_valid=True,
                    h_valid=True, r_valid=True, ro_valid=True):
    """A real InstanceResult shaped for compute_free_cf_sensitivity_ecs: only the
    fields that function reads are populated (H/R/RO sets, cf_contrast_tokens/valid,
    rank_ordering_tokens as ranked pairs)."""
    return InstanceResult(
        instance_id=instance_id, dataset="sst2", model="m", timestamp=datetime(2026, 7, 4),
        text="t", ground_truth_label="positive", predicted_label="positive",
        highlighting_tokens=set(h), highlighting_valid=h_valid,
        rationale_tokens=set(r), rationale_valid=r_valid,
        rank_ordering_tokens=[(t, i + 1) for i, t in enumerate(ro)], rank_ordering_valid=ro_valid,
        cf_contrast_tokens=set(cf_free), cf_contrast_valid=cf_free_valid,
    )


class TestFreeCfSensitivityEcs:
    """Review P1.1: ECS recomputed with cf_contrast_tokens (free/unconstrained CF,
    ~82% validity in the pilot) substituted for the canonical minimal-CF evidence
    (~28% validity) — a zero-extra-API-cost descriptive robustness check."""

    def test_empty_results_returns_zero(self):
        mean, n = rx.compute_free_cf_sensitivity_ecs([], MagicMock())
        assert mean == 0.0
        assert n == 0

    def test_requires_cf_contrast_valid(self):
        from src.metrics.metrics_calculator import MetricsCalculator
        r = _free_cf_result("i0", h={"great"}, r={"great"}, cf_free={"great"}, ro=["great"],
                            cf_free_valid=False)
        mean, n = rx.compute_free_cf_sensitivity_ecs([r], MetricsCalculator())
        assert n == 0

    def test_requires_h_r_ro_all_valid(self):
        from src.metrics.metrics_calculator import MetricsCalculator
        r = _free_cf_result("i0", h={"great"}, r={"great"}, cf_free={"great"}, ro=["great"],
                            h_valid=False)
        mean, n = rx.compute_free_cf_sensitivity_ecs([r], MetricsCalculator())
        assert n == 0

    def test_computes_ecs_over_cross_paradigm_pairs_using_free_cf(self):
        from src.metrics.metrics_calculator import MetricsCalculator
        # H, R, RO all share "great"; free-CF also contains "great" plus an extra
        # token. Cross-paradigm pairs (H-R, H-CF, R-CF, R-RO, CF-RO; H-RO excluded)
        # should reflect this partial overlap.
        r = _free_cf_result("i0", h={"great", "movie"}, r={"great"}, cf_free={"great", "terrible"},
                            ro=["great", "movie"])
        mean, n = rx.compute_free_cf_sensitivity_ecs([r], MetricsCalculator())
        assert n == 1
        assert 0.0 < mean < 1.0

    def test_multiple_instances_averaged(self):
        from src.metrics.metrics_calculator import MetricsCalculator
        # Tokens must be content words that survive normalization: cf_contrast_tokens
        # are now projected into the normalized space before comparison (P0.4), and
        # single-letter/stopword tokens ("a") would normalize away to an empty CF set.
        results = [
            _free_cf_result("i0", h={"great"}, r={"great"}, cf_free={"great"}, ro=["great"]),
            _free_cf_result("i1", h={"movie"}, r={"acting"}, cf_free={"awful"}, ro=["boring"]),  # no overlap anywhere
        ]
        mean, n = rx.compute_free_cf_sensitivity_ecs(results, MetricsCalculator())
        assert n == 2
        # Perfect overlap (i0, ECS=1.0) averaged with zero overlap (i1, ECS=0.0).
        assert mean == pytest.approx(0.5)
