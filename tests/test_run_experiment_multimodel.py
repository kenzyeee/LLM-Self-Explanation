"""Tests for the concurrent multi-model orchestration in run_experiment.

Verifies that `_run_model_on_dataset` (a) produces a correct per-model bundle,
(b) actually runs concurrently across models when fanned out with asyncio.gather,
and (c) counts the three handled failure modes correctly. process_instance,
InferenceEngine, and CheckpointManager are stubbed so no API calls or file writes
happen — only the orchestration logic is exercised.
"""
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from src.utils.config import InferenceConfig, OutputConfig, ModelConfig, DatasetConfig
from src.utils.exceptions import RateLimitExhausted, PromptValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_experiment_mod", ROOT / "scripts" / "run_experiment.py")
rx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rx)


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

    assert bundle["successful"] == 1
    assert bundle["failed"] == 3
    assert bundle["prompt_validation_failures"] == 1
    assert len(bundle["results"]) == 1
