#!/usr/bin/env python
"""Curate a frozen, hand-vetted clean-gold dataset for the full experiment.

No model is used. Two deterministic stages with Claude's hand-vetting in between:

    # Stage A — freeze a candidate pool for review
    python scripts/curate_dataset.py build --dataset sst2
    python scripts/curate_dataset.py build --all

    #   ... Claude reads data/processed/{ds}_candidates.jsonl and authors
    #       data/processed/{ds}_decisions.jsonl  (one line per candidate:
    #       {"instance_id": ..., "decision": "keep"|"drop", "reason": ...}),
    #       dropping mislabeled / ambiguous / low-quality instances ...

    # Stage B — apply decisions, balance to `target`, freeze curated set
    python scripts/curate_dataset.py finalize --dataset sst2
    python scripts/curate_dataset.py finalize --all

Outputs data/processed/{ds}_curated.jsonl and {ds}_datasheet.json. Reproducibility
rests on the seeded candidate pool plus the committed decisions file.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.load.curator import DEFAULTS, DatasetCurator, CurationReport
from src.utils.config_loader import load_and_validate_config
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def _resolve_curation_cfg(ds_cfg, target_override=None) -> dict:
    cfg = dict(DEFAULTS)
    if getattr(ds_cfg, "curation", None):
        cfg.update({k: v for k, v in ds_cfg.curation.items() if v is not None})
    if target_override is not None:
        cfg["target"] = target_override
    return cfg


def _build_pool(ds_cfg, seed: int, cfg: dict):
    """Stages 1–3: load + clean → quality filter → stratified oversampled pool."""
    curator = DatasetCurator(seed=seed)
    name = ds_cfg.name
    report = CurationReport(
        dataset=name, huggingface_id=ds_cfg.huggingface_id, split=ds_cfg.split,
        seed=seed, target=cfg["target"],
    )
    raw = curator.loader.load_dataset(ds_cfg.huggingface_id, ds_cfg.split)
    instances = curator.build_instances(raw, ds_cfg)
    report.raw_count = len(instances)

    instances, drops = curator.quality_filter(instances, cfg)
    report.drop_counts = drops
    logger.info(f"{name}: {report.raw_count} raw -> {len(instances)} after filters {drops}")

    curator.assign_length_buckets(instances)
    pool, n_near = curator.stratified_pool(instances, cfg)
    report.drop_counts["near_dup"] = n_near
    report.pool_size = len(pool)
    logger.info(f"{name}: candidate pool = {len(pool)} (near-dup dropped {n_near})")
    return curator, pool, report


def cmd_build(ds_cfg, seed: int, target_override) -> None:
    cfg = _resolve_curation_cfg(ds_cfg, target_override)
    name = ds_cfg.name
    logger.info(f"=== build {name} (target={cfg['target']}, "
                f"pool~{cfg['target']}x{cfg['oversample_factor']}) ===")
    curator, pool, _ = _build_pool(ds_cfg, seed, cfg)
    candidates_path = PROCESSED_DIR / f"{name}_candidates.jsonl"
    curator.freeze_candidates(pool, candidates_path)
    decisions_path = PROCESSED_DIR / f"{name}_decisions.jsonl"
    logger.info(f"{name}: review {candidates_path} and author {decisions_path}, "
                f"then run: curate_dataset.py finalize --dataset {name}")


def cmd_finalize(ds_cfg, seed: int, target_override) -> bool:
    cfg = _resolve_curation_cfg(ds_cfg, target_override)
    name = ds_cfg.name
    candidates_path = PROCESSED_DIR / f"{name}_candidates.jsonl"
    decisions_path = PROCESSED_DIR / f"{name}_decisions.jsonl"
    if not decisions_path.exists():
        logger.error(f"{name}: missing decisions file {decisions_path}. "
                     f"Run `build --dataset {name}`, author the decisions, then finalize.")
        return False

    # Rebuild the exact same seeded pool (deterministic) so decisions line up,
    # and recompute the build-stage report fields for the datasheet.
    curator, pool, report = _build_pool(ds_cfg, seed, cfg)
    by_id = {i.instance_id: i for i in pool}
    decisions = curator.load_decisions(decisions_path)

    # Warn if the decisions file references the candidate file rather than the pool
    # (e.g. stale candidates after a config change).
    unknown = [iid for iid in decisions if iid not in by_id]
    if unknown:
        logger.warning(f"{name}: {len(unknown)} decision ids not in current pool "
                       f"(stale candidates?). They are ignored.")

    kept, vetted_dropped = curator.apply_decisions(pool, decisions)
    report.vetted_kept = len(kept)
    report.vetted_dropped = vetted_dropped
    logger.info(f"{name}: vetting kept {len(kept)}/{len(pool)} (dropped {vetted_dropped})")

    final, shortfalls = curator.select_balanced(kept, cfg)
    report.shortfalls = shortfalls
    curator.write_outputs(final, report, PROCESSED_DIR)
    logger.info(f"{name}: DONE - {report.final_count} curated "
                f"(labels {report.label_distribution})")
    for s in shortfalls:
        logger.warning(f"{name}: shortfall - {s}")
    return True


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_and_validate_config(config_dir=args.config_dir)
    setup_logging(log_dir=Path("logs"), console_level="INFO")

    selected = [d for d in config.datasets if args.all or d.name == args.dataset]
    if not selected:
        logger.error(f"No dataset matched --dataset={args.dataset}. "
                     f"Available: {[d.name for d in config.datasets]}")
        return 2

    seed = config.experiment.seed
    ok = True
    for ds_cfg in selected:
        if args.command == "build":
            cmd_build(ds_cfg, seed, args.target)
        else:
            ok = cmd_finalize(ds_cfg, seed, args.target) and ok
    return 0 if ok else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Curate a frozen clean-gold dataset (no model)")
    p.add_argument("command", choices=["build", "finalize"],
                   help="build: freeze candidate pool; finalize: apply decisions + balance to target")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", type=str, help="Dataset name (sst2 | mnli | ag_news)")
    g.add_argument("--all", action="store_true", help="Process all configured datasets")
    p.add_argument("--config-dir", type=str, default="config")
    p.add_argument("--target", type=int, default=None, help="Override curated set size")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
