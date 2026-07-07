import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_models import (
    load_instance_results, load_aggregate_metrics
)
from src.utils.pretty_printer import PrettyPrinter

logger = logging.getLogger(__name__)


def analyze_results(results_dir: str) -> None:
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    instance_file = results_path / "instance_results.jsonl"
    aggregate_file = results_path / "aggregate_metrics.json"
    # The erasure pass (scripts/run_validity_tests.py) writes aggregate_erasure.json;
    # the old validity_tests.jsonl file was retired with the deleted Validity_Checker.
    erasure_file = results_path / "aggregate_erasure.json"

    if instance_file.exists():
        instances = load_instance_results(str(instance_file))
        logger.info(f"Loaded {len(instances)} instance results")

        ecs_values = [r.ecs for r in instances if r.ecs is not None]
        if ecs_values:
            import numpy as np
            logger.info(f"ECS statistics:")
            logger.info(f"  Mean: {np.mean(ecs_values):.4f}")
            logger.info(f"  Std:  {np.std(ecs_values):.4f}")
            logger.info(f"  Min:  {min(ecs_values):.4f}")
            logger.info(f"  Max:  {max(ecs_values):.4f}")

        highlight_parsed = sum(1 for r in instances if r.highlighting_parsed)
        rationale_parsed = sum(1 for r in instances if r.rationale_parsed)
        counterfactual_parsed = sum(1 for r in instances if r.counterfactual_parsed)
        rank_parsed = sum(1 for r in instances if r.rank_ordering_parsed)

        logger.info(f"Parsing success rates:")
        logger.info(f"  Highlighting:  {highlight_parsed}/{len(instances)} ({highlight_parsed/len(instances)*100:.1f}%)")
        logger.info(f"  Rationale:     {rationale_parsed}/{len(instances)} ({rationale_parsed/len(instances)*100:.1f}%)")
        logger.info(f"  Counterfactual:{counterfactual_parsed}/{len(instances)} ({counterfactual_parsed/len(instances)*100:.1f}%)")
        logger.info(f"  Rank Ordering: {rank_parsed}/{len(instances)} ({rank_parsed/len(instances)*100:.1f}%)")

    if aggregate_file.exists():
        aggregates = load_aggregate_metrics(str(aggregate_file))
        logger.info(f"Loaded {len(aggregates)} aggregate metric groups")
        for agg in aggregates:
            logger.info(f"  {agg.aggregation_level}={agg.group_name}: n={agg.n_instances}, mean_ECS={agg.mean_ecs:.4f}")

    if erasure_file.exists():
        import json
        with open(erasure_file, encoding="utf-8") as f:
            erasure = json.load(f)
        overall = erasure.get("pooled", {}).get("overall", {})
        operators = overall.get("operators", [])
        logger.info("Erasure pass (second consistency axis — pooled, descriptive):")
        for op in operators:
            cc = overall.get("cc3_flip_rate", {}).get(op)
            rnd = overall.get("random_flip_rate", {}).get(op)
            gap = overall.get("cc3_minus_random", {}).get(op)
            def _fmt(x):
                return f"{x:.3f}" if isinstance(x, (int, float)) else "—"
            logger.info(f"  [{op}] CC3 flip={_fmt(cc)}  random={_fmt(rnd)}  gap={_fmt(gap)}")


def print_instance(results_dir: str, instance_id: str) -> None:
    results_path = Path(results_dir)
    instance_file = results_path / "instance_results.jsonl"

    if not instance_file.exists():
        logger.error(f"Instance results file not found: {instance_file}")
        return

    instances = load_instance_results(str(instance_file))
    for inst in instances:
        if inst.instance_id == instance_id:
            printer = PrettyPrinter()
            print(printer.format_instance(inst.to_dict()))
            print(printer.format_normalized_tokens(inst.to_dict()))
            print(printer.format_pairwise_agreements(inst.to_dict()))
            if inst.cc3_tokens:
                highlighted = printer.highlight_consensus_core(inst.text, inst.cc3_tokens)
                print(f"Consensus Core Highlighted Text:\n{highlighted}")
            return

    logger.warning(f"Instance not found: {instance_id}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("--instance", type=str, help="Print details for a specific instance")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.instance:
        print_instance(args.results_dir, args.instance)
    else:
        analyze_results(args.results_dir)


if __name__ == "__main__":
    main()
