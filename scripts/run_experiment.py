import argparse
import asyncio
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.load.dataset_loader import DatasetLoader, Instance
from src.inference.inference_engine import InferenceEngine
from src.parsing.parser import Parser
from src.normalization.normalizer import Normalizer
from src.metrics.metrics_calculator import MetricsCalculator
from src.utils.config_loader import load_and_validate_config, parse_command_line_args
from src.utils.data_models import InstanceResult, AggregateMetrics, save_instance_results, save_aggregate_metrics
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.execution_summary import ExecutionSummary
from src.utils.exceptions import APIError, ParsingError
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

STRATEGY_IDS = ["H", "R", "CF", "RO"]


def load_prompt(filepath: str) -> str:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding='utf-8').strip()


def format_prompt(template: str, input_text: str, label_set: List[str]) -> str:
    return template.format(input_text=input_text, label_set=", ".join(label_set))


def create_prompt_map(config) -> Dict[str, str]:
    prompts = {}
    for strategy in config.explanation_strategies:
        prompt_text = load_prompt(strategy.prompt_file)
        prompts[strategy.id] = prompt_text
    prompts["classification"] = load_prompt("prompts/classification.txt")
    return prompts


async def process_instance(
    instance: Instance,
    engine: InferenceEngine,
    parser: Parser,
    normalizer: Normalizer,
    calc: MetricsCalculator,
    prompts: Dict[str, str],
    config,
    dataset_config,
) -> InstanceResult:
    text = instance.text
    label_set = dataset_config.labels
    instance_id = instance.instance_id

    # Classification
    class_prompt = format_prompt(prompts["classification"], text, label_set)
    class_result = await engine.classify(class_prompt)
    predicted_label, confidence = parser.parse_classification(class_result.raw_response, label_set)
    correct = predicted_label == instance.label

    # Randomize strategy order
    strategy_order = STRATEGY_IDS.copy()
    random.shuffle(strategy_order)

    raw_responses = {}
    parsed_tokens = {}
    parsed_flags = {}

    for strat_id in STRATEGY_IDS:
        raw_responses[strat_id] = ""
        parsed_tokens[strat_id] = set()
        parsed_flags[strat_id] = False

    for strat_id in strategy_order:
        prompt_template = prompts[strat_id]
        strategy_prompt = format_prompt(prompt_template, text, label_set)
        try:
            result = await engine.explain(strategy_prompt, strat_id)
            raw_responses[strat_id] = result.raw_response
        except APIError as e:
            logger.error(f"API error for {instance_id} strategy {strat_id}: {e}")
            continue

    for strat_id in STRATEGY_IDS:
        raw = raw_responses[strat_id]
        if not raw:
            continue
        try:
            if strat_id == "H":
                tokens = parser.parse_highlighting(raw)
                normalized = normalizer.normalize_tokens(tokens)
                parsed_tokens[strat_id] = normalized
                parsed_flags[strat_id] = len(tokens) > 0
            elif strat_id == "R":
                rationale = parser.parse_rationale(raw)
                content_words = normalizer.extract_content_words_from_rationale(rationale)
                parsed_tokens[strat_id] = content_words
                parsed_flags[strat_id] = len(content_words) > 0
            elif strat_id == "CF":
                cf_text = parser.parse_counterfactual(raw)
                diff = normalizer.extract_counterfactual_diff(text, cf_text)
                parsed_tokens[strat_id] = diff
                parsed_flags[strat_id] = len(diff) > 0
            elif strat_id == "RO":
                ranked = parser.parse_rank_ordering(raw)
                ro_tokens = [t for t, r in ranked]
                normalized_set = normalizer.normalize_tokens(ro_tokens)
                normalized_ranked = []
                for token, rank in ranked:
                    for word in token.split():
                        n = normalizer.normalize(word)
                        if n:
                            normalized_ranked.append((n, rank))
                            break
                parsed_tokens["RO_set"] = normalized_set
                parsed_tokens["RO_ranked"] = normalized_ranked
                parsed_flags[strat_id] = len(ranked) > 0
        except ParsingError as e:
            logger.warning(f"Parsing error for {instance_id} strategy {strat_id}: {e}")

    ro_set = parsed_tokens.get("RO_set", set())
    ro_ranked = parsed_tokens.get("RO_ranked", [])

    explanations = {
        "H": parsed_tokens.get("H", set()),
        "R": parsed_tokens.get("R", set()),
        "CF": parsed_tokens.get("CF", set()),
        "RO": ro_set,
    }

    agreements = calc.compute_pairwise_agreements(explanations)

    result = InstanceResult(
        instance_id=instance_id,
        dataset=instance.dataset,
        model=engine.model_name,
        timestamp=datetime.now(),
        text=text,
        ground_truth_label=instance.label,
        predicted_label=predicted_label,
        confidence=confidence,
        correct=correct,
        raw_highlighting=raw_responses.get("H", ""),
        raw_rationale=raw_responses.get("R", ""),
        raw_counterfactual=raw_responses.get("CF", ""),
        raw_rank_ordering=raw_responses.get("RO", ""),
        highlighting_tokens=parsed_tokens.get("H", set()),
        rationale_tokens=parsed_tokens.get("R", set()),
        counterfactual_tokens=parsed_tokens.get("CF", set()),
        rank_ordering_tokens=ro_ranked,
        highlighting_parsed=parsed_flags.get("H", False),
        rationale_parsed=parsed_flags.get("R", False),
        counterfactual_parsed=parsed_flags.get("CF", False),
        rank_ordering_parsed=parsed_flags.get("RO", False),
        jaccard_H_R=agreements.get(("H", "R")),
        jaccard_H_CF=agreements.get(("H", "CF")),
        jaccard_H_RO=agreements.get(("H", "RO")),
        jaccard_R_CF=agreements.get(("R", "CF")),
        jaccard_R_RO=agreements.get(("R", "RO")),
        jaccard_CF_RO=agreements.get(("CF", "RO")),
        ecs=calc.compute_ecs(agreements),
    )

    # Compute consensus cores
    all_parsed = all(parsed_flags.get(s, False) for s in STRATEGY_IDS)
    if all_parsed:
        result.cc3_tokens = calc.compute_consensus_core(explanations, 3)
        result.cc4_tokens = calc.compute_consensus_core(explanations, 4)
        result.cc3_size = len(result.cc3_tokens)
        result.cc4_size = len(result.cc4_tokens)

    return result


def compute_aggregate_metrics(results: List[InstanceResult], level: str, group: str) -> AggregateMetrics:
    from src.statistics.statistical_tests import compute_confidence_ecs_correlation
    import numpy as np

    ecs_values = [r.ecs for r in results if r.ecs is not None]
    confidences = [r.confidence for r in results if r.ecs is not None]

    if not ecs_values:
        return AggregateMetrics(
            aggregation_level=level, group_name=group, n_instances=len(results),
            mean_ecs=0, std_ecs=0, median_ecs=0,
            ecs_ci_lower=0, ecs_ci_upper=0,
            mean_jaccard_H_R=0, mean_jaccard_H_CF=0, mean_jaccard_H_RO=0,
            mean_jaccard_R_CF=0, mean_jaccard_R_RO=0, mean_jaccard_CF_RO=0,
            mean_kendall_H_RO=0,
            mean_cc3_size=0, mean_cc4_size=0,
            pct_instances_with_cc3=0, pct_instances_with_cc4=0,
            spearman_rho=0, spearman_p_value=1.0,
            correlation_ci_lower=0, correlation_ci_upper=0,
            highlighting_success_rate=0, rationale_success_rate=0,
            counterfactual_success_rate=0, rank_ordering_success_rate=0,
        )

    corr = compute_confidence_ecs_correlation(confidences, ecs_values, n_bootstrap=100)

    def safe_mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    def jaccard_mean(key):
        vals = [getattr(r, key) for r in results if getattr(r, key) is not None]
        return safe_mean(vals)

    return AggregateMetrics(
        aggregation_level=level,
        group_name=group,
        n_instances=len(results),
        mean_ecs=safe_mean(ecs_values),
        std_ecs=float(np.std(ecs_values)) if len(ecs_values) > 1 else 0.0,
        median_ecs=float(np.median(ecs_values)) if ecs_values else 0.0,
        ecs_ci_lower=corr.ci_lower,
        ecs_ci_upper=corr.ci_upper,
        mean_jaccard_H_R=jaccard_mean('jaccard_H_R'),
        mean_jaccard_H_CF=jaccard_mean('jaccard_H_CF'),
        mean_jaccard_H_RO=jaccard_mean('jaccard_H_RO'),
        mean_jaccard_R_CF=jaccard_mean('jaccard_R_CF'),
        mean_jaccard_R_RO=jaccard_mean('jaccard_R_RO'),
        mean_jaccard_CF_RO=jaccard_mean('jaccard_CF_RO'),
        mean_kendall_H_RO=jaccard_mean('kendall_H_RO'),
        mean_cc3_size=safe_mean([r.cc3_size for r in results]),
        mean_cc4_size=safe_mean([r.cc4_size for r in results]),
        pct_instances_with_cc3=sum(1 for r in results if r.cc3_size > 0) / max(len(results), 1) * 100,
        pct_instances_with_cc4=sum(1 for r in results if r.cc4_size > 0) / max(len(results), 1) * 100,
        spearman_rho=corr.rho,
        spearman_p_value=corr.p_value,
        correlation_ci_lower=corr.ci_lower,
        correlation_ci_upper=corr.ci_upper,
        highlighting_success_rate=sum(1 for r in results if r.highlighting_parsed) / max(len(results), 1),
        rationale_success_rate=sum(1 for r in results if r.rationale_parsed) / max(len(results), 1),
        counterfactual_success_rate=sum(1 for r in results if r.counterfactual_parsed) / max(len(results), 1),
        rank_ordering_success_rate=sum(1 for r in results if r.rank_ordering_parsed) / max(len(results), 1),
    )


async def run_experiment(config, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output.base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=output_dir / "logs", console_level=config.output.log_level)
    logger.info(f"Starting experiment: {config.experiment.name} v{config.experiment.version}")
    logger.info(f"Output directory: {output_dir}")

    loader = DatasetLoader(seed=config.experiment.seed)
    parser = Parser()
    normalizer = Normalizer()
    calc = MetricsCalculator()

    summary = ExecutionSummary(
        start_time=datetime.now(),
        end_time=datetime.now(),
        duration_seconds=0.0,
        total_instances=0,
        successful_instances=0,
        failed_instances=0,
    )

    all_results: List[InstanceResult] = []
    aggregate_list: List[AggregateMetrics] = []

    for dataset_config_obj in config.datasets:
        dataset_name = dataset_config_obj.name
        logger.info(f"Processing dataset: {dataset_name}")

        dataset_config = next(
            (d for d in config.datasets if d.name == dataset_name),
            None
        )
        if not dataset_config:
            continue

        try:
            dataset = loader.load_dataset(
                dataset_config.huggingface_id,
                dataset_config.split
            )
            instances = loader.sample_balanced(
                dataset=dataset,
                n_samples=dataset_config.sample_size,
                label_field=getattr(dataset_config, 'label_field', 'label'),
                text_field=getattr(dataset_config, 'text_field', 'text'),
                secondary_text_field=getattr(dataset_config, 'secondary_text_field', None),
                dataset_name=dataset_name,
                split=dataset_config.split,
            )
            logger.info(f"Loaded {len(instances)} instances for {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue

        summary.total_instances += len(instances)

        for model_config in config.models:
            logger.info(f"Processing model: {model_config.name} on {dataset_name}")

            engine = InferenceEngine(
                model_name=model_config.groq_model_id,
                max_retries=config.inference.max_retries,
                concurrent_requests=config.inference.concurrent_requests,
            )

            prompts = create_prompt_map(config)
            model_results: List[InstanceResult] = []

            for i, instance in enumerate(instances):
                logger.info(f"Processing {instance.instance_id} ({i+1}/{len(instances)})")
                try:
                    result = await process_instance(
                        instance, engine, parser, normalizer, calc, prompts, config, dataset_config
                    )
                    model_results.append(result)
                    summary.successful_instances += 1
                except Exception as e:
                    logger.error(f"Failed to process {instance.instance_id}: {e}")
                    summary.failed_instances += 1
                    continue

                if (i + 1) % config.output.checkpoint_frequency == 0:
                    cp = CheckpointManager(output_dir / f"checkpoint_{dataset_name}_{model_config.name}.jsonl")
                    cp.save_checkpoint([r.to_dict() for r in model_results])

            all_results.extend(model_results)

            # Compute aggregate metrics per model-dataset
            agg = compute_aggregate_metrics(model_results, "model_dataset", f"{model_config.name}_{dataset_name}")
            aggregate_list.append(agg)

    # Compute overall aggregate
    if all_results:
        overall = compute_aggregate_metrics(all_results, "overall", "all")
        aggregate_list.append(overall)

    # Save results
    save_instance_results(all_results, str(output_dir / "instance_results.jsonl"))
    save_aggregate_metrics(aggregate_list, str(output_dir / "aggregate_metrics.json"))

    # Save summary
    summary.end_time = datetime.now()
    summary.duration_seconds = (summary.end_time - summary.start_time).total_seconds()
    summary.api_requests_total = len(all_results) * 5  # 1 class + 4 explanations

    with open(output_dir / "execution_summary.txt", 'w') as f:
        f.write(summary.generate_report())

    logger.info(f"Experiment complete. Results saved to {output_dir}")
    logger.info(summary.generate_report())

    return all_results, aggregate_list


def main():
    args = parse_command_line_args()
    config = load_and_validate_config(args=args)
    asyncio.run(run_experiment(config, args))


if __name__ == "__main__":
    main()
