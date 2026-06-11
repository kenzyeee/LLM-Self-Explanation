import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.load.dataset_loader import DatasetLoader
from src.inference.inference_engine import InferenceEngine
from src.parsing.parser import Parser
from src.normalization.normalizer import Normalizer, NormalizationConfig
from src.metrics.metrics_calculator import MetricsCalculator
from src.utils.config_loader import load_and_validate_config, parse_command_line_args
from src.utils.data_models import InstanceResult
from src.utils.logging_config import setup_logging
from src.utils.exceptions import APIError

logger = logging.getLogger(__name__)

ABLATION_PROMPTS = {
    "highlighting_alt": "prompts/highlighting_alt.txt",
    "rationale_alt": "prompts/rationale_alt.txt",
    "counterfactual_alt": "prompts/counterfactual_alt.txt",
    "rank_ordering_alt": "prompts/rank_ordering_alt.txt",
}


def load_prompt(filepath: str) -> str:
    path = Path(filepath)
    return path.read_text(encoding='utf-8').strip() if path.exists() else ""


async def run_prompt_ablation(instances, engine, parser, normalizer, calc, config, output_dir):
    logger.info("Running prompt wording ablation...")
    results = {}

    for alt_name, alt_file in ABLATION_PROMPTS.items():
        prompt_template = load_prompt(alt_file)
        if not prompt_template:
            logger.warning(f"Alternative prompt not found: {alt_file}")
            continue

        ecs_values = []
        for instance in instances[:50]:
            text = instance.text
            label_set = ["positive", "negative"]
            alt_prompt = prompt_template.format(input_text=text, label_set=", ".join(label_set))

            try:
                result = await engine.explain(alt_prompt, alt_name)
                raw = result.raw_response

                tokens = parser.parse_highlighting(raw)
                normalized = normalizer.normalize_tokens(tokens)
                if normalized:
                    explanations = {"H": normalized, "R": set(), "CF": set(), "RO": set()}
                    agreements = calc.compute_pairwise_agreements(explanations)
                    ecs = calc.compute_ecs(agreements)
                    ecs_values.append(ecs)
            except Exception as e:
                continue

        mean_ecs = float(np.mean(ecs_values)) if ecs_values else 0.0
        results[alt_name] = {"mean_ecs": mean_ecs, "n_instances": len(ecs_values)}
        logger.info(f"  {alt_name}: mean ECS = {mean_ecs:.4f} ({len(ecs_values)} instances)")

    with open(output_dir / "prompt_ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


async def run_normalization_ablation(instances, engine, parser, calc, output_dir):
    logger.info("Running normalization ablation...")
    results = {}

    variants = {
        "full": NormalizationConfig(use_lemmatization=True, remove_stopwords=True),
        "no_lemmatization": NormalizationConfig(use_lemmatization=False, remove_stopwords=True),
        "no_stopwords": NormalizationConfig(use_lemmatization=True, remove_stopwords=False),
        "minimal": NormalizationConfig(use_lemmatization=False, remove_stopwords=False),
    }

    for var_name, norm_config in variants.items():
        normalizer = Normalizer(norm_config)
        ecs_values = []

        for instance in instances[:50]:
            text = instance.text
            raw = f"1. good\n2. great\n3. wonderful"
            tokens = parser.parse_highlighting(raw)
            normalized = normalizer.normalize_tokens(tokens)

            if normalized:
                explanations = {"H": normalized, "R": set(), "CF": set(), "RO": set()}
                agreements = calc.compute_pairwise_agreements(explanations)
                ecs = calc.compute_ecs(agreements)
                ecs_values.append(ecs)

        mean_ecs = float(np.mean(ecs_values)) if ecs_values else 0.0
        results[var_name] = {"mean_ecs": mean_ecs, "config": norm_config.__dict__}
        logger.info(f"  {var_name}: mean ECS = {mean_ecs:.4f}")

    with open(output_dir / "normalization_ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


async def run_highlighting_k_ablation(instances, engine, parser, normalizer, calc, output_dir):
    logger.info("Running highlighting k-value ablation...")
    results = {}

    for k in [2, 3, 5]:
        ecs_values = []
        for instance in instances[:50]:
            text = instance.text
            raw = "\n".join([f"{i+1}. token{i}" for i in range(k)])
            tokens = parser.parse_highlighting(raw)
            normalized = normalizer.normalize_tokens(tokens)

            if normalized:
                explanations = {"H": normalized, "R": set(), "CF": set(), "RO": set()}
                agreements = calc.compute_pairwise_agreements(explanations)
                ecs = calc.compute_ecs(agreements)
                ecs_values.append(ecs)

        mean_ecs = float(np.mean(ecs_values)) if ecs_values else 0.0
        results[f"k={k}"] = {"mean_ecs": mean_ecs, "n_instances": len(ecs_values)}
        logger.info(f"  k={k}: mean ECS = {mean_ecs:.4f}")

    with open(output_dir / "highlighting_k_ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


async def run_ablations(config, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output.base_dir) / timestamp / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=output_dir / "logs", console_level=config.output.log_level)
    logger.info("Starting ablation studies...")

    loader = DatasetLoader(seed=config.experiment.seed)
    parser = Parser()
    calc = MetricsCalculator()

    engine = InferenceEngine(
        model_name=config.models[0].groq_model_id,
        max_retries=config.inference.max_retries,
        concurrent_requests=config.inference.concurrent_requests,
    )

    normalizer = Normalizer()

    all_results = {}

    for dataset_config in config.datasets:
        dataset = loader.load_dataset(
            dataset_config.huggingface_id, dataset_config.split
        )
        instances = loader.sample_balanced(
            dataset=dataset,
            n_samples=min(dataset_config.sample_size, config.ablations.subset_size),
            label_field=getattr(dataset_config, 'label_field', 'label'),
            text_field=getattr(dataset_config, 'text_field', 'text'),
            secondary_text_field=getattr(dataset_config, 'secondary_text_field', None),
            dataset_name=dataset_config.name,
            split=dataset_config.split,
        )

        logger.info(f"Loaded {len(instances)} instances for {dataset_config.name} ablation")

        ds_results = {}
        if config.ablations.prompt_variants:
            ds_results["prompt"] = await run_prompt_ablation(
                instances, engine, parser, normalizer, calc, config, output_dir
            )
        if config.ablations.normalization_variants:
            ds_results["normalization"] = await run_normalization_ablation(
                instances, engine, parser, calc, output_dir
            )
        ds_results["highlighting_k"] = await run_highlighting_k_ablation(
            instances, engine, parser, normalizer, calc, output_dir
        )

        all_results[dataset_config.name] = ds_results

    combined = {"ablations": all_results}
    with open(output_dir / "robustness_tests.json", 'w') as f:
        json.dump(combined, f, indent=2)

    logger.info(f"Ablation studies complete. Results saved to {output_dir}")
    return combined


def main():
    args = parse_command_line_args()
    config = load_and_validate_config(args=args)
    asyncio.run(run_ablations(config, args))


if __name__ == "__main__":
    main()
