import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from src.load.dataset_loader import DatasetLoader
from src.inference.inference_engine import InferenceEngine
from src.parsing.parser import Parser
from src.normalization.normalizer import Normalizer
from src.metrics.metrics_calculator import MetricsCalculator
from src.plots.visualization_generator import VisualizationGenerator
from src.utils.config_loader import load_and_validate_config, parse_command_line_args
from src.utils.logging_config import setup_logging
from src.utils.exceptions import APIError, ParsingError

logger = logging.getLogger(__name__)

STRATEGY_IDS = ["H", "R", "CF", "RO"]

ALT_PROMPTS = {
    "H": "prompts/highlighting_alt.txt",
    "R": "prompts/rationale_alt.txt",
    "CF": "prompts/counterfactual_alt.txt",
    "RO": "prompts/rank_ordering_alt.txt",
}

K_PROMPTS = {2: "prompts/highlighting_k2.txt", 5: "prompts/highlighting_k5.txt"}

NORMALIZATION_VARIANTS = {
    "full": {"use_lemmatization": True, "remove_stopwords": True},
    "no_lemmatization": {"use_lemmatization": False, "remove_stopwords": True},
    "no_stopwords": {"use_lemmatization": True, "remove_stopwords": False},
    "minimal": {"use_lemmatization": False, "remove_stopwords": False},
}

BASELINE_K = 3


def load_prompt(filepath: str) -> str:
    path = Path(filepath)
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def format_class_prompt(template: str, input_text: str, label_set: List[str]) -> str:
    return template.format(input_text=input_text, label_set=", ".join(label_set))


def format_explain_prompt(template: str, predicted_label: str) -> str:
    return template.format(predicted_label=predicted_label)


def format_alt_prompt(template: str, predicted_label: str, input_text: str, label_set: List[str]) -> str:
    return template.format(
        predicted_label=predicted_label,
        input_text=input_text,
        label_set=", ".join(label_set),
    )


def load_strategy_explain_prompts(config) -> Dict[str, str]:
    prompts = {}
    for s in config.explanation_strategies:
        explain_file = s.prompt_file.replace(".txt", "_explain.txt")
        prompts[s.id] = load_prompt(explain_file)
    return prompts


def load_classification_prompt(config, dataset_name: str = None) -> str:
    class_path = "prompts/classification.txt"
    if dataset_name:
        ds_path = f"prompts/classification_{dataset_name}.txt"
        if Path(ds_path).exists():
            class_path = ds_path
    return load_prompt(class_path)


def parse_raw_tokens(strategy_id: str, raw: str, text: str,
                     predicted_label: str, label_set: List[str],
                     parser: Parser, normalizer: Normalizer) -> List[str]:
    """Parse raw response and return raw token strings (pre-normalization)."""
    if not raw:
        return []
    try:
        if strategy_id == "H":
            return parser.parse_highlighting(raw, text, normalizer, skip_validation=True)
        elif strategy_id == "R":
            _, evidence = parser.parse_rationale(raw, text, normalizer, skip_validation=True)
            return evidence
        elif strategy_id == "CF":
            cf_text, _ = parser.parse_counterfactual(raw, text, predicted_label, label_set,
                                                     normalizer, skip_validation=True)
            orig_words = set(text.lower().split())
            cf_words = set(cf_text.lower().split())
            return list((orig_words - cf_words) | (cf_words - orig_words))
        elif strategy_id == "RO":
            ranked = parser.parse_rank_ordering(raw, text, normalizer, skip_validation=True)
            return [t for t, _ in ranked]
    except (ParsingError, json.JSONDecodeError) as e:
        logger.debug(f"Parse failed for {strategy_id}: {e}")
        return []


def normalize_token_list(raw_tokens: List[str], normalizer: Normalizer) -> Set[str]:
    return normalizer.normalize_tokens(raw_tokens)


def compute_ecs_from_token_sets(sets: Dict[str, Set[str]], calc: MetricsCalculator) -> float:
    explanations = {s: sets.get(s, set()) for s in STRATEGY_IDS}
    agreements = calc.compute_pairwise_agreements(explanations)
    return calc.compute_ecs(agreements)


async def run_classify(engine, class_prompt, parser, label_set):
    result = await engine.classify(class_prompt)
    predicted_label = parser.parse_classification(result.raw_response, label_set)
    return predicted_label, result.raw_response


async def run_explain(engine, class_prompt, class_raw, explain_prompt, max_tokens=512) -> str:
    messages = [
        {"role": "user", "content": class_prompt},
        {"role": "assistant", "content": class_raw},
        {"role": "user", "content": explain_prompt},
    ]
    return await engine.chat(messages, max_tokens=max_tokens)


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
        model_name=config.models[0].model_id,
        max_retries=config.inference.max_retries,
        concurrent_requests=config.inference.concurrent_requests,
    )

    all_results = {}
    all_plot_data = []

    for dataset_config in config.datasets:
        dataset = loader.load_dataset(dataset_config.huggingface_id, dataset_config.split)
        instances = loader.sample_balanced(
            dataset=dataset,
            n_samples=min(dataset_config.sample_size, config.ablations.subset_size),
            label_field=getattr(dataset_config, "label_field", "label"),
            text_field=getattr(dataset_config, "text_field", "text"),
            secondary_text_field=getattr(dataset_config, "secondary_text_field", None),
            dataset_name=dataset_config.name,
            split=dataset_config.split,
        )
        logger.info(f"Loaded {len(instances)} instances for {dataset_config.name} ablation")

        class_prompt_template = load_classification_prompt(config, dataset_config.name)
        strategy_prompts = load_strategy_explain_prompts(config)
        label_set = dataset_config.labels

        # --- Step 1: Compute baseline for all instances ---
        # baseline_data[i] = {predicted_label, class_raw, class_prompt,
        #                     raw_tokens: {s: [str]}, baseline_ecs}
        baseline_data = []
        for instance in instances:
            text = instance.text
            class_prompt = format_class_prompt(class_prompt_template, text, label_set)

            try:
                predicted_label, class_raw = await run_classify(engine, class_prompt, parser, label_set)
            except (APIError, ParsingError) as e:
                logger.warning(f"Classification failed for {instance.instance_id}: {e}")
                continue

            normalizer = Normalizer()
            raw_tokens = {}
            for s in STRATEGY_IDS:
                try:
                    raw_response = await run_explain(engine, class_prompt, class_raw,
                                                     strategy_prompts[s])
                    raw_tokens[s] = parse_raw_tokens(s, raw_response, text, predicted_label,
                                                     label_set, parser, normalizer)
                except (APIError, ParsingError, json.JSONDecodeError) as e:
                    logger.debug(f"  {s} failed for {instance.instance_id}: {e}")
                    raw_tokens[s] = []

            token_sets = {s: normalize_token_list(raw_tokens[s], normalizer) for s in STRATEGY_IDS}
            baseline_ecs = compute_ecs_from_token_sets(token_sets, calc)

            baseline_data.append({
                "instance": instance,
                "predicted_label": predicted_label,
                "class_raw": class_raw,
                "class_prompt": class_prompt,
                "raw_tokens": raw_tokens,
                "token_sets": token_sets,
                "baseline_ecs": baseline_ecs,
            })

        # --- Step 2: Prompt Ablation ---
        prompt_results = {}
        if config.ablations.prompt_variants and baseline_data:
            logger.info("Running prompt wording ablation...")
            for s in STRATEGY_IDS:
                alt_template = load_prompt(ALT_PROMPTS[s])
                if not alt_template:
                    continue

                deltas = []
                for bd in baseline_data:
                    inst = bd["instance"]
                    alt_prompt = format_alt_prompt(alt_template, bd["predicted_label"],
                                                   inst.text, label_set)

                    try:
                        raw_alt = await run_explain(engine, bd["class_prompt"],
                                                    bd["class_raw"], alt_prompt)
                        alt_raw_tokens = parse_raw_tokens(
                            s, raw_alt, inst.text, bd["predicted_label"],
                            label_set, parser, Normalizer())
                        alt_normalizer = Normalizer()
                        alt_set = normalize_token_list(alt_raw_tokens, alt_normalizer)
                    except (APIError, ParsingError, json.JSONDecodeError) as e:
                        logger.debug(f"  alt {s} failed for {inst.instance_id}: {e}")
                        alt_set = set()

                    variant_sets = dict(bd["token_sets"])
                    variant_sets[s] = alt_set
                    variant_ecs = compute_ecs_from_token_sets(variant_sets, calc)
                    deltas.append(variant_ecs - bd["baseline_ecs"])

                mean_delta = float(np.mean(deltas)) if deltas else 0.0
                prompt_results[f"{s}_alt"] = {
                    "mean_delta": mean_delta,
                    "n_instances": len(deltas),
                    "deltas": deltas,
                }
                logger.info(f"  {s}_alt: mean delta = {mean_delta:.4f} ({len(deltas)} instances)")
                for d in deltas:
                    all_plot_data.append({
                        "Variation": f"prompt_{s}_alt",
                        "ECS_delta": d,
                        "Dataset": dataset_config.name,
                        "Ablation": "prompt",
                    })

            with open(output_dir / f"prompt_ablation_{dataset_config.name}.json", "w") as f:
                json.dump(prompt_results, f, indent=2)
            all_results[f"{dataset_config.name}_prompt"] = prompt_results

        # --- Step 3: Normalization Ablation ---
        if config.ablations.normalization_variants and baseline_data:
            logger.info("Running normalization ablation...")
            norm_results = {}
            for var_name, norm_kwargs in NORMALIZATION_VARIANTS.items():
                var_normalizer = Normalizer(**norm_kwargs)
                deltas = []
                for bd in baseline_data:
                    variant_sets = {}
                    for s in STRATEGY_IDS:
                        variant_sets[s] = normalize_token_list(bd["raw_tokens"][s], var_normalizer)
                    variant_ecs = compute_ecs_from_token_sets(variant_sets, calc)
                    deltas.append(variant_ecs - bd["baseline_ecs"])

                mean_delta = float(np.mean(deltas)) if deltas else 0.0
                norm_results[var_name] = {"mean_delta": mean_delta, "n_instances": len(deltas), "deltas": deltas}
                logger.info(f"  {var_name}: mean delta = {mean_delta:.4f}")
                for d in deltas:
                    all_plot_data.append({
                        "Variation": f"norm_{var_name}",
                        "ECS_delta": d,
                        "Dataset": dataset_config.name,
                        "Ablation": "normalization",
                    })

            with open(output_dir / f"normalization_ablation_{dataset_config.name}.json", "w") as f:
                json.dump(norm_results, f, indent=2)
            all_results[f"{dataset_config.name}_normalization"] = norm_results

        # --- Step 4: K-value Ablation ---
        if baseline_data:
            logger.info("Running highlighting k-value ablation...")
            k_results = {}
            for k in [k for k in K_PROMPTS if k != BASELINE_K]:
                k_template = load_prompt(K_PROMPTS[k])
                if not k_template:
                    logger.warning(f"K prompt not found for k={k}")
                    continue

                deltas = []
                for bd in baseline_data:
                    inst = bd["instance"]
                    k_prompt = format_explain_prompt(k_template, bd["predicted_label"])

                    try:
                        raw_k = await run_explain(engine, bd["class_prompt"],
                                                  bd["class_raw"], k_prompt)
                        k_raw_tokens = parse_raw_tokens(
                            "H", raw_k, inst.text, bd["predicted_label"],
                            label_set, parser, Normalizer())
                        k_normalizer = Normalizer()
                        k_set = normalize_token_list(k_raw_tokens, k_normalizer)
                    except (APIError, ParsingError, json.JSONDecodeError) as e:
                        logger.debug(f"  k={k} failed for {inst.instance_id}: {e}")
                        k_set = set()

                    variant_sets = dict(bd["token_sets"])
                    variant_sets["H"] = k_set
                    variant_ecs = compute_ecs_from_token_sets(variant_sets, calc)
                    deltas.append(variant_ecs - bd["baseline_ecs"])

                mean_delta = float(np.mean(deltas)) if deltas else 0.0
                k_results[f"k={k}"] = {"mean_delta": mean_delta, "n_instances": len(deltas), "deltas": deltas}
                logger.info(f"  k={k}: mean delta = {mean_delta:.4f} ({len(deltas)} instances)")
                for d in deltas:
                    all_plot_data.append({
                        "Variation": f"k_{k}",
                        "ECS_delta": d,
                        "Dataset": dataset_config.name,
                        "Ablation": "highlighting_k",
                    })

            with open(output_dir / f"highlighting_k_ablation_{dataset_config.name}.json", "w") as f:
                json.dump(k_results, f, indent=2)
            all_results[f"{dataset_config.name}_highlighting_k"] = k_results

    # Generate robustness plot
    if all_plot_data:
        plot_df = pd.DataFrame(all_plot_data)
        viz = VisualizationGenerator(output_dir, dpi=config.output.figure_dpi)
        viz.plot_robustness_analysis(plot_df)
        logger.info(f"Robustness analysis plot saved to {output_dir}")

    # Save combined results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
        logger.info(f"Ablation studies complete. Results saved to {output_dir}")

    return all_results


def main():
    load_dotenv()
    args = parse_command_line_args()
    config = load_and_validate_config(args=args)
    asyncio.run(run_ablations(config, args))


if __name__ == "__main__":
    main()
