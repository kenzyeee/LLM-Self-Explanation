import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.stats
from src.inference.inference_engine import InferenceEngine
from src.utils.data_models import ValidityTestResult, AggregateValidityResults, save_validity_results
from src.utils.config_loader import load_and_validate_config, parse_command_line_args
from src.utils.logging_config import setup_logging
from src.utils.exceptions import APIError

logger = logging.getLogger(__name__)


def load_instance_results(filepath: str) -> List[Dict[str, Any]]:
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def build_masked_text(text: str, tokens_to_mask: set, mask_token: str = "[MASK]") -> str:
    words = text.split()
    masked = []
    for word in words:
        clean_word = word.strip('.,!?;:()[]{}\'"')
        if clean_word in tokens_to_mask:
            prefix = word[:len(word)-len(clean_word)] if clean_word else ""
            suffix = word[len(clean_word):] if clean_word else ""
            masked.append(f"{prefix}{mask_token}{suffix}")
        else:
            masked.append(word)
    return " ".join(masked)


def random_token_selection(text: str, n: int) -> list:
    words = list(set(text.split()))
    if not words or n <= 0:
        return []
    if n >= len(words):
        return words
    return list(np.random.choice(words, n, replace=False))


async def process_validity_instance(
    instance_data: Dict[str, Any],
    engine: InferenceEngine,
    classification_prompt: str,
    label_set: list,
) -> ValidityTestResult:
    text = instance_data["text"]
    cc3_tokens = set(instance_data.get("cc3_tokens", []))
    cc4_tokens = set(instance_data.get("cc4_tokens", []))
    original_prediction = instance_data.get("predicted_label", "")
    instance_id = instance_data["instance_id"]

    result = ValidityTestResult(
        instance_id=instance_id,
        dataset=instance_data.get("dataset", ""),
        model=instance_data.get("model", ""),
    )

    # CC3 removal test
    if cc3_tokens:
        masked_text = build_masked_text(text, cc3_tokens)
        prompt = classification_prompt.format(input_text=masked_text, label_set=", ".join(label_set))
        try:
            resp = await engine._make_request(prompt, max_tokens=50)
            from src.parsing.parser import Parser
            parser = Parser()
            predicted, _ = parser.parse_classification(resp, label_set)
            result.cc3_tokens = cc3_tokens
            result.cc3_original_prediction = original_prediction
            result.cc3_masked_prediction = predicted
            result.cc3_flipped = (original_prediction != predicted)
        except APIError as e:
            logger.error(f"CC3 test failed for {instance_id}: {e}")

    # CC4 removal test
    if cc4_tokens:
        masked_text = build_masked_text(text, cc4_tokens)
        prompt = classification_prompt.format(input_text=masked_text, label_set=", ".join(label_set))
        try:
            resp = await engine._make_request(prompt, max_tokens=50)
            parser = Parser()
            predicted, _ = parser.parse_classification(resp, label_set)
            result.cc4_tokens = cc4_tokens
            result.cc4_original_prediction = original_prediction
            result.cc4_masked_prediction = predicted
            result.cc4_flipped = (original_prediction != predicted)
        except APIError as e:
            logger.error(f"CC4 test failed for {instance_id}: {e}")

    # Random baseline
    n_random = len(cc3_tokens) if cc3_tokens else 1
    random_toks = random_token_selection(text, n_random)
    masked_text = build_masked_text(text, set(random_toks))
    prompt = classification_prompt.format(input_text=masked_text, label_set=", ".join(label_set))
    try:
        resp = await engine._make_request(prompt, max_tokens=50)
        parser = Parser()
        predicted, _ = parser.parse_classification(resp, label_set)
        result.random_tokens = set(random_toks)
        result.random_original_prediction = original_prediction
        result.random_masked_prediction = predicted
        result.random_flipped = (original_prediction != predicted)
    except APIError as e:
        logger.error(f"Random baseline test failed for {instance_id}: {e}")

    return result


def compute_aggregate_validity(results: List[ValidityTestResult]) -> AggregateValidityResults:
    if not results:
        return AggregateValidityResults(
            dataset="", model="", n_instances=0,
            cc3_flip_rate=0, cc4_flip_rate=0, random_flip_rate=0,
            t_statistic=0, p_value=1.0, effect_size=0,
            cc3_flip_ci_lower=0, cc3_flip_ci_upper=0,
            random_flip_ci_lower=0, random_flip_ci_upper=0,
        )

    cc3_flips = [1 if r.cc3_flipped else 0 for r in results if r.cc3_tokens]
    random_flips = [1 if r.random_flipped else 0 for r in results if r.random_tokens]
    cc4_flips = [1 if r.cc4_flipped else 0 for r in results if r.cc4_tokens]

    cc3_rate = np.mean(cc3_flips) if cc3_flips else 0
    random_rate = np.mean(random_flips) if random_flips else 0
    cc4_rate = np.mean(cc4_flips) if cc4_flips else 0

    if len(cc3_flips) > 1 and len(random_flips) > 1:
        min_len = min(len(cc3_flips), len(random_flips))
        t_stat, p_val = scipy.stats.ttest_rel(cc3_flips[:min_len], random_flips[:min_len])
        # Cohen's d
        diff = np.array(cc3_flips[:min_len]) - np.array(random_flips[:min_len])
        effect_size = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0
    else:
        t_stat, p_val, effect_size = 0.0, 1.0, 0.0

    # Bootstrap CIs
    n_bootstrap = 1000
    cc3_rates = []
    random_rates = []
    if cc3_flips and random_flips:
        n = len(cc3_flips)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            boot_cc3 = [cc3_flips[i] for i in idx]
            boot_random = [random_flips[i % len(random_flips)] for i in idx]
            cc3_rates.append(np.mean(boot_cc3))
            random_rates.append(np.mean(boot_random))
    else:
        cc3_rates = [0.0]
        random_rates = [0.0]

    return AggregateValidityResults(
        dataset=results[0].dataset,
        model=results[0].model,
        n_instances=len(results),
        cc3_flip_rate=float(cc3_rate),
        cc4_flip_rate=float(cc4_rate),
        random_flip_rate=float(random_rate),
        t_statistic=float(t_stat),
        p_value=float(p_val),
        effect_size=float(effect_size),
        cc3_flip_ci_lower=float(np.percentile(cc3_rates, 2.5)),
        cc3_flip_ci_upper=float(np.percentile(cc3_rates, 97.5)),
        random_flip_ci_lower=float(np.percentile(random_rates, 2.5)),
        random_flip_ci_upper=float(np.percentile(random_rates, 97.5)),
    )


async def run_validity_tests(config, args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output.base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=output_dir / "logs", console_level=config.output.log_level)
    logger.info("Starting validity tests...")

    # Find latest results
    results_base = args.results_dir if hasattr(args, 'results_dir') and args.results_dir else Path(config.output.base_dir)
    results_file = Path(results_base) / "instance_results.jsonl"
    if not results_file.exists():
        # Try to find latest
        candidates = sorted(Path(results_base).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in candidates:
            if (cand / "instance_results.jsonl").exists():
                results_file = cand / "instance_results.jsonl"
                break

    if not results_file.exists():
        logger.error(f"No instance results found. Run experiment first.")
        return

    instances = load_instance_results(str(results_file))
    logger.info(f"Loaded {len(instances)} instance results from {results_file}")

    engine = InferenceEngine(
        model_name=config.models[0].groq_model_id,
        max_retries=config.inference.max_retries,
        concurrent_requests=config.inference.concurrent_requests,
    )

    classification_prompt = Path("prompts/classification.txt").read_text(encoding='utf-8')

    all_validity_results = []
    for instance_data in instances:
        dataset_name = instance_data.get("dataset", "")
        ds_specific = Path(f"prompts/classification_{dataset_name}.txt")
        if ds_specific.exists():
            classification_prompt = ds_specific.read_text(encoding='utf-8')
        dataset_config = config.get_dataset_by_name(dataset_name)
        label_set = dataset_config.labels if dataset_config else ["positive", "negative"]

        result = await process_validity_instance(instance_data, engine, classification_prompt, label_set)
        all_validity_results.append(result)
        logger.info(f"Processed validity test for {result.instance_id}")

    save_validity_results(all_validity_results, str(output_dir / "validity_tests.jsonl"))

    # Aggregate
    agg = compute_aggregate_validity(all_validity_results)
    with open(output_dir / "aggregate_validity.json", 'w') as f:
        json.dump(agg.to_dict(), f, indent=2)

    logger.info(f"Validity tests complete. Results saved to {output_dir}")
    logger.info(f"CC3 flip rate: {agg.cc3_flip_rate:.3f}")
    logger.info(f"Random flip rate: {agg.random_flip_rate:.3f}")
    logger.info(f"Paired t-test p-value: {agg.p_value:.4f}")

    return all_validity_results, agg


def main():
    parser = argparse.ArgumentParser(description="Run validity tests")
    parser.add_argument("--results-dir", type=str, help="Directory containing instance_results.jsonl")
    args, _ = parser.parse_known_args()
    config_args = parse_command_line_args()
    config = load_and_validate_config(args=config_args)
    asyncio.run(run_validity_tests(config, args))


if __name__ == "__main__":
    main()
