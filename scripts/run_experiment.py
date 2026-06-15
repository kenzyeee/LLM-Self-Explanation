import argparse
import asyncio
import hashlib
import json
import logging
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.load.dataset_loader import DatasetLoader, Instance
from src.inference.inference_engine import InferenceEngine
from src.parsing.parser import Parser
from src.normalization.normalizer import Normalizer
from src.metrics.metrics_calculator import MetricsCalculator
from src.utils.config_loader import load_and_validate_config, parse_command_line_args, save_config_to_file
from src.utils.data_models import (
    InstanceResult, AggregateMetrics, save_instance_results, save_aggregate_metrics,
    generate_md_report, save_metrics_csv, save_metadata_table, save_environment_snapshot,
)
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.execution_summary import ExecutionSummary
from src.utils.exceptions import APIError, ParsingError
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

STRATEGY_IDS = ["H", "R", "CF", "RO"]

REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i'm unable", "i am unable", "not able to",
    "i apologize", "i'm sorry", "i am sorry", "cannot fulfill",
    "cannot complete", "not appropriate", "i cannot provide",
    "i can't provide", "against policy", "not permitted",
]

_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            import tiktoken
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKENIZER = None
    return _TOKENIZER


def count_tokens(text: str) -> int:
    enc = _get_tokenizer()
    if enc:
        return len(enc.encode(text))
    return len(text.split())


def compute_prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:16]


def is_model_refusal(response: str) -> bool:
    lower = response.lower().strip()
    for pattern in REFUSAL_PATTERNS:
        if pattern in lower:
            return True
    return False


def load_prompt(filepath: str) -> str:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding='utf-8').strip()


def format_prompt(template: str, input_text: str, label_set: List[str]) -> str:
    return template.format(input_text=input_text, label_set=", ".join(label_set))


def format_explain_prompt(template: str, predicted_label: str, **kwargs) -> str:
    return template.format(predicted_label=predicted_label, **kwargs)


def create_prompt_map(config, dataset_name: str = None) -> Dict[str, str]:
    prompts = {}
    for strategy in config.explanation_strategies:
        prompt_text = load_prompt(strategy.prompt_file)
        prompts[strategy.id] = prompt_text
        explain_file = strategy.prompt_file.replace(".txt", "_explain.txt")
        ds_specific_explain = None
        if dataset_name:
            ds_specific_explain = explain_file.replace("_explain.txt", f"_explain_{dataset_name}.txt")
            if not Path(ds_specific_explain).exists():
                ds_specific_explain = None
        prompts[f"{strategy.id}_explain"] = load_prompt(ds_specific_explain or explain_file)
        multiclass_variant = explain_file.replace("_explain.txt", "_explain_multiclass.txt")
        if Path(multiclass_variant).exists():
            prompts[f"{strategy.id}_explain_multiclass"] = load_prompt(multiclass_variant)

    class_prompt_path = "prompts/classification.txt"
    if dataset_name:
        ds_specific = f"prompts/classification_{dataset_name}.txt"
        if Path(ds_specific).exists():
            class_prompt_path = ds_specific
    prompts["classification"] = load_prompt(class_prompt_path)

    calib_path = "prompts/confidence_calibration.txt"
    if Path(calib_path).exists():
        prompts["confidence_calibration"] = load_prompt(calib_path)

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

    # Classification (label only — no confidence in this context)
    class_prompt = format_prompt(prompts["classification"], text, label_set)
    class_result = await engine.classify(class_prompt)

    predicted_label = ""
    classification_valid = False
    try:
        predicted_label, _ = parser.parse_classification(class_result.raw_response, label_set, require_confidence=False)
        classification_valid = True
    except ParsingError as e:
        logger.warning(f"Classification parsing error for {instance_id}: {e}")

    # Standalone confidence call — separate context, does not branch into explanations
    confidence = 0.0
    if classification_valid and "confidence_calibration" in prompts:
        conf_prompt = prompts["confidence_calibration"].format(
            predicted_label=predicted_label, input_text=text
        )
        conf_messages = [{"role": "user", "content": conf_prompt}]
        try:
            conf_result = await engine.chat(conf_messages, max_tokens=128)
            confidence = parser.parse_confidence(conf_result)
        except Exception as e:
            logger.debug(f"Standalone confidence call failed for {instance_id}: {e}")

    # Blank-label guard
    if not predicted_label or predicted_label.strip() == "":
        logger.error(f"Empty predicted_label for {instance_id}")
    if "{" in predicted_label or "}" in predicted_label:
        logger.error(f"Unrendered placeholder in label for {instance_id}: {predicted_label}")

    correct = predicted_label == instance.label
    model_refused = is_model_refusal(class_result.raw_response)
    prompt_hash = compute_prompt_hash(class_prompt)
    prompt_tokens = count_tokens(class_prompt)
    response_tokens = count_tokens(class_result.raw_response)

    if not correct:
        logger.warning(f"{instance_id}: wrong prediction ({instance.label} -> {predicted_label}), filtering from analysis")
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
            raw_highlighting="", raw_rationale="",
            raw_counterfactual="", raw_rank_ordering="",
            classification_prompt=class_prompt,
            classification_raw_response=class_result.raw_response,
            model_refused=model_refused,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            ecs=None, ecs_primary=None, ecs_primary_pairs=0, n_valid_strategies=0,
        )
        return result

    prompt_tokens += count_tokens(format_explain_prompt(prompts["H_explain"], predicted_label)) * 4

    # Randomize strategy order
    strategy_order = STRATEGY_IDS.copy()
    random.shuffle(strategy_order)

    raw_responses = {}
    explain_prompts = {}
    parsed_tokens = {}
    parsed_flags = {}
    valid_flags = {}
    rationale_text = ""

    for strat_id in STRATEGY_IDS:
        raw_responses[strat_id] = ""
        explain_prompts[strat_id] = ""
        parsed_tokens[strat_id] = set()
        parsed_flags[strat_id] = False
        valid_flags[strat_id] = False

    for strat_id in strategy_order:
        if strat_id == "CF":
            cf_prompt_key = f"{strat_id}_explain"
            if len(label_set) > 2 and f"{cf_prompt_key}_multiclass" in prompts:
                cf_prompt_key = f"{cf_prompt_key}_multiclass"
            other_labels = ", ".join(l for l in label_set if l != predicted_label)
            explain_prompt = format_explain_prompt(prompts[cf_prompt_key], predicted_label,
                                                   other_labels=other_labels)
        else:
            explain_prompt = format_explain_prompt(prompts[f"{strat_id}_explain"], predicted_label)
        explain_prompts[strat_id] = explain_prompt
        messages = [
            {"role": "user", "content": class_prompt},
            {"role": "assistant", "content": class_result.raw_response},
            {"role": "user", "content": explain_prompt},
        ]
        try:
            raw = await engine.chat(messages, max_tokens=512)
            raw_responses[strat_id] = raw
            response_tokens += count_tokens(raw)
            if is_model_refusal(raw):
                model_refused = True
        except APIError as e:
            logger.error(f"API error for {instance_id} strategy {strat_id}: {e}")
            continue

    for strat_id in STRATEGY_IDS:
        raw = raw_responses[strat_id]
        if not raw:
            continue
        try:
            if strat_id == "H":
                tokens = parser.parse_highlighting(raw, text, normalizer)
                normalized = normalizer.normalize_tokens(tokens)
                if not normalized:
                    raise ParsingError("H evidence set is empty after normalization")
                parsed_tokens[strat_id] = normalized
                parsed_tokens["H_ordered"] = tokens
                parsed_flags[strat_id] = True
                valid_flags[strat_id] = True
            elif strat_id == "R":
                r_text, evidence = parser.parse_rationale(raw, text, normalizer)
                # Track verbatim compliance violations
                n_violations = normalizer.check_evidence_compliance(evidence, text)
                if n_violations:
                    logger.warning(f"R evidence for {instance_id}: {n_violations} token(s) not verbatim in input")
                rationale_text = r_text
                normalized = normalizer.normalize_tokens(evidence)
                if not normalized:
                    raise ParsingError("R evidence set is empty after normalization")
                parsed_tokens[strat_id] = normalized
                parsed_flags[strat_id] = True
                valid_flags[strat_id] = True
            elif strat_id == "CF":
                cf_max_ratio = getattr(dataset_config, 'cf_max_edit_ratio', 0.3)
                cf_text, new_pred = parser.parse_counterfactual(
                    raw, text, predicted_label, label_set, normalizer, max_edit_ratio=cf_max_ratio
                )
                diff = normalizer.extract_counterfactual_diff(text, cf_text)
                if not diff:
                    logger.warning(f"CF evidence empty for {instance_id} — reclassifying as invalid")
                    raise ParsingError("CF evidence set is empty after diff extraction")
                # CF flip verification via re-query (observational, not a gate)
                cf_flip_verified = False
                cf_actual_label = ""
                try:
                    cf_class_prompt = format_prompt(prompts["classification"], cf_text, label_set)
                    cf_class_result = await engine.classify(cf_class_prompt)
                    cf_actual_label, _ = parser.parse_classification(cf_class_result.raw_response, label_set)
                    cf_flip_verified = (cf_actual_label != predicted_label)
                    if not cf_flip_verified:
                        logger.warning(f"CF flip not verified for {instance_id}: re-classification gave {cf_actual_label}, same as original")
                    elif cf_actual_label != new_pred:
                        logger.warning(f"CF flip verified but label mismatch for {instance_id}: model said {new_pred}, actual {cf_actual_label}")
                except Exception as e:
                    logger.warning(f"CF flip verification failed for {instance_id}: {e}")
                parsed_tokens[strat_id] = diff
                parsed_flags[strat_id] = True
                valid_flags[strat_id] = True
            elif strat_id == "RO":
                ranked = parser.parse_rank_ordering(raw, text, normalizer)
                ro_tokens = [t for t, r in ranked]
                # One-shot self-correction for hallucinated tokens
                json_obj = parser._extract_json(raw)
                raw_ranking = json_obj.get("ranking", []) if json_obj else []
                if raw_ranking and len(ro_tokens) < len(raw_ranking):
                    discarded = len(raw_ranking) - len(ro_tokens)
                    logger.warning(f"RO for {instance_id}: {discarded} token(s) hallucinated, attempting self-correction")
                    invalid_tokens = [t for t in raw_ranking if isinstance(t, str) and not normalizer.is_anchored(t, text)]
                    correction_prompt = (
                        f"Your previous ranking contained token(s) not found in the original text: {invalid_tokens}\n\n"
                        f"Return a corrected ranking using ONLY words that appear verbatim in this text:\n"
                        f"\"{text}\"\n\n"
                        f"Return only valid JSON: {{\"ranking\": [...]}}"
                    )
                    try:
                        correction_messages = messages + [
                            {"role": "assistant", "content": raw},
                            {"role": "user", "content": correction_prompt},
                        ]
                        correction_raw = await engine.chat(correction_messages, max_tokens=512)
                        ranked = parser.parse_rank_ordering(correction_raw, text, normalizer)
                        ro_tokens = [t for t, r in ranked]
                        if len(ro_tokens) >= 3:
                            logger.info(f"RO self-correction for {instance_id}: recovered {len(ro_tokens)} valid tokens")
                        else:
                            logger.warning(f"RO self-correction for {instance_id}: still insufficient valid tokens")
                    except Exception as e:
                        logger.warning(f"RO self-correction for {instance_id} failed: {e}")
                normalized_set = normalizer.normalize_tokens(ro_tokens)
                if not normalized_set:
                    raise ParsingError("RO evidence set is empty after normalization")
                normalized_ranked = []
                for token, rank in ranked:
                    words = token.split()
                    for word in words:
                        n = normalizer.normalize(word)
                        if n:
                            normalized_ranked.append((n, rank))
                parsed_tokens["RO_set"] = normalized_set
                parsed_tokens["RO_ranked"] = normalized_ranked
                parsed_flags[strat_id] = True
                valid_flags[strat_id] = True

        except (ParsingError, json.JSONDecodeError) as e:
            logger.warning(f"Parsing error for {instance_id} strategy {strat_id}: {e}")

    cf_flip_verified = parsed_flags.get("CF", False) and valid_flags.get("CF", False) and locals().get("cf_flip_verified", False)
    cf_actual_label = locals().get("cf_actual_label", "")
    ro_set = parsed_tokens.get("RO_set", set())
    ro_ranked = parsed_tokens.get("RO_ranked", [])

    explanations = {
        "H": parsed_tokens.get("H", set()),
        "R": parsed_tokens.get("R", set()),
        "CF": parsed_tokens.get("CF", set()),
        "RO": ro_set,
    }

    agreements = calc.compute_pairwise_agreements(explanations)

    # Compute Kendall tau between H (ordered) and RO (ranked)
    h_ordered = parsed_tokens.get("H_ordered", [])
    kendall_val = None
    normalized_kendall_val = None
    if h_ordered and ro_ranked:
        h_ranks = calc.assign_implicit_ranks(h_ordered)
        kendall_val = calc.compute_kendalls_tau(h_ranks, ro_ranked)
        normalized_kendall_val = calc.compute_normalized_kendalls_tau(kendall_val)

    # Count valid strategies
    n_valid = sum(1 for s in STRATEGY_IDS if valid_flags.get(s, False))

    ecs_value = None
    ecs_primary = None
    ecs_primary_pairs = 0
    if n_valid >= 3:
        ecs_value = calc.compute_ecs(agreements)
        ecs_primary, ecs_primary_pairs = calc.compute_ecs_primary(agreements)
    else:
        logger.warning(f"Only {n_valid} valid strategies for {instance_id} — ECS not computed")

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
        classification_prompt=class_prompt,
        classification_raw_response=class_result.raw_response,
        highlighting_explain_prompt=explain_prompts.get("H", ""),
        rationale_explain_prompt=explain_prompts.get("R", ""),
        counterfactual_explain_prompt=explain_prompts.get("CF", ""),
        rank_ordering_explain_prompt=explain_prompts.get("RO", ""),
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
        highlighting_valid=valid_flags.get("H", False),
        rationale_valid=valid_flags.get("R", False),
        counterfactual_valid=valid_flags.get("CF", False),
        rank_ordering_valid=valid_flags.get("RO", False),
        rationale_text=rationale_text,
        model_refused=model_refused,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        prompt_hash=prompt_hash,
        jaccard_H_R=agreements.get(("H", "R")),
        jaccard_H_CF=agreements.get(("H", "CF")),
        jaccard_H_RO=agreements.get(("H", "RO")),
        jaccard_R_CF=agreements.get(("R", "CF")),
        jaccard_R_RO=agreements.get(("R", "RO")),
        jaccard_CF_RO=agreements.get(("CF", "RO")),
        kendall_H_RO=kendall_val,
        normalized_kendall_H_RO=normalized_kendall_val,
        ecs=ecs_value,
        ecs_primary=ecs_primary,
        ecs_primary_pairs=ecs_primary_pairs,
        n_valid_strategies=n_valid,
        cf_flip_verified=cf_flip_verified,
        cf_actual_label=cf_actual_label,
    )

    # Compute consensus cores over valid strategies only
    valid_strategies = [s for s in STRATEGY_IDS if valid_flags.get(s, False)]
    n_valid = len(valid_strategies)
    if n_valid >= 3:
        reduced_explanations = {s: explanations[s] for s in valid_strategies}
        result.cc3_tokens = calc.compute_consensus_core(reduced_explanations, 3)
        result.cc3_size = len(result.cc3_tokens)
        if n_valid == 4:
            result.cc4_tokens = calc.compute_consensus_core(reduced_explanations, 4)
            result.cc4_size = len(result.cc4_tokens)
        else:
            # CC4 requires all N valid strategies
            result.cc4_tokens = calc.compute_consensus_core(reduced_explanations, n_valid)
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
            mean_ecs=0, mean_ecs_primary=0, std_ecs=0, median_ecs=0,
            ecs_ci_lower=0, ecs_ci_upper=0,
            mean_jaccard_H_R=0, mean_jaccard_H_CF=0, mean_jaccard_H_RO=0,
            mean_jaccard_R_CF=0, mean_jaccard_R_RO=0, mean_jaccard_CF_RO=0,
            mean_kendall_H_RO=0,
            mean_normalized_kendall_H_RO=0,
            mean_cc3_size=0, mean_cc4_size=0,
            pct_instances_with_cc3=0, pct_instances_with_cc4=0,
            spearman_rho=0, spearman_p_value=1.0,
            correlation_ci_lower=0, correlation_ci_upper=0,
            highlighting_success_rate=0, rationale_success_rate=0,
            counterfactual_success_rate=0, rank_ordering_success_rate=0,
        )

    corr = compute_confidence_ecs_correlation(confidences, ecs_values, n_bootstrap=100)

    # Bootstrap CI for mean ECS
    if len(ecs_values) > 1:
        boot_means = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            sample = rng.choice(ecs_values, size=len(ecs_values), replace=True)
            boot_means.append(float(np.mean(sample)))
        ecs_ci_lower = float(np.percentile(boot_means, 2.5))
        ecs_ci_upper = float(np.percentile(boot_means, 97.5))
    else:
        ecs_ci_lower = 0.0
        ecs_ci_upper = 0.0

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
        mean_ecs_primary=safe_mean([r.ecs_primary for r in results if r.ecs_primary is not None]),
        std_ecs=float(np.std(ecs_values)) if len(ecs_values) > 1 else 0.0,
        median_ecs=float(np.median(ecs_values)) if ecs_values else 0.0,
        ecs_ci_lower=ecs_ci_lower,
        ecs_ci_upper=ecs_ci_upper,
        mean_jaccard_H_R=jaccard_mean('jaccard_H_R'),
        mean_jaccard_H_CF=jaccard_mean('jaccard_H_CF'),
        mean_jaccard_H_RO=jaccard_mean('jaccard_H_RO'),
        mean_jaccard_R_CF=jaccard_mean('jaccard_R_CF'),
        mean_jaccard_R_RO=jaccard_mean('jaccard_R_RO'),
        mean_jaccard_CF_RO=jaccard_mean('jaccard_CF_RO'),
        mean_kendall_H_RO=jaccard_mean('kendall_H_RO'),
        mean_normalized_kendall_H_RO=jaccard_mean('normalized_kendall_H_RO'),
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
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output.base_dir) / f"{timestamp}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=output_dir / "logs", console_level=config.output.log_level)
    logger.info(f"Starting experiment: {config.experiment.name} v{config.experiment.version}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Random seed: {config.experiment.seed}")
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
        run_id=run_id,
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
                label_names=dataset_config.labels if hasattr(dataset_config, 'labels') else None,
            )
            logger.info(f"Loaded {len(instances)} instances for {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue

        summary.total_instances += len(instances) * len(config.models)

        for model_config in config.models:
            logger.info(f"Processing model: {model_config.name} on {dataset_name}")

            engine = InferenceEngine(
                model_name=model_config.groq_model_id,
                max_retries=config.inference.max_retries,
                concurrent_requests=config.inference.concurrent_requests,
            )

            prompts = create_prompt_map(config, dataset_name=dataset_name)
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

    # Compute pure dataset-level aggregates (collapsing models)
    dataset_names = set(r.dataset for r in all_results)
    for ds in dataset_names:
        ds_results = [r for r in all_results if r.dataset == ds]
        agg = compute_aggregate_metrics(ds_results, "dataset", ds)
        aggregate_list.append(agg)

    # Compute pure model-level aggregates (collapsing datasets)
    model_names = set(r.model for r in all_results)
    for mn in model_names:
        md_results = [r for r in all_results if r.model == mn]
        agg = compute_aggregate_metrics(md_results, "model", mn)
        aggregate_list.append(agg)

    # Save results
    save_instance_results(all_results, str(output_dir / "instance_results.jsonl"))
    save_aggregate_metrics(aggregate_list, str(output_dir / "aggregate_metrics.json"))

    # Save frozen config
    save_config_to_file(config, output_dir / "config_snapshot.yaml")

    # Save CSV export
    save_metrics_csv(all_results, str(output_dir / "instance_metrics.csv"))

    # Save metadata tables
    save_metadata_table(
        [d.to_dict() for d in config.datasets],
        "datasets", str(output_dir / "dataset_metadata.json")
    )
    save_metadata_table(
        [m.to_dict() for m in config.models],
        "models", str(output_dir / "model_metadata.json")
    )

    # Save environment snapshot
    if config.reproducibility.log_git_commit or config.reproducibility.log_package_versions:
        save_environment_snapshot(output_dir / "environment_snapshot.json")

    # Save summary
    summary.end_time = datetime.now()
    summary.duration_seconds = (summary.end_time - summary.start_time).total_seconds()
    summary.api_requests_total = len(all_results) * 5  # 1 class + 4 explanations
    summary.api_requests_failed = summary.failed_instances * 5  # approximating 5 calls per failed instance
    summary.avg_time_per_instance = summary.duration_seconds / max(len(all_results), 1)

    with open(output_dir / "execution_summary.txt", 'w') as f:
        f.write(summary.generate_report())

    # Generate markdown report
    report_md = generate_md_report(aggregate_list, all_results, config)
    with open(output_dir / "report.md", 'w', encoding='utf-8') as f:
        f.write(report_md)
    logger.info(f"Report saved to {output_dir / 'report.md'}")

    logger.info(f"Experiment complete. Results saved to {output_dir}")
    logger.info(summary.generate_report())

    return all_results, aggregate_list


def main():
    from dotenv import load_dotenv
    load_dotenv()
    args = parse_command_line_args()
    config = load_and_validate_config(args=args)
    asyncio.run(run_experiment(config, args))


if __name__ == "__main__":
    main()
