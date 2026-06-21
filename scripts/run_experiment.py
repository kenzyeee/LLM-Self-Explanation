import argparse
import asyncio
import hashlib
import json
import logging
import random
import re
import string
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
from src.metrics.redaction_test import RedactionTest
from src.utils.config_loader import load_and_validate_config, parse_command_line_args, save_config_to_file
from src.utils.data_models import (
    InstanceResult, SamplingLog, AggregateMetrics, save_instance_results, save_aggregate_metrics,
    generate_md_report, save_metrics_csv, save_metadata_table, save_environment_snapshot,
)
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.execution_summary import ExecutionSummary
from src.utils.exceptions import APIError, RateLimitExhausted, ParsingError, PromptValidationError
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

STRATEGY_IDS = ["H", "R", "CF", "RO"]

REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i'm unable", "i am unable", "not able to",
    "i apologize", "i'm sorry", "i am sorry", "cannot fulfill",
    "cannot complete", "not appropriate", "i cannot provide",
    "i can't provide", "against policy", "not permitted",
]

_KNOWN_PLACEHOLDERS = re.compile(r'\{label_set\}|\{input_text\}|\{predicted_label\}|\{other_labels\}')


def validate_prompt(prompt_text: str, label_set: Optional[List[str]] = None, context: str = "") -> None:
    """Reject prompts with unrendered placeholders. Only checks known template placeholders."""
    unrendered = _KNOWN_PLACEHOLDERS.findall(prompt_text)
    if unrendered:
        raise PromptValidationError(
            f"Unrendered placeholders in {context}: {list(set(unrendered))}",
            details={"placeholders": list(set(unrendered)), "prompt_preview": prompt_text[:200]}
        )


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


def format_explain_prompt(template: str, predicted_label: str, input_text: str = "", **kwargs) -> str:
    return template.format(predicted_label=predicted_label, input_text=input_text, **kwargs)


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
        ds_specific_multiclass = None
        if dataset_name:
            ds_specific_multiclass = explain_file.replace("_explain.txt", f"_explain_multiclass_{dataset_name}.txt")
            if not Path(ds_specific_multiclass).exists():
                ds_specific_multiclass = None
        multiclass_path = ds_specific_multiclass or multiclass_variant
        if Path(multiclass_path).exists():
            prompts[f"{strategy.id}_explain_multiclass"] = load_prompt(multiclass_path)

    # CF-free (unconstrained) prompts — the canonical CF used for ECS pairs.
    # The existing CF_explain / CF_explain_multiclass prompts remain the MINIMAL variant.
    cf_free_path = "prompts/counterfactual_explain_free.txt"
    if dataset_name and Path(f"prompts/counterfactual_explain_free_{dataset_name}.txt").exists():
        cf_free_path = f"prompts/counterfactual_explain_free_{dataset_name}.txt"
    if Path(cf_free_path).exists():
        prompts["CF_explain_free"] = load_prompt(cf_free_path)
    cf_free_mc_path = "prompts/counterfactual_explain_free_multiclass.txt"
    if dataset_name and Path(f"prompts/counterfactual_explain_free_multiclass_{dataset_name}.txt").exists():
        cf_free_mc_path = f"prompts/counterfactual_explain_free_multiclass_{dataset_name}.txt"
    if Path(cf_free_mc_path).exists():
        prompts["CF_explain_free_multiclass"] = load_prompt(cf_free_mc_path)

    class_prompt_path = "prompts/classification.txt"
    if dataset_name:
        ds_specific = f"prompts/classification_{dataset_name}.txt"
        if Path(ds_specific).exists():
            class_prompt_path = ds_specific
    prompts["classification"] = load_prompt(class_prompt_path)

    return prompts


def pre_clean_text(text: str) -> str:
    """Clean HTML entities, markup, and orphaned numeric entities from text before prompting."""
    import html as html_mod
    text = html_mod.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'#(\d+);', lambda m: chr(int(m.group(1))), text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    return text


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

    # Pre-clean text for HTML/markup artifacts
    clean_text = pre_clean_text(text)

    # Validate and run classification
    class_prompt = format_prompt(prompts["classification"], clean_text, label_set)
    validate_prompt(class_prompt, label_set, context=f"classification for {instance_id}")
    class_result = await engine.classify(class_prompt)

    predicted_label = ""
    classification_valid = False
    try:
        predicted_label = parser.parse_classification(class_result.raw_response, label_set)
        classification_valid = True
    except ParsingError as e:
        logger.warning(f"Classification parsing error for {instance_id}: {e}")

    if not predicted_label or predicted_label.strip() == "":
        logger.error(f"Empty predicted_label for {instance_id}")
    if "{" in predicted_label or "}" in predicted_label:
        logger.error(f"Unrendered placeholder in label for {instance_id}: {predicted_label}")

    correct = predicted_label == instance.label
    model_refused = is_model_refusal(class_result.raw_response)
    prompt_hash = compute_prompt_hash(class_prompt)
    prompt_tokens = count_tokens(class_prompt)
    response_tokens = count_tokens(class_result.raw_response)
    raw_response_length = len(class_result.raw_response)

    # D5: include misclassified instances. We only short-circuit when the
    # classification itself is UNPARSEABLE (no usable label) — genuine
    # misclassifications proceed and are tagged correct=False for the
    # incorrect-prediction stratum.
    if not classification_valid or not predicted_label:
        logger.warning(f"{instance_id}: SKIPPED — classification unparseable ({instance.label} -> '{predicted_label}'). "
                       f"No explanation strategies elicited, no metrics computed.")
        result = InstanceResult(
            instance_id=instance_id,
            dataset=instance.dataset,
            model=engine.model_name,
            timestamp=datetime.now(),
            text=text,
            input_length=len(text.split()),
            ground_truth_label=instance.label,
            predicted_label=predicted_label,
            confidence=0.0,
            correct=correct,
            raw_highlighting="", raw_rationale="",
            raw_counterfactual="", raw_rank_ordering="",
            classification_prompt=class_prompt,
            classification_raw_response=class_result.raw_response,
            model_refused=model_refused,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            raw_response_length=raw_response_length,
            ecs=None, ecs_extraction_rationale=None, ecs_extraction_perturbation=None, ecs_complete=None, ecs_primary_pairs=0, n_valid_strategies=0,
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
    # CF state — initialized up front so it is always defined even if the CF
    # branch never runs (avoids reading via locals() after the loop).
    cf_json_valid = False
    cf_rules_compliant = False
    cf_flip_verified = False
    cf_actual_label = ""
    # CF-minimal variant state (D2) — always defined even if the branch never runs.
    cf_minimal_valid = False
    cf_minimal_flip_verified = False
    cf_minimal_minimality = None
    cf_minimal_tokens = set()
    cf_minimal_text = ""

    for strat_id in STRATEGY_IDS:
        raw_responses[strat_id] = ""
        explain_prompts[strat_id] = ""
        parsed_tokens[strat_id] = set()
        parsed_flags[strat_id] = False
        valid_flags[strat_id] = False

    for strat_id in strategy_order:
        if strat_id == "CF":
            # Canonical CF for ECS is the UNCONSTRAINED (free) variant — it flips
            # reliably (Mayne et al. 2025). The minimal variant is elicited separately below.
            cf_prompt_key = "CF_explain_free" if "CF_explain_free" in prompts else "CF_explain"
            if len(label_set) > 2:
                if "CF_explain_free_multiclass" in prompts:
                    cf_prompt_key = "CF_explain_free_multiclass"
                elif "CF_explain_multiclass" in prompts:
                    cf_prompt_key = "CF_explain_multiclass"
            other_labels = ", ".join(l for l in label_set if l != predicted_label)
            explain_prompt = format_explain_prompt(prompts[cf_prompt_key], predicted_label,
                                                   input_text=clean_text, other_labels=other_labels)
        else:
            explain_prompt = format_explain_prompt(prompts[f"{strat_id}_explain"], predicted_label,
                                                   input_text=clean_text)
        explain_prompts[strat_id] = explain_prompt
        messages = [
            {"role": "user", "content": class_prompt},
            {"role": "assistant", "content": class_result.raw_response},
            {"role": "user", "content": explain_prompt},
        ]
        try:
            strat_max_tokens = 1500 if strat_id == "H" else 500
            raw = await engine.chat(messages, max_tokens=strat_max_tokens)
            raw_responses[strat_id] = raw or ""
            response_tokens += count_tokens(raw) if raw else 0
            if raw is None or not raw.strip():
                logger.warning(f"Empty response from model for {instance_id} strategy {strat_id}, retrying with max_tokens=2500...")
                raw = await engine.chat(messages, max_tokens=2500)
                raw_responses[strat_id] = raw or ""
                response_tokens += count_tokens(raw) if raw else 0
                if raw is None or not raw.strip():
                    logger.warning(f"Empty response on retry for {instance_id} strategy {strat_id}")
            if raw and is_model_refusal(raw):
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
                tokens = parser.parse_highlighting(raw, clean_text, normalizer)
                normalized = normalizer.normalize_tokens(tokens)
                if not normalized:
                    raise ParsingError("H evidence set is empty after normalization")
                parsed_tokens[strat_id] = normalized
                parsed_tokens["H_ordered"] = tokens
                parsed_flags[strat_id] = True
                valid_flags[strat_id] = True
            elif strat_id == "R":
                r_text, evidence = parser.parse_rationale(raw, clean_text, normalizer)
                rationale_text = r_text
                normalized = normalizer.normalize_tokens(evidence)
                if not normalized:
                    raise ParsingError("R evidence set is empty after normalization")
                parsed_tokens[strat_id] = normalized
                parsed_tokens["R_evidence"] = evidence
                parsed_tokens["R_introduced"] = list(getattr(parser, "_r_introduced", []))
                parsed_flags[strat_id] = True
                valid_flags[strat_id] = True
            elif strat_id == "CF":
                cf_max_ratio = getattr(dataset_config, 'cf_max_edit_ratio', 0.3)
                cf_text_used = ""
                from_tokens = set()
                new_pred = ""
                # Stage 1: JSON parsing
                cf_json_obj = parser._extract_json(raw)
                cf_json_valid = cf_json_obj is not None
                if not cf_json_valid:
                    logger.warning(f"CF for {instance_id}: JSON not parseable (raw: {raw[:100] if raw else 'empty'})")
                    raise ParsingError("CF JSON not parseable")
                # Stage 2: Rules compliance (formerly parse_counterfactual)
                try:
                    # Free/canonical CF is unconstrained — skip the edit-ratio cap.
                    # (skip_validation only bypasses the ratio check; a real flip and
                    # non-empty change are still required.)
                    cf_text_used, new_pred, from_tokens = parser.parse_counterfactual(
                        raw, clean_text, predicted_label, label_set, normalizer,
                        max_edit_ratio=cf_max_ratio, skip_validation=True
                    )
                    cf_rules_compliant = bool(from_tokens)
                except ParsingError:
                    cf_rules_compliant = False
                    raise
                if not from_tokens:
                    cf_rules_compliant = False
                    logger.warning(f"CF evidence empty for {instance_id}")
                    raise ParsingError("CF evidence set is empty after edit extraction")
                # Stage 3: Flip verification — iterative correction on failure
                cf_raw_used = raw
                for cf_attempt in range(2):
                    try:
                        cf_class_prompt = format_prompt(prompts["classification"], cf_text_used, label_set)
                        validate_prompt(cf_class_prompt, label_set, context=f"CF re-classification for {instance_id}")
                        cf_class_result = await engine.classify(cf_class_prompt)
                        cf_actual_label = parser.parse_classification(cf_class_result.raw_response, label_set)
                        cf_flip_verified = (cf_actual_label != predicted_label)
                        if cf_flip_verified:
                            if cf_actual_label != new_pred:
                                logger.warning(f"CF flip verified but label mismatch for {instance_id}: model said {new_pred}, actual {cf_actual_label}")
                            break
                        logger.warning(f"CF flip NOT verified for {instance_id} (attempt {cf_attempt+1}): "
                                      f"rewrote '{clean_text[:60]}...' to '{cf_text_used[:60]}...' but classifier still predicted '{cf_actual_label}'")
                        if cf_attempt == 0:
                            other_labels = ", ".join(l for l in label_set if l != predicted_label)
                            correction_prompt = (
                                f"Your previous rewrite was: \"{cf_text_used}\"\n\n"
                                f"But the classifier still predicts '{cf_actual_label}', not '{new_pred}'.\n"
                                f"Try changing a DIFFERENT word — pick a more impactful word this time — "
                                f"to flip the prediction to one of: {other_labels}\n\n"
                                f"Return only valid JSON: {{\"rewritten\": \"<minimally changed text>\", \"new_prediction\": \"<target>\"}}"
                            )
                            correction_messages = messages + [
                                {"role": "assistant", "content": cf_raw_used},
                                {"role": "user", "content": correction_prompt},
                            ]
                            correction_raw = await engine.chat(correction_messages, max_tokens=500)
                            cf_raw_used = correction_raw
                            cf_json_obj2 = parser._extract_json(correction_raw)
                            if cf_json_obj2 is None:
                                logger.warning(f"CF correction for {instance_id}: correction JSON not parseable")
                                raise ParsingError("CF correction JSON not parseable")
                            cf_text_used, new_pred, from_tokens = parser.parse_counterfactual(
                                correction_raw, clean_text, predicted_label, label_set, normalizer,
                                max_edit_ratio=cf_max_ratio, skip_validation=True
                            )
                            cf_rules_compliant = bool(from_tokens)
                        else:
                            raise ParsingError(f"CF flip not verified after correction: {cf_actual_label}")
                    except ParsingError:
                        raise
                    except Exception as e:
                        logger.warning(f"CF flip verification failed for {instance_id}: {e}")
                        if cf_attempt == 1:
                            raise ParsingError(f"CF flip verification exception: {e}")
                        raise
                parsed_tokens[strat_id] = from_tokens
                parsed_tokens["CF_reconstructed"] = cf_text_used
                parsed_tokens["CF_free_minimality"] = len(from_tokens) / max(len(clean_text.split()), 1)
                parsed_flags[strat_id] = True
                valid_flags[strat_id] = True
            elif strat_id == "RO":
                ranked = parser.parse_rank_ordering(raw, clean_text, normalizer)
                ro_tokens = [t for t, r in ranked]
                # One-shot self-correction for hallucinated tokens
                json_obj = parser._extract_json(raw)
                raw_ranking = json_obj.get("ranking", []) if json_obj else []
                if raw_ranking and len(ro_tokens) < len(raw_ranking):
                    discarded = len(raw_ranking) - len(ro_tokens)
                    logger.warning(f"RO for {instance_id}: {discarded} token(s) hallucinated, attempting self-correction")
                    invalid_tokens = [t for t in raw_ranking if isinstance(t, str) and not normalizer.is_anchored(t, clean_text)]
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
                        correction_raw = await engine.chat(correction_messages, max_tokens=500)
                        ranked = parser.parse_rank_ordering(correction_raw, clean_text, normalizer)
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

    cf_flip_verified = parsed_flags.get("CF", False) and valid_flags.get("CF", False) and cf_flip_verified
    cf_counterfactual_text = parsed_tokens.get("CF_reconstructed", "")
    ro_set = parsed_tokens.get("RO_set", set())
    ro_ranked = parsed_tokens.get("RO_ranked", [])

    explanations = {
        "H": parsed_tokens.get("H", set()),
        "R": parsed_tokens.get("R", set()),
        "CF": parsed_tokens.get("CF", set()),
        "RO": ro_set,
    }

    agreements = calc.compute_pairwise_agreements(explanations)
    overlaps = calc.compute_pairwise_overlaps(explanations)

    h_ordered = parsed_tokens.get("H_ordered", [])
    kendall_val = None
    normalized_kendall_val = None
    rbo_val = None
    if h_ordered and ro_ranked:
        h_ranks = calc.assign_implicit_ranks(h_ordered)
        kendall_val = calc.compute_kendalls_tau(h_ranks, ro_ranked)
        normalized_kendall_val = calc.compute_normalized_kendalls_tau(kendall_val)
        ro_ordered = [t for t, r in ro_ranked]
        rbo_val = calc.compute_rbo(h_ordered, ro_ordered)

    n_valid = sum(1 for s in STRATEGY_IDS if valid_flags.get(s, False))

    # Vocabulary size from normalized input text (content-word unique tokens only)
    input_tokens = normalizer.normalize_input_text(clean_text).split()
    STRUCTURAL_LABELS = {"premise:", "hypothesis:", "sentence1:", "sentence2:", "text:", "label:"}
    content_tokens = [t for t in input_tokens if t not in STRUCTURAL_LABELS and
                      t.strip(string.punctuation) and
                      t not in string.punctuation]
    vocab_size = len(set(content_tokens))
    SHORT_VOCAB_THRESHOLD = 20
    short_vocab = vocab_size <= SHORT_VOCAB_THRESHOLD

    # D6: introduced-concept rate — rationale concepts with NO input anchor
    # (post-hoc rationalization signal). R_introduced is populated by the parser.
    r_introduced = parsed_tokens.get("R_introduced", [])
    r_hallucinated = list(r_introduced)  # retained field name for back-compat
    r_introduced_concept_rate = None
    n_anchored_r = len(parsed_tokens.get("R", set()))
    n_introduced_r = len(r_introduced)
    if (n_anchored_r + n_introduced_r) > 0:
        r_introduced_concept_rate = n_introduced_r / (n_anchored_r + n_introduced_r)

    ecs_value = None
    ecs_extraction_rationale = None
    ecs_extraction_perturbation = None
    ecs_primary_pairs = 0
    if n_valid >= 3:
        ecs_value = calc.compute_ecs(agreements)
        ecs_extraction_rationale, ecs_extraction_perturbation, ecs_primary_pairs = calc.compute_ecs_primary(agreements)
    else:
        logger.warning(f"Only {n_valid} valid strategies for {instance_id} — ECS not computed")

    # D1: report ECS as LIFT over a random-selection baseline (chance agreement given
    # each strategy's set size and the instance content-vocabulary).
    ecs_random = None
    ecs_lift = None
    if ecs_value is not None and vocab_size > 0:
        _excluded = {("H", "RO")}
        _strats = ["H", "R", "CF", "RO"]
        _rand_vals = []
        for _i in range(len(_strats)):
            for _j in range(_i + 1, len(_strats)):
                s1, s2 = _strats[_i], _strats[_j]
                if (s1, s2) in _excluded:
                    continue
                set1, set2 = explanations.get(s1, set()), explanations.get(s2, set())
                if set1 and set2:
                    ej, _ = calc.expected_random_overlap(len(set1), len(set2), vocab_size)
                    _rand_vals.append(ej)
        if _rand_vals:
            ecs_random = sum(_rand_vals) / len(_rand_vals)
            ecs_lift = ecs_value - ecs_random

    # D2: CF-minimal variant — the Mayne et al. validity-minimality probe. Elicited
    # separately, NOT used in ECS. Substitution-only; a length change => invalid.
    if predicted_label:
        cf_min_key = "CF_explain"
        if len(label_set) > 2 and "CF_explain_multiclass" in prompts:
            cf_min_key = "CF_explain_multiclass"
        if cf_min_key in prompts:
            try:
                _other = ", ".join(l for l in label_set if l != predicted_label)
                cf_min_prompt = format_explain_prompt(prompts[cf_min_key], predicted_label,
                                                      input_text=clean_text, other_labels=_other)
                cf_min_messages = [
                    {"role": "user", "content": class_prompt},
                    {"role": "assistant", "content": class_result.raw_response},
                    {"role": "user", "content": cf_min_prompt},
                ]
                cf_min_raw = await engine.chat(cf_min_messages, max_tokens=500)
                cf_max_ratio_m = getattr(dataset_config, 'cf_max_edit_ratio', 0.3)
                cf_min_text, _cf_min_pred, cf_min_from = parser.parse_counterfactual(
                    cf_min_raw, clean_text, predicted_label, label_set, normalizer,
                    max_edit_ratio=cf_max_ratio_m,
                )
                if len(cf_min_text.split()) != len(clean_text.split()):
                    logger.info(f"CF-minimal for {instance_id}: length changed — invalid (not substitution-only)")
                    cf_minimal_text = cf_min_text
                    cf_minimal_minimality = len(cf_min_from) / max(len(clean_text.split()), 1)
                else:
                    cf_minimal_text = cf_min_text
                    cf_minimal_tokens = cf_min_from
                    cf_minimal_minimality = len(cf_min_from) / max(len(clean_text.split()), 1)
                    cf_min_class_prompt = format_prompt(prompts["classification"], cf_min_text, label_set)
                    cf_min_class_raw = (await engine.classify(cf_min_class_prompt)).raw_response
                    cf_min_actual = parser.parse_classification(cf_min_class_raw, label_set)
                    cf_minimal_flip_verified = (cf_min_actual != predicted_label)
                    cf_minimal_valid = cf_minimal_flip_verified
            except (ParsingError, json.JSONDecodeError) as e:
                logger.info(f"CF-minimal for {instance_id}: invalid ({e})")
            except APIError as e:
                logger.warning(f"CF-minimal for {instance_id}: API error ({e})")

    result = InstanceResult(
        instance_id=instance_id,
        dataset=instance.dataset,
        model=engine.model_name,
        timestamp=datetime.now(),
        text=text,
        input_length=len(text.split()),
        ground_truth_label=instance.label,
        predicted_label=predicted_label,
        confidence=0.0,
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
        raw_response_length=raw_response_length,
        prompt_hash=prompt_hash,
        jaccard_H_R=agreements.get(("H", "R")),
        jaccard_H_CF=agreements.get(("H", "CF")),
        jaccard_H_RO=agreements.get(("H", "RO")),
        jaccard_R_CF=agreements.get(("R", "CF")),
        jaccard_R_RO=agreements.get(("R", "RO")),
        jaccard_CF_RO=agreements.get(("CF", "RO")),
        overlap_H_R=overlaps.get(("H", "R")),
        overlap_H_CF=overlaps.get(("H", "CF")),
        overlap_H_RO=overlaps.get(("H", "RO")),
        overlap_R_CF=overlaps.get(("R", "CF")),
        overlap_R_RO=overlaps.get(("R", "RO")),
        overlap_CF_RO=overlaps.get(("CF", "RO")),
        rbo_H_RO=rbo_val,
        kendall_H_RO=kendall_val,
        normalized_kendall_H_RO=normalized_kendall_val,
        ecs=ecs_value,
        ecs_extraction_rationale=ecs_extraction_rationale,
        ecs_extraction_perturbation=ecs_extraction_perturbation,
        ecs_primary_pairs=ecs_primary_pairs,
        n_valid_strategies=n_valid,
        vocab_size=vocab_size,
        short_vocab=short_vocab,
        r_hallucinated_concepts=r_hallucinated,
        cf_json_valid=cf_json_valid,
        cf_rules_compliant=cf_rules_compliant,
        cf_flip_verified=cf_flip_verified,
        cf_actual_label=cf_actual_label,
        cf_counterfactual_text=cf_counterfactual_text,
        ecs_random=ecs_random,
        ecs_lift=ecs_lift,
        cf_free_minimality=parsed_tokens.get("CF_free_minimality"),
        cf_minimal_valid=cf_minimal_valid,
        cf_minimal_flip_verified=cf_minimal_flip_verified,
        cf_minimal_minimality=cf_minimal_minimality,
        cf_minimal_tokens=cf_minimal_tokens,
        cf_minimal_text=cf_minimal_text,
        r_introduced_concept_rate=r_introduced_concept_rate,
    )

    # Consensus cores over valid strategies only
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
            result.cc4_tokens = calc.compute_consensus_core(reduced_explanations, n_valid)
            result.cc4_size = len(result.cc4_tokens)

    # Redaction test: progressive token removal for H and RO
    h_ordered_tokens = parsed_tokens.get("H_ordered", [])
    ro_ranked_tokens = [t for t, r in parsed_tokens.get("RO_ranked", [])]
    # D5: erasure is defined relative to the model's OWN prediction, so it runs
    # for misclassified instances too (not gated on correctness).
    if predicted_label:
        async def redact_classify(redacted_text: str) -> str:
            cfp = format_prompt(prompts["classification"], redacted_text, label_set)
            cr = await engine.classify(cfp)
            return parser.parse_classification(cr.raw_response, label_set)

        rt = RedactionTest(redact_classify)
        if h_ordered_tokens:
            try:
                result.redaction_H = await rt.run(h_ordered_tokens, clean_text, predicted_label)
            except Exception as e:
                logger.warning(f"Redaction test H failed for {instance_id}: {e}")
        if ro_ranked_tokens:
            try:
                result.redaction_RO = await rt.run(ro_ranked_tokens, clean_text, predicted_label)
            except Exception as e:
                logger.warning(f"Redaction test RO failed for {instance_id}: {e}")

    return result


def compute_aggregate_metrics(results: List[InstanceResult], level: str, group: str,
                               sampling_log: Optional[SamplingLog] = None) -> AggregateMetrics:
    import numpy as np

    ecs_values = [r.ecs for r in results if r.ecs is not None]

    if not ecs_values:
        return AggregateMetrics(
            aggregation_level=level, group_name=group, n_instances=len(results),
            mean_ecs=0, mean_ecs_extraction_rationale=0, mean_ecs_extraction_perturbation=0, std_ecs=0, median_ecs=0,
            ecs_ci_lower=0, ecs_ci_upper=0,
            mean_jaccard_H_R=0, mean_jaccard_H_CF=0, mean_jaccard_H_RO=0,
            mean_jaccard_R_CF=0, mean_jaccard_R_RO=0, mean_jaccard_CF_RO=0,
            mean_overlap_H_R=0, mean_overlap_H_CF=0, mean_overlap_H_RO=0,
            mean_overlap_R_CF=0, mean_overlap_R_RO=0,             mean_overlap_CF_RO=0,
            mean_rbo_H_RO=0,
            mean_kendall_H_RO=0,
            mean_normalized_kendall_H_RO=0,
            mean_cc3_size=0, mean_cc4_size=0,
            pct_instances_with_cc3=0, pct_instances_with_cc4=0,
            spearman_rho=0, spearman_p_value=1.0,
            correlation_ci_lower=0, correlation_ci_upper=0,
            highlighting_success_rate=0, rationale_success_rate=0,
            counterfactual_success_rate=0, rank_ordering_success_rate=0,
        )

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

    # Stratify by length
    length_data = MetricsCalculator.compute_length_stratified_ecs(results)
    short_ecs = length_data.get("short", {}).get("mean_ecs", 0.0)
    n_short = length_data.get("short", {}).get("n", 0)
    medium_ecs = length_data.get("medium", {}).get("mean_ecs", 0.0)
    n_medium = length_data.get("medium", {}).get("n", 0)
    long_ecs = length_data.get("long", {}).get("mean_ecs", 0.0)
    n_long = length_data.get("long", {}).get("n", 0)

    # Stratify by vocabulary size
    normal_vocab = [r.ecs for r in results if r.ecs is not None and not r.short_vocab]
    short_vocab_ecs = [r.ecs for r in results if r.ecs is not None and r.short_vocab]
    mean_ecs_normal_vocab = safe_mean(normal_vocab)
    n_normal_vocab = len(normal_vocab)
    mean_ecs_short_vocab = safe_mean(short_vocab_ecs)
    n_short_vocab = len(short_vocab_ecs)

    # Complete-case analysis
    cc_metrics = MetricsCalculator.compute_complete_case_metrics(results)
    complete_mean_ecs = cc_metrics["complete_mean_ecs"]
    n_complete = cc_metrics["n_complete"]

    # D1 lift / D2 CF trade-off / D6 introduced-concept rate / correctness split
    ecs_lift_values = [r.ecs_lift for r in results if r.ecs_lift is not None]
    ecs_random_values = [r.ecs_random for r in results if r.ecs_random is not None]
    icr_values = [r.r_introduced_concept_rate for r in results if r.r_introduced_concept_rate is not None]
    cf_free_min_values = [r.cf_free_minimality for r in results if r.cf_free_minimality is not None]
    cf_minimal_min_values = [r.cf_minimal_minimality for r in results if r.cf_minimal_minimality is not None]
    ecs_correct_vals = [r.ecs for r in results if r.ecs is not None and r.correct]
    ecs_incorrect_vals = [r.ecs for r in results if r.ecs is not None and not r.correct]
    n_results_safe = max(len(results), 1)
    cf_free_validity_rate = sum(1 for r in results if r.counterfactual_valid) / n_results_safe
    cf_minimal_validity_rate = sum(1 for r in results if r.cf_minimal_valid) / n_results_safe

    return AggregateMetrics(
        aggregation_level=level,
        group_name=group,
        n_instances=len(results),
        mean_ecs=safe_mean(ecs_values),
        mean_ecs_extraction_rationale=safe_mean([r.ecs_extraction_rationale for r in results if r.ecs_extraction_rationale is not None]),
        mean_ecs_extraction_perturbation=safe_mean([r.ecs_extraction_perturbation for r in results if r.ecs_extraction_perturbation is not None]),
        mean_ecs_complete=complete_mean_ecs,
        n_complete_cases=n_complete,
        pct_complete_cases=n_complete / max(len(results), 1) * 100,
        mean_ecs_short=short_ecs,
        n_short=n_short,
        mean_ecs_medium=medium_ecs,
        n_medium=n_medium,
        mean_ecs_long=long_ecs,
        n_long=n_long,
        mean_ecs_normal_vocab=mean_ecs_normal_vocab,
        n_normal_vocab=n_normal_vocab,
        mean_ecs_short_vocab=mean_ecs_short_vocab,
        n_short_vocab=n_short_vocab,
        requested_samples=sampling_log.requested if sampling_log else 0,
        sampled_samples=sampling_log.sampled if sampling_log else 0,
        dropped_wrong_pred=sampling_log.wrong_predictions if sampling_log else 0,
        dropped_other=sampling_log.dropped_by_reason if sampling_log else {},
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
        mean_overlap_H_R=jaccard_mean('overlap_H_R'),
        mean_overlap_H_CF=jaccard_mean('overlap_H_CF'),
        mean_overlap_H_RO=jaccard_mean('overlap_H_RO'),
        mean_overlap_R_CF=jaccard_mean('overlap_R_CF'),
        mean_overlap_R_RO=jaccard_mean('overlap_R_RO'),
        mean_overlap_CF_RO=jaccard_mean('overlap_CF_RO'),
        mean_rbo_H_RO=jaccard_mean('rbo_H_RO'),
        mean_kendall_H_RO=jaccard_mean('kendall_H_RO'),
        mean_normalized_kendall_H_RO=jaccard_mean('normalized_kendall_H_RO'),
        mean_cc3_size=safe_mean([r.cc3_size for r in results]),
        mean_cc4_size=safe_mean([r.cc4_size for r in results]),
        pct_instances_with_cc3=sum(1 for r in results if r.cc3_size > 0) / max(len(results), 1) * 100,
        pct_instances_with_cc4=sum(1 for r in results if r.cc4_size > 0) / max(len(results), 1) * 100,
        spearman_rho=0.0,
        spearman_p_value=1.0,
        correlation_ci_lower=0.0,
        correlation_ci_upper=0.0,
        highlighting_success_rate=sum(1 for r in results if r.highlighting_parsed) / max(len(results), 1),
        rationale_success_rate=sum(1 for r in results if r.rationale_parsed) / max(len(results), 1),
        counterfactual_success_rate=sum(1 for r in results if r.counterfactual_parsed) / max(len(results), 1),
        rank_ordering_success_rate=sum(1 for r in results if r.rank_ordering_parsed) / max(len(results), 1),
        mean_redaction_faithfulness_H=safe_mean([r.redaction_H.faithfulness for r in results if r.redaction_H is not None]),
        mean_redaction_faithfulness_RO=safe_mean([r.redaction_RO.faithfulness for r in results if r.redaction_RO is not None]),
        n_redaction_H=sum(1 for r in results if r.redaction_H is not None),
        n_redaction_RO=sum(1 for r in results if r.redaction_RO is not None),
        mean_ecs_lift=safe_mean(ecs_lift_values),
        mean_ecs_random=safe_mean(ecs_random_values),
        introduced_concept_rate=safe_mean(icr_values),
        cf_free_validity_rate=cf_free_validity_rate,
        cf_minimal_validity_rate=cf_minimal_validity_rate,
        mean_cf_free_minimality=safe_mean(cf_free_min_values),
        mean_cf_minimal_minimality=safe_mean(cf_minimal_min_values),
        mean_ecs_correct=safe_mean(ecs_correct_vals),
        n_correct=len(ecs_correct_vals),
        mean_ecs_incorrect=safe_mean(ecs_incorrect_vals),
        n_incorrect=len(ecs_incorrect_vals),
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
    normalizer = Normalizer(
        use_lemmatization=config.normalization.use_lemmatization,
        remove_stopwords=config.normalization.remove_stopwords,
        lemmatizer=config.normalization.lemmatizer,
    )
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
    sampling_logs: List[SamplingLog] = []

    for dataset_config in config.datasets:
        dataset_name = dataset_config.name
        logger.info(f"Processing dataset: {dataset_name}")

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

        # Build sampling log for this dataset
        slog = SamplingLog(
            dataset=dataset_name,
            requested=dataset_config.sample_size,
            sampled=len(instances),
        )

        for model_config in config.models:
            logger.info(f"Processing model: {model_config.name} on {dataset_name}")

            engine = InferenceEngine(
                model_name=model_config.model_id,
                max_retries=config.inference.max_retries,
                concurrent_requests=config.inference.concurrent_requests,
            )

            prompts = create_prompt_map(config, dataset_name=dataset_name)
            model_results: List[InstanceResult] = []
            wrong_pred_count = 0
            cp = CheckpointManager(output_dir / f"checkpoint_{dataset_name}_{model_config.name}.jsonl")
            last_checkpointed = 0

            def _flush_checkpoint():
                nonlocal last_checkpointed
                new_results = model_results[last_checkpointed:]
                if new_results:
                    cp.save_checkpoint([r.to_dict() for r in new_results])
                    last_checkpointed = len(model_results)

            for i, instance in enumerate(instances):
                logger.info(f"Processing {instance.instance_id} ({i+1}/{len(instances)})")
                try:
                    result = await process_instance(
                        instance, engine, parser, normalizer, calc, prompts, config, dataset_config
                    )
                    model_results.append(result)
                    summary.successful_instances += 1
                    if not result.correct:
                        wrong_pred_count += 1
                except RateLimitExhausted as e:
                    logger.error(f"Rate limit exhausted for {instance.instance_id}: {e}")
                    summary.failed_instances += 1
                    _flush_checkpoint()
                    logger.info(f"Partial output saved ({last_checkpointed} instances) after rate limit.")
                    continue
                except PromptValidationError as e:
                    logger.error(f"Prompt validation failed for {instance.instance_id}: {e}")
                    summary.prompt_validation_failures += 1
                    summary.failed_instances += 1
                    continue
                except Exception as e:
                    logger.error(f"Failed to process {instance.instance_id}: {e}")
                    summary.failed_instances += 1
                    continue

                if (i + 1) % config.output.checkpoint_frequency == 0:
                    _flush_checkpoint()

            slog.wrong_predictions += wrong_pred_count
            all_results.extend(model_results)

            agg = compute_aggregate_metrics(model_results, "model_dataset", f"{model_config.name}_{dataset_name}", sampling_log=slog)
            aggregate_list.append(agg)

        sampling_logs.append(slog)

    # Compute overall aggregate
    if all_results:
        overall_slog = SamplingLog(
            dataset="all",
            requested=sum(s.requested for s in sampling_logs),
            sampled=sum(s.sampled for s in sampling_logs),
            wrong_predictions=sum(s.wrong_predictions for s in sampling_logs),
        )
        overall = compute_aggregate_metrics(all_results, "overall", "all", sampling_log=overall_slog)
        aggregate_list.append(overall)

    # Pure dataset-level aggregates
    dataset_names = set(r.dataset for r in all_results)
    for ds in dataset_names:
        ds_results = [r for r in all_results if r.dataset == ds]
        ds_slog = next((s for s in sampling_logs if s.dataset == ds), None)
        agg = compute_aggregate_metrics(ds_results, "dataset", ds, sampling_log=ds_slog)
        aggregate_list.append(agg)

    # Pure model-level aggregates
    model_names = set(r.model for r in all_results)
    for mn in model_names:
        md_results = [r for r in all_results if r.model == mn]
        agg = compute_aggregate_metrics(md_results, "model", mn)
        aggregate_list.append(agg)

    # Save results
    save_instance_results(all_results, str(output_dir / "instance_results.jsonl"))
    save_aggregate_metrics(aggregate_list, str(output_dir / "aggregate_metrics.json"))

    save_config_to_file(config, output_dir / "config_snapshot.yaml")
    save_metrics_csv(all_results, str(output_dir / "instance_metrics.csv"))

    save_metadata_table(
        [d.to_dict() for d in config.datasets],
        "datasets", str(output_dir / "dataset_metadata.json")
    )
    save_metadata_table(
        [m.to_dict() for m in config.models],
        "models", str(output_dir / "model_metadata.json")
    )

    if config.reproducibility.log_git_commit or config.reproducibility.log_package_versions:
        save_environment_snapshot(output_dir / "environment_snapshot.json")

    summary.end_time = datetime.now()
    summary.duration_seconds = (summary.end_time - summary.start_time).total_seconds()
    summary.api_requests_total = len(all_results) * 5
    summary.api_requests_failed = summary.failed_instances * 5
    summary.avg_time_per_instance = summary.duration_seconds / max(len(all_results), 1)
    summary.sampling_logs = [s.to_dict() for s in sampling_logs]

    with open(output_dir / "execution_summary.txt", 'w') as f:
        f.write(summary.generate_report())

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
