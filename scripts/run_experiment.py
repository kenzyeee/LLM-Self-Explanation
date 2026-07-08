import argparse
import asyncio
import hashlib
import json
import logging
import random
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.load.dataset_loader import DatasetLoader, Instance
from src.inference.inference_engine import InferenceEngine
from src.parsing.parser import Parser, dynamic_k, ensure_spacy_available
from src.normalization.normalizer import Normalizer
from src.metrics.metrics_calculator import MetricsCalculator
from src.statistics.statistical_tests import (
    sign_flip_permutation_test, holm_correction, compute_confidence_ecs_correlation,
)
from src.utils.config_loader import load_and_validate_config, parse_command_line_args, save_config_to_file
from src.utils.data_models import (
    InstanceResult, SamplingLog, AggregateMetrics, save_instance_results, save_aggregate_metrics,
    generate_md_report, save_metrics_csv, save_metadata_table, save_environment_snapshot,
)
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.execution_summary import ExecutionSummary
from src.utils.exceptions import (
    APIError, RateLimitExhausted, ParsingError, PromptValidationError,
)
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

STRATEGY_IDS = ["H", "R", "CF", "RO"]

REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i'm unable", "i am unable", "not able to",
    "i apologize", "i'm sorry", "i am sorry", "cannot fulfill",
    "cannot complete", "not appropriate", "i cannot provide",
    "i can't provide", "against policy", "not permitted",
]

_KNOWN_PLACEHOLDERS = re.compile(r'\{label_set\}|\{input_text\}|\{predicted_label\}|\{other_labels_quoted\}|\{other_labels\}')


def validate_prompt(prompt_text: str, label_set: Optional[List[str]] = None, context: str = "") -> None:
    """Reject prompts with unrendered placeholders. Only checks known template placeholders."""
    unrendered = _KNOWN_PLACEHOLDERS.findall(prompt_text)
    if unrendered:
        raise PromptValidationError(
            f"Unrendered placeholders in {context}: {list(set(unrendered))}",
            details={"placeholders": list(set(unrendered)), "prompt_preview": prompt_text[:200]}
        )


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


def quote_labels(labels: List[str]) -> str:
    """Quoted, 'or'-joined target labels for unambiguous CF prose.

    e.g. ['neutral', 'contradiction'] -> '"neutral" or "contradiction"'. Without the
    quotes, a multiclass target list reads ambiguously ("classified as neutral,
    contradiction instead of entailment"), which is what the MNLI CF prompt suffered.
    """
    return " or ".join(f'"{label}"' for label in labels)


def create_prompt_map(config, dataset_name: str = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Resolve and load every prompt used for a dataset.

    ``strategy.prompt_file`` names the EXECUTED elicitation prompt (the
    ``*_explain.txt`` file); dataset-specific and multiclass variants are derived
    from it by suffix. Returns ``(prompts, sources)`` where ``sources`` maps every
    prompt key to the file path actually loaded — the run's ``prompt_manifest.json``
    is generated from it, so the provenance record names the prompts that ran
    (review §1.3/§8: the old snapshot named base files that were never executed).
    """
    prompts: Dict[str, str] = {}
    sources: Dict[str, str] = {}

    def _load(key: str, path: str) -> None:
        prompts[key] = load_prompt(path)
        sources[key] = path

    for strategy in config.explanation_strategies:
        explain_file = strategy.prompt_file
        if not explain_file.endswith("_explain.txt"):
            raise PromptValidationError(
                f"Strategy {strategy.id}: prompt_file must be the executed '*_explain.txt' "
                f"prompt, got: {explain_file}")
        ds_specific_explain = None
        if dataset_name:
            ds_specific_explain = explain_file.replace("_explain.txt", f"_explain_{dataset_name}.txt")
            if not Path(ds_specific_explain).exists():
                ds_specific_explain = None
        _load(f"{strategy.id}_explain", ds_specific_explain or explain_file)
        multiclass_variant = explain_file.replace("_explain.txt", "_explain_multiclass.txt")
        ds_specific_multiclass = None
        if dataset_name:
            ds_specific_multiclass = explain_file.replace("_explain.txt", f"_explain_multiclass_{dataset_name}.txt")
            if not Path(ds_specific_multiclass).exists():
                ds_specific_multiclass = None
        multiclass_path = ds_specific_multiclass or multiclass_variant
        if Path(multiclass_path).exists():
            _load(f"{strategy.id}_explain_multiclass", multiclass_path)

    # CF-free (unconstrained) prompts — the secondary validity-minimality CONTRAST,
    # NOT used in ECS. The CF_explain / CF_explain_multiclass prompts are the MINIMAL
    # variant and are the canonical CF used for ECS pairs.
    cf_free_path = "prompts/counterfactual_explain_free.txt"
    if dataset_name and Path(f"prompts/counterfactual_explain_free_{dataset_name}.txt").exists():
        cf_free_path = f"prompts/counterfactual_explain_free_{dataset_name}.txt"
    if Path(cf_free_path).exists():
        _load("CF_explain_free", cf_free_path)
    cf_free_mc_path = "prompts/counterfactual_explain_free_multiclass.txt"
    if dataset_name and Path(f"prompts/counterfactual_explain_free_multiclass_{dataset_name}.txt").exists():
        cf_free_mc_path = f"prompts/counterfactual_explain_free_multiclass_{dataset_name}.txt"
    if Path(cf_free_mc_path).exists():
        _load("CF_explain_free_multiclass", cf_free_mc_path)

    # Verbalized confidence (0-100), elicited right after classification. The
    # no-logprob confidence signal (Tian et al. 2023; Xiong et al. 2024, ICLR).
    if getattr(config, "confidence", None) is not None and config.confidence.enabled:
        _load("confidence", config.confidence.prompt_file)

    class_prompt_path = "prompts/classification.txt"
    if dataset_name:
        ds_specific = f"prompts/classification_{dataset_name}.txt"
        if Path(ds_specific).exists():
            class_prompt_path = ds_specific
    _load("classification", class_prompt_path)

    return prompts, sources


def build_prompt_manifest(sources_by_dataset: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Machine-checkable prompt provenance: {dataset: {key: {file, sha256}}}."""
    manifest: Dict[str, Any] = {}
    for dataset_name, sources in sources_by_dataset.items():
        entry = {}
        for key, path in sorted(sources.items()):
            content = Path(path).read_text(encoding="utf-8")
            entry[key] = {
                "file": path,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        manifest[dataset_name] = entry
    return manifest


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
    engine.record_request("classification")

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

    # Real token accounting: accumulate the API's reported usage from EVERY call this
    # instance makes (classification, explanations, CF re-classification/corrections,
    # CF-minimal, confidence). count_tokens() string re-tokenization is gone.
    prompt_tokens = 0
    response_tokens = 0

    def _acct(usage):
        nonlocal prompt_tokens, response_tokens
        if usage is not None:
            prompt_tokens += usage.prompt_tokens
            response_tokens += usage.completion_tokens

    _acct(class_result.usage)
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
            confidence=None,
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

    # Per-strategy output budget, sourced from config (with a sane floor). The engine
    # auto-expands on truncation, but a reasonable starting budget avoids needless retries.
    # Highlighting scores EVERY word (~6-10 output tokens per ["word", score] entry), so
    # a flat doubled baseline is fine for short/medium inputs but silently truncates on
    # long ones (MNLI runs up to 206 words -> 1,300-2,000+ tokens needed, vs the 2048
    # a flat 2x baseline gives here) with truncation-retry hitting the SAME cap and
    # failing again — invisible in the pilot (max sampled ~50 words), but it would bias
    # "ECS by Input Length" by silently dropping exactly the long-input stratum (P0.2).
    # H's budget is therefore length-proportional: max(flat 2x baseline, 12*n_words+200).
    base_max_tokens = max((getattr(config.inference, "max_tokens", 512) or 512), 800)
    n_words_clean = len(clean_text.split())
    h_max_tokens = max(base_max_tokens * 2, 12 * n_words_clean + 200)
    truncated_strategies = []

    # Verbalized confidence (0-100 -> [0,1]) — elicited BEFORE any explanation strategy,
    # so it is conditioned only on the committed label (Tian et al. 2023; Xiong et al.
    # 2024). Failure to parse leaves confidence=None (unknown), never a fake 0.
    confidence_value: Optional[float] = None
    # Persisted regardless of outcome (review P1.4: confidence elicitation was
    # invisible in instance_results.jsonl and the per-instance report — every other
    # strategy's prompt/raw response is stored, but a paper claiming Tian/Xiong-style
    # elicitation needs this one too). Set as soon as each piece exists, so a partial
    # failure (e.g. API succeeds but parsing fails) still records what was sent/received.
    confidence_prompt = ""
    confidence_raw_response = ""
    if getattr(config, "confidence", None) is not None and config.confidence.enabled \
            and "confidence" in prompts:
        try:
            conf_prompt = format_explain_prompt(prompts["confidence"], predicted_label,
                                                input_text=clean_text)
            confidence_prompt = conf_prompt
            conf_messages = [
                {"role": "user", "content": class_prompt},
                {"role": "assistant", "content": class_result.raw_response},
                {"role": "user", "content": conf_prompt},
            ]
            conf_raw, conf_usage = await engine.chat_with_usage(conf_messages, max_tokens=200)
            confidence_raw_response = conf_raw or ""
            engine.record_request("confidence")
            _acct(conf_usage)
            confidence_value = parser.parse_confidence(conf_raw)
        except (ParsingError, json.JSONDecodeError) as e:
            logger.warning(f"Confidence elicitation unparseable for {instance_id}: {e}")
        except APIError as e:
            logger.warning(f"Confidence elicitation API error for {instance_id}: {e}")

    # Strategy elicitation order is FIXED (config order). Each strategy is an
    # independent 3-message conversation (classification prompt + label + one explain
    # prompt) — no shared context accumulates across strategies, so execution order
    # cannot influence the model, and randomizing it added unseeded nondeterminism
    # while controlling nothing (review §8.6).
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
    # Single-shot vs coached stratum (review §2.6): the correction loop makes reported
    # rates multi-shot-search rates; these flags let analyses stratify by whether the
    # first, uncoached elicitation already succeeded.
    cf_valid_first_attempt = False
    cf_corrected = False
    ro_self_corrected = False
    # Span-restricted CF (MNLI edits only the Hypothesis): validate the protected
    # prefix and compute the minimal-edit ratio over the editable span (review §8.6c).
    cf_span_marker = "Hypothesis:" if getattr(dataset_config, "secondary_text_field", None) else None
    # CF-free contrast state (D2) — always defined even if the branch never runs.
    cf_contrast_valid = False
    cf_contrast_flip_verified = False
    cf_contrast_minimality = None
    cf_contrast_tokens = set()
    cf_contrast_text = ""

    for strat_id in STRATEGY_IDS:
        raw_responses[strat_id] = ""
        explain_prompts[strat_id] = ""
        parsed_tokens[strat_id] = set()
        parsed_flags[strat_id] = False
        valid_flags[strat_id] = False

    for strat_id in STRATEGY_IDS:
        if strat_id == "CF":
            # Canonical CF for ECS is the MINIMAL contrastive edit (MiCE; Ross et al.
            # 2021): the smallest flip-inducing change is the informative attribution.
            # The unconstrained "free" variant is elicited separately below as a
            # validity-minimality contrast (Mayne et al. 2025).
            cf_prompt_key = "CF_explain"
            if len(label_set) > 2 and "CF_explain_multiclass" in prompts:
                cf_prompt_key = "CF_explain_multiclass"
            _cf_targets = [l for l in label_set if l != predicted_label]
            other_labels = ", ".join(_cf_targets)
            explain_prompt = format_explain_prompt(prompts[cf_prompt_key], predicted_label,
                                                   input_text=clean_text, other_labels=other_labels,
                                                   other_labels_quoted=quote_labels(_cf_targets))
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
            strat_max_tokens = h_max_tokens if strat_id == "H" else base_max_tokens
            raw, usage = await engine.chat_with_usage(messages, max_tokens=strat_max_tokens)
            engine.record_request(strat_id)
            _acct(usage)
            raw_responses[strat_id] = raw or ""
            if usage.truncated:
                truncated_strategies.append(strat_id)
                logger.warning(f"{instance_id} strategy {strat_id}: response truncated at token limit even after retry")
            if raw is None or not raw.strip():
                logger.warning(f"Empty response from model for {instance_id} strategy {strat_id}, retrying with max_tokens=2500...")
                raw, usage = await engine.chat_with_usage(messages, max_tokens=2500)
                engine.record_request(f"{strat_id}_empty_retry")
                _acct(usage)
                raw_responses[strat_id] = raw or ""
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
                # parse_highlighting returns NORMALIZED tokens ranked by salience —
                # the same canonical token space every other strategy's evidence uses,
                # so the H_ordered ranking is directly comparable with RO's normalized
                # ranking for Kendall τ / RBO (review §8.6b: raw-case H keys vs
                # normalized RO tokens zeroed rank agreement on capitalized datasets).
                tokens = parser.parse_highlighting(raw, clean_text, normalizer)
                normalized = set(tokens)
                if not normalized:
                    raise ParsingError("H evidence set is empty after normalization")
                parsed_tokens[strat_id] = normalized
                parsed_tokens["H_ordered"] = tokens
                # Full graded salience vector (normalized token -> max score) for the
                # salience-weighted secondary random baseline (review §2.5).
                parsed_tokens["H_salience_weights"] = dict(getattr(parser, "_h_salience_weights", {}) or {})
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
                    # Canonical CF is the MINIMAL edit — enforce the MiCE-style edit-ratio
                    # cap so non-minimal rewrites are rejected (a real flip and a non-empty
                    # change are also required). For span-restricted datasets (MNLI) the
                    # ratio is computed over the editable span and the Premise must be
                    # unchanged (review §8.6c).
                    cf_text_used, new_pred, from_tokens = parser.parse_counterfactual(
                        raw, clean_text, predicted_label, label_set, normalizer,
                        max_edit_ratio=cf_max_ratio, skip_validation=False,
                        edit_span_marker=cf_span_marker,
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
                        engine.record_request("CF_verify")
                        _acct(cf_class_result.usage)
                        cf_actual_label = parser.parse_classification(cf_class_result.raw_response, label_set)
                        cf_flip_verified = (cf_actual_label != predicted_label)
                        if cf_flip_verified:
                            cf_valid_first_attempt = (cf_attempt == 0)
                            cf_corrected = (cf_attempt > 0)
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
                            correction_raw, _cf_corr_usage = await engine.chat_with_usage(correction_messages, max_tokens=base_max_tokens)
                            engine.record_request("CF_correct")
                            _acct(_cf_corr_usage)
                            cf_raw_used = correction_raw
                            cf_json_obj2 = parser._extract_json(correction_raw)
                            if cf_json_obj2 is None:
                                logger.warning(f"CF correction for {instance_id}: correction JSON not parseable")
                                raise ParsingError("CF correction JSON not parseable")
                            cf_text_used, new_pred, from_tokens = parser.parse_counterfactual(
                                correction_raw, clean_text, predicted_label, label_set, normalizer,
                                max_edit_ratio=cf_max_ratio, skip_validation=False,
                                edit_span_marker=cf_span_marker,
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
                # CF evidence must live in the SAME normalized token space as H/R/RO
                # (review §8.3: raw difflib tokens retain stopwords the other sets can
                # never contain, deflating every CF pair and mis-modelling the null).
                # The raw edited surface tokens are kept separately for the erasure
                # pass and minimality accounting.
                cf_evidence = normalizer.normalize_tokens(sorted(from_tokens))
                parsed_tokens["CF_edited_raw"] = set(from_tokens)
                parsed_tokens["CF_reconstructed"] = cf_text_used
                parsed_tokens["CF_canonical_minimality"] = len(from_tokens) / max(len(clean_text.split()), 1)
                if not cf_evidence:
                    # Flip stats above remain informative; only the ECS attribution is
                    # unusable (the edit touched no content/polarity token).
                    raise ParsingError("CF evidence set is empty after normalization "
                                       "(edited tokens are all function words)")
                parsed_tokens[strat_id] = cf_evidence
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
                        correction_raw, _ro_corr_usage = await engine.chat_with_usage(correction_messages, max_tokens=base_max_tokens)
                        engine.record_request("RO_correct")
                        _acct(_ro_corr_usage)
                        ranked = parser.parse_rank_ordering(correction_raw, clean_text, normalizer)
                        ro_tokens = [t for t, r in ranked]
                        ro_self_corrected = True
                        if len(ro_tokens) >= 3:
                            logger.info(f"RO self-correction for {instance_id}: recovered {len(ro_tokens)} valid tokens")
                        else:
                            logger.warning(f"RO self-correction for {instance_id}: still insufficient valid tokens")
                    except Exception as e:
                        logger.warning(f"RO self-correction for {instance_id} failed: {e}")
                # Use the same length-proportional top-k as Highlighting for the
                # set-overlap comparison (H and RO are the same extraction paradigm);
                # the full ranked list is retained below for Kendall τ / RBO.
                k_ro = dynamic_k(clean_text, cap=len(ro_tokens))
                normalized_set = normalizer.normalize_tokens(ro_tokens[:k_ro])
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
        except Exception as e:
            # Safety net: an unexpected error in ONE strategy's parse must not discard the
            # whole instance (and the other strategies' valid evidence + spent API tokens).
            # Logged loudly (with traceback) so it is never silently masked.
            logger.error(f"Unexpected error parsing {instance_id} strategy {strat_id}: {e}", exc_info=True)

    # cf_flip_verified reports the flip-verification stage's own outcome. It is NOT
    # conjoined with the evidence-validity flags: a verified flip whose edit touched
    # only function words is a real flip with unusable ECS attribution — staged
    # reporting keeps those two facts separate.
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

    # Vocabulary size in the SAME normalized token space the evidence sets live in
    # (review P0.1): both nulls (uniform expected_random_overlap and the exact
    # hypergeometric expected_jaccard_exact behind ecs_adj) model "random draws from
    # the vocabulary the strategies select from", so V must be counted over normalized
    # content lemmas — stopwords/discourse removed, polarity kept, lemmatized to the
    # fixed point — NOT raw surface words. Counting surface words inflated V ~1.4-1.9x
    # (dataset-differentially), mis-centering every chance correction (ecs_random/
    # ecs_lift and ecs_adj's E[J]) and making the pre-registered lift test
    # anti-conservative. STRUCTURAL_LABELS are excluded explicitly (they also happen to
    # be DISCOURSE_WORDS the normalizer drops, but the guard keeps intent visible).
    input_tokens = normalizer.normalize_input_text(clean_text).split()
    STRUCTURAL_LABELS = {"premise:", "hypothesis:", "sentence1:", "sentence2:", "text:", "label:"}
    vocab_tokens = set()
    for t in input_tokens:
        if t in STRUCTURAL_LABELS:
            continue
        norm = normalizer.normalize(t)
        if norm:
            vocab_tokens.add(norm)
    # Support-closure guarantee (review P1.1, 2026-07-08): union in every token any
    # strategy actually selected, so the hypergeometric null's urn provably contains
    # the atoms the strategies drew from. Whitespace tokenization with edge-punctuation
    # stripping keeps a few evidence atoms out of the input-derived vocab (AG News'
    # glued ellipses "senate...supercomputer" -> one vocab token but "senate" and
    # "supercomputer" as separate evidence; possessives "turkey's" vs "turkey"), which
    # both under-counts V and, worse, leaves the null modelling draws from an urn that
    # cannot contain the selected token — a reviewer attack on the chance correction.
    # Measured pilot impact (recomputed from raw, 2026-07-08): 64 evidence tokens across
    # 42 instance-strategy combos were outside the urn. Unioning them in moves the
    # POOLED estimand negligibly (complete-case +0.4413 -> +0.4392; available
    # +0.4742 -> +0.4823, both <0.01) and each population gains exactly one instance as
    # a larger V resolves a borderline degenerate pair. Per-CELL movement is larger than
    # the plan's original "<=0.001" estimate implied — 6/9 complete-case cell means move
    # by >0.001 (worst ~-0.055 at N=17 and +0.062 at N=2), and the largest single
    # instance swings ~0.87 (a small-V instance whose pair crosses the degeneracy guard).
    # No cell changes sign and no tested cell changes significance, so conclusions are
    # invariant — but this is a real support-closure change on small-V instances, not a
    # cosmetic <=0.001 nudge. The evidence sets are already in the same normalized lemma
    # space as vocab_tokens, so this is a pure set union.
    for _ev in explanations.values():
        vocab_tokens |= _ev
    vocab_size = len(vocab_tokens)
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
    ecs_overlap_value = None
    ecs_extraction_rationale = None
    ecs_extraction_perturbation = None
    ecs_primary_pairs = 0
    if n_valid >= 3:
        ecs_value = calc.compute_ecs(agreements)
        # Size-robust secondary composite over the same cross-paradigm pairs
        # (overlap coefficient — immune to the Jaccard set-size ceiling, §2.2).
        ecs_overlap_value = calc.compute_ecs_overlap(overlaps)
        ecs_extraction_rationale, ecs_extraction_perturbation, ecs_primary_pairs = calc.compute_ecs_primary(agreements)
    else:
        logger.warning(f"Only {n_valid} valid strategies for {instance_id} — ECS not computed")

    # ECS-adj (ECS_ROBUSTNESS_PLAN_2026-07-05.md): chance- and ceiling-adjusted,
    # paradigm-balanced composite computed alongside legacy ECS (never replacing it
    # pre-adoption). Independent of the n_valid>=3 gate above — it degrades
    # gracefully per-component (a missing/degenerate pair just drops out of its
    # component's mean), so it is attempted whenever the vocabulary is known.
    ecs_adj_eps = getattr(getattr(config, "metrics", None), "ecs_adj_epsilon", 0.10)
    ecs_adj_result = calc.compute_ecs_adjusted(explanations, vocab_size, eps=ecs_adj_eps)

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

    # Secondary, salience-weighted null (review §2.5): the uniform null understates
    # chance agreement when all methods gravitate to the same few high-salience
    # tokens for reasons unrelated to consensus. Sampling ∝ H's own graded salience
    # over the content vocabulary gives a harder null; lift over it is the
    # conservative secondary lift.
    ecs_random_weighted = None
    ecs_lift_weighted = None
    h_weights = parsed_tokens.get("H_salience_weights") or {}
    if ecs_value is not None and len(h_weights) >= 2:
        _excluded_w = {("H", "RO")}
        _strats_w = ["H", "R", "CF", "RO"]
        _wvals = []
        for _i in range(len(_strats_w)):
            for _j in range(_i + 1, len(_strats_w)):
                s1, s2 = _strats_w[_i], _strats_w[_j]
                if (s1, s2) in _excluded_w:
                    continue
                set1, set2 = explanations.get(s1, set()), explanations.get(s2, set())
                if set1 and set2:
                    ew = calc.expected_random_overlap_weighted(len(set1), len(set2), h_weights)
                    if ew is not None:
                        _wvals.append(ew)
        if _wvals:
            ecs_random_weighted = sum(_wvals) / len(_wvals)
            ecs_lift_weighted = ecs_value - ecs_random_weighted

    # D2: CF-free (unconstrained) variant — the validity-minimality CONTRAST to the
    # canonical minimal CF (Mayne et al. 2025). Elicited separately, NOT used in ECS;
    # it reliably flips but is far from minimal, which is exactly the contrast we report.
    if predicted_label:
        cf_free_key = "CF_explain_free" if "CF_explain_free" in prompts else None
        if len(label_set) > 2 and "CF_explain_free_multiclass" in prompts:
            cf_free_key = "CF_explain_free_multiclass"
        if cf_free_key and cf_free_key in prompts:
            try:
                _free_targets = [l for l in label_set if l != predicted_label]
                _other = ", ".join(_free_targets)
                cf_free_prompt = format_explain_prompt(prompts[cf_free_key], predicted_label,
                                                       input_text=clean_text, other_labels=_other,
                                                       other_labels_quoted=quote_labels(_free_targets))
                cf_free_messages = [
                    {"role": "user", "content": class_prompt},
                    {"role": "assistant", "content": class_result.raw_response},
                    {"role": "user", "content": cf_free_prompt},
                ]
                cf_free_raw, _cf_free_usage = await engine.chat_with_usage(cf_free_messages, max_tokens=base_max_tokens)
                engine.record_request("CF_free")
                _acct(_cf_free_usage)
                # Unconstrained: no edit-ratio cap (skip_validation=True); a real flip and
                # a non-empty change are still required. The span restriction (Premise
                # unchanged) still applies on span-restricted datasets — the free
                # variant relaxes minimality, not the editable region.
                cf_free_text, _cf_free_pred, cf_free_from = parser.parse_counterfactual(
                    cf_free_raw, clean_text, predicted_label, label_set, normalizer,
                    skip_validation=True, edit_span_marker=cf_span_marker,
                )
                cf_contrast_text = cf_free_text
                cf_contrast_minimality = len(cf_free_from) / max(len(clean_text.split()), 1)
                cf_free_class_prompt = format_prompt(prompts["classification"], cf_free_text, label_set)
                cf_free_class_result = await engine.classify(cf_free_class_prompt)
                engine.record_request("CF_free_verify")
                _acct(cf_free_class_result.usage)
                cf_free_actual = parser.parse_classification(cf_free_class_result.raw_response, label_set)
                cf_contrast_flip_verified = (cf_free_actual != predicted_label)
                cf_contrast_valid = cf_contrast_flip_verified
                if cf_contrast_valid:
                    cf_contrast_tokens = cf_free_from
            except (ParsingError, json.JSONDecodeError) as e:
                logger.info(f"CF-free contrast for {instance_id}: invalid ({e})")
            except APIError as e:
                logger.warning(f"CF-free contrast for {instance_id}: API error ({e})")

    result = InstanceResult(
        instance_id=instance_id,
        dataset=instance.dataset,
        model=engine.model_name,
        timestamp=datetime.now(),
        text=text,
        input_length=len(text.split()),
        ground_truth_label=instance.label,
        predicted_label=predicted_label,
        confidence=confidence_value,
        confidence_prompt=confidence_prompt,
        confidence_raw_response=confidence_raw_response,
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
        rank_ordering_set=ro_set,
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
        ecs_overlap=ecs_overlap_value,
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
        cf_valid_first_attempt=cf_valid_first_attempt,
        cf_corrected=cf_corrected,
        ro_self_corrected=ro_self_corrected,
        cf_actual_label=cf_actual_label,
        cf_counterfactual_text=cf_counterfactual_text,
        cf_edited_tokens_raw=parsed_tokens.get("CF_edited_raw", set()),
        ecs_random=ecs_random,
        ecs_lift=ecs_lift,
        ecs_random_weighted=ecs_random_weighted,
        ecs_lift_weighted=ecs_lift_weighted,
        cf_canonical_minimality=parsed_tokens.get("CF_canonical_minimality"),
        cf_contrast_valid=cf_contrast_valid,
        cf_contrast_flip_verified=cf_contrast_flip_verified,
        cf_contrast_minimality=cf_contrast_minimality,
        cf_contrast_tokens=cf_contrast_tokens,
        cf_contrast_text=cf_contrast_text,
        r_introduced_concept_rate=r_introduced_concept_rate,
        ecs_adj_er=ecs_adj_result["ecs_adj_er"],
        ecs_adj_ep=ecs_adj_result["ecs_adj_ep"],
        ecs_adj_rp=ecs_adj_result["ecs_adj_rp"],
        ecs_adj=ecs_adj_result["ecs_adj"],
        ecs_adj_n_components=ecs_adj_result["ecs_adj_n_components"],
        ecs_adj_complete=ecs_adj_result["ecs_adj_complete"],
        n_degenerate_pairs=ecs_adj_result["n_degenerate_pairs"],
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

    # NOTE: the former inline "redaction test" (per-token progressive re-classification
    # of H/RO rankings during collection) was removed (FIX_PLAN §P3.1): it duplicated
    # the dedicated post-hoc erasure pass (scripts/run_validity_tests.py) with a
    # coarser ad-hoc metric on only 2 of 4 strategies, at the cost of up to k extra
    # API calls per strategy per instance, and its "faithfulness" naming leaked the
    # reading the study's framing forbids. Erasure has ONE instrument now.

    # Re-sync token totals: confidence/CF calls were accounted into the local
    # accumulator AFTER the result object was constructed, so refresh the fields.
    result.prompt_tokens = prompt_tokens
    result.response_tokens = response_tokens
    result.truncated_strategies = list(truncated_strategies)

    return result


def compute_free_cf_sensitivity_ecs(results: List[InstanceResult], calc: MetricsCalculator,
                                    normalizer: Optional[Normalizer] = None) -> Tuple[float, int]:
    """Sensitivity analysis (review P1.1): ECS recomputed with cf_contrast_tokens (the
    unconstrained/free CF rewrite — ~82% validity in the pilot) substituted for the
    canonical minimal-CF evidence (~28% validity, gated on counterfactual_valid) in
    the H-CF/CF-RO/R-CF pairs. Pure post-processing over fields already stored on
    every InstanceResult — zero additional API calls. Answers whether conclusions
    survive when the perturbation paradigm isn't gated by the minimal-edit
    constraint. Descriptive robustness check only: NOT a primary estimand, NOT
    NHST-tested, and NOT used for complete-case selection anywhere else — minimal-CF
    ECS remains primary precisely because minimality is part of the CF construct
    (MiCE); this analysis exists to show the alternative doesn't overturn the story.

    cf_contrast_tokens are stored as RAW edited surface words (the difflib diff output),
    so they MUST be projected into the shared normalized token space before Jaccarding
    them against H/R/RO — which are already normalized (review §8.3/P0.4). Comparing raw
    surface tokens (retaining stopwords/inflections the other sets can never contain)
    against normalized sets structurally deflates every free-CF pair, which would falsely
    read as "conclusions don't survive without the minimal-edit gate" — the opposite of
    this check's purpose. The normalizer defaults to the v3.0 shared-space settings.
    """
    if normalizer is None:
        normalizer = Normalizer()
    values = []
    for r in results:
        if not (r.highlighting_valid and r.rationale_valid and r.rank_ordering_valid
                and r.cf_contrast_valid and r.cf_contrast_tokens):
            continue
        cf_set = normalizer.normalize_tokens(list(r.cf_contrast_tokens))
        if not cf_set:
            continue
        # Use the SAME top-k RO evidence set primary ECS used (review P1.1), falling
        # back to the full ranked set for legacy records without rank_ordering_set.
        ro_set = r.rank_ordering_set or {t for t, _ in r.rank_ordering_tokens}
        explanations = {"H": r.highlighting_tokens, "R": r.rationale_tokens,
                        "CF": cf_set, "RO": ro_set}
        agreements = calc.compute_pairwise_agreements(explanations)
        ecs = calc.compute_ecs(agreements)
        if ecs is not None:
            values.append(ecs)
    import numpy as np
    return (float(np.mean(values)) if values else 0.0), len(values)


def compute_free_cf_sensitivity_ecs_adj(results: List[InstanceResult], calc: MetricsCalculator,
                                        normalizer: Optional[Normalizer] = None,
                                        eps: float = 0.10) -> Tuple[float, int, float, int]:
    """Free-CF sensitivity on the PRIMARY (ECS-adj) scale — the pre-registered analysis
    the plan §3.4 requires ("the free-CF sensitivity ... should also be computed in AJ
    form"), now that ECS-adj is primary (P0.3, 2026-07-08). Same substitution as
    compute_free_cf_sensitivity_ecs (the unconstrained/free CF rewrite replaces the
    minimal-CF evidence in the perturbation pairs), but scored with compute_ecs_adjusted
    instead of the legacy flat ECS, so the MNAR robustness check lives on the same
    chance- AND ceiling-corrected scale as the headline estimand.

    Returns (mean_complete, n_complete, mean_available, n_available):
      * complete = ecs_adj over rows where all three paradigm components are defined
        with the free CF substituted (mirrors the primary complete-case estimand);
      * available = ecs_adj over every row with a defined free-CF ecs_adj.
    Pure post-processing over already-stored fields — zero additional API calls.
    Descriptive robustness check only: NOT a primary estimand, NOT NHST-tested (see
    compute_free_cf_sensitivity_ecs's docstring for why minimal-CF stays primary).

    cf_contrast_tokens are stored as RAW edited surface words, so they are projected
    into the shared normalized token space before scoring (same reason as the legacy
    version — comparing raw surface tokens against normalized sets structurally
    deflates every free-CF pair).
    """
    if normalizer is None:
        normalizer = Normalizer()
    complete_vals = []
    avail_vals = []
    for r in results:
        if not (r.highlighting_valid and r.rationale_valid and r.rank_ordering_valid
                and r.cf_contrast_valid and r.cf_contrast_tokens):
            continue
        cf_set = normalizer.normalize_tokens(list(r.cf_contrast_tokens))
        if not cf_set:
            continue
        ro_set = r.rank_ordering_set or {t for t, _ in r.rank_ordering_tokens}
        explanations = {"H": r.highlighting_tokens, "R": r.rationale_tokens,
                        "CF": cf_set, "RO": ro_set}
        res = calc.compute_ecs_adjusted(explanations, r.vocab_size, eps=eps)
        if res["ecs_adj"] is not None:
            avail_vals.append(res["ecs_adj"])
            if res["ecs_adj_complete"]:
                complete_vals.append(res["ecs_adj"])
    import numpy as np
    return (
        float(np.mean(complete_vals)) if complete_vals else 0.0, len(complete_vals),
        float(np.mean(avail_vals)) if avail_vals else 0.0, len(avail_vals),
    )


def compute_aggregate_metrics(results: List[InstanceResult], level: str, group: str,
                               sampling_log: Optional[SamplingLog] = None,
                               normalizer: Optional[Normalizer] = None) -> AggregateMetrics:
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

    # Bootstrap CI on mean ECS. Pooled levels ("overall", "dataset") contain up to
    # one row per (instance, model): the SAME instance appears under every model, so
    # rows are CLUSTERED by instance and a row-level bootstrap treats correlated rows
    # as independent (CI too narrow — review §8.6a). Resample instance clusters for
    # pooled levels; row-level for per-model levels where rows are distinct instances.
    if len(ecs_values) > 1:
        rng = np.random.default_rng(42)
        boot_means = []
        if level in ("overall", "dataset"):
            from collections import defaultdict
            clusters = defaultdict(list)
            for r in results:
                if r.ecs is not None:
                    clusters[(r.dataset, r.instance_id)].append(r.ecs)
            cluster_vals = list(clusters.values())
            n_clusters = len(cluster_vals)
            for _ in range(1000):
                idx = rng.integers(0, n_clusters, size=n_clusters)
                sample = [v for i in idx for v in cluster_vals[i]]
                boot_means.append(float(np.mean(sample)))
        else:
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

    # Free-CF sensitivity analysis (review P1.1): descriptive robustness check, zero
    # extra API cost — see compute_free_cf_sensitivity_ecs's docstring.
    _free_cf_ecs, _n_free_cf = compute_free_cf_sensitivity_ecs(results, MetricsCalculator(), normalizer)
    # Free-CF sensitivity on the PRIMARY (ECS-adj) scale (P0.3, 2026-07-08): the plan
    # §3.4-required AJ-form version of the MNAR robustness check. complete = primary
    # scale; available = wider-N companion. Zero extra API cost.
    (_free_cf_adj_c, _n_free_cf_adj_c,
     _free_cf_adj_a, _n_free_cf_adj_a) = compute_free_cf_sensitivity_ecs_adj(
        results, MetricsCalculator(), normalizer)

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

    # ECS-adj aggregation (ECS_ROBUSTNESS_PLAN_2026-07-05.md §3.4 missing-data policy):
    # complete-case mean over ecs_adj_complete==True rows only (candidate primary
    # estimand, mirrors the legacy n_valid_strategies==4 gate above), plus the
    # available-component secondary over every row with a defined ecs_adj.
    ecs_adj_values = [r.ecs_adj for r in results if r.ecs_adj is not None]
    ecs_adj_complete_values = [r.ecs_adj for r in results if r.ecs_adj_complete and r.ecs_adj is not None]
    # ECS-adj length/vocab strata (P1.2): same buckets as legacy ECS strata but on the
    # primary scale — the brevity/short-vocab confounds present in raw ECS should NOT
    # reproduce here (the adjustment removes exactly those). Computed over rows with a
    # defined ecs_adj (available-component) so the strata Ns track the primary scale.
    def _adj_len_bucket(bucket):
        vals = [r.ecs_adj for r in results if r.ecs_adj is not None
                and MetricsCalculator.classify_length(r.text) == bucket]
        return (float(np.mean(vals)) if vals else 0.0), len(vals)
    adj_short_mean, adj_short_n = _adj_len_bucket("short")
    adj_medium_mean, adj_medium_n = _adj_len_bucket("medium")
    adj_long_mean, adj_long_n = _adj_len_bucket("long")
    adj_normal_vocab = [r.ecs_adj for r in results if r.ecs_adj is not None and not r.short_vocab]
    adj_short_vocab = [r.ecs_adj for r in results if r.ecs_adj is not None and r.short_vocab]
    ecs_adj_er_values = [r.ecs_adj_er for r in results if r.ecs_adj_er is not None]
    ecs_adj_ep_values = [r.ecs_adj_ep for r in results if r.ecs_adj_ep is not None]
    ecs_adj_rp_values = [r.ecs_adj_rp for r in results if r.ecs_adj_rp is not None]
    n_degenerate_pairs_total = sum(r.n_degenerate_pairs for r in results)

    # D1 lift / D2 CF trade-off / D6 introduced-concept rate / correctness split
    ecs_lift_values = [r.ecs_lift for r in results if r.ecs_lift is not None]
    ecs_random_values = [r.ecs_random for r in results if r.ecs_random is not None]
    ecs_lift_weighted_values = [r.ecs_lift_weighted for r in results if r.ecs_lift_weighted is not None]
    ecs_overlap_values = [r.ecs_overlap for r in results if r.ecs_overlap is not None]
    icr_values = [r.r_introduced_concept_rate for r in results if r.r_introduced_concept_rate is not None]
    cf_canonical_min_values = [r.cf_canonical_minimality for r in results if r.cf_canonical_minimality is not None]
    cf_contrast_min_values = [r.cf_contrast_minimality for r in results if r.cf_contrast_minimality is not None]
    # Correct-vs-incorrect ECS is only meaningful WITHIN a model×dataset cell —
    # pooled, it is confounded by dataset/model composition (review §2.11: all
    # incorrect instances came from one dataset, so the pooled contrast was a
    # dataset effect wearing a correctness costume).
    if level == "model_dataset":
        ecs_correct_vals = [r.ecs for r in results if r.ecs is not None and r.correct]
        ecs_incorrect_vals = [r.ecs for r in results if r.ecs is not None and not r.correct]
    else:
        ecs_correct_vals = []
        ecs_incorrect_vals = []
    n_results_safe = max(len(results), 1)
    cf_canonical_validity_rate = sum(1 for r in results if r.counterfactual_valid) / n_results_safe
    cf_contrast_validity_rate = sum(1 for r in results if r.cf_contrast_valid) / n_results_safe
    # Single-shot vs coached stratum (review §2.6): validity of the FIRST, uncoached
    # elicitation, and how many valid CFs needed the correction loop.
    cf_first_attempt_validity_rate = sum(1 for r in results if r.cf_valid_first_attempt) / n_results_safe
    cf_corrected_count = sum(1 for r in results if r.cf_corrected)
    ro_self_corrected_count = sum(1 for r in results if r.ro_self_corrected)
    # Verbalized-confidence <-> ECS association (Spearman + seeded bootstrap CI);
    # computed per cell over instances with both quantities present.
    conf_pairs = [(r.confidence, r.ecs, r.instance_id) for r in results
                  if r.confidence is not None and r.ecs is not None]
    if len(conf_pairs) >= 3:
        # Pooled levels (overall/dataset) hold one row PER MODEL for the same
        # instance; cluster the bootstrap by instance_id there so correlated rows
        # aren't resampled as if independent (review P2.3 — the same fix already
        # applied to the pooled ECS bootstrap CI below).
        cluster_ids = [iid for _, _, iid in conf_pairs] if level in ("overall", "dataset") else None
        corr = compute_confidence_ecs_correlation([c for c, _, _ in conf_pairs],
                                                  [e for _, e, _ in conf_pairs],
                                                  cluster_ids=cluster_ids)
        spearman_rho, spearman_p = corr.rho, corr.p_value
        corr_ci_lo, corr_ci_hi = corr.ci_lower, corr.ci_upper
        kendall_tau_b_confidence = corr.kendall_tau_b
    else:
        spearman_rho, spearman_p, corr_ci_lo, corr_ci_hi = 0.0, 1.0, 0.0, 0.0
        kendall_tau_b_confidence = None
    conf_values = [c for c, _, _ in conf_pairs]
    # Per-pair coverage: how many instances actually contributed to each pairwise mean
    # (a pooled pairwise mean over 3 instances reads very differently than over 300).
    pair_ns = {}
    for key in ("jaccard_H_R", "jaccard_H_CF", "jaccard_H_RO",
                "jaccard_R_CF", "jaccard_R_RO", "jaccard_CF_RO",
                "rbo_H_RO", "kendall_H_RO", "normalized_kendall_H_RO"):
        pair_ns[key] = sum(1 for r in results if getattr(r, key) is not None)

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
        wrong_pred_kept=sampling_log.wrong_predictions if sampling_log else 0,
        dropped_other=sampling_log.dropped_by_reason if sampling_log else {},
        std_ecs=float(np.std(ecs_values, ddof=1)) if len(ecs_values) > 1 else 0.0,
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
        spearman_rho=spearman_rho,
        spearman_p_value=spearman_p,
        correlation_ci_lower=corr_ci_lo,
        correlation_ci_upper=corr_ci_hi,
        kendall_tau_b_confidence=kendall_tau_b_confidence,
        n_confidence=len(conf_values),
        mean_confidence=safe_mean(conf_values),
        highlighting_success_rate=sum(1 for r in results if r.highlighting_parsed) / max(len(results), 1),
        rationale_success_rate=sum(1 for r in results if r.rationale_parsed) / max(len(results), 1),
        counterfactual_success_rate=sum(1 for r in results if r.counterfactual_parsed) / max(len(results), 1),
        rank_ordering_success_rate=sum(1 for r in results if r.rank_ordering_parsed) / max(len(results), 1),
        mean_ecs_lift=safe_mean(ecs_lift_values),
        mean_ecs_random=safe_mean(ecs_random_values),
        mean_ecs_lift_weighted=safe_mean(ecs_lift_weighted_values),
        n_lift_weighted=len(ecs_lift_weighted_values),
        mean_ecs_overlap=safe_mean(ecs_overlap_values),
        n_lift=len(ecs_lift_values),
        introduced_concept_rate=safe_mean(icr_values),
        cf_canonical_validity_rate=cf_canonical_validity_rate,
        cf_contrast_validity_rate=cf_contrast_validity_rate,
        cf_first_attempt_validity_rate=cf_first_attempt_validity_rate,
        n_cf_corrected=cf_corrected_count,
        n_ro_self_corrected=ro_self_corrected_count,
        mean_cf_canonical_minimality=safe_mean(cf_canonical_min_values),
        mean_cf_contrast_minimality=safe_mean(cf_contrast_min_values),
        mean_ecs_correct=safe_mean(ecs_correct_vals),
        n_correct=len(ecs_correct_vals),
        mean_ecs_incorrect=safe_mean(ecs_incorrect_vals),
        n_incorrect=len(ecs_incorrect_vals),
        pair_ns=pair_ns,
        mean_ecs_free_cf=_free_cf_ecs,
        n_free_cf=_n_free_cf,
        mean_ecs_adj_free_cf_complete=_free_cf_adj_c,
        n_ecs_adj_free_cf_complete=_n_free_cf_adj_c,
        mean_ecs_adj_free_cf=_free_cf_adj_a,
        n_ecs_adj_free_cf=_n_free_cf_adj_a,
        mean_ecs_adj=safe_mean(ecs_adj_values),
        n_ecs_adj=len(ecs_adj_values),
        mean_ecs_adj_complete=safe_mean(ecs_adj_complete_values),
        n_ecs_adj_complete=len(ecs_adj_complete_values),
        mean_ecs_adj_er=safe_mean(ecs_adj_er_values),
        mean_ecs_adj_ep=safe_mean(ecs_adj_ep_values),
        mean_ecs_adj_rp=safe_mean(ecs_adj_rp_values),
        n_degenerate_pairs_total=n_degenerate_pairs_total,
        mean_ecs_adj_short=adj_short_mean,
        n_ecs_adj_short=adj_short_n,
        mean_ecs_adj_medium=adj_medium_mean,
        n_ecs_adj_medium=adj_medium_n,
        mean_ecs_adj_long=adj_long_mean,
        n_ecs_adj_long=adj_long_n,
        mean_ecs_adj_normal_vocab=safe_mean(adj_normal_vocab),
        n_ecs_adj_normal_vocab=len(adj_normal_vocab),
        mean_ecs_adj_short_vocab=safe_mean(adj_short_vocab),
        n_ecs_adj_short_vocab=len(adj_short_vocab),
    )


def _load_checkpointed_results(output_dir: Path, dataset_name: str, model_name: str) -> List[InstanceResult]:
    """Load already-completed instances for one (dataset, model) pair from an
    existing run's checkpoint file, so a resumed run does not re-process them.

    De-duplicates by instance_id (last occurrence wins) as a defensive guard —
    normal operation never appends the same instance twice (see last_checkpointed
    in _run_model_on_dataset), but a corrupted/hand-edited checkpoint must not
    silently double-count an instance in the resumed aggregates.
    """
    cp_path = output_dir / f"checkpoint_{dataset_name}_{model_name}.jsonl"
    if not cp_path.exists():
        return []
    by_id: Dict[str, InstanceResult] = {}
    with open(cp_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = InstanceResult.from_dict(json.loads(line))
                by_id[r.instance_id] = r
    return list(by_id.values())


async def _run_model_on_dataset(model_config, instances, prompts, dataset_config,
                                parser, normalizer, calc, config, output_dir,
                                existing_results: Optional[List[InstanceResult]] = None,
                                force_restart: bool = False):
    """Run one model over every instance of one dataset and return a result bundle.

    Isolated so the models configured for a run can execute concurrently
    (asyncio.gather) rather than one after another — each call owns its engine,
    checkpoint file, and result list and shares no mutable state with its siblings.
    Instances within a model are processed in order; cross-model concurrency combined
    with each engine's own request semaphore bounds the total in-flight Bedrock calls.

    ``existing_results`` (from a prior, interrupted process's checkpoint file) seeds
    the result list and their instance_ids are skipped — this is what makes
    scripts/resume_experiment.py's resume cheap: only genuinely unfinished instances
    incur new API calls. ``force_restart`` discards them and clears the checkpoint
    file instead (used by --force-restart).
    """
    dataset_name = dataset_config.name
    tag = f"{model_config.name}/{dataset_name}"

    seed_results = [] if force_restart else list(existing_results or [])
    already_done = {r.instance_id for r in seed_results}
    remaining_instances = [inst for inst in instances if inst.instance_id not in already_done]
    if already_done:
        logger.info(f"[{tag}] Resuming: {len(already_done)}/{len(instances)} instance(s) already "
                    f"checkpointed, {len(remaining_instances)} remaining")
    logger.info(f"Processing model: {model_config.name} on {dataset_name}"
                + (" (resumed)" if already_done else ""))

    engine = InferenceEngine(
        model_name=model_config.model_id,
        max_retries=config.inference.max_retries,
        concurrent_requests=config.inference.concurrent_requests,
        context_window=getattr(model_config, "context_window", 8192),
    )

    model_results: List[InstanceResult] = list(seed_results)
    wrong_pred_count = sum(1 for r in seed_results if not r.correct)
    successful = len(seed_results)
    failed = 0
    prompt_validation_failures = 0
    cp = CheckpointManager(output_dir / f"checkpoint_{dataset_name}_{model_config.name}.jsonl",
                           force_restart=force_restart)
    # Seeded results are already durably persisted from the prior process — start
    # the "unflushed" offset past them so _flush_checkpoint never re-appends
    # already-checkpointed lines (which would double-count them on the NEXT resume).
    last_checkpointed = len(model_results)

    def _flush_checkpoint():
        nonlocal last_checkpointed
        new_results = model_results[last_checkpointed:]
        if new_results:
            cp.save_checkpoint([r.to_dict() for r in new_results])
            last_checkpointed = len(model_results)

    for i, instance in enumerate(remaining_instances):
        progress = len(already_done) + i + 1
        logger.info(f"[{model_config.name}] Processing {instance.instance_id} ({progress}/{len(instances)})")
        try:
            result = await process_instance(
                instance, engine, parser, normalizer, calc, prompts, config, dataset_config
            )
            model_results.append(result)
            successful += 1
            if not result.correct:
                wrong_pred_count += 1
        except RateLimitExhausted as e:
            # A sustained rate/quota exhaustion (all configured retries+backoff already
            # failed) will not clear up by immediately trying the NEXT instance — Bedrock
            # daily-token-quota exhaustion persists for hours, so retrying instance after
            # instance only burns wall-clock on guaranteed failures. Flush what succeeded
            # and STOP this model's loop; scripts/resume_experiment.py continues later
            # from exactly this point instead of the process churning through the rest.
            logger.error(f"[{model_config.name}] Rate limit exhausted for {instance.instance_id}: {e}")
            failed += 1
            _flush_checkpoint()
            logger.warning(f"[{model_config.name}] Stopping early after sustained rate limiting — "
                           f"{last_checkpointed}/{len(instances)} instances saved. Resume later with: "
                           f"python scripts/resume_experiment.py {output_dir.name}")
            break
        except PromptValidationError as e:
            logger.error(f"[{model_config.name}] Prompt validation failed for {instance.instance_id}: {e}")
            prompt_validation_failures += 1
            failed += 1
            continue
        except Exception as e:
            logger.error(f"[{model_config.name}] Failed to process {instance.instance_id}: {e}")
            failed += 1
            continue

        if (i + 1) % config.output.checkpoint_frequency == 0:
            _flush_checkpoint()

    # Final flush: guarantees any tail batch smaller than checkpoint_frequency is
    # durably on disk before this coroutine returns, not just held in memory pending
    # run_experiment()'s end-of-run write — a crash between here and there must not
    # lose newly-completed instances that a resume could otherwise have skipped.
    _flush_checkpoint()

    # Cross-check: the engine's authoritative cumulative usage vs. the sum attributed
    # to per-instance records. A divergence flags an uncounted call path.
    engine_total = engine.total_prompt_tokens + engine.total_completion_tokens
    instance_total = sum(r.prompt_tokens + r.response_tokens for r in model_results)
    logger.info(f"Token usage for {tag}: engine={engine_total} "
                f"(prompt={engine.total_prompt_tokens}, completion={engine.total_completion_tokens}), "
                f"per-instance sum={instance_total}, truncated_calls={engine.n_truncated}")
    if engine_total and abs(engine_total - instance_total) > 0.05 * engine_total:
        logger.warning(f"Token attribution mismatch for {tag}: "
                       f"engine={engine_total} vs per-instance={instance_total} — an API call path may be unaccounted.")

    # Per-(model,dataset) aggregate uses THIS model's own sampling log, so its
    # wrong_pred_kept reflects only this model rather than a cross-model tally.
    model_slog = SamplingLog(
        dataset=dataset_name,
        model=model_config.name,
        requested=dataset_config.sample_size,
        sampled=len(instances),
        wrong_predictions=wrong_pred_count,
    )
    agg = compute_aggregate_metrics(
        model_results, "model_dataset", f"{model_config.name}_{dataset_name}", sampling_log=model_slog,
        normalizer=normalizer,
    )

    return {
        "results": model_results,
        "agg": agg,
        "wrong_pred_count": wrong_pred_count,
        "successful": successful,
        "failed": failed,
        "sampling_log": model_slog,
        "prompt_validation_failures": prompt_validation_failures,
        # Real request accounting (review P0.4) — this model's own engine's authoritative
        # counters, summed across all per-(dataset,model) engines by the caller.
        "api_requests": engine.total_requests,
        "api_requests_failed": engine.total_requests_failed,
        "api_requests_by_category": dict(engine.requests_by_category),
    }


async def run_experiment(config, args):
    ensure_spacy_available()  # fail fast, before any API calls, if R extraction would silently degrade

    resume_dir = getattr(args, "resume_dir", None)
    force_restart = bool(getattr(args, "force_restart", False))
    if resume_dir:
        output_dir = Path(resume_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"--resume-dir path does not exist: {output_dir}")
        run_id = output_dir.name.rsplit("_", 1)[-1]
    else:
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.output.base_dir) / f"{timestamp}_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=output_dir / "logs", console_level=config.output.log_level)
    logger.info(f"{'Resuming' if resume_dir else 'Starting'} experiment: "
                f"{config.experiment.name} v{config.experiment.version}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Random seed: {config.experiment.seed}")
    logger.info(f"Output directory: {output_dir}")

    # Provenance written EARLY (not just at the end of a successful run) so a
    # crashed/interrupted run still has a durable record of exactly what config
    # produced it — this is what lets scripts/resume_experiment.py resume from just
    # the output directory name, without needing the live config/ (which may have
    # changed since). Only written if absent: on resume this preserves the ORIGINAL
    # run's provenance rather than overwriting it with the resume-time environment.
    config_snapshot_path = output_dir / "config_snapshot.yaml"
    if not config_snapshot_path.exists():
        save_config_to_file(config, config_snapshot_path)
    env_snapshot_path = output_dir / "environment_snapshot.json"
    if not env_snapshot_path.exists() and (config.reproducibility.log_git_commit
                                           or config.reproducibility.log_package_versions):
        save_environment_snapshot(env_snapshot_path)

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
    sampling_logs: List[SamplingLog] = []          # dataset-pooled (feeds overall_slog)
    per_model_sampling_logs: List[SamplingLog] = []  # one per (model,dataset) — feeds execution_summary.txt
    prompt_sources_by_dataset: Dict[str, Dict[str, str]] = {}

    for dataset_config in config.datasets:
        dataset_name = dataset_config.name
        logger.info(f"Processing dataset: {dataset_name}")

        try:
            # Prefer a frozen curated set (data/processed/{dataset}_curated.jsonl,
            # produced by scripts/curate_dataset.py) so every run analyses the same
            # quality-controlled instances. Fall back to live balanced sampling.
            curated_path = Path("data/processed") / f"{dataset_name}_curated.jsonl"
            if curated_path.exists():
                instances = loader.load_curated(str(curated_path))
                logger.info(f"Loaded {len(instances)} CURATED instances for {dataset_name} "
                            f"from {curated_path}")
                # Respect sample_size for pilots/subsets; shuffle reproducibly before slicing.
                if dataset_config.sample_size < len(instances):
                    rng_slice = random.Random(config.experiment.seed)
                    rng_slice.shuffle(instances)
                    instances = instances[:dataset_config.sample_size]
                    logger.info(f"Sliced to {len(instances)} instances (sample_size={dataset_config.sample_size})")
            else:
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
                logger.info(f"Loaded {len(instances)} sampled instances for {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue

        summary.total_instances += len(instances) * len(config.models)
        # Prompts depend only on config + dataset, not the model, so build once and
        # share (read-only) across the concurrent model runs. The resolved sources go
        # into prompt_manifest.json — the provenance record of what actually ran.
        prompts, prompt_sources = create_prompt_map(config, dataset_name=dataset_name)
        prompt_sources_by_dataset[dataset_name] = prompt_sources

        # Run every configured model on this dataset CONCURRENTLY ("at once"). Each
        # model gets its own engine, checkpoint file, and result list, so the
        # coroutines share no mutable state; asyncio.gather returns them in config
        # order, keeping merged outputs deterministic regardless of finish order.
        logger.info(f"Running {len(config.models)} models concurrently on {dataset_name}: "
                    f"{', '.join(m.name for m in config.models)}")
        model_bundles = await asyncio.gather(*[
            _run_model_on_dataset(
                model_config, instances, prompts, dataset_config,
                parser, normalizer, calc, config, output_dir,
                existing_results=_load_checkpointed_results(output_dir, dataset_name, model_config.name),
                force_restart=force_restart,
            )
            for model_config in config.models
        ])

        # Merge the per-model bundles in config order.
        dataset_wrong_pred = 0
        for bundle in model_bundles:
            all_results.extend(bundle["results"])
            aggregate_list.append(bundle["agg"])
            summary.successful_instances += bundle["successful"]
            summary.failed_instances += bundle["failed"]
            summary.prompt_validation_failures += bundle["prompt_validation_failures"]
            dataset_wrong_pred += bundle["wrong_pred_count"]
            # Real request accounting (review P0.4): sum each model's own engine's
            # authoritative counters — replaces the end-of-run `len(all_results) * 5` guess.
            summary.api_requests_total += bundle.get("api_requests", 0)
            summary.api_requests_failed += bundle.get("api_requests_failed", 0)
            for category, count in bundle.get("api_requests_by_category", {}).items():
                summary.api_requests_by_category[category] = (
                    summary.api_requests_by_category.get(category, 0) + count
                )
            # Per-(model,dataset) sampling log (review P1.3): requested/sampled/
            # wrong_predictions are all THIS model's own numbers, so they read
            # consistently together — unlike the dataset-pooled log below, whose
            # wrong_predictions sums across models against a single model's sample size.
            if bundle.get("sampling_log") is not None:
                per_model_sampling_logs.append(bundle["sampling_log"])

        # Dataset-level sampling log (wrong_predictions summed across all models),
        # used for the dataset-level and overall aggregates below.
        sampling_logs.append(SamplingLog(
            dataset=dataset_name,
            requested=dataset_config.sample_size,
            sampled=len(instances),
            wrong_predictions=dataset_wrong_pred,
        ))

    # Compute overall aggregate
    if all_results:
        overall_slog = SamplingLog(
            dataset="all",
            requested=sum(s.requested for s in sampling_logs),
            sampled=sum(s.sampled for s in sampling_logs),
            wrong_predictions=sum(s.wrong_predictions for s in sampling_logs),
        )
        overall = compute_aggregate_metrics(all_results, "overall", "all", sampling_log=overall_slog,
                                            normalizer=normalizer)
        aggregate_list.append(overall)

    # Pure dataset-level aggregates
    dataset_names = set(r.dataset for r in all_results)
    for ds in dataset_names:
        ds_results = [r for r in all_results if r.dataset == ds]
        ds_slog = next((s for s in sampling_logs if s.dataset == ds), None)
        agg = compute_aggregate_metrics(ds_results, "dataset", ds, sampling_log=ds_slog,
                                        normalizer=normalizer)
        aggregate_list.append(agg)

    # Pure model-level aggregates
    model_names = set(r.model for r in all_results)
    for mn in model_names:
        md_results = [r for r in all_results if r.model == mn]
        agg = compute_aggregate_metrics(md_results, "model", mn, normalizer=normalizer)
        aggregate_list.append(agg)

    # Pre-registered NHST family (a): per model×dataset cell, is mean ECS-lift > 0?
    # One-sided sign-flip permutation on the per-instance paired (ecs − ecs_random)
    # differences; Holm correction across the cells of this run (one family). Cells
    # below metrics.min_n_for_test report the estimate but skip the test (p=None) —
    # a permutation test on 1-5 points is noise, not evidence.
    min_n = getattr(config.metrics, "min_n_for_test", 6)
    n_perms = getattr(config.metrics, "permutation_tests", 10000)
    md_aggs = [a for a in aggregate_list if a.aggregation_level == "model_dataset"]
    md_results_by_group = {
        a.group_name: [r.ecs_lift for r in all_results
                       if r.ecs_lift is not None
                       and f"{next((m.name for m in config.models if m.model_id == r.model), r.model)}_{r.dataset}" == a.group_name]
        for a in md_aggs
    }
    raw_ps: List[Optional[float]] = []
    for a in md_aggs:
        lifts = md_results_by_group.get(a.group_name, [])
        if len(lifts) >= min_n:
            p = sign_flip_permutation_test(lifts, n_permutations=n_perms,
                                           seed=config.experiment.seed, alternative="greater")
        else:
            p = None
        raw_ps.append(p)
    if getattr(config.metrics, "correction", "holm") == "holm":
        adj_ps = holm_correction(raw_ps)
    else:
        adj_ps = list(raw_ps)
    for a, p_raw, p_adj in zip(md_aggs, raw_ps, adj_ps):
        a.ecs_lift_p_value = p_raw
        a.ecs_lift_p_holm = p_adj

    # PRIMARY test family (a) — mean ECS-adj > 0 on the COMPLETE-CASE population,
    # per model×dataset cell (ECS_ROBUSTNESS_PLAN_2026-07-05.md §3.5 + the 2026-07-08
    # P0.1 amendment). The complete-case ECS-adj (all three paradigm components
    # defined) IS the primary estimand, so the pre-registered sign-flip test must run
    # on exactly that population — not on the available-component pool, >half of which
    # is a single-paradigm-pair statement (E-R only). Same sign-flip machinery applied
    # DIRECTLY to ecs_adj (AJ's null is 0 by construction — no baseline subtraction).
    # Holm across this run's cells (its own family).
    adj_complete_by_group = {
        a.group_name: [r.ecs_adj for r in all_results
                       if r.ecs_adj_complete and r.ecs_adj is not None
                       and f"{next((m.name for m in config.models if m.model_id == r.model), r.model)}_{r.dataset}" == a.group_name]
        for a in md_aggs
    }
    raw_adj_c_ps: List[Optional[float]] = []
    for a in md_aggs:
        adj_vals = adj_complete_by_group.get(a.group_name, [])
        if len(adj_vals) >= min_n:
            p = sign_flip_permutation_test(adj_vals, n_permutations=n_perms,
                                           seed=config.experiment.seed, alternative="greater")
        else:
            p = None
        raw_adj_c_ps.append(p)
    if getattr(config.metrics, "correction", "holm") == "holm":
        adj_c_ps = holm_correction(raw_adj_c_ps)
    else:
        adj_c_ps = list(raw_adj_c_ps)
    for a, p_raw, p_adj in zip(md_aggs, raw_adj_c_ps, adj_c_ps):
        a.ecs_adj_complete_p_value = p_raw
        a.ecs_adj_complete_p_holm = p_adj

    # SENSITIVITY test family (a2) — the same test on the AVAILABLE-COMPONENT ECS-adj
    # (larger N; framed as "above-chance agreement across whichever paradigm pairs
    # were elicitable"). Reported as the wider-N robustness companion to (a), never as
    # the headline. A separate Holm family from (a).
    adj_results_by_group = {
        a.group_name: [r.ecs_adj for r in all_results
                       if r.ecs_adj is not None
                       and f"{next((m.name for m in config.models if m.model_id == r.model), r.model)}_{r.dataset}" == a.group_name]
        for a in md_aggs
    }
    raw_adj_ps: List[Optional[float]] = []
    for a in md_aggs:
        adj_vals = adj_results_by_group.get(a.group_name, [])
        if len(adj_vals) >= min_n:
            p = sign_flip_permutation_test(adj_vals, n_permutations=n_perms,
                                           seed=config.experiment.seed, alternative="greater")
        else:
            p = None
        raw_adj_ps.append(p)
    if getattr(config.metrics, "correction", "holm") == "holm":
        adj_adj_ps = holm_correction(raw_adj_ps)
    else:
        adj_adj_ps = list(raw_adj_ps)
    for a, p_raw, p_adj in zip(md_aggs, raw_adj_ps, adj_adj_ps):
        a.ecs_adj_p_value = p_raw
        a.ecs_adj_p_holm = p_adj

    # Cross-model same-strategy agreement (zero extra API calls): compares
    # within-model cross-strategy consensus against cross-model same-strategy
    # agreement — privileged-self-knowledge vs generic-task-prior (arXiv:2602.02639,
    # arXiv:2603.15821). Only computable when >=2 models ran.
    cross_model = MetricsCalculator.compute_cross_model_agreement(all_results)
    if cross_model:
        with open(output_dir / "cross_model_agreement.json", "w", encoding="utf-8") as f:
            json.dump(cross_model, f, indent=2)

    # Save results (config_snapshot.yaml / environment_snapshot.json were already
    # written near the top of this function, before any API calls, so a crashed run
    # still has them — not repeated here).
    save_instance_results(all_results, str(output_dir / "instance_results.jsonl"))
    save_aggregate_metrics(aggregate_list, str(output_dir / "aggregate_metrics.json"))
    save_metrics_csv(all_results, str(output_dir / "instance_metrics.csv"))

    # Machine-checkable prompt provenance: which prompt files (and content hashes)
    # were EXECUTED per dataset. The config snapshot alone was previously misleading
    # (review §1.3: it named base files that never ran).
    with open(output_dir / "prompt_manifest.json", "w", encoding="utf-8") as f:
        json.dump(build_prompt_manifest(prompt_sources_by_dataset), f, indent=2)

    save_metadata_table(
        [d.to_dict() for d in config.datasets],
        "datasets", str(output_dir / "dataset_metadata.json")
    )
    save_metadata_table(
        [m.to_dict() for m in config.models],
        "models", str(output_dir / "model_metadata.json")
    )

    summary.end_time = datetime.now()
    summary.duration_seconds = (summary.end_time - summary.start_time).total_seconds()
    # api_requests_total/failed/by_category are already real (summed from each
    # engine's authoritative counters in the merge loop above) — no longer computed
    # here from a fabricated len(all_results) * 5 formula (review P0.4).
    summary.avg_time_per_instance = summary.duration_seconds / max(len(all_results), 1)
    # Per-(model,dataset) logs (review P1.3), not the dataset-pooled ones (those still
    # feed overall_slog above) — each line's requested/sampled/wrong_predictions are
    # all the SAME model's own numbers, so "requested=10, wrong_pred=12" (3 models'
    # wrong counts vs 1 model's sample size) can no longer happen.
    summary.sampling_logs = [s.to_dict() for s in per_model_sampling_logs]
    # Parsing failures by strategy (review P1.3): counts of instances where that
    # strategy's raw response could not be parsed at all (*_parsed == False) — the
    # per-instance flags already recorded during collection, never surfaced here.
    summary.parsing_failures = {
        "H": sum(1 for r in all_results if not r.highlighting_parsed),
        "R": sum(1 for r in all_results if not r.rationale_parsed),
        "CF": sum(1 for r in all_results if not r.counterfactual_parsed),
        "RO": sum(1 for r in all_results if not r.rank_ordering_parsed),
    }

    with open(output_dir / "execution_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary.generate_report())

    report_md = generate_md_report(aggregate_list, all_results, config, cross_model=cross_model)
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
