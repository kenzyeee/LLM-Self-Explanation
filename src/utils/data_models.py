import csv
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Set, List, Tuple, Dict, Optional, Any


@dataclass
class SamplingLog:
    dataset: str = ""
    requested: int = 0
    sampled: int = 0
    wrong_predictions: int = 0
    dropped_by_reason: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SamplingLog':
        return cls(**data)


@dataclass
class InstanceResult:
    instance_id: str
    dataset: str
    model: str
    timestamp: datetime
    text: str
    ground_truth_label: str
    predicted_label: str
    # Verbalized confidence in [0,1] (elicited 0-100, Tian et al. 2023 / Xiong et al.
    # 2024); None = not elicited or unparseable — never a fake 0.0.
    confidence: Optional[float] = None
    correct: bool = False
    raw_highlighting: str = ""
    raw_rationale: str = ""
    raw_counterfactual: str = ""
    raw_rank_ordering: str = ""
    input_length: int = 0
    highlighting_tokens: Set[str] = field(default_factory=set)
    rationale_tokens: Set[str] = field(default_factory=set)
    counterfactual_tokens: Set[str] = field(default_factory=set)
    rank_ordering_tokens: List[Tuple[str, int]] = field(default_factory=list)
    highlighting_parsed: bool = False
    rationale_parsed: bool = False
    counterfactual_parsed: bool = False
    rank_ordering_parsed: bool = False
    highlighting_valid: bool = False
    rationale_valid: bool = False
    counterfactual_valid: bool = False
    rank_ordering_valid: bool = False
    rationale_text: str = ""
    ecs_extraction_rationale: Optional[float] = None
    ecs_extraction_perturbation: Optional[float] = None
    cf_json_valid: bool = False
    cf_rules_compliant: bool = False
    cf_flip_verified: bool = False
    cf_actual_label: str = ""
    cf_counterfactual_text: str = ""
    classification_prompt: str = ""
    classification_raw_response: str = ""
    highlighting_explain_prompt: str = ""
    rationale_explain_prompt: str = ""
    counterfactual_explain_prompt: str = ""
    rank_ordering_explain_prompt: str = ""
    model_refused: bool = False
    prompt_tokens: int = 0
    response_tokens: int = 0
    prompt_hash: str = ""
    raw_response_length: int = 0
    jaccard_H_R: Optional[float] = None
    jaccard_H_CF: Optional[float] = None
    jaccard_H_RO: Optional[float] = None
    jaccard_R_CF: Optional[float] = None
    jaccard_R_RO: Optional[float] = None
    jaccard_CF_RO: Optional[float] = None
    overlap_H_R: Optional[float] = None
    overlap_H_CF: Optional[float] = None
    overlap_H_RO: Optional[float] = None
    overlap_R_CF: Optional[float] = None
    overlap_R_RO: Optional[float] = None
    overlap_CF_RO: Optional[float] = None
    rbo_H_RO: Optional[float] = None
    kendall_H_RO: Optional[float] = None
    normalized_kendall_H_RO: Optional[float] = None
    ecs: Optional[float] = None
    # Size-robust secondary composite: mean Overlap Coefficient over the same 5
    # cross-paradigm pairs (immune to the Jaccard set-size ceiling; Krishna-style
    # feature-agreement analogue).
    ecs_overlap: Optional[float] = None
    ecs_complete: Optional[float] = None
    ecs_primary_pairs: int = 0
    n_valid_strategies: int = 0
    vocab_size: int = 0
    short_vocab: bool = False
    r_hallucinated_concepts: List[str] = field(default_factory=list)
    cc3_tokens: Set[str] = field(default_factory=set)
    cc4_tokens: Set[str] = field(default_factory=set)
    cc3_size: int = 0
    cc4_size: int = 0
    # --- D1: ECS as lift over a random-selection baseline (the headline number).
    #     *_weighted is the secondary lift over a salience-weighted null (harder null:
    #     random draws ∝ H's graded salience — review §2.5). ---
    ecs_random: Optional[float] = None
    ecs_lift: Optional[float] = None
    ecs_random_weighted: Optional[float] = None
    ecs_lift_weighted: Optional[float] = None
    # --- D2: CF dual variant. The generic cf_* / counterfactual_* fields describe the
    #         CANONICAL CF = the minimal contrastive edit (MiCE), used in ECS;
    #         cf_canonical_minimality is its minimality. cf_contrast_* is the secondary
    #         FREE/unconstrained edit, a validity-minimality reference NOT used in ECS. ---
    cf_canonical_minimality: Optional[float] = None
    cf_contrast_valid: bool = False
    cf_contrast_flip_verified: bool = False
    cf_contrast_minimality: Optional[float] = None
    cf_contrast_tokens: Set[str] = field(default_factory=set)
    cf_contrast_text: str = ""
    # Single-shot vs coached stratum (review §2.6): whether the FIRST uncoached CF
    # elicitation was already flip-valid, whether the correction loop produced the
    # valid CF, and whether RO needed its hallucination self-correction.
    cf_valid_first_attempt: bool = False
    cf_corrected: bool = False
    ro_self_corrected: bool = False
    # Raw edited SURFACE tokens from the CF diff (pre-normalization) — used by the
    # erasure pass and minimality accounting. counterfactual_tokens (above) is the
    # NORMALIZED evidence set used in ECS (same token space as H/R/RO — review §8.3).
    cf_edited_tokens_raw: Set[str] = field(default_factory=set)
    # --- D6: introduced-concept rate (fraction of rationale concepts absent from input) ---
    r_introduced_concept_rate: Optional[float] = None
    # Strategies whose elicitation hit the token limit (finish_reason="length") and could
    # not be recovered — these are invalid for a token-limit reason, not a model-content one.
    truncated_strategies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        base = {'instance_id': self.instance_id, 'dataset': self.dataset, 'model': self.model,
                'timestamp': self.timestamp.isoformat(),
                'text': self.text, 'input_length': self.input_length,
                'ground_truth_label': self.ground_truth_label,
                'predicted_label': self.predicted_label, 'confidence': self.confidence, 'correct': self.correct,
                'raw_highlighting': self.raw_highlighting, 'raw_rationale': self.raw_rationale,
                'raw_counterfactual': self.raw_counterfactual, 'raw_rank_ordering': self.raw_rank_ordering,
                'highlighting_tokens': sorted(list(self.highlighting_tokens)),
                'rationale_tokens': sorted(list(self.rationale_tokens)),
                'counterfactual_tokens': sorted(list(self.counterfactual_tokens)),
                'rank_ordering_tokens': [[token, rank] for token, rank in self.rank_ordering_tokens],
                'highlighting_parsed': self.highlighting_parsed, 'rationale_parsed': self.rationale_parsed,
                'counterfactual_parsed': self.counterfactual_parsed, 'rank_ordering_parsed': self.rank_ordering_parsed,
                'highlighting_valid': self.highlighting_valid, 'rationale_valid': self.rationale_valid,
                'counterfactual_valid': self.counterfactual_valid, 'rank_ordering_valid': self.rank_ordering_valid,
                'rationale_text': self.rationale_text,
                'classification_prompt': self.classification_prompt,
                'classification_raw_response': self.classification_raw_response,
                'highlighting_explain_prompt': self.highlighting_explain_prompt,
                'rationale_explain_prompt': self.rationale_explain_prompt,
                'counterfactual_explain_prompt': self.counterfactual_explain_prompt,
                'rank_ordering_explain_prompt': self.rank_ordering_explain_prompt,
                'model_refused': self.model_refused, 'prompt_tokens': self.prompt_tokens,
                'response_tokens': self.response_tokens, 'raw_response_length': self.raw_response_length,
                'prompt_hash': self.prompt_hash}
        metrics = {'jaccard_H_R': self.jaccard_H_R, 'jaccard_H_CF': self.jaccard_H_CF,
                   'jaccard_H_RO': self.jaccard_H_RO, 'jaccard_R_CF': self.jaccard_R_CF,
                   'jaccard_R_RO': self.jaccard_R_RO, 'jaccard_CF_RO': self.jaccard_CF_RO,
                   'overlap_H_R': self.overlap_H_R, 'overlap_H_CF': self.overlap_H_CF,
                   'overlap_H_RO': self.overlap_H_RO, 'overlap_R_CF': self.overlap_R_CF,
                   'overlap_R_RO': self.overlap_R_RO, 'overlap_CF_RO': self.overlap_CF_RO,
                   'rbo_H_RO': self.rbo_H_RO,
                   'kendall_H_RO': self.kendall_H_RO, 'normalized_kendall_H_RO': self.normalized_kendall_H_RO,
                   'ecs': self.ecs, 'ecs_overlap': self.ecs_overlap, 'ecs_complete': self.ecs_complete,
                   'ecs_primary_pairs': self.ecs_primary_pairs,
                   'ecs_extraction_rationale': self.ecs_extraction_rationale,
                   'ecs_extraction_perturbation': self.ecs_extraction_perturbation,
                    'n_valid_strategies': self.n_valid_strategies,
                    'vocab_size': self.vocab_size, 'short_vocab': self.short_vocab,
                    'r_hallucinated_concepts': list(self.r_hallucinated_concepts),
                    'cc3_tokens': sorted(list(self.cc3_tokens)), 'cc4_tokens': sorted(list(self.cc4_tokens)),
                    'cc3_size': self.cc3_size, 'cc4_size': self.cc4_size,
                     'cf_json_valid': self.cf_json_valid,
                     'cf_rules_compliant': self.cf_rules_compliant,
                     'cf_flip_verified': self.cf_flip_verified,
                     'cf_valid_first_attempt': self.cf_valid_first_attempt,
                     'cf_corrected': self.cf_corrected,
                     'ro_self_corrected': self.ro_self_corrected,
                     'cf_actual_label': self.cf_actual_label,
                     'cf_counterfactual_text': self.cf_counterfactual_text,
                     'cf_edited_tokens_raw': sorted(list(self.cf_edited_tokens_raw)),
                     'ecs_random': self.ecs_random, 'ecs_lift': self.ecs_lift,
                     'ecs_random_weighted': self.ecs_random_weighted,
                     'ecs_lift_weighted': self.ecs_lift_weighted,
                     'cf_canonical_minimality': self.cf_canonical_minimality,
                     'cf_contrast_valid': self.cf_contrast_valid,
                     'cf_contrast_flip_verified': self.cf_contrast_flip_verified,
                     'cf_contrast_minimality': self.cf_contrast_minimality,
                     'cf_contrast_tokens': sorted(list(self.cf_contrast_tokens)),
                     'cf_contrast_text': self.cf_contrast_text,
                     'r_introduced_concept_rate': self.r_introduced_concept_rate,
                     'truncated_strategies': list(self.truncated_strategies)}
        base.update(metrics)
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstanceResult':
        return cls(
            instance_id=data['instance_id'], dataset=data['dataset'], model=data['model'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            text=data['text'],
            input_length=data.get('input_length', len(data['text'])),
            ground_truth_label=data['ground_truth_label'],
            predicted_label=data['predicted_label'], confidence=data.get('confidence'), correct=data['correct'],
            raw_highlighting=data['raw_highlighting'], raw_rationale=data['raw_rationale'],
            raw_counterfactual=data['raw_counterfactual'], raw_rank_ordering=data['raw_rank_ordering'],
            highlighting_tokens=set(data['highlighting_tokens']),
            rationale_tokens=set(data['rationale_tokens']),
            counterfactual_tokens=set(data['counterfactual_tokens']),
            rank_ordering_tokens=[(token, rank) for token, rank in data['rank_ordering_tokens']],
            highlighting_parsed=data.get('highlighting_parsed', False), rationale_parsed=data.get('rationale_parsed', False),
            counterfactual_parsed=data.get('counterfactual_parsed', False), rank_ordering_parsed=data.get('rank_ordering_parsed', False),
            highlighting_valid=data.get('highlighting_valid', False), rationale_valid=data.get('rationale_valid', False),
            counterfactual_valid=data.get('counterfactual_valid', False), rank_ordering_valid=data.get('rank_ordering_valid', False),
            rationale_text=data.get('rationale_text', ''),
            classification_prompt=data.get('classification_prompt', ''),
            classification_raw_response=data.get('classification_raw_response', ''),
            highlighting_explain_prompt=data.get('highlighting_explain_prompt', ''),
            rationale_explain_prompt=data.get('rationale_explain_prompt', ''),
            counterfactual_explain_prompt=data.get('counterfactual_explain_prompt', ''),
            rank_ordering_explain_prompt=data.get('rank_ordering_explain_prompt', ''),
            model_refused=data.get('model_refused', False),
            prompt_tokens=data.get('prompt_tokens', 0),
            response_tokens=data.get('response_tokens', 0),
            raw_response_length=data.get('raw_response_length', 0),
            prompt_hash=data.get('prompt_hash', ''),
            jaccard_H_R=data.get('jaccard_H_R'), jaccard_H_CF=data.get('jaccard_H_CF'),
            jaccard_H_RO=data.get('jaccard_H_RO'), jaccard_R_CF=data.get('jaccard_R_CF'),
            jaccard_R_RO=data.get('jaccard_R_RO'), jaccard_CF_RO=data.get('jaccard_CF_RO'),
            overlap_H_R=data.get('overlap_H_R'), overlap_H_CF=data.get('overlap_H_CF'),
            overlap_H_RO=data.get('overlap_H_RO'), overlap_R_CF=data.get('overlap_R_CF'),
            overlap_R_RO=data.get('overlap_R_RO'), overlap_CF_RO=data.get('overlap_CF_RO'),
            rbo_H_RO=data.get('rbo_H_RO'),
            kendall_H_RO=data.get('kendall_H_RO'),
            normalized_kendall_H_RO=data.get('normalized_kendall_H_RO'),
            ecs=data.get('ecs'), ecs_overlap=data.get('ecs_overlap'), ecs_complete=data.get('ecs_complete'),
            ecs_primary_pairs=data.get('ecs_primary_pairs', 0),
            ecs_extraction_rationale=data.get('ecs_extraction_rationale'),
            ecs_extraction_perturbation=data.get('ecs_extraction_perturbation'),
            n_valid_strategies=data.get('n_valid_strategies', 0),
            vocab_size=data.get('vocab_size', 0),
            short_vocab=data.get('short_vocab', False),
            r_hallucinated_concepts=data.get('r_hallucinated_concepts', []),
            cc3_tokens=set(data['cc3_tokens']), cc4_tokens=set(data['cc4_tokens']),
            cc3_size=data['cc3_size'], cc4_size=data['cc4_size'],
            cf_json_valid=data.get('cf_json_valid', False),
            cf_rules_compliant=data.get('cf_rules_compliant', False),
            cf_flip_verified=data.get('cf_flip_verified', False),
            cf_valid_first_attempt=data.get('cf_valid_first_attempt', False),
            cf_corrected=data.get('cf_corrected', False),
            ro_self_corrected=data.get('ro_self_corrected', False),
            cf_actual_label=data.get('cf_actual_label', ''),
            cf_counterfactual_text=data.get('cf_counterfactual_text', ''),
            cf_edited_tokens_raw=set(data.get('cf_edited_tokens_raw', [])),
            ecs_random=data.get('ecs_random'), ecs_lift=data.get('ecs_lift'),
            ecs_random_weighted=data.get('ecs_random_weighted'),
            ecs_lift_weighted=data.get('ecs_lift_weighted'),
            cf_canonical_minimality=data.get('cf_canonical_minimality'),
            cf_contrast_valid=data.get('cf_contrast_valid', False),
            cf_contrast_flip_verified=data.get('cf_contrast_flip_verified', False),
            cf_contrast_minimality=data.get('cf_contrast_minimality'),
            cf_contrast_tokens=set(data.get('cf_contrast_tokens', [])),
            cf_contrast_text=data.get('cf_contrast_text', ''),
            r_introduced_concept_rate=data.get('r_introduced_concept_rate'),
            truncated_strategies=data.get('truncated_strategies', []),
        )


@dataclass
class AggregateMetrics:
    aggregation_level: str
    group_name: str
    n_instances: int
    mean_ecs: float
    std_ecs: float
    median_ecs: float
    ecs_ci_lower: float
    ecs_ci_upper: float
    mean_jaccard_H_R: float
    mean_jaccard_H_CF: float
    mean_jaccard_H_RO: float
    mean_jaccard_R_CF: float
    mean_jaccard_R_RO: float
    mean_jaccard_CF_RO: float
    mean_rbo_H_RO: float
    mean_kendall_H_RO: float
    mean_normalized_kendall_H_RO: float
    mean_overlap_H_R: float
    mean_overlap_H_CF: float
    mean_overlap_H_RO: float
    mean_overlap_R_CF: float
    mean_overlap_R_RO: float
    mean_overlap_CF_RO: float
    mean_cc3_size: float
    mean_cc4_size: float
    pct_instances_with_cc3: float
    pct_instances_with_cc4: float
    spearman_rho: float
    spearman_p_value: float
    correlation_ci_lower: float
    correlation_ci_upper: float
    highlighting_success_rate: float
    rationale_success_rate: float
    counterfactual_success_rate: float
    rank_ordering_success_rate: float
    mean_ecs_extraction_rationale: float = 0.0
    mean_ecs_extraction_perturbation: float = 0.0
    mean_ecs_complete: float = 0.0
    n_complete_cases: int = 0
    pct_complete_cases: float = 0.0
    mean_ecs_short: float = 0.0
    n_short: int = 0
    mean_ecs_medium: float = 0.0
    n_medium: int = 0
    mean_ecs_long: float = 0.0
    n_long: int = 0
    mean_ecs_normal_vocab: float = 0.0
    n_normal_vocab: int = 0
    mean_ecs_short_vocab: float = 0.0
    n_short_vocab: int = 0
    requested_samples: int = 0
    sampled_samples: int = 0
    dropped_wrong_pred: int = 0
    dropped_other: Dict[str, int] = field(default_factory=dict)
    # --- D1/D2/D6 + correctness split (correctness only set at model_dataset level —
    #     pooled, the contrast is confounded by cell composition, review §2.11) ---
    mean_ecs_lift: float = 0.0
    mean_ecs_random: float = 0.0
    n_lift: int = 0
    # Secondary lift over the salience-weighted null (review §2.5).
    mean_ecs_lift_weighted: float = 0.0
    n_lift_weighted: int = 0
    # Size-robust secondary composite (mean overlap coefficient over ECS pairs).
    mean_ecs_overlap: float = 0.0
    introduced_concept_rate: float = 0.0
    cf_canonical_validity_rate: float = 0.0
    cf_contrast_validity_rate: float = 0.0
    # Single-shot vs coached stratum (review §2.6).
    cf_first_attempt_validity_rate: float = 0.0
    n_cf_corrected: int = 0
    n_ro_self_corrected: int = 0
    mean_cf_canonical_minimality: float = 0.0
    mean_cf_contrast_minimality: float = 0.0
    mean_ecs_correct: float = 0.0
    n_correct: int = 0
    mean_ecs_incorrect: float = 0.0
    n_incorrect: int = 0
    # Verbalized confidence coverage.
    mean_confidence: float = 0.0
    n_confidence: int = 0
    # Per-pair instance counts behind each pairwise mean.
    pair_ns: Dict[str, int] = field(default_factory=dict)
    # Pre-registered test (a): sign-flip permutation p for mean ECS-lift > 0 in this
    # cell (model_dataset level only), raw and Holm-adjusted across the run's cells.
    # None = test not run (pooled level, or cell below metrics.min_n_for_test).
    ecs_lift_p_value: Optional[float] = None
    ecs_lift_p_holm: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AggregateMetrics':
        return cls(**data)


@dataclass
class ValidityTestResult:
    instance_id: str
    dataset: str
    model: str
    cc3_tokens: Set[str] = field(default_factory=set)
    cc3_original_prediction: str = ""
    cc3_masked_prediction: str = ""
    cc3_flipped: bool = False
    cc4_tokens: Set[str] = field(default_factory=set)
    cc4_original_prediction: str = ""
    cc4_masked_prediction: str = ""
    cc4_flipped: bool = False
    random_tokens: Set[str] = field(default_factory=set)
    random_original_prediction: str = ""
    random_masked_prediction: str = ""
    random_flipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'instance_id': self.instance_id, 'dataset': self.dataset, 'model': self.model,
            'cc3_tokens': sorted(list(self.cc3_tokens)),
            'cc3_original_prediction': self.cc3_original_prediction,
            'cc3_masked_prediction': self.cc3_masked_prediction,
            'cc3_flipped': self.cc3_flipped,
            'cc4_tokens': sorted(list(self.cc4_tokens)),
            'cc4_original_prediction': self.cc4_original_prediction,
            'cc4_masked_prediction': self.cc4_masked_prediction,
            'cc4_flipped': self.cc4_flipped,
            'random_tokens': sorted(list(self.random_tokens)),
            'random_original_prediction': self.random_original_prediction,
            'random_masked_prediction': self.random_masked_prediction,
            'random_flipped': self.random_flipped,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidityTestResult':
        return cls(
            instance_id=data['instance_id'], dataset=data['dataset'], model=data['model'],
            cc3_tokens=set(data['cc3_tokens']),
            cc3_original_prediction=data['cc3_original_prediction'],
            cc3_masked_prediction=data['cc3_masked_prediction'],
            cc3_flipped=data['cc3_flipped'],
            cc4_tokens=set(data['cc4_tokens']),
            cc4_original_prediction=data['cc4_original_prediction'],
            cc4_masked_prediction=data['cc4_masked_prediction'],
            cc4_flipped=data['cc4_flipped'],
            random_tokens=set(data['random_tokens']),
            random_original_prediction=data['random_original_prediction'],
            random_masked_prediction=data['random_masked_prediction'],
            random_flipped=data['random_flipped'],
        )


@dataclass
class AggregateValidityResults:
    dataset: str
    model: str
    n_instances: int
    cc3_flip_rate: float
    cc4_flip_rate: float
    random_flip_rate: float
    t_statistic: float
    p_value: float
    effect_size: float
    cc3_flip_ci_lower: float
    cc3_flip_ci_upper: float
    random_flip_ci_lower: float
    random_flip_ci_upper: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AggregateValidityResults':
        return cls(**data)


@dataclass
class ExecutionSummary:
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_instances: int
    successful_instances: int
    failed_instances: int
    run_id: str = ""
    parsing_failures: Dict[str, int] = field(default_factory=dict)
    api_failures: int = 0
    normalization_failures: int = 0
    avg_time_per_instance: float = 0.0
    api_requests_total: int = 0
    api_requests_failed: int = 0
    prompt_validation_failures: int = 0
    sampling_logs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'total_instances': self.total_instances,
            'successful_instances': self.successful_instances,
            'failed_instances': self.failed_instances,
            'run_id': self.run_id,
            'parsing_failures': self.parsing_failures,
            'api_failures': self.api_failures,
            'normalization_failures': self.normalization_failures,
            'avg_time_per_instance': self.avg_time_per_instance,
            'api_requests_total': self.api_requests_total,
            'api_requests_failed': self.api_requests_failed,
            'prompt_validation_failures': self.prompt_validation_failures,
            'sampling_logs': self.sampling_logs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionSummary':
        return cls(
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
            duration_seconds=data['duration_seconds'],
            total_instances=data['total_instances'],
            successful_instances=data['successful_instances'],
            failed_instances=data['failed_instances'],
            run_id=data.get('run_id', ''),
            parsing_failures=data.get('parsing_failures', {}),
            api_failures=data.get('api_failures', 0),
            normalization_failures=data.get('normalization_failures', 0),
            avg_time_per_instance=data.get('avg_time_per_instance', 0.0),
            api_requests_total=data.get('api_requests_total', 0),
            api_requests_failed=data.get('api_requests_failed', 0),
            prompt_validation_failures=data.get('prompt_validation_failures', 0),
            sampling_logs=data.get('sampling_logs', []),
        )

    def generate_report(self) -> str:
        success_rate = (self.successful_instances / self.total_instances * 100) if self.total_instances > 0 else 0
        failure_rate = (self.failed_instances / self.total_instances * 100) if self.total_instances > 0 else 0
        api_failure_rate = (self.api_requests_failed / self.api_requests_total * 100) if self.api_requests_total > 0 else 0
        report = f"""
Execution Summary
=================
Run ID: {self.run_id}
Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
End Time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {self.duration_seconds:.2f} seconds ({self.duration_seconds / 60:.2f} minutes)

Processing Statistics
---------------------
Total Instances: {self.total_instances}
Successful: {self.successful_instances} ({success_rate:.1f}%)
Failed: {self.failed_instances} ({failure_rate:.1f}%)

Failure Breakdown
-----------------
API Failures: {self.api_failures}
Normalization Failures: {self.normalization_failures}
Prompt Validation Failures: {self.prompt_validation_failures}
Parsing Failures by Strategy:
"""
        for strategy, count in sorted(self.parsing_failures.items()):
            report += f"  {strategy}: {count}\n"

        if self.sampling_logs:
            report += "\nSampling Log:\n"
            report += "-" * 40 + "\n"
            for slog in self.sampling_logs:
                report += f"  {slog['dataset']}: requested={slog['requested']}, sampled={slog['sampled']}, wrong_pred={slog['wrong_predictions']}\n"
                for reason, count in slog.get('dropped_by_reason', {}).items():
                    report += f"    dropped: {reason} -> {count}\n"

        report += f"""
Performance Metrics
-------------------
Average Time per Instance: {self.avg_time_per_instance:.2f} seconds
Total API Requests: {self.api_requests_total}
Failed API Requests: {self.api_requests_failed} ({api_failure_rate:.1f}%)
"""
        return report.strip()


def _check_unresolved_placeholders(report: str) -> None:
    """Fail-fast: reject reports containing unresolved template braces like {overall.n_short}."""
    unresolved = re.findall(r'\{[a-zA-Z_][a-zA-Z0-9_.]*\}', report)
    if unresolved:
        raise RuntimeError(
            f"Report contains {len(unresolved)} unresolved placeholder(s): {unresolved[:10]}"
        )


def generate_md_report(
    aggregate_list: List[AggregateMetrics],
    all_results: List[InstanceResult],
    config,
    cross_model: Optional[Dict[str, Any]] = None,
) -> str:
    overall = next((m for m in aggregate_list if m.aggregation_level == "overall"), None)
    model_dataset = [m for m in aggregate_list if m.aggregation_level == "model_dataset"]

    # InstanceResult.model stores the Bedrock model_id; aggregate group_names use the
    # study-facing label (config model .name). Map between them so the report can label
    # models and match per-(model,dataset) accuracy correctly for multi-model runs.
    id_to_label = {m.model_id: m.name for m in getattr(config, "models", [])}
    label_to_id = {m.name: m.model_id for m in getattr(config, "models", [])}

    def _model_label(model_id: str) -> str:
        return id_to_label.get(model_id, model_id)

    lines = []
    exp_name = config.experiment.name if hasattr(config, 'experiment') else "experiment"
    lines.append(f"# Experiment Report: {exp_name}")
    lines.append("")

    if all_results:
        t0 = min(r.timestamp for r in all_results)
        t1 = max(r.timestamp for r in all_results)
        dur = (t1 - t0).total_seconds()
        n_refused = sum(1 for r in all_results if r.model_refused)
        total_tokens = sum(r.prompt_tokens + r.response_tokens for r in all_results)
        n_truncated = sum(len(getattr(r, 'truncated_strategies', None) or []) for r in all_results)
        lines.append(f"- **Date:** {t0.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Duration:** {dur:.1f}s ({dur/60:.1f}m)")
        # Distinct models actually present in the results (config order, then any extras),
        # each shown as "label (`model_id`)". A single-model run keeps the old "Model:" line.
        present_ids = list(dict.fromkeys(r.model for r in all_results))
        model_strs = [f"{_model_label(mid)} (`{mid}`)" for mid in present_ids]
        if len(model_strs) <= 1:
            lines.append(f"- **Model:** {model_strs[0] if model_strs else '—'}")
        else:
            lines.append(f"- **Models ({len(model_strs)}):** {', '.join(model_strs)}")
        lines.append(f"- **Total instances:** {len(all_results)}")
        lines.append(f"- **Model refusals:** {n_refused} ({n_refused/max(len(all_results),1)*100:.1f}%)")
        lines.append(f"- **Total tokens processed:** {total_tokens}")
        lines.append(f"- **Avg tokens per instance:** {total_tokens/max(len(all_results),1):.0f}")
        if n_truncated:
            lines.append(f"- **Responses truncated at token limit:** {n_truncated} (strategy elicitations cut off even after retry)")

    lines.append("")
    lines.append("## Per-Dataset Summary")
    lines.append("")
    lines.append("| Model | Dataset | Instances | Accuracy | Mean ECS | H | R | CF | RO |")
    lines.append("|-------|---------|-----------|----------|----------|---|---|------|---|")

    for md in model_dataset:
        parts = md.group_name.split("_", 1)
        ds = parts[1] if len(parts) > 1 else md.group_name
        model_label = parts[0] if len(parts) > 1 else ""
        # Match results by model_id (results store the Bedrock id, not the label).
        model_id = label_to_id.get(model_label, "")
        ds_results = [r for r in all_results
                      if r.dataset == ds and (not model_id or r.model == model_id)]
        correct = sum(1 for r in ds_results if r.correct)
        acc = f"{correct}/{len(ds_results)} ({correct/max(len(ds_results),1)*100:.0f}%)" if ds_results else "—"
        lines.append(
            f"| {model_label or '—'} | {ds} | {md.n_instances} | {acc} | {md.mean_ecs:.3f} | "
            f"{md.highlighting_success_rate*100:.0f}% | {md.rationale_success_rate*100:.0f}% | "
            f"{md.counterfactual_success_rate*100:.0f}% | {md.rank_ordering_success_rate*100:.0f}% |"
        )

    if overall:
        lines.append("")
        lines.append("## Sampling Log")
        lines.append("")
        lines.append("| Model | Dataset | Requested | Sampled | Wrong Pred |")
        lines.append("|-------|---------|-----------|---------|------------|")
        for md in model_dataset:
            parts = md.group_name.split("_", 1)
            ds = parts[1] if len(parts) > 1 else md.group_name
            model_label = parts[0] if len(parts) > 1 else "—"
            lines.append(f"| {model_label} | {ds} | {md.requested_samples} | {md.sampled_samples} | {md.dropped_wrong_pred} |")
        lines.append(f"| **Total** | | {sum(m.requested_samples for m in model_dataset)} | {sum(m.sampled_samples for m in model_dataset)} | {sum(m.dropped_wrong_pred for m in model_dataset)} |")

        lines.append("")
        lines.append(f"**Note:** No long inputs (>50 words) were sampled. All instances are short (≤20 words, N={overall.n_short}) or medium-length (21–50 words, N={overall.n_medium}). ECS may partly reflect brevity.")
        lines.append("")
        lines.append(f"**Short-vocab filter:** Instances with ≤20 unique normalized tokens are flagged `short_vocab` (N={overall.n_short_vocab}). These degenerate inputs yield inflated/trivial ECS via near-identical evidence across strategies. Compare filtered results below.")

        lines.append("")
        lines.append("## Overview — Two Analysis Regimes")
        lines.append("")
        lines.append(f"### Analysis A: Complete Cases (all 4 strategies valid, N={overall.n_complete_cases})")
        lines.append("")
        lines.append("This is the clean statistical analysis: instances where Highlighting, Rationale, Counterfactual, and Rank Ordering all produced valid evidence.")
        lines.append("")
        lines.append("### Analysis B: Coverage")
        lines.append("")
        lines.append("| Strategy | Valid / Total | Rate |")
        lines.append("|----------|---------------|------|")
        n_total = len(all_results)
        h_valid = sum(1 for r in all_results if r.highlighting_valid)
        r_valid = sum(1 for r in all_results if r.rationale_valid)
        cf_valid = sum(1 for r in all_results if r.counterfactual_valid)
        ro_valid = sum(1 for r in all_results if r.rank_ordering_valid)
        lines.append(f"| Highlighting | {h_valid}/{n_total} | {h_valid/max(n_total,1)*100:.0f}% |")
        lines.append(f"| Rationale | {r_valid}/{n_total} | {r_valid/max(n_total,1)*100:.0f}% |")
        lines.append(f"| Counterfactual | {cf_valid}/{n_total} | {cf_valid/max(n_total,1)*100:.0f}% |")
        lines.append(f"| Rank Ordering | {ro_valid}/{n_total} | {ro_valid/max(n_total,1)*100:.0f}% |")
        lines.append("")
        lines.append("Coverage varies dramatically — Counterfactual is the least reliable method. This is itself a finding: explanation methods differ substantially in reliability.")
        lines.append("")
        lines.append("**Design note:** H (graded salience) and RO (ranked selection) belong to the same extraction-based paradigm. Their pairwise agreement is excluded from ECS; overlap is measured via Overlap Coefficient, rank agreement via Kendall τ and RBO. The three genuinely distinct paradigms are: extraction (H/RO), rationalization (R), and perturbation (CF).")
        lines.append("")

        lines.append("")
        lines.append("## Overall Metrics")
        lines.append("")
        lines.append("> **Pooled across models and datasets — descriptive context only.** The primary "
                     "reporting unit is the model×dataset cell (tables below): pooled numbers mix "
                     "heterogeneous tasks and models, and their bootstrap CI resamples instance "
                     "clusters (the same instance appears under every model). At scale, "
                     "**complete-case ECS is the primary estimand** — partial-case ECS averages "
                     "whichever pairs survived, which changes the construct per instance.")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| **Mean ECS (complete cases, N={overall.n_complete_cases}) — primary estimand** | {overall.mean_ecs_complete:.4f} |")
        lines.append(f"| Complete cases | {overall.n_complete_cases}/{overall.n_instances} ({overall.pct_complete_cases:.0f}%) |")
        lines.append(f"| Mean ECS (all with ≥3 valid, N={overall.n_instances}) | {overall.mean_ecs:.4f} |")
        lines.append(f"| Mean ECS-overlap (size-robust secondary, same pairs) | {overall.mean_ecs_overlap:.4f} |")
        lines.append(f"| Mean ECS (extraction–rationale: H,R,RO) | {overall.mean_ecs_extraction_rationale:.4f} |")
        lines.append(f"| Mean ECS (extraction–perturbation: H,CF,RO) | {overall.mean_ecs_extraction_perturbation:.4f} |")

        # Flag degenerate ECS — instances where <2 pairs were used
        reduced_primary = [r for r in all_results if r.ecs is not None and r.ecs_primary_pairs < 2]
        if reduced_primary:
            lines.append(f"| ECS pairs <3 | {len(reduced_primary)} instances |")
            for r in reduced_primary[:5]:
                er_s = f"{r.ecs_extraction_rationale:.4f}" if r.ecs_extraction_rationale is not None else "NA"
                ep_s = f"{r.ecs_extraction_perturbation:.4f}" if r.ecs_extraction_perturbation is not None else "NA"
                lines.append(f"  → {r.instance_id}: {r.ecs_primary_pairs} pair(s), er={er_s}, ep={ep_s}")
            if len(reduced_primary) > 5:
                lines.append(f"  → ... and {len(reduced_primary)-5} more")
        lines.append(f"| Std ECS | {overall.std_ecs:.4f} |")
        lines.append(f"| Median ECS | {overall.median_ecs:.4f} |")
        lines.append(f"| **Mean ECS lift over chance** (ECS − uniform random) | {overall.mean_ecs_lift:+.4f} |")
        lines.append(f"| Mean ECS random baseline (uniform) | {overall.mean_ecs_random:.4f} |")
        lines.append(f"| Mean ECS lift over salience-weighted null (secondary, N={overall.n_lift_weighted}) | {overall.mean_ecs_lift_weighted:+.4f} |")
        lines.append("")
        lines.append("> **Significance testing: pre-registered tests only.** Exactly two test families "
                     "run (FIX_PLAN §P1.3): (a) sign-flip permutation on per-instance ECS-lift per "
                     "model×dataset cell, Holm-corrected across cells — results in the table below; "
                     "(b) CC-erasure vs random control in the separate erasure pass. Every other "
                     "number in this report — strata, splits, contrasts — is descriptive, and cells "
                     "below the configured minimum N report estimates without a test.")
        lines.append("")
        lines.append(f"| Introduced-concept rate (R) | {overall.introduced_concept_rate:.3f} |")
        lines.append(f"| CF canonical (minimal) validity rate | {overall.cf_canonical_validity_rate*100:.0f}% |")
        lines.append(f"| CF canonical validity — first attempt (single-shot, uncoached) | {overall.cf_first_attempt_validity_rate*100:.0f}% |")
        lines.append(f"| CF valid only after correction loop (coached stratum) | {overall.n_cf_corrected} instances |")
        lines.append(f"| RO required hallucination self-correction | {overall.n_ro_self_corrected} instances |")
        lines.append(f"| CF contrast (free) validity rate | {overall.cf_contrast_validity_rate*100:.0f}% |")
        lines.append(f"| CF canonical (minimal) minimality (edits/len) | {overall.mean_cf_canonical_minimality:.3f} |")
        lines.append(f"| CF contrast (free) minimality (edits/len) | {overall.mean_cf_contrast_minimality:.3f} |")
        lines.append(f"| Verbalized confidence — mean (N={overall.n_confidence}) | {overall.mean_confidence:.3f} |")
        lines.append(f"| Mean CC3 size | {overall.mean_cc3_size:.2f} |")
        lines.append(f"| Mean CC4 size | {overall.mean_cc4_size:.2f} |")
        lines.append(f"| % instances with CC3 | {overall.pct_instances_with_cc3:.1f}% |")
        lines.append(f"| % instances with CC4 | {overall.pct_instances_with_cc4:.1f}% |")
        lines.append("")
        lines.append("> The CF minimal-vs-free validity contrast replicates the validity–minimality "
                     "trade-off of arXiv:2509.09396 (*LLMs Don't Know Their Own Decision Boundaries*): "
                     "unconstrained rewrites flip reliably but over-edit; minimal edits often fail to "
                     "flip. Single-shot rates are reported separately from the coached loop because the "
                     "correction re-prompt makes headline rates multi-shot-search rates (not comparable "
                     "to single-shot elicitation in Madsen et al. 2024 / MiCE).")
        lines.append("")
        lines.append("> **Framing.** ECS is an inter-method **consistency** measure, reported as **lift over a random "
                     "baseline** (raw overlap is confounded by set size and vocabulary). It is **not** a faithfulness "
                     "score. The erasure pass (`aggregate_erasure.json`, run separately via `run_validity_tests.py`) is a "
                     "**second consistency axis** (comprehensiveness-style, stated-vs-revealed input sensitivity) — not "
                     "ground truth; its headline is Consensus-Core erasure vs. a same-size random control, by ECS-lift tier.")

        lines.append("")
        lines.append("### Pre-registered test (a): mean ECS-lift > 0, per model×dataset cell")
        lines.append("")
        lines.append("One-sided sign-flip permutation on per-instance (ECS − ECS_random) differences; "
                     "Holm-corrected across this run's cells. `—` = cell below the configured minimum N "
                     "for testing (estimate reported, test skipped).")
        lines.append("")
        lines.append("| Model | Dataset | N (lift) | Mean lift | p (raw) | p (Holm) |")
        lines.append("|-------|---------|----------|-----------|---------|----------|")
        for md in model_dataset:
            parts = md.group_name.split("_", 1)
            ds = parts[1] if len(parts) > 1 else md.group_name
            model_label = parts[0] if len(parts) > 1 else "—"
            p_raw = f"{md.ecs_lift_p_value:.4f}" if md.ecs_lift_p_value is not None else "—"
            p_holm = f"{md.ecs_lift_p_holm:.4f}" if md.ecs_lift_p_holm is not None else "—"
            lines.append(f"| {model_label} | {ds} | {md.n_lift} | {md.mean_ecs_lift:+.4f} | {p_raw} | {p_holm} |")

        lines.append("")
        lines.append("### ECS by prediction correctness (per cell only)")
        lines.append("")
        lines.append("Reported per model×dataset cell only — pooled, this contrast is confounded by "
                     "cell composition. Descriptive; no test.")
        lines.append("")
        lines.append("| Model | Dataset | Mean ECS correct (N) | Mean ECS incorrect (N) |")
        lines.append("|-------|---------|----------------------|------------------------|")
        for md in model_dataset:
            parts = md.group_name.split("_", 1)
            ds = parts[1] if len(parts) > 1 else md.group_name
            model_label = parts[0] if len(parts) > 1 else "—"
            lines.append(f"| {model_label} | {ds} | {md.mean_ecs_correct:.4f} ({md.n_correct}) | "
                         f"{md.mean_ecs_incorrect:.4f} ({md.n_incorrect}) |")

        lines.append("")
        lines.append("### Verbalized confidence ↔ ECS (Spearman, per cell)")
        lines.append("")
        lines.append("Association estimate with a seeded bootstrap CI (pre-registered as an estimate, "
                     "not a hypothesis test). Confidence is the model's verbalized 0–100 probability "
                     "that its classification is correct (Tian et al. 2023; Xiong et al. 2024).")
        lines.append("")
        lines.append("| Model | Dataset | N pairs | Spearman ρ | 95% CI |")
        lines.append("|-------|---------|---------|------------|--------|")
        for md in model_dataset:
            parts = md.group_name.split("_", 1)
            ds = parts[1] if len(parts) > 1 else md.group_name
            model_label = parts[0] if len(parts) > 1 else "—"
            lines.append(f"| {model_label} | {ds} | {md.n_confidence} | {md.spearman_rho:.3f} | "
                         f"[{md.correlation_ci_lower:.3f}, {md.correlation_ci_upper:.3f}] |")

        lines.append("")
        lines.append("### Pairwise Agreement (Overlap Coefficient)")
        lines.append("")
        _pn = overall.pair_ns or {}
        lines.append("| Pair | Overlap Coeff (mean) | Jaccard (mean) | N instances |")
        lines.append("|------|----------------------|----------------|-------------|")
        lines.append(f"| H–R | {overall.mean_overlap_H_R:.4f} | {overall.mean_jaccard_H_R:.4f} | {_pn.get('jaccard_H_R', 0)} |")
        lines.append(f"| H–CF | {overall.mean_overlap_H_CF:.4f} | {overall.mean_jaccard_H_CF:.4f} | {_pn.get('jaccard_H_CF', 0)} |")
        lines.append(f"| H–RO | {overall.mean_overlap_H_RO:.4f} | {overall.mean_jaccard_H_RO:.4f} | {_pn.get('jaccard_H_RO', 0)} |")
        lines.append(f"| R–CF | {overall.mean_overlap_R_CF:.4f} | {overall.mean_jaccard_R_CF:.4f} | {_pn.get('jaccard_R_CF', 0)} |")
        lines.append(f"| R–RO | {overall.mean_overlap_R_RO:.4f} | {overall.mean_jaccard_R_RO:.4f} | {_pn.get('jaccard_R_RO', 0)} |")
        lines.append(f"| CF–RO | {overall.mean_overlap_CF_RO:.4f} | {overall.mean_jaccard_CF_RO:.4f} | {_pn.get('jaccard_CF_RO', 0)} |")
        lines.append("")
        lines.append("### Rank-Based Agreement (H vs RO)")
        lines.append("")
        rbo_val = overall.mean_rbo_H_RO if hasattr(overall, 'mean_rbo_H_RO') else 0.0
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| RBO (H,RO) | {rbo_val:.4f} |" if rbo_val != 0.0 else "| RBO (H,RO) | — |")
        lines.append(f"| Kendall τ (H,RO) | {overall.mean_kendall_H_RO:.4f} |")
        lines.append(f"| Normalized τ | {overall.mean_normalized_kendall_H_RO:.4f} |")
        lines.append("")
        lines.append("> Jaccard is what feeds the headline ECS and ECS-lift (both computed from Jaccard pairwise agreement, never Overlap Coefficient). Overlap Coefficient is reported as a size-robust complement — unlike Jaccard, it is not penalized by smaller salience sets — and its cross-paradigm mean is the `ECS-overlap` secondary composite. RBO measures top-weighted rank agreement between H's graded salience order and RO's selected ranking (both rankings live in the same normalized token space); Kendall τ provides a complementary rank correlation measure.")

        if cross_model:
            lines.append("")
            lines.append("## Cross-Model Agreement (same strategy, different models)")
            lines.append("")
            lines.append("For every instance run under ≥2 models: the Jaccard between DIFFERENT models' "
                         "evidence sets for the SAME strategy, next to the within-model cross-strategy "
                         "ECS of the same instances. If within-model consensus systematically exceeds "
                         "cross-model same-strategy agreement, stated evidence tracks model-specific "
                         "computation (privileged self-knowledge, arXiv:2602.02639); if not, it is "
                         "closer to a generic task prior shared across models (cf. the cross-model "
                         "explanation lottery, arXiv:2603.15821). Zero extra API calls; descriptive.")
            lines.append("")
            lines.append("| Dataset | N instances | H | R | CF | RO | Cross-model mean | Within-model mean ECS |")
            lines.append("|---------|-------------|---|---|----|----|------------------|------------------------|")
            for ds, entry in sorted(cross_model.items()):
                strat = entry.get("strategies", {})

                def _cell(s):
                    v = strat.get(s, {}).get("mean_jaccard")
                    n = strat.get(s, {}).get("n_pairs", 0)
                    return f"{v:.3f} ({n})" if v is not None else f"— ({n})"

                xm = entry.get("cross_model_same_strategy_mean")
                wm = entry.get("within_model_cross_strategy_mean_ecs")
                xm_s = f"{xm:.3f}" if xm is not None else "—"
                wm_s = f"{wm:.3f}" if wm is not None else "—"
                lines.append(
                    f"| {ds} | {entry.get('n_instances_multi_model', 0)} | {_cell('H')} | {_cell('R')} | "
                    f"{_cell('CF')} | {_cell('RO')} | {xm_s} | {wm_s} |")

        lines.append("")
        lines.append("### Validation Rates (strict)")
        lines.append("")
        lines.append("| Strategy | Parsed | Valid |")
        lines.append("|----------|--------|-------|")
        n_total = len(all_results)
        h_parsed = sum(1 for r in all_results if r.highlighting_parsed)
        h_valid = sum(1 for r in all_results if r.highlighting_valid)
        r_parsed = sum(1 for r in all_results if r.rationale_parsed)
        r_valid = sum(1 for r in all_results if r.rationale_valid)
        cf_parsed = sum(1 for r in all_results if r.counterfactual_parsed)
        cf_valid = sum(1 for r in all_results if r.counterfactual_valid)
        ro_parsed = sum(1 for r in all_results if r.rank_ordering_parsed)
        ro_valid = sum(1 for r in all_results if r.rank_ordering_valid)
        lines.append(f"| Highlighting | {h_parsed}/{n_total} ({h_parsed/max(n_total,1)*100:.0f}%) | {h_valid}/{n_total} ({h_valid/max(n_total,1)*100:.0f}%) |")
        lines.append(f"| Rationale | {r_parsed}/{n_total} ({r_parsed/max(n_total,1)*100:.0f}%) | {r_valid}/{n_total} ({r_valid/max(n_total,1)*100:.0f}%) |")
        cf_json_ok = sum(1 for r in all_results if r.cf_json_valid)
        cf_rules_ok = sum(1 for r in all_results if r.cf_rules_compliant)
        cf_verified = sum(1 for r in all_results if r.cf_flip_verified)
        lines.append(f"| Counterfactual (JSON parsed) | {cf_json_ok}/{n_total} ({cf_json_ok/max(n_total,1)*100:.0f}%) |")
        lines.append(f"| Counterfactual (rules compliant) | {cf_rules_ok}/{n_total} ({cf_rules_ok/max(n_total,1)*100:.0f}%) |")
        lines.append(f"| Counterfactual (flip verified) | {cf_verified}/{cf_rules_ok} ({cf_verified/max(cf_rules_ok,1)*100:.0f}%) |")
        lines.append(f"| Rank Ordering | {ro_parsed}/{n_total} ({ro_parsed/max(n_total,1)*100:.0f}%) | {ro_valid}/{n_total} ({ro_valid/max(n_total,1)*100:.0f}%) |")
        lines.append(f"| All 4 valid | {sum(1 for r in all_results if r.n_valid_strategies == 4)}/{n_total} |")
        lines.append(f"| 3 valid (CC3 adjusted) | {sum(1 for r in all_results if r.n_valid_strategies == 3)}/{n_total} |")
        lines.append("")
        lines.append("CC is computed over valid strategies only. When CF is missing, CC3-of-3 replaces CC4-of-4.")
        lines.append("ECS is computed using validated outputs only. The H–RO pair is excluded (same paradigm family). Full ECS averages the 5 remaining cross-paradigm pairs (all Jaccard). Primary ECS is two composites: extraction–rationale averages (H,R) and (R,RO); extraction–perturbation averages (H,CF) and (CF,RO). H–RO agreement is measured via Overlap Coefficient (set), Kendall τ (rank correlation), and RBO (top-weighted rank).")

        lines.append("")
        lines.append("### ECS by Input Length")
        lines.append("")
        lines.append("| Length | N | Mean ECS |")
        lines.append("|--------|---|----------|")
        lines.append(f"| Short (≤20 words) | {overall.n_short} | {overall.mean_ecs_short:.4f} |")
        lines.append(f"| Medium (21–50) | {overall.n_medium} | {overall.mean_ecs_medium:.4f} |")
        lines.append(f"| Long (>50 words) | {overall.n_long} | {overall.mean_ecs_long:.4f} |")

        lines.append("")
        lines.append("")
        lines.append("### ECS by Vocabulary Size")
        lines.append("")
        lines.append("| Vocab Bucket | N | Mean ECS |")
        lines.append("|--------------|---|----------|")
        lines.append(f"| Normal vocab (>20 unique tokens) | {overall.n_normal_vocab} | {overall.mean_ecs_normal_vocab:.4f} |")
        lines.append(f"| Short vocab (≤20 unique tokens) | {overall.n_short_vocab} | {overall.mean_ecs_short_vocab:.4f} |")
        if overall.n_normal_vocab > 0:
            lines.append("")
            lines.append("> The normal-vocab column gives a more conservative, reliable ECS estimate by excluding degenerate instances.")

        lines.append("")
        lines.append("### Confidence Intervals")
        lines.append("")
        lines.append("| Metric | Estimate | 95% CI |")
        lines.append("|--------|----------|--------|")
        lines.append(f"| Mean ECS | {overall.mean_ecs:.4f} | [{overall.ecs_ci_lower:.4f}, {overall.ecs_ci_upper:.4f}] |")

    lines.append("")
    lines.append("## Per-Instance Details")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r.instance_id} — {_model_label(r.model)}")
        lines.append("")
        # ONLY the unparseable-classification case is actually skipped by the pipeline
        # (no usable label -> nothing to explain). Wrong-but-parseable predictions get
        # the full elicitation per D5 and are rendered in full below with a stratum
        # banner — the old renderer claimed they were skipped while the aggregates
        # included their ECS (review §8.5, a report-integrity contradiction).
        _pipeline_skipped = (r.n_valid_strategies == 0 and not r.raw_highlighting
                             and not r.raw_rationale and not r.raw_counterfactual
                             and not r.raw_rank_ordering and r.ecs is None)
        if _pipeline_skipped:
            lines.append("> **SKIPPED**: classification response unparseable — no usable label, "
                         "so no explanation strategies were elicited and no ECS was computed.")
            lines.append("")
            lines.append(f"- **Ground truth:** `{r.ground_truth_label}`")
            lines.append(f"- **Predicted:** `{r.predicted_label}` ✗")
            lines.append(f"- **Input length:** {r.input_length} words")
            lines.append(f"- **Raw response length:** {r.raw_response_length} chars")
            lines.append("")
            continue
        if not r.correct:
            lines.append("> ⚠ **Wrong prediction** — included per D5 (incorrect-prediction stratum). "
                         "Explanations below are elicited for the model's OWN (incorrect) label; "
                         "ECS/erasure are defined relative to that prediction.")
            lines.append("")

        lines.append("#### 1. Classification Prompt")
        lines.append("```")
        lines.append(r.classification_prompt)
        lines.append("```")
        lines.append("")
        lines.append("#### 2. Classification Response")
        lines.append("```")
        lines.append(r.classification_raw_response)
        lines.append("```")
        lines.append("")
        lines.append("#### 3. Ground Truth & Prediction")
        lines.append(f"- **Ground truth:** `{r.ground_truth_label}`")
        lines.append(f"- **Predicted:** `{r.predicted_label}` {'✓' if r.correct else '✗'}")
        lines.append(f"- **Verbalized confidence:** {f'{r.confidence:.2f}' if r.confidence is not None else '—'}")
        lines.append(f"- **Model refused:** {'Yes' if r.model_refused else 'No'}")
        lines.append(f"- **Input length:** {r.input_length} words")
        lines.append(f"- **Raw response length:** {r.raw_response_length} chars")
        lines.append(f"- **Prompt hash:** `{r.prompt_hash}`")
        lines.append(f"- **ECS (full):** {r.ecs:.4f}" if r.ecs is not None else "- **ECS (full):** —")
        lines.append(f"- **ECS (extraction–rationale):** {r.ecs_extraction_rationale:.4f}" if r.ecs_extraction_rationale is not None else "- **ECS (extraction–rationale):** —")
        lines.append(f"- **ECS (extraction–perturbation):** {r.ecs_extraction_perturbation:.4f}" if r.ecs_extraction_perturbation is not None else "- **ECS (extraction–perturbation):** —")
        lines.append(f"- **ECS primary pairs:** {r.ecs_primary_pairs}")
        lines.append(f"- **Valid strategies:** {r.n_valid_strategies}")
        lines.append(f"- **CF flip verified:** {'Yes' if r.cf_flip_verified else 'No'}")
        if r.cf_actual_label:
            lines.append(f"- **CF actual label:** `{r.cf_actual_label}`")
        lines.append(f"- **CC3 size:** {r.cc3_size} | **CC4 size:** {r.cc4_size}")
        lines.append("")
        strategy_info = [
            ("H", "Highlighting", r.highlighting_explain_prompt, r.raw_highlighting,
             r.highlighting_tokens, r.highlighting_valid, r.highlighting_parsed),
            ("R", "Rationale", r.rationale_explain_prompt, r.raw_rationale,
             r.rationale_tokens, r.rationale_valid, r.rationale_parsed),
            ("CF", "Counterfactual", r.counterfactual_explain_prompt, r.raw_counterfactual,
             r.counterfactual_tokens, r.counterfactual_valid, r.counterfactual_parsed),
            ("RO", "Rank Ordering", r.rank_ordering_explain_prompt, r.raw_rank_ordering,
             r.rank_ordering_tokens, r.rank_ordering_valid, r.rank_ordering_parsed),
        ]
        for sid, sname, sprompt, sraw, stokens, svalid, sparsed in strategy_info:
            lines.append(f"#### 4. {sname} Explanation")
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("```")
            lines.append(sprompt)
            lines.append("```")
            lines.append("")
            lines.append("**Response:**")
            lines.append("```")
            lines.append(sraw)
            lines.append("```")
            lines.append("")
            if sid == "R" and r.rationale_text:
                lines.append("**Rationale text:**")
                lines.append(f"> {r.rationale_text}")
                lines.append("")
            if sid == "CF":
                lines.append("**CF stages:**")
                lines.append(f"- JSON parsed: {'Yes' if r.cf_json_valid else 'No'}")
                lines.append(f"- Rules compliant: {'Yes' if r.cf_rules_compliant else 'No'}")
                lines.append(f"- Flip verified: {'Yes' if r.cf_flip_verified else 'No'}")
                if r.cf_actual_label:
                    lines.append(f"- Actual label: `{r.cf_actual_label}`")
                lines.append("")
            if sid == "CF" and r.cf_counterfactual_text:
                lines.append("**Reconstructed text:**")
                lines.append(f"> {r.cf_counterfactual_text}")
                lines.append("")
            lines.append("**Parsed tokens:**")
            if sparsed:
                if svalid:
                    status = "Valid"
                else:
                    status = "Parsed but failed validation"
                lines.append(f"- Status: **{status}**")
                if sid == "RO":
                    token_list = ", ".join(f"`{t}`({r})" for t, r in stokens)
                else:
                    token_list = ", ".join(f"`{t}`" for t in sorted(stokens))
                lines.append(f"- Tokens: {token_list}")
            else:
                lines.append("- *Not parsed*")
            lines.append("")
        lines.append("#### 5. Pairwise Agreement")
        lines.append("")
        lines.append("| Pair | Overlap Coeff | Jaccard |")
        lines.append("|------|---------------|---------|")
        ov_pairs = [("H–R", "overlap_H_R", "jaccard_H_R"), ("H–CF", "overlap_H_CF", "jaccard_H_CF"),
                    ("H–RO", "overlap_H_RO", "jaccard_H_RO"), ("R–CF", "overlap_R_CF", "jaccard_R_CF"),
                    ("R–RO", "overlap_R_RO", "jaccard_R_RO"), ("CF–RO", "overlap_CF_RO", "jaccard_CF_RO")]
        for label, ov_attr, j_attr in ov_pairs:
            ov = getattr(r, ov_attr, None)
            jv = getattr(r, j_attr, None)
            ov_str = f"{ov:.4f}" if ov is not None else "—"
            jv_str = f"{jv:.4f}" if jv is not None else "—"
            lines.append(f"| {label} | {ov_str} | {jv_str} |")
        kv = r.kendall_H_RO
        nkv = r.normalized_kendall_H_RO
        rbo = r.rbo_H_RO
        lines.append(f"| Kendall τ (H,RO) | {kv:.4f} |" if kv is not None else "| Kendall τ (H,RO) | — |")
        lines.append(f"| Normalized τ | {nkv:.4f} |" if nkv is not None else "| Normalized τ | — |")
        lines.append(f"| RBO (H,RO) | {rbo:.4f} |" if rbo is not None else "| RBO (H,RO) | — |")
        lines.append("")

    # High/Low ECS case studies
    extremes = extract_high_low_ecs_examples(all_results, n=3)
    for category, label in [("high_ecs", "High-ECS Examples"), ("low_ecs", "Low-ECS Examples")]:
        examples = extremes.get(category, [])
        if examples:
            lines.append("")
            lines.append(f"## {label}")
            lines.append("")
            for r in examples:
                ecs_s = f"{r.ecs:.4f}" if r.ecs is not None else "NA"
                lines.append(f"### {r.instance_id} (ECS={ecs_s})")
                lines.append(f"- **Dataset:** {r.dataset}")
                lines.append(f"- **Text:** {r.text[:200]}{'…' if len(r.text) > 200 else ''}")
                lines.append(f"- **Ground truth:** `{r.ground_truth_label}` → **Predicted:** `{r.predicted_label}` {'✓' if r.correct else '✗'}")
                lines.append("")

    report = "\n".join(lines)
    _check_unresolved_placeholders(report)
    return report


def save_instance_results(results: List[InstanceResult], filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result.to_dict(), f)
            f.write('\n')


def load_instance_results(filepath: str) -> List[InstanceResult]:
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                results.append(InstanceResult.from_dict(data))
    return results


def save_aggregate_metrics(metrics: List[AggregateMetrics], filepath: str) -> None:
    data = [m.to_dict() for m in metrics]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_aggregate_metrics(filepath: str) -> List[AggregateMetrics]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [AggregateMetrics.from_dict(m) for m in data]


def save_validity_results(results: List[ValidityTestResult], filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        for result in results:
            json.dump(result.to_dict(), f)
            f.write('\n')


def load_validity_results(filepath: str) -> List[ValidityTestResult]:
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                results.append(ValidityTestResult.from_dict(data))
    return results


def save_metrics_csv(results: List[InstanceResult], filepath: str) -> None:
    if not results:
        return
    fieldnames = [
        'instance_id', 'dataset', 'model', 'ground_truth_label', 'predicted_label',
        'confidence', 'correct', 'model_refused', 'prompt_tokens', 'response_tokens',
        'prompt_hash',
        'jaccard_H_R', 'jaccard_H_CF', 'jaccard_H_RO', 'jaccard_R_CF', 'jaccard_R_RO', 'jaccard_CF_RO',
        'overlap_H_R', 'overlap_H_CF', 'overlap_H_RO', 'overlap_R_CF', 'overlap_R_RO', 'overlap_CF_RO',
        'rbo_H_RO', 'kendall_H_RO', 'normalized_kendall_H_RO', 'ecs', 'ecs_overlap',
        'ecs_random', 'ecs_lift', 'ecs_random_weighted', 'ecs_lift_weighted',
        'ecs_extraction_rationale', 'ecs_extraction_perturbation', 'ecs_primary_pairs',
        'cf_flip_verified', 'cf_valid_first_attempt', 'cf_corrected', 'ro_self_corrected',
        'cf_actual_label',
        'highlighting_parsed', 'rationale_parsed', 'counterfactual_parsed', 'rank_ordering_parsed',
        'cc3_size', 'cc4_size',
    ]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def save_metadata_table(data: List[Dict[str, Any]], table_name: str, filepath: str) -> None:
    metadata = {"table": table_name, "entries": data, "timestamp": datetime.now().isoformat()}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def save_environment_snapshot(output_path: Path) -> None:
    import subprocess
    import importlib.metadata as md
    snapshot = {}

    # Git commit
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            snapshot["git_commit"] = result.stdout.strip()
        result = subprocess.run(["git", "log", "-1", "--format=%ai"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            snapshot["git_commit_date"] = result.stdout.strip()
    except Exception:
        snapshot["git_commit"] = "unknown"

    # Python version
    import sys as _sys
    snapshot["python_version"] = f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}"

    # Package versions
    packages = {}
    for dist in md.distributions():
        name = dist.metadata.get("Name", "")
        ver = dist.version
        packages[name.lower()] = ver
    snapshot["packages"] = packages

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2)


def extract_high_low_ecs_examples(results: List[InstanceResult], n: int = 5) -> Dict[str, List[InstanceResult]]:
    valid = [r for r in results if r.ecs is not None]
    valid.sort(key=lambda r: r.ecs)
    low = valid[:n]
    high = valid[-n:] if len(valid) >= n else valid[::-1]
    high.reverse()
    return {"high_ecs": high, "low_ecs": low}
