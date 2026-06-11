import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Set, List, Tuple, Dict, Optional, Any


@dataclass
class InstanceResult:
    instance_id: str
    dataset: str
    model: str
    timestamp: datetime
    text: str
    ground_truth_label: str
    predicted_label: str
    confidence: float
    correct: bool
    raw_highlighting: str
    raw_rationale: str
    raw_counterfactual: str
    raw_rank_ordering: str
    highlighting_tokens: Set[str] = field(default_factory=set)
    rationale_tokens: Set[str] = field(default_factory=set)
    counterfactual_tokens: Set[str] = field(default_factory=set)
    rank_ordering_tokens: List[Tuple[str, int]] = field(default_factory=list)
    highlighting_parsed: bool = False
    rationale_parsed: bool = False
    counterfactual_parsed: bool = False
    rank_ordering_parsed: bool = False
    jaccard_H_R: Optional[float] = None
    jaccard_H_CF: Optional[float] = None
    jaccard_H_RO: Optional[float] = None
    jaccard_R_CF: Optional[float] = None
    jaccard_R_RO: Optional[float] = None
    jaccard_CF_RO: Optional[float] = None
    kendall_H_RO: Optional[float] = None
    ecs: Optional[float] = None
    cc3_tokens: Set[str] = field(default_factory=set)
    cc4_tokens: Set[str] = field(default_factory=set)
    cc3_size: int = 0
    cc4_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'instance_id': self.instance_id, 'dataset': self.dataset, 'model': self.model,
            'timestamp': self.timestamp.isoformat(),
            'text': self.text, 'ground_truth_label': self.ground_truth_label,
            'predicted_label': self.predicted_label, 'confidence': self.confidence, 'correct': self.correct,
            'raw_highlighting': self.raw_highlighting, 'raw_rationale': self.raw_rationale,
            'raw_counterfactual': self.raw_counterfactual, 'raw_rank_ordering': self.raw_rank_ordering,
            'highlighting_tokens': sorted(list(self.highlighting_tokens)),
            'rationale_tokens': sorted(list(self.rationale_tokens)),
            'counterfactual_tokens': sorted(list(self.counterfactual_tokens)),
            'rank_ordering_tokens': [[token, rank] for token, rank in self.rank_ordering_tokens],
            'highlighting_parsed': self.highlighting_parsed, 'rationale_parsed': self.rationale_parsed,
            'counterfactual_parsed': self.counterfactual_parsed, 'rank_ordering_parsed': self.rank_ordering_parsed,
            'jaccard_H_R': self.jaccard_H_R, 'jaccard_H_CF': self.jaccard_H_CF,
            'jaccard_H_RO': self.jaccard_H_RO, 'jaccard_R_CF': self.jaccard_R_CF,
            'jaccard_R_RO': self.jaccard_R_RO, 'jaccard_CF_RO': self.jaccard_CF_RO,
            'kendall_H_RO': self.kendall_H_RO,
            'ecs': self.ecs,
            'cc3_tokens': sorted(list(self.cc3_tokens)), 'cc4_tokens': sorted(list(self.cc4_tokens)),
            'cc3_size': self.cc3_size, 'cc4_size': self.cc4_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstanceResult':
        return cls(
            instance_id=data['instance_id'], dataset=data['dataset'], model=data['model'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            text=data['text'], ground_truth_label=data['ground_truth_label'],
            predicted_label=data['predicted_label'], confidence=data['confidence'], correct=data['correct'],
            raw_highlighting=data['raw_highlighting'], raw_rationale=data['raw_rationale'],
            raw_counterfactual=data['raw_counterfactual'], raw_rank_ordering=data['raw_rank_ordering'],
            highlighting_tokens=set(data['highlighting_tokens']),
            rationale_tokens=set(data['rationale_tokens']),
            counterfactual_tokens=set(data['counterfactual_tokens']),
            rank_ordering_tokens=[(token, rank) for token, rank in data['rank_ordering_tokens']],
            highlighting_parsed=data['highlighting_parsed'], rationale_parsed=data['rationale_parsed'],
            counterfactual_parsed=data['counterfactual_parsed'], rank_ordering_parsed=data['rank_ordering_parsed'],
            jaccard_H_R=data.get('jaccard_H_R'), jaccard_H_CF=data.get('jaccard_H_CF'),
            jaccard_H_RO=data.get('jaccard_H_RO'), jaccard_R_CF=data.get('jaccard_R_CF'),
            jaccard_R_RO=data.get('jaccard_R_RO'), jaccard_CF_RO=data.get('jaccard_CF_RO'),
            kendall_H_RO=data.get('kendall_H_RO'),
            ecs=data.get('ecs'),
            cc3_tokens=set(data['cc3_tokens']), cc4_tokens=set(data['cc4_tokens']),
            cc3_size=data['cc3_size'], cc4_size=data['cc4_size'],
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
    mean_kendall_H_RO: float
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
class CorrelationResult:
    rho: float
    p_value: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorrelationResult':
        return cls(**data)


@dataclass
class StatisticalTest:
    test_statistic: float
    p_value: float
    mean_diff: Optional[float] = None
    effect_size: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StatisticalTest':
        return cls(**data)


@dataclass
class FlipResult:
    original_prediction: str
    masked_prediction: str
    flipped: bool
    masked_tokens: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_prediction': self.original_prediction,
            'masked_prediction': self.masked_prediction,
            'flipped': self.flipped,
            'masked_tokens': sorted(list(self.masked_tokens)),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlipResult':
        return cls(
            original_prediction=data['original_prediction'],
            masked_prediction=data['masked_prediction'],
            flipped=data['flipped'],
            masked_tokens=set(data['masked_tokens']),
        )


@dataclass
class ExecutionSummary:
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_instances: int
    successful_instances: int
    failed_instances: int
    parsing_failures: Dict[str, int] = field(default_factory=dict)
    api_failures: int = 0
    normalization_failures: int = 0
    avg_time_per_instance: float = 0.0
    api_requests_total: int = 0
    api_requests_failed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'total_instances': self.total_instances,
            'successful_instances': self.successful_instances,
            'failed_instances': self.failed_instances,
            'parsing_failures': self.parsing_failures,
            'api_failures': self.api_failures,
            'normalization_failures': self.normalization_failures,
            'avg_time_per_instance': self.avg_time_per_instance,
            'api_requests_total': self.api_requests_total,
            'api_requests_failed': self.api_requests_failed,
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
            parsing_failures=data.get('parsing_failures', {}),
            api_failures=data.get('api_failures', 0),
            normalization_failures=data.get('normalization_failures', 0),
            avg_time_per_instance=data.get('avg_time_per_instance', 0.0),
            api_requests_total=data.get('api_requests_total', 0),
            api_requests_failed=data.get('api_requests_failed', 0),
        )

    def generate_report(self) -> str:
        success_rate = (self.successful_instances / self.total_instances * 100) if self.total_instances > 0 else 0
        api_success_rate = ((self.api_requests_total - self.api_requests_failed) / self.api_requests_total * 100) if self.api_requests_total > 0 else 0
        report = f"""
Execution Summary
=================
Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
End Time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {self.duration_seconds:.2f} seconds ({self.duration_seconds / 60:.2f} minutes)

Processing Statistics
---------------------
Total Instances: {self.total_instances}
Successful: {self.successful_instances} ({success_rate:.1f}%)
Failed: {self.failed_instances} ({100 - success_rate:.1f}%)

Failure Breakdown
-----------------
API Failures: {self.api_failures}
Normalization Failures: {self.normalization_failures}
Parsing Failures by Strategy:
"""
        for strategy, count in sorted(self.parsing_failures.items()):
            report += f"  {strategy}: {count}\n"

        report += f"""
Performance Metrics
-------------------
Average Time per Instance: {self.avg_time_per_instance:.2f} seconds
Total API Requests: {self.api_requests_total}
Failed API Requests: {self.api_requests_failed} ({100 - api_success_rate:.1f}%)
"""
        return report.strip()


def generate_md_report(
    aggregate_list: List[AggregateMetrics],
    all_results: List[InstanceResult],
    config,
) -> str:
    overall = next((m for m in aggregate_list if m.aggregation_level == "overall"), None)
    model_dataset = [m for m in aggregate_list if m.aggregation_level == "model_dataset"]

    lines = []
    exp_name = config.experiment.name if hasattr(config, 'experiment') else "experiment"
    lines.append(f"# Experiment Report: {exp_name}")
    lines.append("")

    if all_results:
        t0 = min(r.timestamp for r in all_results)
        t1 = max(r.timestamp for r in all_results)
        dur = (t1 - t0).total_seconds()
        lines.append(f"- **Date:** {t0.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Duration:** {dur:.1f}s ({dur/60:.1f}m)")
        lines.append(f"- **Model:** {all_results[0].model}")
        lines.append(f"- **Total instances:** {len(all_results)}")

    lines.append("")
    lines.append("## Per-Dataset Summary")
    lines.append("")
    lines.append("| Dataset | Instances | Accuracy | Mean ECS | H | R | CF | RO |")
    lines.append("|---------|-----------|----------|----------|---|---|------|---|")

    for md in model_dataset:
        parts = md.group_name.split("_", 1)
        ds = parts[1] if len(parts) > 1 else md.group_name
        model_name = parts[0] if len(parts) > 1 else ""
        ds_results = [r for r in all_results if r.dataset == ds and (not model_name or r.model.startswith(model_name))]
        correct = sum(1 for r in ds_results if r.correct)
        acc = f"{correct}/{len(ds_results)} ({correct/max(len(ds_results),1)*100:.0f}%)" if ds_results else "—"
        lines.append(
            f"| {ds} | {md.n_instances} | {acc} | {md.mean_ecs:.3f} | "
            f"{md.highlighting_success_rate*100:.0f}% | {md.rationale_success_rate*100:.0f}% | "
            f"{md.counterfactual_success_rate*100:.0f}% | {md.rank_ordering_success_rate*100:.0f}% |"
        )

    if overall:
        lines.append("")
        lines.append("## Overall Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Mean ECS | {overall.mean_ecs:.4f} |")
        lines.append(f"| Std ECS | {overall.std_ecs:.4f} |")
        lines.append(f"| Median ECS | {overall.median_ecs:.4f} |")
        lines.append(f"| Spearman ρ | {overall.spearman_rho:.4f} (p={overall.spearman_p_value:.4f}) |")
        lines.append(f"| Mean CC3 size | {overall.mean_cc3_size:.2f} |")
        lines.append(f"| Mean CC4 size | {overall.mean_cc4_size:.2f} |")
        lines.append(f"| % instances with CC3 | {overall.pct_instances_with_cc3:.1f}% |")
        lines.append(f"| % instances with CC4 | {overall.pct_instances_with_cc4:.1f}% |")

        lines.append("")
        lines.append("### Pairwise Jaccard Similarity")
        lines.append("")
        lines.append("| Pair | Mean |")
        lines.append("|------|------|")
        lines.append(f"| H–R | {overall.mean_jaccard_H_R:.4f} |")
        lines.append(f"| H–CF | {overall.mean_jaccard_H_CF:.4f} |")
        lines.append(f"| H–RO | {overall.mean_jaccard_H_RO:.4f} |")
        lines.append(f"| R–CF | {overall.mean_jaccard_R_CF:.4f} |")
        lines.append(f"| R–RO | {overall.mean_jaccard_R_RO:.4f} |")
        lines.append(f"| CF–RO | {overall.mean_jaccard_CF_RO:.4f} |")

    lines.append("")
    lines.append("## Per-Instance Details")
    lines.append("")
    for r in all_results:
        lines.append(f"### {r.instance_id}")
        lines.append(f"- **Dataset:** {r.dataset}")
        lines.append(f"- **Text:** {r.text[:120]}{'…' if len(r.text) > 120 else ''}")
        lines.append(f"- **Ground truth:** `{r.ground_truth_label}`")
        lines.append(f"- **Predicted:** `{r.predicted_label}`")
        lines.append(f"- **Confidence:** {r.confidence * 100:.1f}%")
        lines.append(f"- **Correct:** {'✓' if r.correct else '✗'}")
        lines.append(f"- **ECS:** {r.ecs:.4f}" if r.ecs is not None else "- **ECS:** —")
        lines.append(f"- **CC3 size:** {r.cc3_size} | **CC4 size:** {r.cc4_size}")
        lines.append("")

    return "\n".join(lines)


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
