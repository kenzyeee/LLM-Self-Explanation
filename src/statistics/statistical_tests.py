import numpy as np
import scipy.stats
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class CorrelationResult:
    rho: float
    p_value: float
    ci_lower: float
    ci_upper: float


@dataclass
class StatisticalTest:
    t_statistic: float
    p_value: float
    mean_diff: float = 0.0
    effect_size: float = 0.0


def compute_confidence_ecs_correlation(confidences: List[float], ecs_values: List[float],
                                        n_bootstrap: int = 1000) -> CorrelationResult:
    if len(confidences) < 3 or len(ecs_values) < 3:
        return CorrelationResult(rho=0.0, p_value=1.0, ci_lower=0.0, ci_upper=0.0)
    rho, p_value = scipy.stats.spearmanr(confidences, ecs_values)
    if np.isnan(rho):
        return CorrelationResult(rho=0.0, p_value=1.0, ci_lower=0.0, ci_upper=0.0)
    boot_rhos = []
    n = len(confidences)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            boot_rho, _ = scipy.stats.spearmanr([confidences[i] for i in idx], [ecs_values[i] for i in idx])
            if not np.isnan(boot_rho):
                boot_rhos.append(boot_rho)
        except Exception:
            continue
    ci_lower = float(np.percentile(boot_rhos, 2.5)) if boot_rhos else 0.0
    ci_upper = float(np.percentile(boot_rhos, 97.5)) if boot_rhos else 0.0
    return CorrelationResult(rho=float(rho), p_value=float(p_value), ci_lower=ci_lower, ci_upper=ci_upper)


def permutation_test(group1: List[float], group2: List[float], n_permutations: int = 10000) -> float:
    observed = abs(np.mean(group1) - np.mean(group2))
    combined = group1 + group2
    n1 = len(group1)
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_mean1 = np.mean(combined[:n1])
        perm_mean2 = np.mean(combined[n1:])
        if abs(perm_mean1 - perm_mean2) >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


def apply_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    n = len(p_values)
    if n == 0:
        return []
    return [min(p * n, 1.0) for p in p_values]


def are_significant(p_values: List[float], corrected: bool = True, alpha: float = 0.05) -> List[bool]:
    adjusted = apply_bonferroni_correction(p_values, alpha) if corrected else p_values
    return [p <= alpha for p in adjusted]


def wilcoxon_signed_rank_test(group1: List[float], group2: List[float]) -> StatisticalTest:
    if len(group1) < 2 or len(group2) < 2:
        return StatisticalTest(t_statistic=0.0, p_value=1.0)
    if len(group1) != len(group2):
        min_len = min(len(group1), len(group2))
        group1 = group1[:min_len]
        group2 = group2[:min_len]
    try:
        stat, p_value = scipy.stats.wilcoxon(group1, group2)
        if np.isnan(stat):
            return StatisticalTest(t_statistic=0.0, p_value=1.0)
        diff = np.mean(group1) - np.mean(group2)
        return StatisticalTest(t_statistic=float(stat), p_value=float(p_value), mean_diff=float(diff))
    except ValueError:
        return StatisticalTest(t_statistic=0.0, p_value=1.0)


def one_sample_ttest(values: List[float], popmean: float = 0.0) -> Tuple[float, float]:
    if len(values) < 2:
        return (0.0, 1.0)
    try:
        t_stat, p_value = scipy.stats.ttest_1samp(values, popmean)
        if np.isnan(t_stat):
            return (0.0, 1.0)
        return (float(t_stat), float(p_value))
    except Exception:
        return (0.0, 1.0)


def paired_ttest(group1: List[float], group2: List[float]) -> StatisticalTest:
    if len(group1) < 2 or len(group2) < 2:
        return StatisticalTest(t_statistic=0.0, p_value=1.0)
    min_len = min(len(group1), len(group2))
    group1 = group1[:min_len]
    group2 = group2[:min_len]
    try:
        t_stat, p_value = scipy.stats.ttest_rel(group1, group2)
        if np.isnan(t_stat):
            return StatisticalTest(t_statistic=0.0, p_value=1.0)
        diff = np.mean(group1) - np.mean(group2)
        std_diff = np.std([group1[i] - group2[i] for i in range(min_len)])
        effect_size = float(diff / std_diff) if std_diff > 0 else 0.0
        return StatisticalTest(t_statistic=float(t_stat), p_value=float(p_value), mean_diff=float(diff), effect_size=effect_size)
    except Exception:
        return StatisticalTest(t_statistic=0.0, p_value=1.0)
