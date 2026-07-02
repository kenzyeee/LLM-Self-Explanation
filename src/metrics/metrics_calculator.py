import numpy as np
import scipy.stats
from functools import lru_cache
from typing import List, Tuple, Dict, Set, Optional


@lru_cache(maxsize=8192)
def _expected_random_overlap(a_size: int, b_size: int, vocab_size: int,
                             n_sims: int, seed: int) -> Tuple[float, float]:
    """Monte-Carlo expectation of Jaccard and Overlap-coefficient between two
    random token subsets drawn (without replacement) from a shared content
    vocabulary of `vocab_size`, with the given subset sizes. Cached on the size
    triple so repeated instances with identical geometry are free.
    """
    a_size = min(a_size, vocab_size)
    b_size = min(b_size, vocab_size)
    if vocab_size <= 0 or a_size <= 0 or b_size <= 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    vocab = np.arange(vocab_size)
    smaller = min(a_size, b_size)  # loop-invariant: a_size/b_size are fixed and > 0 here
    jac_sum = 0.0
    ovl_sum = 0.0
    for _ in range(n_sims):
        a = set(rng.choice(vocab, size=a_size, replace=False).tolist())
        b = set(rng.choice(vocab, size=b_size, replace=False).tolist())
        inter = len(a & b)
        union = len(a | b)
        jac_sum += (inter / union) if union else 0.0
        ovl_sum += (inter / smaller) if smaller else 0.0
    return jac_sum / n_sims, ovl_sum / n_sims


class MetricsCalculator:
    @staticmethod
    def expected_random_overlap(set_size_a: int, set_size_b: int, vocab_size: int,
                                n_sims: int = 2000, seed: int = 42) -> Tuple[float, float]:
        """Expected (Jaccard, Overlap-coefficient) under random token selection.

        Used to report ECS as *lift over chance* (ECS - ECS_random): raw overlap
        is confounded by set sizes and vocabulary size, so the agreement that
        matters is the increment above what random selection would yield.
        """
        return _expected_random_overlap(int(set_size_a), int(set_size_b),
                                        int(vocab_size), int(n_sims), int(seed))

    def compute_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def compute_overlap_coefficient(self, set1: Set[str], set2: Set[str]) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        smaller = min(len(set1), len(set2))
        return intersection / smaller if smaller > 0 else 0.0

    def assign_implicit_ranks(self, token_sequence: List[str]) -> List[Tuple[str, int]]:
        seen = set()
        result = []
        rank = 1
        for token in token_sequence:
            if token not in seen:
                seen.add(token)
                result.append((token, rank))
                rank += 1
        return result

    def compute_kendalls_tau(self, ranks1: List[Tuple[str, int]], ranks2: List[Tuple[str, int]]) -> Optional[float]:
        rank_dict1 = {t: r for t, r in ranks1}
        rank_dict2 = {t: r for t, r in ranks2}
        common_tokens = set(rank_dict1.keys()) & set(rank_dict2.keys())
        # Require >=4 overlapping tokens: with n<=3 Kendall tau is forced to
        # near-degenerate values (+/-1) and is not meaningful.
        if len(common_tokens) < 4:
            return None
        rank_vec1 = [rank_dict1[t] for t in sorted(common_tokens)]
        rank_vec2 = [rank_dict2[t] for t in sorted(common_tokens)]
        tau, _ = scipy.stats.kendalltau(rank_vec1, rank_vec2)
        return tau if not np.isnan(tau) else None

    def compute_normalized_kendalls_tau(self, tau: Optional[float]) -> Optional[float]:
        return (tau + 1.0) / 2.0 if tau is not None else None

    def compute_pairwise_agreements(self, explanations: Dict[str, Set[str]]) -> Dict[Tuple[str, str], float]:
        strategies = ["H", "R", "CF", "RO"]
        agreements = {}
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                s1, s2 = strategies[i], strategies[j]
                pair = (s1, s2)
                set1 = explanations.get(s1, set())
                set2 = explanations.get(s2, set())
                if set1 and set2:
                    agreements[pair] = self.compute_jaccard_similarity(set1, set2)
        return agreements

    def compute_pairwise_overlaps(self, explanations: Dict[str, Set[str]]) -> Dict[Tuple[str, str], float]:
        strategies = ["H", "R", "CF", "RO"]
        overlaps = {}
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                s1, s2 = strategies[i], strategies[j]
                pair = (s1, s2)
                set1 = explanations.get(s1, set())
                set2 = explanations.get(s2, set())
                if set1 and set2:
                    overlaps[pair] = self.compute_overlap_coefficient(set1, set2)
        return overlaps

    def compute_rbo(self, list1: List[str], list2: List[str], p: float = 0.9) -> Optional[float]:
        """Rank-Biased Overlap between two ranked lists.

        RBO = (1-p) * sum_{d=1..max(len1,len2)} p^{d-1} * A_d
        where A_d = overlap of top-d elements at depth d.

        Args:
            list1: First ranked list (higher rank = more important first)
            list2: Second ranked list
            p: Weight parameter (0 < p < 1), lower = more top-weighted

        Returns:
            RBO score in [0, 1], or None if either list is empty.
        """
        if not list1 or not list2:
            return None
        max_depth = max(len(list1), len(list2))
        set1: Set[str] = set()
        set2: Set[str] = set()
        agreement_sum = 0.0
        for d in range(1, max_depth + 1):
            if d <= len(list1):
                set1.add(list1[d - 1])
            if d <= len(list2):
                set2.add(list2[d - 1])
            agreement = len(set1 & set2) / d
            agreement_sum += (p ** (d - 1)) * agreement
        return (1 - p) * agreement_sum

    def compute_ecs(self, pairwise_agreements: Dict[Tuple[str, str], float]) -> Optional[float]:
        """ECS over cross-paradigm pairs only (excludes H-RO, same paradigm family)."""
        excluded = {("H", "RO")}
        values = [v for p, v in pairwise_agreements.items() if p not in excluded]
        return float(np.mean(values)) if values else None

    def compute_ecs_primary(self, pairwise_agreements: Dict[Tuple[str, str], float]) -> Tuple[Optional[float], Optional[float], int]:
        """Primary ECS as two composites:
           - extraction_rationale: average of (H,R) and (R,RO) — extraction methods agreeing with rationale
           - extraction_perturbation: average of (H,CF) and (CF,RO) — extraction methods agreeing with perturbation

        Pair keys must match the ordering compute_pairwise_agreements stores them under
        (index order within ["H","R","CF","RO"], i.e. ("R","RO") and ("CF","RO"), never
        ("RO","R")/("RO","CF")) or the lookups silently miss and these composites collapse
        to a single pair.
        """
        er_pairs = [("H", "R"), ("R", "RO")]
        ep_pairs = [("H", "CF"), ("CF", "RO")]
        er_values = [pairwise_agreements.get(p) for p in er_pairs if pairwise_agreements.get(p) is not None]
        ep_values = [pairwise_agreements.get(p) for p in ep_pairs if pairwise_agreements.get(p) is not None]
        er_mean = float(np.mean(er_values)) if er_values else None
        ep_mean = float(np.mean(ep_values)) if ep_values else None
        n_pairs = len(er_values) + len(ep_values)
        return er_mean, ep_mean, n_pairs

    def compute_consensus_core(self, explanations: Dict[str, Set[str]], threshold: int) -> Set[str]:
        strategy_token_counts = {}
        for strategy, tokens in explanations.items():
            for token in tokens:
                strategy_token_counts[token] = strategy_token_counts.get(token, 0) + 1
        return {token for token, count in strategy_token_counts.items() if count >= threshold}

    @staticmethod
    def classify_length(text: str) -> str:
        wc = len(text.split())
        if wc <= 20:
            return "short"
        elif wc <= 50:
            return "medium"
        else:
            return "long"

    @staticmethod
    def compute_length_stratified_ecs(results) -> Dict[str, Dict]:
        """Return {length_bucket: {mean_ecs, n}} for 'short','medium','long'."""
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in results:
            if r.ecs is not None:
                bucket = MetricsCalculator.classify_length(r.text)
                buckets[bucket].append(r.ecs)
        out = {}
        for bucket, vals in sorted(buckets.items()):
            out[bucket] = {"mean_ecs": float(np.mean(vals)) if vals else 0.0, "n": len(vals)}
        return out

    @staticmethod
    def compute_complete_case_metrics(results) -> Dict:
        """Compute metrics using only instances where all 4 strategies are valid."""
        complete = [r for r in results if r.n_valid_strategies == 4]
        partial = [r for r in results if r.n_valid_strategies < 4 and r.ecs is not None]
        return {
            "n_complete": len(complete),
            "n_partial": len(partial),
            "complete_mean_ecs": float(np.mean([r.ecs for r in complete])) if complete else 0.0,
            "partial_mean_ecs": float(np.mean([r.ecs for r in partial])) if partial else 0.0,
            "complete_instances": [r.instance_id for r in complete],
            "partial_instances": [r.instance_id for r in partial],
        }
