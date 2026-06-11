import numpy as np
import scipy.stats
from typing import List, Tuple, Dict, Set


class MetricsCalculator:
    def compute_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

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

    def compute_kendalls_tau(self, ranks1: List[Tuple[str, int]], ranks2: List[Tuple[str, int]]) -> float:
        rank_dict1 = {t: r for t, r in ranks1}
        rank_dict2 = {t: r for t, r in ranks2}
        common_tokens = set(rank_dict1.keys()) & set(rank_dict2.keys())
        if len(common_tokens) < 2:
            return 0.0
        rank_vec1 = [rank_dict1[t] for t in sorted(common_tokens)]
        rank_vec2 = [rank_dict2[t] for t in sorted(common_tokens)]
        tau, _ = scipy.stats.kendalltau(rank_vec1, rank_vec2)
        return tau if not np.isnan(tau) else 0.0

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

    def compute_ecs(self, pairwise_agreements: Dict[Tuple[str, str], float]) -> float:
        values = list(pairwise_agreements.values())
        return float(np.mean(values)) if values else 0.0

    def compute_consensus_core(self, explanations: Dict[str, Set[str]], threshold: int) -> Set[str]:
        strategy_token_counts = {}
        for strategy, tokens in explanations.items():
            for token in tokens:
                strategy_token_counts[token] = strategy_token_counts.get(token, 0) + 1
        return {token for token, count in strategy_token_counts.items() if count >= threshold}
