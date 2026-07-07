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

    @staticmethod
    def expected_jaccard_exact(a_size: int, b_size: int, vocab_size: int) -> float:
        """Exact E[Jaccard] between two sets of sizes a,b drawn uniformly without
        replacement from a shared vocabulary of size V (ECS-adj plan §3.1, fixes W1).

        Intersection size K ~ Hypergeometric(V, a, b); Jaccard at K=k is
        k/(a+b-k). Summed in closed form over k in [max(0,a+b-V), min(a,b)]
        (<=~10 terms in practice) instead of Monte-Carlo simulation — cheaper
        and exact, no seed/cache machinery needed.
        """
        a = min(int(a_size), int(vocab_size))
        b = min(int(b_size), int(vocab_size))
        V = int(vocab_size)
        if V <= 0 or a <= 0 or b <= 0:
            return 0.0
        k_lo = max(0, a + b - V)
        k_hi = min(a, b)
        total = 0.0
        for k in range(k_lo, k_hi + 1):
            pmf = scipy.stats.hypergeom.pmf(k, V, a, b)
            union = a + b - k
            j = (k / union) if union > 0 else 0.0
            total += pmf * j
        return float(total)

    @staticmethod
    def expected_overlap_exact(a_size: int, b_size: int, vocab_size: int) -> float:
        """Exact E[Overlap coefficient] = E[K]/min(a,b) = a*b / (V*min(a,b)),
        a one-liner from the hypergeometric mean E[K] = a*b/V (ECS-adj plan §3.1)."""
        a = min(int(a_size), int(vocab_size))
        b = min(int(b_size), int(vocab_size))
        V = int(vocab_size)
        smaller = min(a, b)
        if V <= 0 or a <= 0 or b <= 0 or smaller <= 0:
            return 0.0
        return float(a * b) / (V * smaller)

    def adjusted_jaccard(self, set1: Set[str], set2: Set[str], vocab_size: int,
                         eps: float = 0.10) -> Tuple[Optional[float], bool]:
        """Kappa-style chance- AND ceiling-adjusted Jaccard (ECS-adj plan §3.2,
        fixes W2/W7): AJ = (J - E[J]) / (J_max - E[J]).

        J_max = min(a,b)/max(a,b) is the ceiling attainable given the set sizes
        (attained when the smaller set nests in the larger). 0 = chance,
        1 = maximum agreement achievable given the sizes; below-chance values
        go negative with an asymmetric floor of -E[J]/(J_max-E[J]) (not -1).
        Unifies the current lift (chance-only) and overlap-ECS (ceiling-only)
        patches into one estimator comparable across pair-size geometries.

        Degeneracy guard: returns (None, True) when J_max - E[J] < eps — the
        pair's geometry puts the ceiling too close to the chance floor for the
        ratio to be a meaningful estimate (dividing by near-zero amplifies
        noise). eps is pre-registered at 0.10 (plan §3.2).
        """
        a, b = len(set1), len(set2)
        if a == 0 or b == 0 or vocab_size <= 0:
            return None, True
        j = self.compute_jaccard_similarity(set1, set2)
        ej = self.expected_jaccard_exact(a, b, vocab_size)
        j_max = min(a, b) / max(a, b)
        denom = j_max - ej
        if denom < eps:
            return None, True
        return (j - ej) / denom, False

    def compute_ecs_adjusted(self, explanations: Dict[str, Set[str]], vocab_size: int,
                             eps: float = 0.10) -> Dict[str, object]:
        """Chance- and ceiling-adjusted ECS (ECS-adj, plan §3): paradigm-balanced
        aggregation of adjusted_jaccard pairs, replacing the flat 5-pair mean
        that let CF dominate 60% of the composite (fixes W3, W4).

        Components (each is a mean over its sub-pairs, so paradigm weight stays
        fixed at 1/3 regardless of how many sub-pairs survive):
          E-R (extraction-rationalization)   = mean(AJ(H,R), AJ(RO,R))
          E-P (extraction-perturbation)      = mean(AJ(H,CF), AJ(RO,CF))
          R-P (rationalization-perturbation) =      AJ(R,CF)
        ecs_adj = mean(E-R, E-P, R-P) over whichever components are defined
        (the AVAILABLE-COMPONENT secondary, never silently the headline).
        ecs_adj_complete (bool) flags whether ALL 3 components were defined —
        callers filter on this flag to build the primary complete-case
        estimand (mirroring the existing n_valid_strategies==4 gate for
        legacy ECS), exactly as the plan's §3.4 missing-data policy specifies.

        Returns: {ecs_adj_er, ecs_adj_ep, ecs_adj_rp, ecs_adj,
                  ecs_adj_n_components (0-3), ecs_adj_complete (bool),
                  n_degenerate_pairs (0-5)}.
        """
        pair_keys = [("H", "R"), ("RO", "R"), ("H", "CF"), ("RO", "CF"), ("R", "CF")]
        aj: Dict[Tuple[str, str], float] = {}
        n_degenerate = 0
        for (s1, s2) in pair_keys:
            set1, set2 = explanations.get(s1, set()), explanations.get(s2, set())
            if not set1 or not set2:
                continue
            val, degenerate = self.adjusted_jaccard(set1, set2, vocab_size, eps)
            if degenerate:
                n_degenerate += 1
            if val is not None:
                aj[(s1, s2)] = val

        def _mean_of(keys):
            vals = [aj[k] for k in keys if k in aj]
            return float(np.mean(vals)) if vals else None

        er = _mean_of([("H", "R"), ("RO", "R")])
        ep = _mean_of([("H", "CF"), ("RO", "CF")])
        rp = aj.get(("R", "CF"))

        components = [c for c in (er, ep, rp) if c is not None]
        ecs_adj = float(np.mean(components)) if components else None
        complete = er is not None and ep is not None and rp is not None

        return {
            "ecs_adj_er": er,
            "ecs_adj_ep": ep,
            "ecs_adj_rp": rp,
            "ecs_adj": ecs_adj,
            "ecs_adj_n_components": len(components),
            "ecs_adj_complete": complete,
            "n_degenerate_pairs": n_degenerate,
        }

    def compute_ecs_overlap(self, pairwise_overlaps: Dict[Tuple[str, str], float]) -> Optional[float]:
        """Size-robust secondary ECS: mean Overlap Coefficient over the same 5
        cross-paradigm pairs the Jaccard ECS uses (H-RO excluded).

        Jaccard has a structural ceiling when set sizes differ (|CF|≈1-2 vs |H|≈3-5
        caps Jaccard at ~0.2-0.33 regardless of agreement); the overlap coefficient
        |A∩B|/min(|A|,|B|) is invariant to that asymmetry — the closest analogue of
        Krishna et al. (2022)'s feature-agreement@k when the two sets' k differ.
        Reported alongside (never instead of) the Jaccard ECS so composites with
        different set-size geometry are comparable.
        """
        excluded = {("H", "RO")}
        values = [v for p, v in pairwise_overlaps.items() if p not in excluded]
        return float(np.mean(values)) if values else None

    @staticmethod
    def expected_random_overlap_weighted(set_size_a: int, set_size_b: int,
                                         weights: Dict[str, float],
                                         n_sims: int = 2000, seed: int = 42) -> Optional[float]:
        """Expected Jaccard when both token subsets are drawn WITHOUT replacement with
        probability proportional to per-token salience weights.

        The uniform null (expected_random_overlap) understates chance agreement when
        every method is drawn toward the same few high-salience tokens for reasons
        unrelated to consensus (one obvious sentiment word). Sampling ∝ the model's
        own graded salience (H's full 1-10 vector over the content vocabulary) gives
        a harder, salience-aware null; lift over THIS null is the conservative
        secondary check (review §2.5). Returns None when the weight vector is
        degenerate (fewer weighted tokens than either set size needs).
        """
        vocab = sorted(weights.keys())
        if not vocab:
            return None
        w = np.array([max(float(weights[t]), 1e-9) for t in vocab], dtype=float)
        p = w / w.sum()
        n = len(vocab)
        a = min(int(set_size_a), n)
        b = min(int(set_size_b), n)
        if a <= 0 or b <= 0:
            return None
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        jac_sum = 0.0
        for _ in range(n_sims):
            sa = set(rng.choice(idx, size=a, replace=False, p=p).tolist())
            sb = set(rng.choice(idx, size=b, replace=False, p=p).tolist())
            union = len(sa | sb)
            jac_sum += (len(sa & sb) / union) if union else 0.0
        return jac_sum / n_sims

    @staticmethod
    def compute_cross_model_agreement(results, n_bootstrap: int = 1000, seed: int = 42) -> Dict[str, Dict]:
        """Cross-model SAME-strategy agreement: for each (dataset, instance) present
        under >=2 models, the pairwise Jaccard between different models' evidence
        sets for the SAME strategy — plus the within-model cross-strategy ECS of the
        same instances for direct comparison.

        This is the free analysis the multi-model design enables: if within-model
        cross-strategy consensus is systematically higher than cross-model
        same-strategy agreement, explanations track model-specific computation
        ("privileged self-knowledge", arXiv:2602.02639); if it is not, stated
        evidence is closer to a generic task prior shared across models
        (cf. the cross-model "explanation lottery", arXiv:2603.15821). All evidence
        sets are already in the shared normalized token space, so the comparison is
        apples-to-apples. Zero extra API calls.

        Also computes a PAIRED per-instance contrast (review P2.4): for each
        multi-model instance, Δ = (that instance's own cross-model same-strategy
        mean Jaccard) − (that instance's own within-model mean ECS across its
        rows). This is a genuine paired estimate (unlike diffing the two dataset
        means above, which are each averaged over a DIFFERENT set of pairs/rows)
        with a seeded bootstrap CI, and an explicit stated direction — the report
        previously cited both the "privileged self-knowledge" and "shared task
        prior" hypotheses without ever saying which one the run's own data pointed to.

        Returns {dataset: {"strategies": {S: {mean_jaccard, n_pairs}},
                           "cross_model_same_strategy_mean": float|None,
                           "within_model_cross_strategy_mean_ecs": float|None,
                           "n_instances_multi_model": int,
                           "paired_contrast": {"mean_delta": float|None, "ci_lower": float,
                                               "ci_upper": float, "n": int,
                                               "direction": "cross_model_exceeds"|"within_model_exceeds"|"indeterminate"}}}
        """
        calc = MetricsCalculator()
        from collections import defaultdict
        by_instance = defaultdict(list)
        for r in results:
            by_instance[(r.dataset, r.instance_id)].append(r)

        strategy_sets = {
            "H": lambda r: r.highlighting_tokens if r.highlighting_valid else None,
            "R": lambda r: r.rationale_tokens if r.rationale_valid else None,
            "CF": lambda r: r.counterfactual_tokens if r.counterfactual_valid else None,
            # Top-k RO evidence set ECS scored (review P1.1), falling back to the full
            # ranked list for legacy records that predate rank_ordering_set.
            "RO": lambda r: (getattr(r, "rank_ordering_set", None) or {t for t, _ in r.rank_ordering_tokens}) if r.rank_ordering_valid else None,
        }

        per_dataset: Dict[str, Dict] = {}
        agg = defaultdict(lambda: defaultdict(list))   # dataset -> strategy -> [jaccards]
        within_ecs = defaultdict(list)                 # dataset -> [ecs values]
        multi_counts = defaultdict(set)                # dataset -> instance ids with >=2 models
        deltas = defaultdict(list)                      # dataset -> [per-instance paired deltas]

        for (dataset, iid), rows in by_instance.items():
            if len(rows) < 2:
                continue
            multi_counts[dataset].add(iid)
            instance_ecs = [r.ecs for r in rows if r.ecs is not None]
            for v in instance_ecs:
                within_ecs[dataset].append(v)
            instance_jaccards = []
            for s, getter in strategy_sets.items():
                sets = [getter(r) for r in rows]
                sets = [x for x in sets if x]
                for i in range(len(sets)):
                    for j in range(i + 1, len(sets)):
                        jac = calc.compute_jaccard_similarity(sets[i], sets[j])
                        agg[dataset][s].append(jac)
                        instance_jaccards.append(jac)
            # This instance's OWN paired contrast — both sides averaged over only
            # THIS instance's rows/pairs, not the dataset's pooled sets.
            if instance_jaccards and instance_ecs:
                cross_model_this = float(np.mean(instance_jaccards))
                within_model_this = float(np.mean(instance_ecs))
                deltas[dataset].append(cross_model_this - within_model_this)

        for dataset in sorted(multi_counts.keys()):
            strategies = {}
            all_vals = []
            for s in ["H", "R", "CF", "RO"]:
                vals = agg[dataset].get(s, [])
                strategies[s] = {
                    "mean_jaccard": float(np.mean(vals)) if vals else None,
                    "n_pairs": len(vals),
                }
                all_vals.extend(vals)

            d = deltas[dataset]
            if d:
                rng = np.random.default_rng(seed)
                n = len(d)
                boot_means = [float(np.mean(rng.choice(d, size=n, replace=True))) for _ in range(n_bootstrap)]
                ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
                mean_delta = float(np.mean(d))
                if ci_lo > 0:
                    direction = "cross_model_exceeds"
                elif ci_hi < 0:
                    direction = "within_model_exceeds"
                else:
                    direction = "indeterminate"
                paired_contrast = {"mean_delta": mean_delta, "ci_lower": ci_lo, "ci_upper": ci_hi,
                                   "n": n, "direction": direction}
            else:
                paired_contrast = {"mean_delta": None, "ci_lower": 0.0, "ci_upper": 0.0,
                                   "n": 0, "direction": "indeterminate"}

            per_dataset[dataset] = {
                "strategies": strategies,
                "cross_model_same_strategy_mean": float(np.mean(all_vals)) if all_vals else None,
                "within_model_cross_strategy_mean_ecs": (
                    float(np.mean(within_ecs[dataset])) if within_ecs[dataset] else None),
                "n_instances_multi_model": len(multi_counts[dataset]),
                "paired_contrast": paired_contrast,
            }
        return per_dataset

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
