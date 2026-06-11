from typing import Dict, Any, Set, List


class PrettyPrinter:
    def format_instance(self, instance: Dict[str, Any]) -> str:
        s = f"Instance ID: {instance.get('instance_id', 'N/A')}\n"
        s += f"Input: {instance.get('text', 'N/A')}\n"
        s += f"Ground Truth: {instance.get('ground_truth_label', 'N/A')}\n"
        s += f"Prediction: {instance.get('predicted_label', 'N/A')} "
        s += f"(Confidence: {instance.get('confidence', 'N/A')})\n"
        s += f"Correct: {instance.get('correct', 'N/A')}\n"
        s += "-" * 50 + "\n"
        s += "Explanation Strategies:\n"
        s += self._format_raw_strategy("Highlighting (H)", instance.get('raw_highlighting', ''), instance.get('highlighting_parsed', False))
        s += self._format_raw_strategy("Rationale (R)", instance.get('raw_rationale', ''), instance.get('rationale_parsed', False))
        s += self._format_raw_strategy("Counterfactual (CF)", instance.get('raw_counterfactual', ''), instance.get('counterfactual_parsed', False))
        s += self._format_raw_strategy("Rank Ordering (RO)", instance.get('raw_rank_ordering', ''), instance.get('rank_ordering_parsed', False))
        return s

    def _format_raw_strategy(self, name: str, raw: str, parsed: bool) -> str:
        status = "OK" if parsed else "FAILED"
        s = f"  [{status}] {name}:\n"
        raw_preview = raw[:100] + "..." if len(raw) > 100 else raw
        s += f"    {raw_preview}\n"
        return s

    def format_normalized_tokens(self, instance: Dict[str, Any]) -> str:
        s = "Normalized Tokens:\n"
        strategies = [("H", "highlighting_tokens"), ("R", "rationale_tokens"), ("CF", "counterfactual_tokens")]
        for label, key in strategies:
            tokens = instance.get(key, [])
            s += f"  {label}: {sorted(tokens) if isinstance(tokens, set) else tokens}\n"
        ro_tokens = instance.get('rank_ordering_tokens', [])
        if ro_tokens:
            ro_words = [t[0] if isinstance(t, tuple) else t[0] for t in ro_tokens]
            s += f"  RO: {ro_words}\n"
        else:
            s += f"  RO: []\n"
        return s

    def format_pairwise_agreements(self, instance: Dict[str, Any]) -> str:
        s = "Pairwise Agreements:\n"
        keys = [
            ('jaccard_H_R', 'H-R'), ('jaccard_H_CF', 'H-CF'), ('jaccard_H_RO', 'H-RO'),
            ('jaccard_R_CF', 'R-CF'), ('jaccard_R_RO', 'R-RO'), ('jaccard_CF_RO', 'CF-RO'),
            ('kendall_H_RO', 'H-RO (Kendall)'),
        ]
        for key, label in keys:
            val = instance.get(key)
            display = f"{val:.4f}" if val is not None else "N/A"
            s += f"  {label:15s}: {display}\n"
        ecs = instance.get('ecs')
        display = f"{ecs:.4f}" if ecs is not None else "N/A"
        s += f"  {'ECS':15s}: {display}\n"
        return s

    def highlight_consensus_core(self, text: str, cc_tokens: Set[str]) -> str:
        if not cc_tokens:
            return text
        highlighted = text
        for token in sorted(cc_tokens, key=len, reverse=True):
            highlighted = highlighted.replace(token, f"\033[93m{token}\033[0m")
        return highlighted
