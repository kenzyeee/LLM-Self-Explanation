import logging
from typing import Callable, List, Optional, Awaitable

logger = logging.getLogger(__name__)


class RedactionTestResult:
    def __init__(self, faithfulness: float = 0.0, flip_at_k: Optional[int] = None,
                 n_tokens: int = 0, predictions: Optional[List[str]] = None):
        self.faithfulness = faithfulness
        self.flip_at_k = flip_at_k
        self.n_tokens = n_tokens
        self.predictions = predictions or []

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "flip_at_k": self.flip_at_k,
            "n_tokens": self.n_tokens,
            "predictions": self.predictions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RedactionTestResult":
        return cls(
            faithfulness=data.get("faithfulness", 0.0),
            flip_at_k=data.get("flip_at_k"),
            n_tokens=data.get("n_tokens", 0),
            predictions=data.get("predictions", []),
        )


class RedactionTest:
    def __init__(self, classify_fn: Callable[[str], Awaitable[str]],
                 operator: str = "mask", normalizer=None):
        """Progressive comprehensiveness erasure.

        operator: "mask" replaces erased tokens with [MASK] (in-distribution-ish);
                  "delete" removes them entirely. ICE (2026) shows operator choice
                  flips faithfulness conclusions on short vs long text, so callers
                  should run both and compare.
        normalizer: if given, an evidence token also erases input surface forms
                  that share a WordNet lemma with it (e.g. token "movie" erases
                  input "movies"), matching how tokens were anchored to the input
                  during extraction (Normalizer.is_anchored). Without it, matching
                  is surface-only and an inflected input occurrence silently
                  survives erasure, understating comprehensiveness/flip rate.
        """
        self.classify_fn = classify_fn
        if operator not in ("mask", "delete"):
            raise ValueError(f"operator must be 'mask' or 'delete', got {operator!r}")
        self.operator = operator
        self.normalizer = normalizer

    async def run(self, ordered_tokens: List[str], input_text: str,
                  original_prediction: str) -> RedactionTestResult:
        """Progressively erase ordered_tokens[0], then [0:2], ... until the
        prediction flips at depth k. faithfulness = 1 - k/n: the EARLIER a strategy's
        own top-ranked tokens flip the prediction, the more faithful its ranking.

        This is a first-flip-depth proxy, not ERASER comprehensiveness (DeYoung et
        al. 2020) — comprehensiveness is the drop in predicted-class PROBABILITY
        after removing the top-k%, averaged over k, which needs class probabilities
        this pipeline doesn't have (logprobs are unsupported on this model). 1-k/n
        is the closest available analogue: both erase by importance rank and reward
        early flips, but this pipeline never averages over multiple k or scores
        probability change, so treat it as an ad-hoc rank-erasure metric.
        """
        if not ordered_tokens:
            return RedactionTestResult(faithfulness=0.0, flip_at_k=None,
                                       n_tokens=0)

        n = len(ordered_tokens)
        predictions: List[str] = [original_prediction]

        for k in range(1, n + 1):
            tokens_to_redact = ordered_tokens[:k]
            redacted_text = self._redact_tokens(input_text, tokens_to_redact, self.operator,
                                                 self.normalizer)
            pred = await self.classify_fn(redacted_text)
            predictions.append(pred)
            if pred != original_prediction:
                flip_at_k = k
                faithfulness = 1.0 - (k / n)
                return RedactionTestResult(
                    faithfulness=faithfulness,
                    flip_at_k=flip_at_k,
                    n_tokens=n,
                    predictions=predictions,
                )

        return RedactionTestResult(
            faithfulness=0.0,
            flip_at_k=None,
            n_tokens=n,
            predictions=predictions,
        )

    @staticmethod
    def _redact_tokens(text: str, tokens_to_redact: List[str], operator: str = "mask",
                       normalizer=None) -> str:
        """Erase every occurrence of tokens_to_redact in text (not just the first).

        With a normalizer, a word also matches if it shares a WordNet lemma with a
        token to redact — the same morphology-aware criterion Normalizer.is_anchored
        uses to anchor evidence to the input in the first place, so erasure can't
        under-cover an evidence token just because the input used a different
        inflection.
        """
        words = text.split()
        redacted = []
        redact_set = {t.lower() for t in tokens_to_redact}
        redact_lemmas = set()
        if normalizer is not None:
            for t in redact_set:
                redact_lemmas |= normalizer._anchor_lemmas(t)
        for word in words:
            stripped_lower = word.strip(".,!?;:()[]{}\"'").lower()
            is_match = bool(stripped_lower) and stripped_lower in redact_set
            if not is_match and normalizer is not None and stripped_lower:
                is_match = bool(normalizer._anchor_lemmas(stripped_lower) & redact_lemmas)
            if is_match:
                if operator == "mask":
                    redacted.append("[MASK]")
                # operator == "delete": drop the word entirely
            else:
                redacted.append(word)
        return " ".join(redacted)
