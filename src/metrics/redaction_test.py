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
                 operator: str = "mask"):
        """Progressive comprehensiveness erasure.

        operator: "mask" replaces erased tokens with [MASK] (in-distribution-ish);
                  "delete" removes them entirely. ICE (2026) shows operator choice
                  flips faithfulness conclusions on short vs long text, so callers
                  should run both and compare.
        """
        self.classify_fn = classify_fn
        if operator not in ("mask", "delete"):
            raise ValueError(f"operator must be 'mask' or 'delete', got {operator!r}")
        self.operator = operator

    async def run(self, ordered_tokens: List[str], input_text: str,
                  original_prediction: str) -> RedactionTestResult:
        if not ordered_tokens:
            return RedactionTestResult(faithfulness=0.0, flip_at_k=None,
                                       n_tokens=0)

        n = len(ordered_tokens)
        predictions: List[str] = [original_prediction]

        for k in range(1, n + 1):
            tokens_to_redact = ordered_tokens[:k]
            redacted_text = self._redact_tokens(input_text, tokens_to_redact, self.operator)
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
    def _redact_tokens(text: str, tokens_to_redact: List[str], operator: str = "mask") -> str:
        words = text.split()
        redacted = []
        redact_set = {t.lower() for t in tokens_to_redact}
        remaining = set(redact_set)
        for word in words:
            stripped = word.strip(".,!?;:()[]{}\"'")
            if stripped.lower() in remaining:
                remaining.discard(stripped.lower())
                if operator == "mask":
                    redacted.append("[MASK]")
                # operator == "delete": drop the word entirely
            else:
                redacted.append(word)
        return " ".join(redacted)
