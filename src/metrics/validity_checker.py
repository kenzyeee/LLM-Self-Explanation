import random
from typing import Set


class FlipResult:
    def __init__(self, original_prediction: str = "", masked_prediction: str = "",
                 flipped: bool = False, masked_tokens: Set[str] = None):
        self.original_prediction = original_prediction
        self.masked_prediction = masked_prediction
        self.flipped = flipped
        self.masked_tokens = masked_tokens or set()

    def to_dict(self):
        return {
            'original_prediction': self.original_prediction,
            'masked_prediction': self.masked_prediction,
            'flipped': self.flipped,
            'masked_tokens': sorted(list(self.masked_tokens)),
        }


class ValidityChecker:
    def __init__(self, engine):
        self.engine = engine

    async def test_consensus_core_removal(self, text: str, original_prediction: str,
                                           consensus_core_tokens: Set[str]) -> FlipResult:
        masked_text = self._mask_tokens(text, consensus_core_tokens)
        try:
            result = await self.engine.classify_with_mask(masked_text, consensus_core_tokens)
            masked_prediction = result.predicted_label
        except Exception:
            masked_prediction = ""
        flipped = (masked_prediction != original_prediction) and bool(masked_prediction)
        return FlipResult(
            original_prediction=original_prediction,
            masked_prediction=masked_prediction,
            flipped=flipped,
            masked_tokens=consensus_core_tokens,
        )

    async def test_random_removal_baseline(self, text: str, original_prediction: str,
                                            n_tokens: int) -> FlipResult:
        words = list(set(text.split()))
        if n_tokens >= len(words):
            random_tokens = set(words)
        else:
            rng = random.Random(42)
            random_tokens = set(rng.sample(words, n_tokens))
        masked_text = self._mask_tokens(text, random_tokens)
        try:
            result = await self.engine.classify_with_mask(masked_text, random_tokens)
            masked_prediction = result.predicted_label
        except Exception:
            masked_prediction = ""
        flipped = (masked_prediction != original_prediction) and bool(masked_prediction)
        return FlipResult(
            original_prediction=original_prediction,
            masked_prediction=masked_prediction,
            flipped=flipped,
            masked_tokens=random_tokens,
        )

    def _mask_tokens(self, text: str, tokens_to_mask: Set[str], mask_token: str = "[MASK]") -> str:
        words = text.split()
        masked = []
        for word in words:
            clean_word = word.strip('.,!?;:()[]{}\'"')
            if clean_word in tokens_to_mask:
                masked.append(mask_token)
            else:
                masked.append(word)
        return " ".join(masked)
