import json
import logging
import re
from typing import Tuple, List, Optional

from src.utils.exceptions import ParsingError

logger = logging.getLogger(__name__)


class Parser:
    def parse_classification(self, raw_response: str, label_set: List[str], require_confidence: bool = False) -> Tuple[str, float]:
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in classification response")
        label = obj.get("label", "")
        if not isinstance(label, str) or label not in label_set:
            raise ParsingError(f"Label '{label}' not in allowed set: {label_set}")
        confidence = obj.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                raise ParsingError(f"Confidence must be numeric, got {type(confidence).__name__}")
            confidence_val = float(confidence)
            if confidence_val < 1 or confidence_val > 10:
                raise ParsingError(f"Confidence out of range 1-10: {confidence_val}")
            return label, confidence_val / 10.0
        if require_confidence:
            raise ParsingError("Confidence field missing from classification response")
        return label, 0.0

    def parse_confidence(self, raw_response: str) -> float:
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in confidence response")
        confidence = obj.get("confidence")
        if not isinstance(confidence, (int, float)):
            raise ParsingError(f"Confidence must be numeric, got {type(confidence).__name__}")
        confidence_val = float(confidence)
        if confidence_val < 1 or confidence_val > 10:
            raise ParsingError(f"Confidence out of range 1-10: {confidence_val}")
        return confidence_val / 10.0

    def parse_highlighting(self, raw_response: str, input_text: str, normalizer, skip_validation: bool = False) -> List[str]:
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in highlighting response")
        highlights = obj.get("highlights", [])
        if not isinstance(highlights, list) or len(highlights) < 2:
            raise ParsingError(f"Expected at least 2 highlights, got {len(highlights) if isinstance(highlights, list) else 'not a list'}")
        if not skip_validation:
            valid = []
            for h in highlights:
                if isinstance(h, str) and normalizer.is_anchored(h, input_text):
                    valid.append(h)
                else:
                    logger.warning(f"Highlight '{h}' not anchored in input text - discarding")
            if len(valid) < 2:
                raise ParsingError(f"Only {len(valid)} valid highlights remain after anchoring check (need ≥2)")
            return valid
        return highlights

    def parse_rationale(self, raw_response: str, input_text: str, normalizer, skip_validation: bool = False) -> Tuple[str, List[str]]:
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in rationale response")
        rationale = obj.get("rationale", "")
        evidence = obj.get("evidence", [])
        if not isinstance(rationale, str) or not rationale:
            raise ParsingError("Rationale text is empty or not a string")
        if not isinstance(evidence, list) or len(evidence) < 1:
            raise ParsingError("Evidence list is empty or not a list")
        if len(evidence) > 5:
            logger.warning(f"Evidence list has {len(evidence)} items, truncating to 5")
            evidence = evidence[:5]
        if not skip_validation:
            valid = []
            for e in evidence:
                if isinstance(e, str) and normalizer.is_anchored(e, input_text):
                    valid.append(e)
                else:
                    logger.warning(f"Evidence token '{e}' not anchored in input text — discarding")
            if len(valid) < 1:
                raise ParsingError("No valid evidence tokens remain after anchoring check")
            return rationale, valid
        return rationale, evidence

    def parse_counterfactual(self, raw_response: str, input_text: str,
                             original_label: str, label_set: List[str],
                             normalizer, skip_validation: bool = False,
                             max_edit_ratio: float = 0.3) -> Tuple[str, str]:
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in counterfactual response")
        cf_text = obj.get("counterfactual_text", "")
        new_pred = obj.get("new_prediction", "")
        if not isinstance(cf_text, str) or len(cf_text) < 3:
            raise ParsingError("Counterfactual text is too short or not a string")
        if not isinstance(new_pred, str) or new_pred not in label_set:
            raise ParsingError(f"New prediction '{new_pred}' not in label set")
        if new_pred == original_label:
            raise ParsingError("Counterfactual prediction did not flip")
        if not skip_validation:
            edit_ratio = self._word_edit_ratio(input_text, cf_text)
            if edit_ratio > max_edit_ratio:
                raise ParsingError(f"Counterfactual edit ratio {edit_ratio:.3f} exceeds {max_edit_ratio} threshold")
        return cf_text, new_pred

    def parse_rank_ordering(self, raw_response: str, input_text: str, normalizer, skip_validation: bool = False) -> List[Tuple[str, int]]:
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in rank ordering response")
        ranking = obj.get("ranking", [])
        if not isinstance(ranking, list) or len(ranking) < 3:
            raise ParsingError(f"Expected at least 3 ranked items, got {len(ranking) if isinstance(ranking, list) else 'not a list'}")
        seen = set()
        result = []
        rank = 1
        for token in ranking:
            if not isinstance(token, str):
                continue
            norm = normalizer.pre_normalize(token)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            if not skip_validation and not normalizer.is_anchored(token, input_text):
                logger.warning(f"Rank item '{token}' not anchored in input text — discarding")
                continue
            result.append((token, rank))
            rank += 1
        if len(result) < 3:
            raise ParsingError(f"Only {len(result)} unique valid tokens remain (need ≥3)")
        return result

    def _extract_json(self, text: str) -> Optional[dict]:
        """Try to parse JSON from response text. Handles surrounding text, code fences, etc."""
        text = text.strip()
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try extracting from ```json ... ``` fences
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        # Try finding first { to last }
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _word_edit_ratio(original: str, counterfactual: str) -> float:
        """Word-level Levenshtein distance / max(word count)."""
        orig_words = original.strip().split()
        cf_words = counterfactual.strip().split()
        if not orig_words and not cf_words:
            return 0.0
        if not orig_words or not cf_words:
            return 1.0
        n, m = len(orig_words), len(cf_words)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if orig_words[i - 1] == cf_words[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        distance = dp[n][m]
        return distance / max(n, m) if max(n, m) > 0 else 0.0

    # -- legacy methods kept for backward compatibility --

    def _fuzzy_extract(self, text: str, label_set: List[str]) -> Optional[str]:
        text_lower = text.lower().strip()
        for label in label_set:
            if label.lower() == text_lower:
                return label
        for label in label_set:
            if label.lower() in text_lower:
                return label
        for label in label_set:
            if self._levenshtein_similarity(label.lower(), text_lower) > 0.7:
                return label
        return None

    @staticmethod
    def _levenshtein_similarity(s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                cost = 0 if c1 == c2 else 1
                curr.append(min(curr[-1] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        distance = prev[-1]
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0
