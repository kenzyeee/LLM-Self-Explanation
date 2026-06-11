import re
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class Parser:
    def parse_classification(self, raw_response: str, label_set: List[str]) -> Tuple[str, float]:
        pred_match = re.search(r"Prediction:\s*(.+)", raw_response, re.IGNORECASE)
        conf_match = re.search(r"Confidence:\s*(.+)", raw_response, re.IGNORECASE)
        label = pred_match.group(1).strip() if pred_match else ""
        conf_str = conf_match.group(1).strip() if conf_match else ""
        predicted_label = self._fuzzy_extract(label, label_set) or label
        try:
            conf_str = conf_str.replace('%', '')
            confidence = float(conf_str)
            if confidence > 1.0:
                confidence = confidence / 100.0
        except ValueError:
            confidence = 0.0
        return predicted_label, confidence

    def parse_highlighting(self, raw_response: str) -> List[str]:
        lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
        tokens = []
        for line in lines:
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                tokens.append(m.group(1).strip(' \'"*'))
        if not tokens:
            m = re.findall(r'"([^"]+)"', raw_response)
            if m:
                tokens = m
        if not tokens:
            m = re.findall(r"'([^']+)'", raw_response)
            if m:
                tokens = m
        if not tokens:
            parts = re.split(r'[,;]', raw_response)
            tokens = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]
        return tokens[:3]

    def parse_rationale(self, raw_response: str) -> str:
        m = re.search(r"Rationale:\s*(.*)", raw_response, re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return sentences[0] if sentences else text
        m = re.search(r"Prediction:\s*.+", raw_response, re.IGNORECASE | re.MULTILINE)
        if m:
            after_pred = raw_response[m.end():].strip()
            if after_pred:
                sentences = re.split(r'(?<=[.!?])\s+', after_pred)
                return sentences[0] if sentences else after_pred
        sentences = re.split(r'(?<=[.!?])\s+', raw_response.strip())
        return sentences[0] if sentences else raw_response.strip()

    def parse_counterfactual(self, raw_response: str) -> str:
        m = re.search(r"Counterfactual Text:\s*(.+?)(?=\nCounterfactual Prediction:|\nOriginal Prediction:|$)", raw_response, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"Modified Text:\s*(.+)", raw_response, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
        for i, line in enumerate(lines):
            if 'counterfactual' in line.lower() and ':' in line and i + 1 < len(lines):
                return lines[i + 1]
        return raw_response.strip()

    def parse_rank_ordering(self, raw_response: str) -> List[Tuple[str, int]]:
        lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
        tokens = []
        rank = 1
        for line in lines:
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                tokens.append((m.group(1).strip(' \'"*'), rank))
                rank += 1
        if not tokens:
            patterns = [
                r"(?:most\s+)?important(?:\s*:\s*|\s+is\s+)(.+)",
                r"(?:top|rank)\s*\d*\s*(?:is|:)\s*(.+)",
            ]
            for pattern in patterns:
                m = re.search(pattern, raw_response, re.IGNORECASE)
                if m:
                    parts = re.split(r'[,;]', m.group(1))
                    for i, part in enumerate(parts[:5]):
                        part = part.strip().strip('"\'')
                        if part:
                            tokens.append((part, i + 1))
                    break
        return tokens[:5]

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
