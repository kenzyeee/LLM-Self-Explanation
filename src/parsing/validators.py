import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


def validate_classification(raw_response: str, label_set: List[str]) -> Tuple[bool, str]:
    if not raw_response or not raw_response.strip():
        return False, "Empty classification response"
    try:
        import json
        obj = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        return False, "Response is not valid JSON"
    label = obj.get("label", "")
    confidence = obj.get("confidence", 0)
    if label not in label_set:
        return False, f"Label '{label}' not in {label_set}"
    if not isinstance(confidence, (int, float)):
        return False, f"Confidence must be numeric, got {type(confidence).__name__}"
    if confidence < 0 or confidence > 100:
        return False, f"Confidence {confidence} out of range [0, 100]"
    return True, ""


def validate_highlighting(raw_response: str, input_text: str, normalizer) -> Tuple[bool, str]:
    if not raw_response or not raw_response.strip():
        return False, "Empty highlighting response"
    import json
    try:
        obj = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        # Try extracting from code fences
        import re
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_response, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                return False, "Response is not valid JSON"
        else:
            return False, "Response is not valid JSON"
    highlights = obj.get("highlights", [])
    if not isinstance(highlights, list):
        return False, "Highlights is not a list"
    if len(highlights) != 3:
        return False, f"Expected 3 highlights, got {len(highlights)}"
    for i, h in enumerate(highlights):
        if not isinstance(h, str) or not h.strip():
            return False, f"Highlight {i+1} is empty"
        if not normalizer.is_anchored(h, input_text):
            return False, f"Highlight '{h}' not found in input text"
    return True, ""


def validate_rationale(raw_response: str, input_text: str, normalizer) -> Tuple[bool, str]:
    if not raw_response or not raw_response.strip():
        return False, "Empty rationale response"
    import json
    try:
        obj = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        import re
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_response, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                return False, "Response is not valid JSON"
        else:
            return False, "Response is not valid JSON"
    rationale = obj.get("rationale", "")
    evidence = obj.get("evidence", [])
    if not isinstance(rationale, str) or not rationale.strip():
        return False, "Rationale text is empty"
    if not isinstance(evidence, list) or len(evidence) < 1:
        return False, "Evidence list is empty"
    for i, e in enumerate(evidence):
        if not isinstance(e, str) or not e.strip():
            return False, f"Evidence item {i+1} is empty"
        if not normalizer.is_anchored(e, input_text):
            return False, f"Evidence '{e}' not found in input text"
    return True, ""


def validate_counterfactual(raw_response: str, input_text: str,
                            original_label: str, label_set: List[str]) -> Tuple[bool, str]:
    if not raw_response or not raw_response.strip():
        return False, "Empty counterfactual response"
    import json
    try:
        obj = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        import re
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_response, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                return False, "Response is not valid JSON"
        else:
            return False, "Response is not valid JSON"
    cf_text = obj.get("counterfactual_text", "")
    new_pred = obj.get("new_prediction", "")
    if not isinstance(cf_text, str) or len(cf_text.strip()) < 3:
        return False, "Counterfactual text is too short"
    if new_pred not in label_set:
        return False, f"New prediction '{new_pred}' not in label set"
    if new_pred == original_label:
        return False, "Counterfactual prediction did not flip"
    from src.parsing.parser import Parser
    edit_ratio = Parser._word_edit_ratio(input_text, cf_text)
    if edit_ratio > 0.3:
        return False, f"Edit ratio {edit_ratio:.3f} exceeds 0.3 threshold"
    return True, ""


def validate_rank_ordering(raw_response: str, input_text: str, normalizer) -> Tuple[bool, str]:
    if not raw_response or not raw_response.strip():
        return False, "Empty rank ordering response"
    import json
    try:
        obj = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        import re
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw_response, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                return False, "Response is not valid JSON"
        else:
            return False, "Response is not valid JSON"
    ranking = obj.get("ranking", [])
    if not isinstance(ranking, list):
        return False, "Ranking is not a list"
    if len(ranking) != 5:
        return False, f"Expected 5 ranked items, got {len(ranking)}"
    seen = set()
    for i, token in enumerate(ranking):
        if not isinstance(token, str) or not token.strip():
            return False, f"Rank item {i+1} is empty"
        if not normalizer.is_anchored(token, input_text):
            return False, f"Rank item '{token}' not found in input text"
        norm = normalizer.pre_normalize(token)
        if norm in seen:
            return False, f"Duplicate rank item '{token}'"
        seen.add(norm)
    return True, ""
