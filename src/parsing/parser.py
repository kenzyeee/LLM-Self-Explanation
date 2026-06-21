import difflib
import json
import logging
import re
from typing import Tuple, List, Optional, Set

from src.utils.exceptions import ParsingError
from src.normalization.normalizer import DISCOURSE_WORDS

logger = logging.getLogger(__name__)

POLARITY_WORDS = {"no", "not", "never", "nor", "neither", "none", "nobody", "nothing", "nowhere"}
STOPWORDS = set()
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "over", "after",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "can", "could", "shall",
        "should", "may", "might", "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
        "their", "this", "that", "these", "those", "not", "no", "nor",
    }

# Dependency labels that capture meaningful content from a rationale sentence
# per Algorithm 1 Step 7 in the original paper spec
RATIONALE_DEP_LABELS = {"nsubj", "nsubjpass", "dobj", "attr", "ROOT", "amod", "acomp", "pobj", "agent"}

# spaCy model loaded lazily
_nlp = None


def _get_spacy():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception:
            _nlp = None
    return _nlp


class Parser:
    @staticmethod
    def _is_single_word(s: str) -> bool:
        """Reject multi-word phrases, hyphenated compounds, and empty strings."""
        if not s or not isinstance(s, str):
            return False
        s = s.strip()
        if not s:
            return False
        if ' ' in s:
            return False
        return True

    def parse_classification(self, raw_response: str, label_set: List[str]) -> str:
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in classification response")
        label = obj.get("label", "")
        if not isinstance(label, str) or label not in label_set:
            raise ParsingError(f"Label '{label}' not in allowed set: {label_set}")
        return label

    def parse_highlighting(self, raw_response: str, input_text: str, normalizer, skip_validation: bool = False) -> List[str]:
        """Parse graded salience response.

        Expected format:
          {"salience": {"word1": score, "word2": score, ...}}

        Returns top-5 words by salience score that anchor in the input text.
        Also stores the full score-sorted list in self._h_salience_ordered
        for downstream Kendall τ comparison with RO.
        """
        self._h_salience_ordered: List[str] = []
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in highlighting response")
        salience = obj.get("salience", {})
        if not isinstance(salience, dict) or not salience:
            raise ParsingError("Salience must be a non-empty dict of word -> score")
        # Validate scores and filter anchored words
        scored = []
        for word, score in salience.items():
            if not isinstance(word, str) or not word.strip():
                continue
            if not isinstance(score, (int, float)) or score < 1:
                continue
            if not self._is_single_word(word):
                logger.warning(f"H salience item '{word}' is not a single word — discarding")
                continue
            if not skip_validation and not normalizer.is_anchored(word, input_text):
                logger.warning(f"H salience word '{word}' not anchored — discarding")
                continue
            scored.append((word, float(score)))
        if len(scored) < 2:
            raise ParsingError(f"Only {len(scored)} valid salience entries (need >=2)")
        # Sort descending by score, then return top 5
        scored.sort(key=lambda x: -x[1])
        self._h_salience_ordered = [w for w, _ in scored]
        return [w for w, _ in scored[:5]]

    def parse_rationale(self, raw_response: str, input_text: str, normalizer, skip_validation: bool = False) -> Tuple[str, List[str]]:
        # Reset per-call; populated with rationale concepts that have NO input
        # anchor (post-hoc "introduced" concepts) for the introduced-concept rate.
        self._r_introduced: List[str] = []
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in rationale response")
        rationale = obj.get("rationale", "")
        if not isinstance(rationale, str) or not rationale:
            raise ParsingError("Rationale text is empty or not a string")
        if skip_validation:
            return rationale, []
        nlp = _get_spacy()
        if nlp is None:
            logger.warning("spaCy not available, falling back to word-token extraction from rationale")
            tokens = rationale.split()
            return rationale, tokens[:5]
        # Step 1: extract hyphenated compounds from rationale that appear verbatim in input
        hyphenated_in_rationale = re.findall(r'\b\w+-\w+\b', rationale)
        hyphenated_matches = set()
        for compound in hyphenated_in_rationale:
            compound_lower = compound.lower()
            if normalizer.is_anchored(compound_lower, input_text):
                hyphenated_matches.add(compound_lower)
        # Step 2: dependency-parse the rationale sentence
        doc = nlp(rationale)
        dep_tokens = set()
        for token in doc:
            if token.dep_ in RATIONALE_DEP_LABELS:
                lemma = token.lemma_.lower().strip()
                if (lemma and lemma not in STOPWORDS and lemma not in POLARITY_WORDS
                        and lemma not in DISCOURSE_WORDS and len(lemma) > 1):
                    dep_tokens.add(lemma)
        if not dep_tokens:
            dep_tokens = set()
            for token in doc:
                lemma = token.lemma_.lower().strip()
                if lemma and lemma not in STOPWORDS and lemma not in DISCOURSE_WORDS and len(lemma) > 1:
                    dep_tokens.add(lemma)
        # Merge hyphenated compound matches into dep_tokens
        dep_tokens.update(hyphenated_matches)
        # Step 3: anchored rationale extraction — keep only tokens that directly appear in input text.
        # Tokens with no anchor are logged as INTRODUCED concepts (post-hoc rationalization signal),
        # not silently dropped.
        anchored = []
        introduced = []
        seen = set()
        for tok in dep_tokens:
            if tok in seen:
                continue
            seen.add(tok)
            if normalizer.is_anchored(tok, input_text):
                anchored.append(tok)
            else:
                introduced.append(tok)
                logger.info(f"R token '{tok}' has no input anchor — introduced concept")
        self._r_introduced = introduced
        if not anchored:
            raise ParsingError("No evidence tokens could be extracted from rationale (all unanchored)")
        return rationale, anchored

    def parse_counterfactual(self, raw_response: str, input_text: str,
                             original_label: str, label_set: List[str],
                             normalizer, skip_validation: bool = False,
                             max_edit_ratio: float = 0.3) -> Tuple[str, str, Set[str]]:
        """Parse rewrite-based CF response via difflib.

        Expected format:
          {"rewritten": "<full rewritten text>", "new_prediction": "<target label>"}

        Returns (rewritten_text, new_prediction, from_tokens).
        from_tokens are the original words that differ between input_text and rewritten.
        """
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in counterfactual response")
        rewritten = obj.get("rewritten")
        new_pred = obj.get("new_prediction")
        if rewritten is None or new_pred is None:
            raise ParsingError("Counterfactual declared impossible (null rewritten or prediction)")
        if not isinstance(new_pred, str) or new_pred not in label_set:
            raise ParsingError(f"New prediction '{new_pred}' not in label set")
        if new_pred == original_label:
            raise ParsingError("Counterfactual prediction did not flip")
        # Extract changed tokens via difflib
        from_tokens = self._extract_changed_tokens(input_text, rewritten)
        if not from_tokens:
            raise ParsingError("No tokens changed between original and rewritten text (identical text)")
        if rewritten == input_text:
            raise ParsingError("Counterfactual text is identical to original")
        # Validate edit ratio (word-level Levenshtein)
        edit_ratio = self._word_edit_ratio(input_text, rewritten)
        if not skip_validation and edit_ratio > max_edit_ratio:
            raise ParsingError(f"Counterfactual edit ratio {edit_ratio:.3f} exceeds {max_edit_ratio} threshold")
        return rewritten, new_pred, from_tokens

    @staticmethod
    def _extract_changed_tokens(original: str, rewritten: str) -> Set[str]:
        """Use difflib to find original tokens that were changed or removed."""
        orig_words = original.strip().split()
        rew_words = rewritten.strip().split()
        matcher = difflib.SequenceMatcher(None, orig_words, rew_words)
        changed = set()
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ('replace', 'delete'):
                for w in orig_words[i1:i2]:
                    w_clean = w.strip('.,!?;:\'"()[]{}')
                    if w_clean:
                        changed.add(w_clean.lower())
        return changed

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
            if not self._is_single_word(token):
                logger.warning(f"Rank item '{token}' is not a single word — discarding")
                continue
            norm = normalizer.pre_normalize(token)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            if norm in POLARITY_WORDS:
                pass  # keep polarity words
            elif norm in STOPWORDS:
                logger.warning(f"Rank item '{token}' is a stopword — discarding")
                continue
            if not skip_validation and not normalizer.is_anchored(token, input_text):
                logger.warning(f"Rank item '{token}' not anchored in input text — discarding")
                continue
            result.append((token, rank))
            rank += 1
        if len(result) < 3:
            raise ParsingError(f"Only {len(result)} unique valid tokens remain (need >=3)")
        return result

    def _extract_json(self, text: str) -> Optional[dict]:
        """Try to parse JSON from response text. Handles surrounding text, code fences, etc."""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _word_edit_ratio(original: str, counterfactual: str) -> float:
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
