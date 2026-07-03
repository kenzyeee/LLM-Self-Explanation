import difflib
import json
import logging
import re
import string
from typing import Tuple, List, Optional, Set

from src.utils.exceptions import ParsingError
from src.normalization.normalizer import DISCOURSE_WORDS, POLARITY_WORDS

logger = logging.getLogger(__name__)


def dynamic_k(input_text: str, cap: Optional[int] = None) -> int:
    """Length-proportional top-k for feature-importance set extraction.

    Huang et al. 2023 (arXiv:2310.11207) and "Dynamic Top-k Estimation"
    (arXiv:2310.05619) select k proportional to input length rather than a fixed
    constant, which reduces spurious cross-method disagreement driven by set size.
    k = max(3, round(L / 5)) where L is the content-word count, capped at `cap`
    (the number of available valid tokens) when provided.
    """
    words = input_text.split()
    n_content = sum(1 for w in words if w.strip(string.punctuation))
    k = max(3, round(n_content / 5))
    if cap is not None:
        k = min(k, cap)
    return k

# POLARITY_WORDS is imported from normalizer.py — ONE whitelist shared by every
# evidence path (H, R, CF, RO). A divergent local copy is exactly the asymmetry that
# silently dropped contracted negations from 3 of 4 strategies (review §8.4).
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

# Open-class (content-word) POS tags. Rationale token extraction keeps the lemmas
# of content words (ERASER / rationalization survey arXiv:2301.08912), rather than a
# hand-picked dependency-label subset which has no published precedent.
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

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


def ensure_spacy_available() -> None:
    """Hard-fail at startup if spaCy's POS tagger is unavailable.

    parse_rationale silently falls back to whitespace-split content-word extraction
    per-instance when spaCy is missing (see its docstring) — which changes the R
    token set, and therefore ECS, depending on the machine's environment. That
    fallback exists so a single instance doesn't fail outright; it must not be how
    an entire collection run quietly happens. Call this once before collection
    starts so a missing model is a loud, immediate error instead of a silent
    per-instance degradation discovered only when comparing runs.
    """
    if _get_spacy() is None:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not available, but rationale (R) token "
            "extraction depends on it for reproducible results. Install it with:\n"
            "  pip install spacy && python -m spacy download en_core_web_sm\n"
            "(Without it, R falls back to whitespace-split extraction, which yields a "
            "different token set than POS-lemma extraction and would make this run "
            "silently incomparable to any run made with spaCy available.)"
        )


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

    def parse_confidence(self, raw_response: str) -> float:
        """Parse a verbalized-confidence response into a probability in [0, 1].

        Expected format: {"confidence": <0-100>}. Verbalized numerical confidence is
        the standard no-logprob elicitation (Tian et al. 2023, "Just Ask for
        Calibration"; Xiong et al. 2024, ICLR) — Bedrock's Converse API exposes no
        token logprobs, so this is the only confidence signal available API-only.
        Values are accepted on either a 0-100 or 0-1 scale (a value <= 1 is treated
        as already being a probability) and clamped to [0, 1].
        """
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in confidence response")
        value = obj.get("confidence")
        if isinstance(value, str):
            try:
                value = float(value.strip().rstrip("%"))
            except ValueError:
                raise ParsingError(f"Confidence '{value}' is not numeric")
        if not isinstance(value, (int, float)):
            raise ParsingError(f"Confidence must be numeric, got {type(value).__name__}")
        conf = float(value)
        if conf < 0:
            raise ParsingError(f"Confidence {conf} is negative")
        if conf > 1.0:
            conf = conf / 100.0
        return max(0.0, min(1.0, conf))

    def parse_highlighting(self, raw_response: str, input_text: str, normalizer, skip_validation: bool = False) -> List[str]:
        """Parse graded salience response.

        Accepted formats (the prompt asks for the list-of-pairs form; the dict form is
        accepted for robustness because models sometimes emit it):
          {"salience": [["word1", score], ["word2", score], ...]}   (canonical)
          {"salience": {"word1": score, "word2": score, ...}}       (legacy/fallback)
        A JSON object cannot represent the same word twice (duplicate keys silently
        collapse to the last one — review §8.6d), which is why the canonical schema is
        a list of pairs; repeated words are aggregated by MAX score (the importance of
        a word type = its most important occurrence).

        Selection method (deterministic):
          1. Keep entries whose key is a single word, scores >= 1, that anchor in the
             input text. Pure-punctuation / empty keys (model padding such as "(" or
             ".") are dropped silently — they are noise, not evidence.
          2. Restrict to *content* tokens: anything the normalizer would discard
             (stopwords, discourse/label words, punctuation) is removed BEFORE ranking,
             while polarity words (incl. contracted negations) are retained.
          3. Rank by NORMALIZED token (the same canonical token space every other
             strategy's evidence lives in — review §8.2/§8.6b), deduplicated by max
             score, sort by salience (descending) and return the length-proportional
             top-k (dynamic_k). The returned ranked top-k list is reused by the caller
             for the downstream Kendall τ / RBO comparison with RO, so it MUST be in
             the same normalized space as the RO ranking.

        Full normalized salience weights are exposed on ``self._h_salience_weights``
        (dict normalized_token -> max score over the whole input) for the
        salience-weighted random baseline.
        """
        self._h_salience_weights: dict = {}
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in highlighting response")
        salience = obj.get("salience")
        if isinstance(salience, dict):
            entries = list(salience.items())
        elif isinstance(salience, list):
            entries = []
            for item in salience:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    entries.append((item[0], item[1]))
                elif isinstance(item, dict) and len(item) == 1:
                    entries.append(next(iter(item.items())))
        else:
            entries = []
        if not entries:
            raise ParsingError("Salience must be a non-empty list of [word, score] pairs or a word->score dict")
        # Aggregate by NORMALIZED token, max score. This is both the duplicate-handling
        # rule and the projection into the canonical token space.
        norm_scored: dict = {}
        for word, score in entries:
            if not isinstance(word, str) or not word.strip():
                continue
            if not isinstance(score, (int, float)) or score < 1:
                continue
            if not self._is_single_word(word):
                logger.warning(f"H salience item '{word}' is not a single word — discarding")
                continue
            # Drop pure-punctuation / empty-after-strip keys silently (model noise).
            if not word.strip(string.punctuation + "“”‘’\"'`"):
                continue
            if not skip_validation and not normalizer.is_anchored(word, input_text):
                logger.warning(f"H salience word '{word}' not anchored — discarding")
                continue
            norm = normalizer.normalize(word)
            if norm is None:
                continue
            norm_scored[norm] = max(norm_scored.get(norm, 0.0), float(score))
        if len(norm_scored) < 2:
            raise ParsingError(f"Only {len(norm_scored)} valid content salience entries (need >=2)")
        self._h_salience_weights = dict(norm_scored)
        # Sort descending by score (normalized-token ties broken alphabetically for
        # determinism), then return the length-proportional top-k.
        ranked = sorted(norm_scored.items(), key=lambda x: (-x[1], x[0]))
        k = dynamic_k(input_text, cap=len(ranked))
        return [w for w, _ in ranked[:k]]

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
            logger.warning("spaCy not available, falling back to content-word extraction from rationale")
            anchored = []
            seen = set()
            for raw_tok in rationale.split():
                tok = raw_tok.strip(string.punctuation).lower()
                if not tok or len(tok) <= 1 or tok in STOPWORDS or tok in DISCOURSE_WORDS:
                    continue
                if tok in seen:
                    continue
                seen.add(tok)
                if normalizer.is_anchored(tok, input_text):
                    anchored.append(tok)
            if not anchored:
                raise ParsingError("No evidence tokens could be extracted from rationale (all unanchored)")
            return rationale, anchored
        # Step 1: extract hyphenated compounds from rationale that appear verbatim in input
        hyphenated_in_rationale = re.findall(r'\b\w+-\w+\b', rationale)
        hyphenated_matches = set()
        for compound in hyphenated_in_rationale:
            compound_lower = compound.lower()
            if normalizer.is_anchored(compound_lower, input_text):
                hyphenated_matches.add(compound_lower)
        # Step 2: POS-tag the rationale and keep open-class content-word lemmas.
        # Polarity words (incl. contracted negations) are ALSO kept, regardless of
        # POS — H and RO retain them via the normalizer's polarity whitelist, so R
        # excluding them was an evidence-space asymmetry (review §8.4) that broke
        # cross-strategy comparability precisely on negation-driven (NLI) instances.
        doc = nlp(rationale)
        content_tokens = set()
        for token in doc:
            lower = token.lower_.strip()
            lemma = token.lemma_.lower().strip()
            if lower in POLARITY_WORDS or lemma in POLARITY_WORDS:
                content_tokens.add(lower if lower in POLARITY_WORDS else lemma)
                continue
            if token.pos_ in CONTENT_POS:
                if (lemma and lemma not in STOPWORDS
                        and lemma not in DISCOURSE_WORDS and len(lemma) > 1):
                    content_tokens.add(lemma)
        if not content_tokens:
            content_tokens = set()
            for token in doc:
                lemma = token.lemma_.lower().strip()
                if lemma and lemma not in STOPWORDS and lemma not in DISCOURSE_WORDS and len(lemma) > 1:
                    content_tokens.add(lemma)
        # Merge hyphenated compound matches into content_tokens
        content_tokens.update(hyphenated_matches)
        # Step 3: anchored rationale extraction — keep only tokens that directly appear in input text.
        # Tokens with no anchor are logged as INTRODUCED concepts (post-hoc rationalization signal),
        # not silently dropped.
        anchored = []
        introduced = []
        seen = set()
        for tok in content_tokens:
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

    @staticmethod
    def _split_on_marker(text: str, marker: str) -> Tuple[str, str]:
        """Split into (prefix_through_marker, editable_span_after_marker) at the FIRST
        occurrence of `marker`. Returns (text, "") when the marker is absent."""
        idx = text.find(marker)
        if idx < 0:
            return text, ""
        cut = idx + len(marker)
        return text[:cut], text[cut:]

    @staticmethod
    def _collapse_ws(s: str) -> str:
        return " ".join(s.split()).strip().casefold()

    def parse_counterfactual(self, raw_response: str, input_text: str,
                             original_label: str, label_set: List[str],
                             normalizer, skip_validation: bool = False,
                             max_edit_ratio: float = 0.3,
                             edit_span_marker: Optional[str] = None) -> Tuple[str, str, Set[str]]:
        """Parse rewrite-based CF response via difflib.

        Expected format:
          {"rewritten": "<full rewritten text>", "new_prediction": "<target label>"}

        Returns (rewritten_text, new_prediction, from_tokens).
        from_tokens are the original SURFACE words that differ between input_text and
        rewritten (raw edit attribution — the caller normalizes them into the shared
        evidence token space before any cross-strategy comparison).

        edit_span_marker: when the prompt restricts edits to a span (MNLI: "edit only
        the Hypothesis"), pass the marker ("Hypothesis:"). Then (a) the text before
        and including the marker must be unchanged (whitespace/case-insensitive) —
        an edited Premise is a rules violation; and (b) the minimal-edit ratio is
        computed over the EDITABLE span only. Without this, the effective cap on a
        hypothesis-only edit varies with premise length (review §8.6c): a fully
        legitimate minimal hypothesis edit auto-fails when the premise is short.
        """
        text = raw_response.strip()
        obj = self._extract_json(text)
        if obj is None:
            raise ParsingError("No valid JSON found in counterfactual response")
        rewritten = obj.get("rewritten")
        new_pred = obj.get("new_prediction")
        if rewritten is None or new_pred is None:
            raise ParsingError("Counterfactual declared impossible (null rewritten or prediction)")
        # Guard the field type before any string ops: some models emit a nested object or
        # list for "rewritten" (e.g. {"rewritten": {...}}), and rewritten.strip() below
        # would raise AttributeError — which escapes the caller's ParsingError handler and
        # kills the whole instance. Fail as a ParsingError so only CF is invalidated.
        if not isinstance(rewritten, str):
            raise ParsingError(f"Counterfactual 'rewritten' must be a string, got {type(rewritten).__name__}")
        if not isinstance(new_pred, str) or new_pred not in label_set:
            raise ParsingError(f"New prediction '{new_pred}' not in label set")
        if new_pred == original_label:
            raise ParsingError("Counterfactual prediction did not flip")
        if rewritten.strip() == input_text.strip():
            raise ParsingError("Counterfactual text is identical to original")
        # Span-restricted CF: validate the protected prefix and narrow the ratio window.
        ratio_original, ratio_rewritten = input_text, rewritten
        if edit_span_marker and edit_span_marker in input_text:
            orig_prefix, orig_span = self._split_on_marker(input_text, edit_span_marker)
            rew_prefix, rew_span = self._split_on_marker(rewritten, edit_span_marker)
            if edit_span_marker not in rewritten:
                raise ParsingError(f"Counterfactual dropped the '{edit_span_marker}' structure; "
                                   "cannot verify the protected span was kept")
            if self._collapse_ws(orig_prefix) != self._collapse_ws(rew_prefix):
                raise ParsingError("Counterfactual edited outside the allowed span "
                                   f"(text before '{edit_span_marker}' changed)")
            ratio_original, ratio_rewritten = orig_span, rew_span
            if not ratio_rewritten.strip():
                raise ParsingError("Counterfactual editable span is empty after rewrite")
        # Extract changed tokens via difflib (over the full text — with a validated
        # unchanged prefix this is equivalent to diffing the span, and it keeps
        # single-segment datasets on the same code path). A flip achieved purely by
        # INSERTING a word (e.g. negating a hypothesis by adding "not") changes the
        # text but leaves no original token replaced or deleted, so there is no
        # original-token attribution for ECS — treat as non-attributable rather than
        # mislabelling it "identical".
        from_tokens = self._extract_changed_tokens(input_text, rewritten)
        if not from_tokens:
            raise ParsingError("Counterfactual flips only by insertion(s); no original token "
                               "was replaced or deleted to attribute the prediction to")
        # Validate edit ratio (word-level Levenshtein over the editable window).
        edit_ratio = self._word_edit_ratio(ratio_original, ratio_rewritten)
        if not skip_validation and edit_ratio > max_edit_ratio:
            raise ParsingError(f"Counterfactual edit ratio {edit_ratio:.3f} exceeds {max_edit_ratio} threshold"
                               + (" (computed over the editable span)" if edit_span_marker else ""))
        return rewritten, new_pred, from_tokens

    @staticmethod
    def _extract_changed_tokens(original: str, rewritten: str) -> Set[str]:
        """Use difflib to find original tokens that were changed or removed."""
        orig_words = original.strip().split()
        rew_words = rewritten.strip().split()
        matcher = difflib.SequenceMatcher(None, orig_words, rew_words)
        changed = set()
        # Strip surrounding punctuation including hyphens/unicode dashes so that an
        # edited token like "charm-less," diffs down to the bare word "charm-less".
        strip_chars = string.punctuation + "‐‑‒–—―−"
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ('replace', 'delete'):
                for w in orig_words[i1:i2]:
                    w_clean = w.strip(strip_chars)
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
        """Try to parse JSON from response text. Handles surrounding text, code fences, etc.

        Adds a repair pass for the most common LLM JSON error: unescaped double quotes
        inside a string value (e.g. a rationale that quotes a phrase from the text:
        ``"...the film as being "as seductive as it is haunting" implies..."``). Such
        responses otherwise fail json.loads and the whole strategy is discarded.
        """
        text = text.strip()
        candidates = [text]
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m:
            candidates.append(m.group(1).strip())
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            candidates.append(m.group(0))
        for cand in candidates:
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                pass
        # Repair pass: re-escape stray quotes inside string values, then retry.
        for cand in candidates:
            repaired = self._repair_unescaped_quotes(cand)
            if repaired != cand:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _repair_unescaped_quotes(s: str) -> str:
        """Escape double quotes that appear *inside* a JSON string value.

        Walks the text as a state machine. A quote that opens/closes a string is one
        whose next non-whitespace neighbour is structural (``: , } ]`` or the string
        start). Any other quote encountered while inside a string is content and gets
        backslash-escaped so json.loads can parse it. Already-escaped quotes (\\") and
        backslashes are passed through untouched.
        """
        out = []
        in_string = False
        escaped = False
        n = len(s)
        for i, c in enumerate(s):
            if escaped:
                out.append(c)
                escaped = False
                continue
            if c == '\\':
                out.append(c)
                escaped = True
                continue
            if c == '"':
                if not in_string:
                    in_string = True
                    out.append(c)
                else:
                    j = i + 1
                    while j < n and s[j] in ' \t\r\n':
                        j += 1
                    if j >= n or s[j] in ',}]:':
                        in_string = False
                        out.append(c)
                    else:
                        out.append('\\"')  # inner content quote — escape it
                continue
            out.append(c)
        return ''.join(out)

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
        # MiCE (Ross et al. 2021): normalize the word-level Levenshtein distance by the
        # length of the ORIGINAL input, not max(n, m).
        return distance / n if n > 0 else 0.0
