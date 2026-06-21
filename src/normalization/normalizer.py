import re
import html
import string
import logging
import difflib
from typing import List, Set, Optional

logger = logging.getLogger(__name__)

_NLTK_DATA_READY = False

DISCOURSE_WORDS = {
    "correct", "incorrect", "classification", "classify",
    "text", "sentence", "passage", "article", "statement",
    "indicates", "indicating", "indicated", "indicate",
    "describes", "describing", "described", "describe",
    "supports", "supporting", "supported", "support",
    "based", "basing", "context", "contextual",
    "prediction", "predict", "predicts",
    "evidence", "explanation", "explain",
    "important",
    "positive", "negative", "neutral",
    "entailment", "contradiction",
    "entail", "contradict",
    "premise", "hypothesis",
    "us", "u",
}

SEP_PATTERN = re.compile(r'\[SEP\]', re.IGNORECASE)
HTML_ENTITY_PATTERN = re.compile(r'&[a-zA-Z]+;|&#\d+;')

POLARITY_WORDS = {"no", "not", "never", "nor", "neither", "none", "nobody", "nothing", "nowhere",
                   "every", "some", "any", "all"}
FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "over", "after",
    "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "can", "could", "shall",
    "should", "may", "might", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "this", "that", "these", "those", "very",
    "just", "then", "than", "so", "too", "also", "now", "here", "there",
}


def _ensure_nltk_data() -> None:
    """Download required NLTK corpora once, lazily (never at import time).

    Keeping this out of module import makes importing the package fast and
    offline-safe; the (cached, no-op-after-first) download runs the first time
    a Normalizer is constructed.
    """
    global _NLTK_DATA_READY
    if _NLTK_DATA_READY:
        return
    try:
        import nltk
        for resource in ('stopwords', 'wordnet', 'omw-1.4'):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
    except ImportError:
        pass
    _NLTK_DATA_READY = True


class Normalizer:
    def __init__(self, use_lemmatization: bool = True, remove_stopwords: bool = True,
                 lemmatizer: str = "wordnet"):
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        _ensure_nltk_data()
        if remove_stopwords:
            try:
                from nltk.corpus import stopwords as nltk_stopwords
                self.stop_words = set(nltk_stopwords.words('english'))
            except Exception:
                self.stop_words = FALLBACK_STOPWORDS.copy()
        else:
            self.stop_words = set()
        self._lemmatizer = None
        if use_lemmatization:
            if lemmatizer != "wordnet":
                logger.warning(f"Lemmatizer '{lemmatizer}' not implemented; using WordNet instead.")
            try:
                from nltk.stem import WordNetLemmatizer
                self._lemmatizer = WordNetLemmatizer()
                self._lemmatizer.lemmatize("tests")  # probe: surface a missing corpus now, not per-token
            except Exception:
                logger.warning("WordNet data unavailable; proceeding without lemmatization.")
                self._lemmatizer = None

    def pre_normalize(self, token: str) -> str:
        """Light normalization for input-anchored matching (no lemmatization)."""
        t = token.strip().strip('\'"*')
        t = t.lower()
        t = t.strip(string.punctuation)
        return t

    def normalize(self, token: str) -> Optional[str]:
        """Full normalization pipeline for evidence tokens."""
        if not token:
            return None
        t = token.strip()
        if not t:
            return None

        t = html.unescape(t)
        t = SEP_PATTERN.sub('', t)
        t = t.lower()
        t = t.strip(string.punctuation)
        if not t:
            return None

        if self._lemmatizer is not None:
            # No POS context is available for a lone token, so apply verb-then-noun
            # WordNet lemmatization — a deterministic heuristic that collapses the
            # most common inflections (e.g. "running"->"run", "movies"->"movie").
            t = self._lemmatizer.lemmatize(self._lemmatizer.lemmatize(t, 'v'), 'n')

        if t in POLARITY_WORDS:
            return t
        if t in self.stop_words:
            return None
        if t in DISCOURSE_WORDS:
            return None

        return t if t else None

    def normalize_tokens(self, tokens: List[str]) -> Set[str]:
        """Normalize a list of token strings. Splits multi-word spans into individual tokens."""
        result = set()
        for t in tokens:
            for word in t.split():
                norm = self.normalize(word)
                if norm:
                    result.add(norm)
        return result

    def normalize_input_text(self, text: str) -> str:
        """Return canonical lowercase text for input-anchored matching."""
        t = html.unescape(text)
        t = SEP_PATTERN.sub(' ', t)
        t = t.lower()
        t = t.strip()
        return t

    def is_anchored(self, token: str, input_text: str) -> bool:
        """Check if token appears as a whole word in input_text (word-boundary match)."""
        norm_token = self.pre_normalize(token)
        if not norm_token:
            return False
        norm_input = self.normalize_input_text(input_text)
        pattern = re.compile(r'\b' + re.escape(norm_token) + r'\b')
        return bool(pattern.search(norm_input))

    def _exact_or_fuzzy_match(self, token: str, input_text: str) -> bool:
        if token.lower() in input_text.lower():
            return True
        for input_word in input_text.lower().split():
            ratio = difflib.SequenceMatcher(None, token.lower(), input_word).ratio()
            if ratio >= 0.85:
                return True
        return False

    def is_verbatim_in_input(self, token: str, input_text: str) -> bool:
        """Check if a token appears verbatim in the input text (before lemmatization).
        Uses exact substring match first, then fuzzy match with 0.85 threshold.
        Returns True if match found."""
        return self._exact_or_fuzzy_match(token, input_text)

    def check_evidence_compliance(self, evidence_tokens: List[str], input_text: str) -> int:
        """Count how many evidence tokens are NOT verbatim in input (compliance violations).
        Uses exact match first, then fuzzy match at 0.85 threshold.
        Returns number of violations. Logs warnings for each violation."""
        violations = 0
        for token in evidence_tokens:
            if not self.is_verbatim_in_input(token, input_text):
                logger.warning(f"Evidence compliance violation: '{token}' not verbatim in input")
                violations += 1
            else:
                fuzzy_only = not (token.lower() in input_text.lower())
                if fuzzy_only:
                    logger.info(f"Evidence token '{token}' matched via fuzzy similarity (not exact)")
        return violations
