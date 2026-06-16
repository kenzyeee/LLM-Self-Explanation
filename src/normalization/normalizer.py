import re
import html
import string
import logging
import difflib
from typing import List, Set, Optional

try:
    import nltk
    import nltk.data
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    stopwords = None
    WordNetLemmatizer = None

logger = logging.getLogger(__name__)

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
    "world", "sports", "business", "sci/tech",
    "us", "u",
}

SEP_PATTERN = re.compile(r'\[SEP\]', re.IGNORECASE)
HTML_ENTITY_PATTERN = re.compile(r'&[a-zA-Z]+;|&#\d+;')

FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "with",
}


class FallbackLemmatizer:
    def lemmatize(self, token: str) -> str:
        if not token.isalpha():
            return token
        if len(token) > 4 and token.endswith("vies"):
            return token[:-1]
        if len(token) > 3 and token.endswith("ies"):
            return f"{token[:-3]}y"
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token


def _wordnet_available() -> bool:
    try:
        nltk.data.find("corpora/wordnet")
        return True
    except Exception:
        return False


class Normalizer:
    def __init__(self, use_lemmatization=True, remove_stopwords=True):
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        if use_lemmatization and WordNetLemmatizer and _wordnet_available():
            self.lemmatizer = WordNetLemmatizer()
        elif use_lemmatization:
            self.lemmatizer = FallbackLemmatizer()
        else:
            self.lemmatizer = None
        try:
            self.stop_words = set(stopwords.words('english')) if remove_stopwords and stopwords else set()
        except Exception:
            self.stop_words = set()
        if remove_stopwords and not self.stop_words:
            self.stop_words = FALLBACK_STOPWORDS

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

        if self.remove_stopwords and t in self.stop_words:
            return None
        if t in DISCOURSE_WORDS:
            return None

        if self.use_lemmatization and self.lemmatizer:
            t = self.lemmatizer.lemmatize(t)

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
        """Check if a token (after pre-normalization) appears in the input text."""
        norm_token = self.pre_normalize(token)
        if not norm_token:
            return False
        norm_input = self.normalize_input_text(input_text)
        return norm_token in norm_input

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

    def extract_content_words_from_rationale(self, rationale: str) -> Set[str]:
        """Legacy method - kept for backward compatibility. Use extract_evidence_tokens instead."""
        try:
            from nltk.tokenize import word_tokenize
            from nltk import pos_tag
            tokens = word_tokenize(rationale)
            tagged = pos_tag(tokens)
            content_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN',
                            'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
            content_words = [word for word, tag in tagged if tag in content_tags]
            return self.normalize_tokens(content_words)
        except Exception:
            tokens = rationale.split()
            return self.normalize_tokens(tokens)

    def extract_counterfactual_diff(self, original: str, counterfactual: str) -> Set[str]:
        orig_words = original.lower().split()
        cf_words = counterfactual.lower().split()
        min_len = min(len(orig_words), len(cf_words))
        diff = []
        for i in range(min_len):
            if orig_words[i] != cf_words[i]:
                diff.append(orig_words[i])
        if len(cf_words) < len(orig_words):
            diff.extend(orig_words[len(cf_words):])
        return self.normalize_tokens(list(diff))
