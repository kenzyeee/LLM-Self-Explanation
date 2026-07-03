import re
import html
import string
import logging
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

# Contracted negations. NLTK's English stopword list contains these (both the
# apostrophe form "shouldn't" and the bare stem "shouldn"), so without an explicit
# whitelist they are silently dropped from evidence sets — on NLI, negation IS the
# label-critical evidence (review §8.4: "shouldn't" being discarded invalidated an
# entire RO strategy). Apostrophe-stripped variants are included because
# pre_normalize/normalize strip surrounding punctuation and some model tokenizations
# emit "dont"/"shouldnt".
NEGATION_CONTRACTIONS = {
    "n't", "nt", "cannot",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "won't", "wouldn't", "can't", "couldn't", "shouldn't", "shan't",
    "mustn't", "needn't", "hasn't", "haven't", "hadn't", "ain't", "daren't", "mightn't",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "wont", "wouldnt", "cant", "couldnt", "shouldnt", "shant",
    "mustnt", "neednt", "hasnt", "havent", "hadnt", "aint", "darent", "mightnt",
    # NLTK stopword stems of the above (post-apostrophe-split tokenizations)
    "don", "doesn", "didn", "isn", "aren", "wasn", "weren",
    "won", "wouldn", "couldn", "shouldn", "shan", "mustn", "needn",
    "hasn", "haven", "hadn", "ain", "daren", "mightn",
}

# Single polarity whitelist shared by every evidence-extraction path (H, R, CF, RO).
# parser.py imports this — do NOT define a divergent copy elsewhere (that asymmetry
# was review §8.4).
POLARITY_WORDS = {"no", "not", "never", "nor", "neither", "none", "nobody", "nothing", "nowhere",
                   "every", "some", "any", "all"} | NEGATION_CONTRACTIONS
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

        # Anchoring lemmatizer — ALWAYS available, independent of `use_lemmatization`.
        # is_anchored() must match an explanation token to its input occurrence even
        # across inflection (e.g. rationale lemma "scene" vs input surface "scenes",
        # "moved" vs "move"). Evidence-set normalization may run unlemmatized, but
        # anchoring should never be defeated by a plural/tense difference, so this
        # lemmatizer is constructed separately and used only for the anchor fallback.
        self._anchor_lemmatizer = self._lemmatizer
        if self._anchor_lemmatizer is None:
            try:
                from nltk.stem import WordNetLemmatizer
                self._anchor_lemmatizer = WordNetLemmatizer()
                self._anchor_lemmatizer.lemmatize("tests")
            except Exception:
                self._anchor_lemmatizer = None
        self._input_lemma_cache: dict = {}

    def pre_normalize(self, token: str) -> str:
        """Light normalization for input-anchored matching (no lemmatization)."""
        t = token.strip().strip('\'"*')
        t = t.lower()
        t = t.strip(string.punctuation)
        return t

    def _lemmatize_to_fixed_point(self, token: str) -> str:
        """Apply verb→noun→adjective→adverb WordNet lemmatization repeatedly until stable.

        A single pass is not idempotent: e.g. 'canings' reduces only via the noun
        step to 'caning', but feeding 'caning' back finds the verb base 'can'.
        Without iterating, normalize(normalize(x)) can differ from normalize(x).
        Iterating to a fixed point guarantees the returned lemma is stable under
        re-lemmatization. The adjective/adverb steps ('a', 'r') are included so
        comparative/superlative inflections ('happier'→'happy', 'best'→'best'/'good')
        converge to the same canonical form the rationale extractor's spaCy
        lemmatizer produces — evidence sets from different strategies must live in
        ONE token space or their overlap is understated (review §8.2). The ``seen``
        guard bounds the loop so a (theoretical) lemmatization cycle cannot hang.
        """
        seen: Set[str] = set()
        current = token
        while current not in seen:
            seen.add(current)
            nxt = current
            for pos in ('v', 'n', 'a', 'r'):
                nxt = self._lemmatizer.lemmatize(nxt, pos)
            if nxt == current:
                break
            current = nxt
        return current

    def normalize(self, token: str) -> Optional[str]:
        """Full normalization pipeline for evidence tokens."""
        if not token:
            return None
        t = token.strip()
        if not t:
            return None

        t = html.unescape(t)
        t = SEP_PATTERN.sub('', t)
        # Re-strip after unescaping: an entity like "&#10" decodes to whitespace
        # ("\n"), which would otherwise survive this pass and be dropped only on a
        # second normalize() — breaking idempotence (found by the round-trip
        # property test).
        t = t.strip()
        t = t.lower()
        t = t.strip(string.punctuation)
        if not t:
            return None

        if self._lemmatizer is not None:
            # No POS context is available for a lone token, so apply verb-then-noun
            # WordNet lemmatization — a deterministic heuristic that collapses the
            # most common inflections (e.g. "running"->"run", "movies"->"movie").
            t = self._lemmatize_to_fixed_point(t)

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

    # POS tags lemmatized for anchoring: noun, verb, adjective, adverb, satellite-adj.
    # Covering all open classes (not just noun+verb) is what lets adjective inflections
    # — "happier"→"happy", "biggest"→"big", "better"→"good" — anchor correctly.
    _ANCHOR_POS = ('n', 'v', 'a', 'r', 's')

    def _anchor_lemmas(self, word: str) -> Set[str]:
        """All candidate WordNet lemmas of a word across open-class POS, for matching.

        Two surface forms are treated as the same evidence iff their lemma sets share
        a member — a morphological criterion, not a hand-built synonym/word list. The
        word itself is always included so matching degrades to surface comparison when
        no lemmatizer is available (independent of `use_lemmatization`).
        """
        w = self.pre_normalize(word)
        if not w:
            return set()
        lemmas = {w}
        lem = self._anchor_lemmatizer
        if lem is not None:
            for pos in self._ANCHOR_POS:
                try:
                    lemmas.add(lem.lemmatize(w, pos))
                except Exception:
                    pass
        return lemmas

    def _input_anchor_lemmas(self, input_text: str) -> Set[str]:
        """Union of anchor-lemmas over every input word (cached per input_text)."""
        cached = self._input_lemma_cache.get(input_text)
        if cached is not None:
            return cached
        norm_input = self.normalize_input_text(input_text)
        lemmas: Set[str] = set()
        for w in re.findall(r"[\w'-]+", norm_input):
            lemmas |= self._anchor_lemmas(w)
        self._input_lemma_cache[input_text] = lemmas
        return lemmas

    def is_anchored(self, token: str, input_text: str) -> bool:
        """Check if token occurs in input_text, robust to inflection.

        Fast path: whole-word (word-boundary) surface match — exact, no lemmatization.
        Fallback: the token anchors if its lemma set intersects the input's lemma set,
        so an inflected explanation token ("scene", "move", "happy") still anchors to a
        differently-inflected input surface form ("scenes", "moved", "happier"). This
        is generic morphology (WordNet), not per-word rules. Genuine synonyms that are
        NOT morphological variants (e.g. "good"/"excellent") are deliberately NOT
        matched — collapsing those would inflate cross-method agreement.
        """
        norm_token = self.pre_normalize(token)
        if not norm_token:
            return False
        norm_input = self.normalize_input_text(input_text)
        if re.search(r'\b' + re.escape(norm_token) + r'\b', norm_input):
            return True
        return bool(self._anchor_lemmas(norm_token) & self._input_anchor_lemmas(input_text))
