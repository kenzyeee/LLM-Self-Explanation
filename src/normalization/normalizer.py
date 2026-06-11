import re
import string
import logging
from typing import List, Set, Optional

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError:
    pass

logger = logging.getLogger(__name__)


class NormalizationConfig:
    def __init__(self, use_lemmatization=True, remove_stopwords=True, lowercase=True, remove_punctuation=True):
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation


class Normalizer:
    def __init__(self, config: NormalizationConfig = NormalizationConfig()):
        self.config = config
        self.lemmatizer = WordNetLemmatizer() if self.config.use_lemmatization else None
        try:
            self.stop_words = set(stopwords.words('english')) if self.config.remove_stopwords else set()
        except Exception:
            self.stop_words = set()

    def normalize(self, token: str) -> Optional[str]:
        if not token:
            return None
        token = token.strip()
        if not token:
            return None
        if self.config.lowercase:
            token = token.lower()
        if self.config.remove_punctuation:
            token = token.strip(string.punctuation)
        if not token:
            return None
        if self.config.remove_stopwords and token in self.stop_words:
            return None
        if self.config.use_lemmatization and self.lemmatizer:
            token = self.lemmatizer.lemmatize(token)
        return token if token else None

    def normalize_tokens(self, tokens: List[str]) -> Set[str]:
        result = set()
        for t in tokens:
            for word in t.split():
                norm = self.normalize(word)
                if norm:
                    result.add(norm)
        return result

    def extract_content_words_from_rationale(self, rationale: str) -> Set[str]:
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
        orig_tokens = set(original.lower().split())
        cf_tokens = set(counterfactual.lower().split())
        diff = (orig_tokens - cf_tokens) | (cf_tokens - orig_tokens)
        return self.normalize_tokens(list(diff))
