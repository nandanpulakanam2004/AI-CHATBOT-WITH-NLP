import re
from typing import Iterable, List


def ensure_nltk_data() -> None:
    """
    Downloads required NLTK datasets if they are missing.
    This makes the project easier to run on a fresh machine.
    """
    import nltk

    # We use wordpunct_tokenize (no punkt download required).
    required = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for path, name in required:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


_NON_WORD_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """
    Lowercase + remove most punctuation.
    (We keep letters/numbers/spaces to simplify tokenization.)
    """
    text = text.lower().strip()
    text = _NON_WORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Word tokenization using NLTK.
    """
    # wordpunct_tokenize is lightweight and avoids extra punkt dependencies.
    from nltk.tokenize import wordpunct_tokenize

    return wordpunct_tokenize(text)


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    """
    Remove common stopwords like 'the', 'is', 'and', etc.
    """
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t not in stop_words]


def lemmatize(tokens: Iterable[str]) -> List[str]:
    """
    Reduce tokens to their dictionary form (e.g., 'running' -> 'run').
    """
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def stem(tokens: Iterable[str]) -> List[str]:
    """
    Optional: stemming can be more aggressive than lemmatization.
    """
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def preprocess_text(text: str, *, use_stemming: bool = False) -> str:
    """
    Full preprocessing pipeline:
    - normalize (lowercase + punctuation cleanup)
    - tokenize
    - stopword removal
    - lemmatize (or stem)

    Returns a single space-joined string.
    This format works nicely with scikit-learn's TF-IDF vectorizer.
    """
    text = normalize_text(text)
    tokens = tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens) if use_stemming else lemmatize(tokens)
    return " ".join(tokens)

