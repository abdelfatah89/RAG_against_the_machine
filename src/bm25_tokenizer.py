import re
import unicodedata
from typing import List, Optional, Set

try:
    from nltk.stem import SnowballStemmer
    _STEMMER: Optional[SnowballStemmer] = SnowballStemmer("english")
except ImportError:
    _STEMMER = None

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
# Splits "camelCase" / "HTTPServer" style identifiers into word boundaries.
CAMEL_CASE_PATTERN = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

DEFAULT_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for",
    "of", "to", "in", "on", "at", "by", "with", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "as", "from", "into", "not", "no", "do", "does",
    "did", "has", "have", "had", "will", "would", "can", "could",
    "should", "shall", "may", "might", "must",
}


def _split_identifier(token: str) -> List[str]:
    """Split snake_case / camelCase identifiers into sub-words
    (useful for code)."""
    parts = token.split("_")
    result: List[str] = []
    for part in parts:
        if part:
            result.extend(CAMEL_CASE_PATTERN.sub(" ", part).split())
    return result


def tokenize(
    text: str,
    *,
    remove_stopwords: bool = True,
    stem: bool = True,
    split_identifiers: bool = True,
    min_token_length: int = 2,
    stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """
    Tokenize text for BM25.

    Steps:
    1. Normalize unicode (NFKC) so visually-equivalent chars compare equal.
    2. Lowercase.
    3. Extract word tokens via regex.
    4. Optionally split snake_case/camelCase identifiers into sub-words.
    5. Drop stopwords, too-short tokens, and low-signal short numbers.
    6. Optionally stem tokens to collapse morphological variants.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    raw_tokens = TOKEN_PATTERN.findall(text)

    stop = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    tokens: List[str] = []

    for tok in raw_tokens:
        pieces = _split_identifier(tok) if split_identifiers else [tok]
        for piece in pieces:
            piece = piece.lower()
            if len(piece) < min_token_length:
                continue
            if remove_stopwords and piece in stop:
                continue
            if piece.isdigit() and len(piece) < 3:
                continue
            if stem and _STEMMER is not None:
                piece = _STEMMER.stem(piece)
            tokens.append(piece)

    return tokens
