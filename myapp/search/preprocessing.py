"""
Utility helpers to keep query preprocessing aligned with the offline corpus pipeline.
The logic mirrors the notebooks used to build `tokens` inside `processed_corpus.json`
so that user queries are normalized (cleaned, stopword-filtered, stemmed) in the same
way as the indexed content.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable, List

from bs4 import BeautifulSoup
from nltk.stem import SnowballStemmer

# Static fallback copied from the NLTK English stopword list to avoid runtime downloads.
FALLBACK_STOPWORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
}

try:
    from nltk.corpus import stopwords

    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = FALLBACK_STOPWORDS

STEMMER = SnowballStemmer("english")


def clean_text(text: str) -> str:
    """Replicates the corpus cleaning rules (HTML strip, normalization, hyphen fixes)."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    text = text.lower()
    text = re.sub(r"\bt\s+shirt\b", "t-shirt", text)
    text = re.sub(r"\bv\s+neck\b", "v-neck", text)
    text = re.sub(r"\bround\s+neck\b", "round-neck", text)
    text = re.sub(r"\bpolo\s+neck\b", "polo-neck", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization that preserves hyphenated terms."""
    if not text:
        return []
    return text.split()


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    return [token for token in tokens if token not in STOP_WORDS]


def stem_tokens(tokens: Iterable[str]) -> List[str]:
    return [STEMMER.stem(token) for token in tokens if len(token) > 2]


def preprocess_query(query: str) -> List[str]:
    """Full pipeline used before issuing an index lookup."""
    cleaned = clean_text(query)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return tokens


