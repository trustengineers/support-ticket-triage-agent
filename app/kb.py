"""BM25 search over the FAQ markdown files in kb/. No external search dependency."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOP_WORDS = frozenset(
    "a an and are as at be but by can do for from has have how i if in is it my not of on or so that the this to was we what when with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without stop words."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOP_WORDS and len(token) > 1]


@dataclass(frozen=True)
class Article:
    """One FAQ file."""

    file: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchHit:
    """One search result with its BM25 score."""

    file: str
    title: str
    score: float
    text: str


class KnowledgeBase:
    """Okapi BM25 index over a small set of articles."""

    def __init__(self, articles: list[Article], k1: float = 1.5, b: float = 0.75) -> None:
        self.articles = articles
        self._k1 = k1
        self._b = b
        # The title is indexed twice so that a title match outweighs a body match.
        self._term_counts = [Counter(tokenize(f"{a.title} {a.title} {a.text}")) for a in articles]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        self._average_length = (sum(self._lengths) / len(self._lengths)) if articles else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            document_frequency.update(counts.keys())
        total = len(articles)
        self._idf = {term: math.log(1 + (total - df + 0.5) / (df + 0.5)) for term, df in document_frequency.items()}

    @classmethod
    def from_dir(cls, directory: Path) -> "KnowledgeBase":
        """Load every .md file in a folder. The first heading is the title."""
        articles = []
        for path in sorted(directory.glob("*.md")):
            text = path.read_text(encoding="utf-8").strip()
            first_line, _, rest = text.partition("\n")
            title = first_line.lstrip("# ").strip() or path.stem
            articles.append(Article(file=path.name, title=title, text=rest.strip()))
        return cls(articles)

    def _score(self, query_terms: list[str], index: int) -> float:
        counts = self._term_counts[index]
        norm = self._k1 * (1 - self._b + self._b * self._lengths[index] / self._average_length)
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if frequency:
                score += self._idf[term] * frequency * (self._k1 + 1) / (frequency + norm)
        return score

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        """Return the best-matching articles, best first. No match returns an empty list."""
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms or not self.articles:
            return []
        scored = [(self._score(query_terms, i), i) for i in range(len(self.articles))]
        best = sorted((item for item in scored if item[0] > 0), key=lambda item: (-item[0], item[1]))[:top_k]
        return [
            SearchHit(file=self.articles[i].file, title=self.articles[i].title, score=round(score, 3), text=self.articles[i].text)
            for score, i in best
        ]
