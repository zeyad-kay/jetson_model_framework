"""Small helpers shared by the services: transcript segments, fuzzy text matching, JSON shapes."""

import json
import re
from pathlib import Path
from typing import Any, Optional, Union


def read_segments(path: Union[str, Path]) -> list[dict]:
    """The segments in an ASR segments.json, ordered by segment_id; [] if it can't be read right now.

    Each is {"segment_id", "start", "end", "text": [{"word", "start", "end"}]}.
    """
    try:
        with open(path, encoding="utf-8") as f:
            segments = json.load(f)
    except (FileNotFoundError, ValueError):
        return []
    if not isinstance(segments, list):
        return []
    return sorted((s for s in segments if isinstance(s, dict)), key=lambda s: s.get("segment_id", 0))


def normalize(text: Optional[str]) -> str:
    """Lowercase alphanumerics separated by single spaces, for fuzzy text matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def word_tokens(words: list[dict]) -> list[tuple[str, int]]:
    """The normalized tokens of timed words ({"word", "start", "end"}), each with the index of the
    word it came from ("BP's" gives two tokens, "bp" and "s"); the input of quote_time()."""
    tokens = []
    for i, w in enumerate(words):
        tokens.extend((t, i) for t in normalize(w.get("word")).split())
    return tokens


def quote_time(quote: str, tokens: list[tuple[str, int]], words: list[dict],
               threshold: float = 0.5) -> Optional[float]:
    """Start time of the run of words that best matches the quote, or None if nothing is close.

    tokens is word_tokens(words). Slides a quote-sized window over the tokens and scores
    each by word overlap (Jaccard); the best window must score at least threshold.
    """
    q = normalize(quote).split()
    if not q or not tokens:
        return None
    qset = set(q)
    best, best_score = None, 0.0
    for i in range(max(1, len(tokens) - len(q) + 1)):
        window = {t for t, _ in tokens[i:i + len(q)]}
        score = len(window & qset) / len(window | qset)
        if score > best_score:
            best, best_score = i, score
    if best is None or best_score < threshold:
        return None
    return words[tokens[best][1]].get("start")


def as_list(value: Any) -> list:
    """A value stored either as one object or as a list, as a list: {} and None give [],
    another dict gives [it], a list is returned as is, anything else gives []."""
    if isinstance(value, dict):
        return [value] if value else []
    return value if isinstance(value, list) else []
