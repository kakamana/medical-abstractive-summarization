"""Feature engineering for the Medical Abstractive Summarization notebook stack.

- Sentence segmentation (regex fallback).
- TF-IDF vectoriser for the sentence-similarity graph.
- Self-contained ROUGE-L implementation (no external dependency).
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter — robust enough for medical abstracts."""
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def build_sentence_vectorizer(min_df: int = 1) -> TfidfVectorizer:
    return TfidfVectorizer(
        min_df=min_df,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9-]+\b",
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    """Pure-Python LCS length — O(len(a) * len(b)) DP."""
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    prev = np.zeros(m + 1, dtype=np.int32)
    curr = np.zeros(m + 1, dtype=np.int32)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return int(prev[m])


def rouge_l(candidate: str, reference: str) -> dict:
    """ROUGE-L F1 (beta=1) at unigram level."""
    cand_tok = _tokenize(candidate)
    ref_tok = _tokenize(reference)
    if not cand_tok or not ref_tok:
        return dict(precision=0.0, recall=0.0, f1=0.0)
    lcs = lcs_length(cand_tok, ref_tok)
    p = lcs / len(cand_tok)
    r = lcs / len(ref_tok)
    f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return dict(precision=p, recall=r, f1=f)


def rouge_n(candidate: str, reference: str, n: int = 1) -> dict:
    """ROUGE-N F1 — overlap of n-grams."""
    def ngrams(tokens: list[str], k: int) -> list[tuple]:
        return [tuple(tokens[i: i + k]) for i in range(len(tokens) - k + 1)]

    cand_ng = ngrams(_tokenize(candidate), n)
    ref_ng = ngrams(_tokenize(reference), n)
    if not cand_ng or not ref_ng:
        return dict(precision=0.0, recall=0.0, f1=0.0)
    cand_counts: dict[tuple, int] = {}
    for g in cand_ng:
        cand_counts[g] = cand_counts.get(g, 0) + 1
    ref_counts: dict[tuple, int] = {}
    for g in ref_ng:
        ref_counts[g] = ref_counts.get(g, 0) + 1
    overlap = sum(min(cand_counts.get(g, 0), c) for g, c in ref_counts.items())
    p = overlap / max(sum(cand_counts.values()), 1)
    r = overlap / max(sum(ref_counts.values()), 1)
    f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return dict(precision=p, recall=r, f1=f)
