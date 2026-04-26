"""Inference helper for the Medical Abstractive Summarization notebook stack."""
from __future__ import annotations

from functools import lru_cache

from . import models
from .features import rouge_l, rouge_n, split_sentences


@lru_cache(maxsize=1)
def _load():
    return models.load_artefacts()


def summarize(abstract: str, reference: str | None = None, top_k: int = 2) -> dict:
    """Return the extractive summary, the top sentences, and ROUGE scores.

    Args:
        abstract: free-form abstract text.
        reference: optional reference summary for ROUGE scoring.
        top_k: number of sentences to extract.
    """
    try:
        cv, lda = _load()
        summary, top_sents = models.extractive_summary_with_lda(
            abstract, cv, lda, top_k=top_k
        )
    except FileNotFoundError:
        summary, top_sents = models.extractive_summary(abstract, top_k=top_k)

    if not summary:
        sents = split_sentences(abstract)
        summary = " ".join(sents[:top_k])
        top_sents = sents[:top_k]

    out: dict = dict(
        summary=summary,
        top_extractive_sentences=top_sents,
    )
    if reference:
        rl = rouge_l(summary, reference)
        r1 = rouge_n(summary, reference, n=1)
        r2 = rouge_n(summary, reference, n=2)
        out["rouge_l_target"] = round(rl["f1"], 3)
        out["rouge_1"] = round(r1["f1"], 3)
        out["rouge_2"] = round(r2["f1"], 3)
    return out
