"""Train + persist artefacts for the Medical Abstractive Summarization notebook stack.

We persist:
- lda_model.joblib       — sklearn LDA on the abstract corpus
- lda_vectorizer.joblib  — CountVectorizer used to feed the LDA

TextRank itself is computed per-abstract at request time (no model artefact needed).

Run as `python -m med_summarize.models` to fit + save.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .features import build_sentence_vectorizer, split_sentences

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ----- TextRank --------------------------------------------------------------


def textrank_scores(
    sentences: list[str],
    damping: float = 0.85,
    position_decay: float = 0.05,
) -> np.ndarray:
    """Run TF-IDF-similarity-weighted PageRank on the sentence graph."""
    if len(sentences) <= 1:
        return np.ones(len(sentences))
    vec = build_sentence_vectorizer()
    X = vec.fit_transform(sentences)
    sim = (X @ X.T).toarray()
    np.fill_diagonal(sim, 0.0)
    sim = np.clip(sim, 0.0, None)

    g = nx.from_numpy_array(sim)
    pr = nx.pagerank(g, alpha=damping, max_iter=200, tol=1e-6, weight="weight")
    scores = np.array([pr.get(i, 0.0) for i in range(len(sentences))])
    weights = np.array([max(0.1, 1 - position_decay * i) for i in range(len(sentences))])
    return scores * weights


def extractive_summary(
    abstract: str, top_k: int = 2, damping: float = 0.85
) -> tuple[str, list[str]]:
    """Return the top-k sentences in their original order (joined)."""
    sents = split_sentences(abstract)
    if not sents:
        return "", []
    scores = textrank_scores(sents, damping=damping)
    if top_k >= len(sents):
        return " ".join(sents), sents
    top_idx = np.argsort(-scores)[:top_k]
    top_idx = sorted(top_idx)  # preserve narrative order
    chosen = [sents[i] for i in top_idx]
    return " ".join(chosen), chosen


# ----- LDA topic model -------------------------------------------------------


def fit_lda(corpus: list[str], n_topics: int = 5, seed: int = 0):
    cv = CountVectorizer(
        min_df=5, max_df=0.85, token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b"
    )
    X = cv.fit_transform(corpus)
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=seed, learning_method="batch", max_iter=20
    )
    lda.fit(X)
    return cv, lda


def lda_topic_score(
    abstract: str,
    sentences: list[str],
    cv: CountVectorizer,
    lda: LatentDirichletAllocation,
) -> np.ndarray:
    """Per-sentence cosine similarity between sentence-topic and abstract-topic."""
    abs_topic = lda.transform(cv.transform([abstract]))[0]
    if not sentences:
        return np.zeros(0)
    sent_topic = lda.transform(cv.transform(sentences))

    def cos(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    return np.array([cos(abs_topic, t) for t in sent_topic])


def extractive_summary_with_lda(
    abstract: str,
    cv: CountVectorizer,
    lda: LatentDirichletAllocation,
    top_k: int = 2,
    alpha: float = 0.7,
) -> tuple[str, list[str]]:
    sents = split_sentences(abstract)
    if not sents:
        return "", []
    pr = textrank_scores(sents)
    pr_n = pr / (pr.max() + 1e-9)
    topic = lda_topic_score(abstract, sents, cv, lda)
    topic_n = topic / (topic.max() + 1e-9)
    score = alpha * pr_n + (1 - alpha) * topic_n
    if top_k >= len(sents):
        return " ".join(sents), sents
    top_idx = sorted(np.argsort(-score)[:top_k])
    chosen = [sents[i] for i in top_idx]
    return " ".join(chosen), chosen


# ----- pipeline --------------------------------------------------------------


def fit_and_save() -> dict[str, Path]:
    df = pd.read_parquet(DATA_DIR / "papers.parquet")
    cv, lda = fit_lda(df["abstract"].tolist())
    cv_path = MODEL_DIR / "lda_vectorizer.joblib"
    lda_path = MODEL_DIR / "lda_model.joblib"
    joblib.dump(cv, cv_path)
    joblib.dump(lda, lda_path)
    return {"vectorizer": cv_path, "lda": lda_path}


def load_artefacts():
    cv = joblib.load(MODEL_DIR / "lda_vectorizer.joblib")
    lda = joblib.load(MODEL_DIR / "lda_model.joblib")
    return cv, lda


if __name__ == "__main__":
    paths = fit_and_save()
    print(f"saved: {paths}")
