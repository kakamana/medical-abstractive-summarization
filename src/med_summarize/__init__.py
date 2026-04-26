"""Medical Abstractive Summarization.

A small Python package that:
- generates a synthetic 10k abstract / summary corpus across 5 medical topics
- builds an extractive summarizer (TextRank + LDA) for the notebook stack
- exposes a /summarize endpoint via FastAPI
- evaluates with a self-contained ROUGE-L implementation
"""
__version__ = "0.1.0"
