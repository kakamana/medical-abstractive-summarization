# Abstractive Summarization — Medical Papers

> **Compress a 250-word medical abstract into a 1–2 sentence takeaway, with a ROUGE-L score on every result.** Production stack: BART / Pegasus / LongT5 fine-tunes. Notebook stack: extractive (TextRank + sklearn LDA) — Dataiku-friendly and CPU-only.

![Python](https://img.shields.io/badge/python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688) ![Next.js](https://img.shields.io/badge/Next.js-14-black) ![License](https://img.shields.io/badge/license-MIT-green)

## Why this project
- A clinician reading 30 papers a week wants the *gist*, not the whole abstract. A reliable 1–2 sentence summary saves 20+ minutes per scan.
- Production summarization uses a fine-tuned encoder-decoder (BART or LongT5). For a Dataiku-friendly notebook stack we use **extractive** summarization (TextRank over a sentence-similarity graph + sklearn LDA topic-aware ranking) — same evaluation harness, same ROUGE-L scores.
- 10,000 synthetic abstracts + reference summaries are generated from controlled topic templates (cardiology, oncology, infectious_disease, neurology, endocrinology), so ROUGE-L is reproducible by construction.

## Table of contents
- [Business Requirements](./docs/01_business_requirements.md)
- [Feasibility Study](./docs/02_feasibility_study.md)
- [Methodology — TextRank + LDA + ROUGE](./docs/03_methodology.md)
- [Evaluation Plan](./docs/04_evaluation.md)
- [Data card](./data/data_card.md) - [Data sources](./data/data_sources.md)
- [Notebooks](./notebooks/) - [Source](./src/med_summarize/) - [API](./api/main.py) - [UI](./ui/app/page.tsx)
- [CLAUDE.md](./CLAUDE.md) — paste prompt to resume in this folder

## Headline results (target)

| Metric | Lead-1 baseline | Notebook (TextRank + LDA, extractive) | Production (BART fine-tune, abstractive) |
|---|---|---|---|
| ROUGE-1 | 0.32 | 0.41 | **0.48** |
| ROUGE-2 | 0.10 | 0.18 | **0.25** |
| ROUGE-L | 0.27 | 0.36 | **0.44** |
| Compression ratio | 1.0× | **6.5×** | **9.0×** |
| p95 latency | < 5 ms | 60 ms | 350 ms (CPU) |

## Quickstart

```bash
pip install -e ".[dev]"
python -m med_summarize.data                  # generate 10k synthetic abstracts + summaries
jupyter lab notebooks/                        # 01_eda - 02_features - 03_model - 04_eval
uvicorn api.main:app --reload                 # serves POST /summarize
cd ui && npm install && npm run dev           # paste abstract - get extractive summary + ROUGE-L
```

## Stack
Python · pandas · scikit-learn · networkx · nltk · joblib · FastAPI · Next.js · Tailwind  
**Production add-on:** transformers (BART / Pegasus / LongT5) — kept out of the notebook stack for Dataiku compatibility.

## Author
Asad — MADS @ University of Michigan · Dubai HR
