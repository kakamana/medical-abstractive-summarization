# Data Card — Medical Abstractive Summarization

## Dataset composition

| Layer | Source | Rows × cols | Purpose |
|-------|--------|-------------|---------|
| Abstracts + summaries | Synthetic (`src/med_summarize/data.py`) | 10,000 × 5 | Train + eval |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `paper_id` | string | `P-00001` to `P-10000` |
| `topic` | string | One of cardiology, oncology, infectious_disease, neurology, endocrinology |
| `abstract` | string | 100–400-token structured abstract (background, methods, results, conclusion) |
| `summary` | string | 1–2 sentence reference summary |
| `tokens_count` | int | whitespace token count of the abstract |

## Generation rules
- Each topic has its own template bank for background phrasing, intervention vocabulary, sample-size language, and outcome verbs.
- The reference summary is composed from the conclusion sentence + the headline result figure — making the extractive ceiling reachable but not trivial.

## Known biases
- All abstracts are English-language and follow the structured-abstract pattern.
- Underlying biology vocabulary is curated, not exhaustive — a real PubMed loader is documented as the production augment.

## PII
None. No real patient or author data.

## Splits
- 80% train / 10% val / 10% test, topic-stratified.

## Reproducing
```bash
python -m med_summarize.data   # writes data/processed/papers.parquet
```
Deterministic seed = 42.

## Licensing
- Synthetic dataset — MIT (this repo).
