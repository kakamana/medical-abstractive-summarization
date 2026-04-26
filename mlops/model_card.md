# Model Card — Medical Abstractive Summarization

## Intended use
Summarize the abstract of a medical research paper into a 1–2 sentence headline. Designed as a research assistant for clinicians, medical-affairs leads, and analysts. **Not** clinical decision support.

## Training data
Synthetic — `src/med_summarize/data.py` generates 10,000 abstracts + reference summaries across 5 medical topics (cardiology, oncology, infectious_disease, neurology, endocrinology). See `data/data_card.md`.

## Model family
- **Notebook stack:** TextRank (PageRank on a TF-IDF sentence-similarity graph) + sklearn LDA topic-aware re-rank. Self-contained ROUGE-L implementation.
- **Production stack:** BART-base / Pegasus / LongT5 fine-tune on `(abstract, summary)` pairs.

## Metrics (held-out test, to be filled)
| Metric | Notebook target | Production target |
|--------|-----------------|-------------------|
| ROUGE-1 | ≥ 0.41 | ≥ 0.48 |
| ROUGE-L | ≥ 0.36 | ≥ 0.44 |
| Compression ratio | ≥ 6× | ≥ 9× |
| p95 latency | < 100 ms | < 400 ms |
| Topic-stratified ROUGE-L gap | ≤ 5 pts | ≤ 5 pts |

## Limitations
- Synthetic abstracts share a structured-abstract shape; real PubMed abstracts vary more.
- Extractive systems cannot paraphrase; if the reference summary uses a synonym, ROUGE-L will be conservative.
- The production abstractive model can hallucinate — every API response includes the extractive fallback as a sanity check.

## Ethical considerations
- API response carries an explicit "research-assistant only, not clinical decision support" disclaimer.
- ROUGE-L reported on every response so the user can decide whether to trust the summary.

## Retraining
- Quarterly cycle for the production model on the latest PubMed Central OA slice.
- LDA topic model can be re-fitted whenever the topic mix shifts.

## Ownership
- On-call DS: Asad
- Runbook: `mlops/runbook.md` (TBD)
