# Evaluation Plan — Medical Abstractive Summarization

## 1. Held-out test set
10% topic-stratified holdout (5 topics × ~200 abstracts).

## 2. Primary scorecard
| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | p95 latency |
|-------|---------|---------|---------|-------------|-------------|
| Lead-1 baseline | – | – | – | – | – |
| TextRank | – | – | – | – | – |
| TextRank + LDA topic re-rank | – | – | – | – | – |
| BART fine-tune (production) | – | – | – | – | – |

## 3. Topic-stratified ROUGE-L
For each of the 5 topics, compute the mean ROUGE-L on the test slice. Flag any topic > 5 pts below the cross-topic mean.

## 4. Compression ratio
Mean and 25/50/75 percentiles. Sanity-check that the ratio is ≥ 6× without ROUGE collapse.

## 5. Length sensitivity
Bucket abstracts into short (< 150 tokens), medium (150–250), long (250+). Report ROUGE-L in each bucket; long abstracts often hurt extractive systems most because the front-loading heuristic breaks down.

## 6. Hallucination check (production stack only)
For the production abstractive output, count the fraction of unigrams in the summary that do *not* appear in the abstract. Target < 8% for safe biomedical use; report alongside ROUGE-L.

## 7. Robustness
- Drop the conclusion sentence from each abstract → measure ROUGE-L decay (should hurt most for extractive).
- Reorder sentences randomly → measure ROUGE-L decay (should hurt the position-weighted extractive least).

## 8. Latency
Single-abstract request, on CPU. Vary abstract length; report p95.

## 9. Deployment readiness checklist
- [ ] Notebook ROUGE-L ≥ 0.36 on test
- [ ] Production ROUGE-L ≥ 0.44 on test
- [ ] No topic > 5 pts below cross-topic mean
- [ ] Hallucination rate < 8% (production)
- [ ] p95 latency under target
- [ ] Model card published
