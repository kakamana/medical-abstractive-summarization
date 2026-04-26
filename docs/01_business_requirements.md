# Business Requirements — Medical Abstractive Summarization

## 1. Problem Statement
Clinicians, medical-affairs leads, and research analysts read 20–50 abstracts a week to keep up with new evidence. Even at a brisk 2-minute-per-abstract pace, that's 1.5+ hours of reading no one bills for. A reliable 1–2 sentence summary — with a quality score on every result — recovers most of that time and keeps the human in control of when to read the full abstract.

## 2. Stakeholders
| Role | Interest | Success criterion |
|------|----------|-------------------|
| Clinician | Faster scan of new evidence | ≥ 70% of abstracts skimmable from the summary alone |
| Medical-affairs lead | Internal newsletter at scale | One-click summary of 200 abstracts/week |
| Research analyst | Find the relevant paper fast | ROUGE-L ≥ 0.36 on the held-out set |
| Compliance | Auditable summaries | Every output ships its ROUGE-L vs reference |
| Data engineer | Reproduce the pipeline in Dataiku | Notebooks run with sklearn/networkx only |

## 3. Business Objectives
1. Achieve **ROUGE-L ≥ 0.44** with the production BART fine-tune.
2. Achieve **ROUGE-L ≥ 0.36** with the notebook extractive stack (TextRank + LDA).
3. Compression ratio ≥ 6× (notebook), ≥ 9× (production).
4. Single-abstract `/summarize` request returns the summary + ROUGE-L in **< 100 ms** (notebook) and **< 400 ms** on CPU (production).

## 4. KPIs
| KPI | Definition | Target | Baseline |
|-----|-----------|--------|----------|
| ROUGE-L | Longest common subsequence F-score vs reference | ≥ 0.36 (notebook) / ≥ 0.44 (prod) | 0.27 (lead-1) |
| ROUGE-1 | Unigram overlap F-score | ≥ 0.41 / ≥ 0.48 | 0.32 |
| Compression ratio | abstract chars / summary chars | ≥ 6× | 1.0× |
| p95 latency | Single-abstract request | < 100 ms / < 400 ms | – |
| Topic-balanced ROUGE-L | min ROUGE-L across 5 topics | within 5 pts of max | – |

## 5. Scope
**In scope:** abstracts of 100–400 tokens, English only, across 5 medical topics (cardiology, oncology, infectious_disease, neurology, endocrinology); 1–2 sentence output summary; ROUGE-L scored against the reference summary at request time.

**Out of scope:** full-paper summarization (deferred to v2); non-English abstracts; figures and tables.

## 6. Constraints & Assumptions
- **Compute:** notebook stack runs CPU-only inside Dataiku DSS; production stack runs CPU-only with ONNX-quantised BART or LongT5 (acceptable p95 < 400 ms).
- **Latency:** p95 < 100 ms for the notebook stack at single-abstract granularity.
- **Reproducibility:** ROUGE-L is reported on every API response; reference summaries are versioned with the corpus.
- **No PII:** the synthetic corpus contains no real patient data.

## 7. Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Hallucination in production abstractive model | Medium | High | Always show ROUGE-L vs reference + extractive fallback alongside |
| ROUGE-L under-rewards paraphrase | High | Medium | Add ROUGE-1 + BERTScore in evaluation as well |
| Topic imbalance — model best on cardiology, worst on endocrinology | Medium | Medium | Topic-stratified ROUGE; flag any topic > 5 pts below the mean |
| Notebook stack underperforms production | High | Low | Document the gap clearly in `docs/03_methodology.md` |

## 8. Timeline
- **Week 1** — synthetic corpus generator (10k abstracts, 5 topics)
- **Week 2** — extractive baselines (lead-1, TextRank, LDA)
- **Week 3** — production fine-tune of BART-base / LongT5-small
- **Week 4** — API + UI + ROUGE-L scoring on every request
- **Week 5** — content + ship
