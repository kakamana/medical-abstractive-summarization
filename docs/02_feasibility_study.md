# Feasibility Study — Medical Abstractive Summarization

## 1. Data feasibility

### Primary: synthetic corpus (this repo)
- Generator: `src/med_summarize/data.py` builds 10,000 abstracts + 1–2 sentence reference summaries across 5 medical topics.
- Each abstract follows the standard structured format (background → methods → results → conclusion).
- The reference summary is constructed from the conclusion + key result, giving a verifiable target.

### Public augment options (post-launch)
| Source | URL | Use |
|--------|-----|-----|
| PubMed Central Open Access subset | https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/ | Real abstracts + paper bodies |
| arXiv biomedical | https://arxiv.org/ | Pre-prints (with caveat) |
| MIMIC-III discharge summaries (DUA-gated) | https://physionet.org/content/mimiciii/ | Clinical-style summaries |

## 2. Technical feasibility
- **Notebook stack (Dataiku-compatible):**
  - Sentence segmentation — `nltk` + a fallback regex sentence-splitter.
  - TF-IDF sentence embeddings.
  - Sentence-similarity graph → networkx PageRank → top-k sentences.
  - Optional sklearn LDA topic-aware re-rank.
  - ROUGE-L computed via a self-contained LCS implementation (no external rouge package).
- **Production stack (heavy):** BART-base / LongT5-small fine-tune via Hugging Face transformers; ONNX-quantised for CPU serving.

## 3. Economic feasibility
| Line item | Monthly cost |
|-----------|--------------|
| 1× small CPU container | ~$8 |
| Storage (model weights, ~200 MB) | ~$0.10 |
| MLflow self-hosted | $0 |
| **Total** | **~$8 / mo** |

**Value:** at 50 abstracts × 4 minutes saved each × 50 weeks/year × $80/hr (clinician time) = $13k/year per analyst.

## 4. Operational feasibility
- **Re-evaluation:** ROUGE-L on every API response; aggregate weekly into a Grafana panel.
- **Retraining:** quarterly cycle for the production model on the latest PubMed slice.
- **Monitoring:** drift in token-length distribution; per-topic ROUGE drop > 5 pts triggers retrain review.

## 5. Ethical / legal feasibility
- **Hallucination risk:** every API response includes the extractive fallback alongside the abstractive output, so the user can sanity-check.
- **Disclaimer:** the API response includes a non-clinical-decision-support disclaimer.
- **PII:** the synthetic corpus contains no patient identifiers.
- **License:** MIT for this repo; PubMed Central content (when added) follows OA license tags per article.

## 6. Recommendation
**Go.** The notebook stack is honest about its capabilities; the production lift comes from a well-understood transformer fine-tune; the safety surface (hallucination) is mitigated by always showing the extractive fallback and the ROUGE-L score.
