# Methodology — Medical Abstractive Summarization

Two stacks, one evaluation harness:
1. **Notebook (extractive, Dataiku-compatible):** TextRank over a TF-IDF sentence-similarity graph + sklearn LDA topic-aware re-rank.
2. **Production (abstractive, heavy):** BART-base / Pegasus / LongT5 fine-tuned on `(abstract → summary)` pairs.

Both feed the same ROUGE-L scoring harness.

---

## 1. EDA plan
- Token-length distribution of abstract and summary across the 5 topics.
- Sentence-count distribution per abstract.
- Vocabulary overlap between abstract and reference summary (sets the ceiling for extractive ROUGE).
- Topic balance — confirm none of the 5 topics is starved.

## 2. Sentence segmentation
- Primary: `nltk.sent_tokenize` (Punkt).
- Fallback: regex on `[.!?]\s+(?=[A-Z])`. The fallback is what the notebook actually uses to avoid the NLTK-data download requirement inside Dataiku.

## 3. TextRank

For each abstract, build a graph where:
- Nodes are sentences.
- Edges are weighted by TF-IDF cosine similarity between sentences.

Run PageRank on the weighted graph:

$$ \text{PR}(s) = (1 - d) + d \sum_{u \to s} \frac{w_{u,s}}{\sum_{v: u \to v} w_{u,v}} \cdot \text{PR}(u) $$

with damping factor $d = 0.85$. Top-k sentences (by PR score) form the extractive summary.

We additionally penalise sentences after position 6 with a small position-decay factor `(1 - 0.05 * pos)` — front-loaded sentences in medical abstracts are usually background, mid-body sentences are usually methods/results.

## 4. LDA topic-aware re-rank

Fit a sklearn LDA with `n_components = 5` on the abstract corpus (matches our 5 topics). For each candidate sentence:

$$ \text{topic\_score}(s) = \cos\big(\theta_{abstract},\, \theta_{s}\big) $$

where $\theta$ is the topic distribution. Final score is a weighted combo:

$$ \text{score}(s) = \alpha \cdot \text{PR}(s) + (1 - \alpha) \cdot \text{topic\_score}(s) $$

with $\alpha = 0.7$.

## 5. Production stack — BART fine-tune (out of notebook)
- Backbone: `facebook/bart-base` (139M params).
- Training: `(abstract, summary)` pairs, 3 epochs, lr = 3e-5, label-smoothing = 0.1.
- Decoding: beam = 4, length-penalty = 1.0, max-length = 48 tokens.
- Why kept out of the notebook: CPU-only fine-tune is impractically slow; Dataiku users can swap this in by adding the `heavy` extras.

## 6. ROUGE-L
ROUGE-L is the longest-common-subsequence F-score:

$$ \text{ROUGE-L} = \frac{(1+\beta^2) R_L P_L}{R_L + \beta^2 P_L}, \quad \beta = 1 $$

where $R_L = \text{LCS}(c, r) / |r|$ and $P_L = \text{LCS}(c, r) / |c|$, computed at the unigram level on the candidate $c$ and reference $r$. We implement LCS in pure NumPy (DP table) so the notebook does not depend on `rouge_score`.

## 7. Cross-validation
- 80% train / 10% val / 10% test, stratified by topic.
- All hyperparameters tuned on the val split; reported metrics are on test.

## 8. Evaluation metrics
| Metric | Formula |
|--------|---------|
| ROUGE-L | LCS-based F1 |
| ROUGE-1 | unigram-overlap F1 |
| Compression ratio | abstract chars / summary chars |
| Topic-stratified ROUGE-L | min over 5 topics |

## 9. Production-vs-notebook gap
Extractive systems are bounded by the *vocabulary* of the source — they cannot paraphrase. Synthetic abstracts are written so the conclusion sentence is a near-exact reformulation of the reference summary; this gives the extractive stack a fair shot. On real PubMed abstracts, the gap to abstractive will be wider.

## 10. References
- Mihalcea & Tarau, *TextRank: Bringing Order into Texts*, 2004.
- Lin, *ROUGE: A Package for Automatic Evaluation of Summaries*, 2004.
- Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training*, 2020.
- Blei, Ng, Jordan, *Latent Dirichlet Allocation*, 2003.
