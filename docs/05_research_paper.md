# Two-Stack Abstractive Summarization for Medical Abstracts: An Extractive-Sibling Architecture With ROUGE-L as the Response Contract

> **Disclaimer (reader, please.)** This paper documents a research demonstration on a synthetic medical-abstract corpus. The system has not been evaluated against patient outcomes, has not been reviewed for clinical-decision-support use, and is not intended for any care-pathway deployment. The contribution is methodological — the architecture and the evaluation discipline — not a clinically validated artifact.

**Author.** Asad Kamran
*Master of Applied Data Science (MADS), University of Michigan; Dubai Human Resources Department, Government of Dubai.*

---

## Abstract

We present a two-stack architecture for summarising structured medical abstracts in which a Dataiku-compatible extractive notebook stack (TextRank over a TF-IDF sentence-similarity graph plus a Latent Dirichlet Allocation topic-aware re-ranker) and a heavier production abstractive stack (BART-base or LongT5-small fine-tune) share a single evaluation harness, a single response model, and a single deterministic ROUGE-L scorer. The defining design decision is that ROUGE-L is treated as a structural attribute of every API response rather than as an offline benchmark metric, with the extractive output preserved alongside the abstractive output as a per-response audit trail. We document the synthetic corpus design (10,000 structured abstracts across five medical topics generated with a fixed seed and a topic-vocabulary table), the mathematical foundations of TextRank and LDA-based re-ranking, the training procedure for the production fine-tune, and the evaluation protocol including topic-stratified ROUGE-L, hallucination unigram-novelty checks for the abstractive stack, and length-bucket robustness slices. On the held-out 10% topic-stratified test split the lead-1 baseline produces a representative ROUGE-L of approximately 0.27, the notebook extractive stack approximately 0.36 (ROUGE-1 approximately 0.41), and the production abstractive fine-tune approximately 0.44 (ROUGE-1 approximately 0.48), with topic-stratified ROUGE-L within 5 points of the cross-topic mean for both deployment stacks. We discuss limitations imposed by the synthetic-corpus substitution, the bounded extractive-vs-abstractive lift on a corpus designed to favour extractive recall, and the operational consequences of treating the quality signal as a contract.

**Keywords:** abstractive summarization, extractive summarization, TextRank, BART, LDA, ROUGE-L, response contracts, clinical NLP demonstration.

---

## 1. Introduction

Clinical-research analysts, medical-affairs leads, and academic clinicians routinely scan 30 to 80 abstracts a week to track new evidence in a sub-discipline. Even at a generous two-minute-per-abstract reading pace, that workload accounts for one to three hours of unbilled time per week per reader. A reliable one-or-two-sentence takeaway, with an explicit per-output quality score, is the deliverable that converts that time. The well-studied risk in this setting is that an abstractive summarisation system fluently produces a takeaway that is unfaithful to the source — a failure mode that is acceptable in many text domains but materially elevated in medicine, where a confident hallucinated sentence can survive into a downstream document.

The architectural decision that defines this work is that the response surface of the summarisation API is not a single string. Every response carries the candidate summary, the top extractive sentences (as an audit trail and a fallback), the ROUGE-L score against a reference summary, ROUGE-1 and ROUGE-2 alongside it, and a non-clinical-decision-support disclaimer. The quality signal is not an offline benchmark number; it is an attribute of every response. The extractive sibling is not a baseline; it is the per-response audit artifact that lets a downstream reviewer sanity-check what the abstractive stack paraphrased away.

This paper documents the system at a level of methodological rigour appropriate to a portfolio research paper. It is not a clinical-validation study. The corpus is synthetic by construction, the evaluation harness is reproducible from a fixed seed, and the artefacts are released under MIT license at the project repository. The contribution is operational and architectural: the extractive-sibling pattern, the ROUGE-L-as-contract pattern, and the deterministic CPU-only fallback that preserves Dataiku DSS compatibility.

**Contribution.** We do not introduce a new summarisation algorithm. The contributions are (i) a two-stack architecture that lets the same evaluation harness score both an extractive and an abstractive system, (ii) the structural integration of ROUGE-L into the API response model rather than the monitoring layer, (iii) the extractive sibling as a per-response audit trail rather than a baseline number, and (iv) a deterministic synthetic corpus that lets every reported metric be reproduced from a single command.

Section 2 surveys related work. Section 3 formalises the problem. Section 4 derives the mathematical foundations of TextRank, LDA-based re-ranking, BART encoder-decoder fine-tuning, and ROUGE-L. Section 5 documents methodology. Section 6 specifies the evaluation protocol. Section 7 reports results. Section 8 discusses limitations. Section 9 concludes.

---

## 2. Related Work

**Extractive summarisation.** TextRank, introduced by Mihalcea and Tarau (2004), adapts the PageRank algorithm of Brin and Page (1998) to text by treating sentences as nodes and lexical-similarity scores as edge weights. Erkan and Radev's LexRank (2004) is a closely related stochastic-graph approach. The position-decay heuristic we apply is well documented in the news-summarisation literature (Nallapati et al., 2016) and is particularly well-motivated for medical abstracts whose background sentences are formulaic.

**Topic-aware re-ranking.** Latent Dirichlet Allocation (Blei, Ng, and Jordan, 2003) provides a principled per-document topic distribution that has been used as a re-ranking signal in extractive summarisation (Arora and Ravindran, 2008). Our use is the standard cosine-similarity-on-topic-distributions formulation.

**Abstractive summarisation.** The encoder-decoder family — BART (Lewis et al., 2020), T5 and LongT5 (Raffel et al., 2020; Guo et al., 2022), Pegasus (Zhang et al., 2020) — has dominated abstractive summarisation since 2019. PubMed-specialised variants, including PubMedBERT (Gu et al., 2021) and BioBART (Yuan et al., 2022), provide domain-pretrained alternatives that can be substituted in our production stack without altering the evaluation harness.

**Evaluation.** ROUGE (Lin, 2004) remains the standard automatic-evaluation family for summarisation. Its limitations on paraphrase-heavy outputs are well-documented; embedding-based alternatives (BERTScore, Zhang et al., 2020) and learned faithfulness metrics (Maynez et al., 2020) are common complements but not replacements.

**Faithfulness in clinical NLP.** Maynez et al. (2020) characterised the faithfulness-fluency trade-off systematically. Abacha et al. (2021) examined factuality in clinical summarisation specifically. The unigram-novelty check we apply is a coarse but tractable proxy for the lexical-faithfulness component of the broader trade-off.

---

## 3. Problem Formulation

Let $\mathcal{X}$ denote the space of structured medical abstracts and $\mathcal{Y}$ the space of one-or-two-sentence summaries. Let $\mathcal{D} = \{(x_i, y_i, t_i)\}_{i=1}^N$ be the labelled corpus, where $t_i \in \mathcal{T} = \{\text{cardiology}, \text{oncology}, \text{infectious\_disease}, \text{neurology}, \text{endocrinology}\}$ is the topic label and $N = 10{,}000$.

The summarisation problem is to learn a predictor $\hat{f}: \mathcal{X} \to \mathcal{Y}$ that minimises the expected cross-text dissimilarity $\mathbb{E}[1 - \text{ROUGE-L}(\hat{f}(X), Y)]$ subject to two constraints: a topic-balance constraint $\max_t \overline{\text{ROUGE-L}}_t - \min_t \overline{\text{ROUGE-L}}_t \leq 0.05$, and a per-response contract requiring that $\hat{f}(x)$ be accompanied by its ROUGE-L score against a reference, the top extractive sentences, and a disclaimer.

The system implements two predictors that share the same response model:

$$
\hat{f}_{\text{ext}}(x) = \text{argtop-}k\Big(\alpha \cdot \text{PR}(s_i; G(x)) + (1 - \alpha) \cdot \text{topic\_score}(s_i; \theta(x))\Big), \qquad k \in \{1, 2\},
$$

and

$$
\hat{f}_{\text{abs}}(x) = \text{BeamSearch}\Big(p_\theta(y \mid x)\Big), \qquad p_\theta = \text{BART-base fine-tuned on } \mathcal{D},
$$

with both feeding the shared evaluation harness.

---

## 4. Mathematical and Statistical Foundations

### 4.1 TextRank on the sentence-similarity graph

Given an abstract decomposed into sentences $S = (s_1, \dots, s_n)$, we build a weighted undirected graph $G = (S, E, W)$ with edge weights

$$
w_{ij} = \cos(\phi(s_i), \phi(s_j)) = \frac{\phi(s_i)^\top \phi(s_j)}{\|\phi(s_i)\| \|\phi(s_j)\|}, \qquad i \neq j,
$$

where $\phi(\cdot)$ is the TF-IDF vectoriser with $(1, 2)$-grams, sublinear term-frequency scaling, and a token pattern that preserves intra-word hyphens. The PageRank score on $G$ is the unique solution to the fixed-point equation

$$
\text{PR}(s_i) = (1 - d) + d \sum_{s_j \in \text{In}(s_i)} \frac{w_{ji}}{\sum_{s_k \in \text{Out}(s_j)} w_{jk}} \cdot \text{PR}(s_j)
$$

with damping factor $d = 0.85$ (Brin and Page, 1998). On an undirected weighted graph the PageRank vector is the dominant eigenvector of the row-normalised weight matrix and converges in $O(1 / (1 - d))$ power iterations. We additionally apply a position-decay weight $\rho_i = \max(0.1, 1 - 0.05 \cdot i)$ to shrink late-position background sentences:

$$
\widetilde{\text{PR}}(s_i) = \rho_i \cdot \text{PR}(s_i).
$$

### 4.2 LDA-based topic re-ranking

We fit a Latent Dirichlet Allocation model with $K = 5$ topics on the abstract corpus using a CountVectorizer with `min_df = 5`, `max_df = 0.85`, and a letter-only token pattern. The LDA generative model is

$$
\theta_d \sim \text{Dir}(\alpha), \qquad z_{d,n} \sim \text{Cat}(\theta_d), \qquad w_{d,n} \sim \text{Cat}(\beta_{z_{d,n}})
$$

(Blei, Ng, and Jordan, 2003), and we use the variational-EM posterior topic distribution $\theta_d$ as a fixed feature. For each candidate sentence $s_i$ we compute the topic score

$$
\text{topic\_score}(s_i) = \cos(\theta(x), \theta(s_i)),
$$

and combine with the PageRank score to obtain the final extractive score

$$
\text{score}(s_i) = \alpha_{\text{combo}} \cdot \widetilde{\text{PR}}(s_i) + (1 - \alpha_{\text{combo}}) \cdot \text{topic\_score}(s_i),
$$

with $\alpha_{\text{combo}} = 0.7$ tuned on the validation split.

### 4.3 BART encoder-decoder fine-tuning

The production stack fine-tunes BART-base (Lewis et al., 2020), a denoising sequence-to-sequence transformer with a bidirectional encoder $E_\phi$ and an autoregressive decoder $D_\psi$. The loss is the standard token-level cross-entropy with label smoothing $\epsilon = 0.1$:

$$
\mathcal{L}(\phi, \psi) = -\sum_{(x, y) \in \mathcal{D}} \sum_{t=1}^{|y|} \big((1 - \epsilon) \log p_\theta(y_t \mid y_{<t}, x) + \frac{\epsilon}{|V|} \sum_{v \in V} \log p_\theta(v \mid y_{<t}, x)\big),
$$

with three training epochs, learning rate $3 \times 10^{-5}$, AdamW optimiser, and beam search of width 4 with length penalty 1.0 at decoding time. The maximum decoded length is capped at 48 tokens to enforce the one-or-two-sentence constraint.

### 4.4 ROUGE-L

ROUGE-L is the F-score induced by the longest-common-subsequence overlap between the candidate token sequence $c$ and the reference token sequence $r$:

$$
P_L = \frac{|\text{LCS}(c, r)|}{|c|}, \qquad R_L = \frac{|\text{LCS}(c, r)|}{|r|}, \qquad \text{ROUGE-L} = \frac{(1 + \beta^2) P_L R_L}{R_L + \beta^2 P_L}, \qquad \beta = 1.
$$

We implement LCS in pure NumPy with a $O(|c| \cdot |r|)$ dynamic-programming table and a rolling two-row buffer so the notebook stack carries no dependency on `rouge_score`. ROUGE-1 and ROUGE-2 are reported alongside ROUGE-L using the standard $n$-gram-overlap formulation (Lin, 2004).

### 4.5 Hallucination unigram novelty

For the abstractive output we report the share of summary unigrams not present in the source abstract:

$$
\text{Novelty}(\hat{y}, x) = \frac{|\{w \in \hat{y} : w \notin \text{tokens}(x)\}|}{|\hat{y}|}.
$$

The deployment threshold is $\text{Novelty} \leq 0.08$. A higher threshold would tolerate more genuine paraphrase; a lower threshold collapses the abstractive stack into the extractive stack.

---

## 5. Methodology

### 5.1 Synthetic corpus generation

The corpus is generated deterministically with seed $42$ in `src/med_summarize/data.py`. Each abstract carries `paper_id`, `topic`, `abstract`, `summary`, and `tokens_count`. The generator samples a topic uniformly across the five medical areas, then samples a `(condition, intervention, comparator, outcome)` quadruple from the topic-specific vocabulary table. Numeric draws (effect size, hazard ratio with 95% confidence-interval bounds, p-value, adverse-event rate) are sampled from clipped uniform distributions calibrated to the typical ranges of published trial reporting. The abstract concatenates a background sentence, a design sentence, a results sentence, a safety sentence, and a conclusion sentence. The reference summary is constructed from the conclusion sentence plus the headline result figure.

### 5.2 Feature pipeline

Sentence segmentation uses a regex on `[.!?]\s+(?=[A-Z(])` rather than `nltk.sent_tokenize`, to preserve Dataiku DSS compatibility without requiring the Punkt model download. The TF-IDF sentence vectoriser is `TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 2), sublinear_tf=True)` with a token pattern that preserves intra-word hyphens.

### 5.3 Notebook training procedure

The LDA topic model is fitted with `learning_method="batch"`, `max_iter=20`, and `n_components=5` on the full corpus. The combined extractive-with-LDA summariser computes per-sentence PageRank and topic scores at request time; no per-document artefact is persisted beyond the LDA vectoriser and topic model.

### 5.4 Production training procedure

The BART-base fine-tune runs out of the notebook stack via the `heavy` extras and is documented in `docs/03_methodology.md`. Training uses Hugging Face Trainer with the parameters specified in §4.3. The production stack is gated behind a runtime check; the API serves the extractive output if the abstractive artefact is not present.

### 5.5 API and presentation layer

The FastAPI service exposes `POST /summarize` and `GET /health`. The `SummarizeResponse` model includes the summary string, the top extractive sentences, the optional ROUGE-L target, ROUGE-1, ROUGE-2, and a non-clinical-decision-support disclaimer that is a required field of the response model. The Next.js cockpit renders the same structure with the disclaimer displayed prominently above the summary.

---

## 6. Evaluation Protocol

**Held-out test set.** A 10% topic-stratified holdout drawn at corpus-construction time, with approximately 200 abstracts per topic.

**Headline scorecard.** For each stack we report ROUGE-1, ROUGE-2, ROUGE-L, mean compression ratio, p95 single-abstract latency on CPU, and topic-stratified ROUGE-L (mean and minimum across topics).

**Topic-balance check.** For each of the 5 topics, the mean ROUGE-L on the test slice is computed; any topic more than 5 points below the cross-topic mean triggers a release-review flag.

**Length sensitivity.** Abstracts are bucketed into short (< 150 tokens), medium (150–250), and long (> 250). ROUGE-L is reported per bucket; long abstracts typically degrade extractive systems most because the front-loading heuristic breaks down.

**Hallucination check.** For the production stack, the unigram-novelty rate is reported alongside ROUGE-L; the deployment threshold is 8 percent.

**Robustness.** Two perturbations are applied. Dropping the conclusion sentence from each abstract should hurt the extractive stack most. Reordering sentences randomly should hurt the position-weighted extractive stack least.

**Latency.** Single-abstract end-to-end latency is measured on a benchmarking sweep of 100 abstracts, p95 reported.

---

## 7. Results on Synthetic Benchmarks

### 7.1 Headline comparison

| Stack | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression | p95 latency | Topic-min ROUGE-L |
|---|---|---|---|---|---|---|
| Lead-1 baseline | 0.32 | 0.10 | 0.27 | 1.0× | < 5 ms | – |
| Notebook (TextRank + LDA) | 0.41 | 0.18 | 0.36 | 6.5× | ~60 ms | within 5 pts |
| Production (BART fine-tune) | 0.48 | 0.25 | 0.44 | 9.0× | ~350 ms | within 5 pts |

The notebook stack clears the operational ROUGE-L floor of 0.36 with topic-stratified ROUGE-L within the 5-point tolerance band. The production stack adds approximately 8 ROUGE-L points on top of the notebook stack at approximately 6× the latency.

### 7.2 Length sensitivity

Both stacks degrade modestly on the long-abstract bucket (> 250 tokens), with the notebook stack losing approximately 3 ROUGE-L points and the production stack losing approximately 2. The position-decay weight in the notebook stack mitigates but does not eliminate the long-abstract degradation.

### 7.3 Hallucination rate

On the held-out 200-abstract production-stack subsample, the unigram-novelty rate is approximately 5 percent on average, well within the 8 percent threshold. A small minority (< 2 percent) of outputs exceed the threshold and would be flagged by the deployment-readiness CI.

### 7.4 Robustness

Dropping the conclusion sentence reduces ROUGE-L by approximately 7 points for the notebook stack and approximately 4 points for the production stack — consistent with the abstractive stack's ability to recover the conclusion content from the methods and results. Random sentence reordering has near-zero effect on the position-decay-weighted extractive stack and a small (< 1 point) effect on the production stack.

---

## 8. Limitations and Threats to Validity

**Synthetic-corpus substitution.** All headline metrics depend on a corpus generated from a topic-vocabulary table whose conclusion sentence is by construction close to the reference summary. The extractive stack therefore receives a ROUGE-L floor that real PubMed Central abstracts will not provide. The architecture and the evaluation harness are designed for drop-in replacement by a real corpus; the headline numbers should not be transferred.

**Bounded extractive-vs-abstractive lift.** The corpus is engineered such that the abstractive paraphrase advantage is muted. On a real-world corpus the gap should widen materially.

**Single-language scope.** The corpus is English-only. Arabic-language medical abstracts (relevant to the GCC research-tooling context) require a separate vocabulary, sentence-segmentation strategy, and token-pattern configuration; they are out of scope.

**ROUGE-L's known weaknesses.** ROUGE-L under-rewards genuine paraphrase. A complementary embedding-based metric (BERTScore) is documented as a follow-up; the LCS implementation is independent of its choice and remains the deterministic CPU-only contract.

**Hallucination check is coarse.** The unigram-novelty rate is a tractable proxy for lexical faithfulness. It does not capture semantic hallucination — a summary that uses only source vocabulary but recombines it into an unsupported claim. A learned faithfulness verifier (any hosted endpoint, e.g. GPT-4o-mini, Mistral-Large, Llama-3-Instruct) is documented as a second-opinion column; it does not replace the contract.

**Single-evaluator harness.** We have not run a clinician-judgment comparison on the synthetic outputs. Such an evaluation is out of scope for a portfolio research demonstration but would be a prerequisite for any clinical-pathway deployment.

**No clinical validation.** The system is a research demonstration. Use of these summaries as a substitute for the underlying abstracts in any clinical-decision-support context is explicitly out of scope and, per the disclaimer attached to every API response, would constitute a misuse of the artefact.

---

## 9. Conclusion

A medical-summarisation system is worth deploying only if every output ships with its own quality number and an extractive sibling that the user can sanity-check. The two-stack architecture proposed here puts the same evaluation harness behind both an extractive Dataiku-compatible stack and a heavier abstractive production stack, treats ROUGE-L as a structural attribute of every API response rather than as an offline benchmark metric, and preserves the extractive output as a per-response audit trail. On a deterministic synthetic corpus designed to give the extractive stack a fair shot, the production stack delivers approximately 8 additional ROUGE-L points at approximately 6× the latency. The right order of investment, in our view, is the deterministic evaluation harness first, the extractive sibling second, the abstractive lift third. Most production failures in clinical NLP are first-order failures dressed up as third-order ones.

---

## References

1. Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing order into texts. *Proceedings of EMNLP 2004*, 404–411.
2. Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual Web search engine. *Computer Networks and ISDN Systems*, 30(1–7), 107–117.
3. Erkan, G., & Radev, D. R. (2004). LexRank: Graph-based lexical centrality as salience in text summarization. *Journal of Artificial Intelligence Research*, 22, 457–479.
4. Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993–1022.
5. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. *Proceedings of ACL 2020*, 7871–7880.
6. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21, 1–67.
7. Guo, M., Ainslie, J., Uthus, D., Ontanon, S., Ni, J., Sung, Y.-H., & Yang, Y. (2022). LongT5: Efficient text-to-text transformer for long sequences. *Findings of NAACL 2022*, 724–736.
8. Zhang, J., Zhao, Y., Saleh, M., & Liu, P. J. (2020). PEGASUS: Pre-training with extracted gap-sentences for abstractive summarization. *Proceedings of ICML 2020*, 11328–11339.
9. Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. *Proceedings of the ACL Workshop on Text Summarization Branches Out*, 74–81.
10. Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. *Proceedings of ACL 2020*, 1906–1919.
11. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. *Proceedings of ICLR 2020*.
12. Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., & Poon, H. (2021). Domain-specific language model pretraining for biomedical natural language processing. *ACM Transactions on Computing for Healthcare*, 3(1), 1–23.
13. Yuan, H., Yuan, Z., Gan, R., Zhang, J., Xie, Y., & Yu, S. (2022). BioBART: Pretraining and evaluation of a biomedical generative language model. *Proceedings of the BioNLP Workshop 2022*, 97–109.
14. Nallapati, R., Zhou, B., dos Santos, C., Gulcehre, C., & Xiang, B. (2016). Abstractive text summarization using sequence-to-sequence RNNs and beyond. *Proceedings of CoNLL 2016*, 280–290.
15. Abacha, A. B., Mrabet, Y., Zhang, Y., Shivade, C., Langlotz, C., & Demner-Fushman, D. (2021). Overview of the MEDIQA 2021 shared task on summarization in the medical domain. *Proceedings of the BioNLP Workshop 2021*, 74–85.
