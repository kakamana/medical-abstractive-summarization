# Data Sources — Medical Abstractive Summarization

## Primary
| # | Source | URL | Fields used | License |
|---|--------|-----|-------------|---------|
| 1 | Synthetic generator (this repo) | `src/med_summarize/data.py` | All | MIT |

## Reference / future augment
| Source | URL | Use |
|--------|-----|-----|
| PubMed Central Open Access | https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/ | Real abstracts + papers |
| arXiv q-bio | https://arxiv.org/list/q-bio/recent | Pre-prints |
| MIMIC-III (DUA-gated) | https://physionet.org/content/mimiciii/ | Clinical narrative summarization |
| Cochrane plain-language summaries | https://www.cochrane.org/ | Reference for plain-English summaries |

## Regenerate
```bash
python -m med_summarize.data
# writes data/processed/papers.parquet  (10,000 rows)
```

## Attribution
If you publish results derived from this generator, please cite this repo and the underlying templates' inspirations (PubMed structured-abstract style; CONSORT-style results reporting).
