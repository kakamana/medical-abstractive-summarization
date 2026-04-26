"""Synthetic data generator for the Medical Abstractive Summarization project.

Produces one parquet file under data/processed/:
- papers.parquet  (10,000 rows: paper_id, topic, abstract, summary, tokens_count)

Each abstract follows the standard structured-abstract format (background,
methods, results, conclusion). The reference summary is composed from the
conclusion sentence + the headline result figure.

Run:
    python -m med_summarize.data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED = DATA_DIR / "processed"

N_PAPERS = 10_000
SEED = 42

TOPICS = [
    "cardiology",
    "oncology",
    "infectious_disease",
    "neurology",
    "endocrinology",
]

# ----- topic-specific vocabularies -------------------------------------------

TOPIC_VOCAB: dict[str, dict[str, list[str]]] = {
    "cardiology": {
        "condition": [
            "atrial fibrillation", "heart failure with reduced ejection fraction",
            "stable angina", "acute coronary syndrome", "non-valvular atrial fibrillation",
        ],
        "intervention": [
            "low-dose rivaroxaban", "high-intensity statin therapy",
            "catheter ablation", "SGLT2 inhibitor add-on",
            "guideline-directed medical therapy",
        ],
        "comparator": ["standard of care", "placebo", "warfarin", "usual care"],
        "outcome": [
            "major adverse cardiovascular events", "30-day rehospitalisation",
            "cardiovascular mortality", "stroke or systemic embolism",
        ],
        "background": (
            "Cardiovascular disease remains a leading cause of mortality, and optimal "
            "management of {condition} is still a clinical challenge."
        ),
    },
    "oncology": {
        "condition": [
            "metastatic non-small cell lung cancer",
            "HER2-positive breast cancer", "triple-negative breast cancer",
            "advanced melanoma", "platinum-resistant ovarian cancer",
        ],
        "intervention": [
            "pembrolizumab plus chemotherapy",
            "trastuzumab deruxtecan", "olaparib monotherapy",
            "nivolumab plus ipilimumab", "CDK4/6 inhibitor maintenance",
        ],
        "comparator": ["chemotherapy alone", "physician's choice therapy", "placebo"],
        "outcome": [
            "progression-free survival", "objective response rate",
            "overall survival", "duration of response",
        ],
        "background": (
            "Despite recent advances, treatment of {condition} continues to face "
            "limited durable response and resistance development."
        ),
    },
    "infectious_disease": {
        "condition": [
            "community-acquired pneumonia", "complicated urinary tract infection",
            "MRSA bloodstream infection", "post-COVID-19 long syndrome",
            "carbapenem-resistant Enterobacteriaceae infection",
        ],
        "intervention": [
            "ceftolozane-tazobactam", "remdesivir plus dexamethasone",
            "shorter five-day antibiotic course", "monoclonal antibody prophylaxis",
            "phage-therapy adjunct",
        ],
        "comparator": ["standard antibiotic regimen", "placebo", "longer ten-day course"],
        "outcome": [
            "30-day all-cause mortality", "clinical cure rate at day 14",
            "time to defervescence", "microbiological eradication",
        ],
        "background": (
            "Antimicrobial resistance and treatment-duration optimisation are "
            "active questions in the management of {condition}."
        ),
    },
    "neurology": {
        "condition": [
            "early Alzheimer's disease", "relapsing-remitting multiple sclerosis",
            "treatment-resistant epilepsy", "Parkinson's disease motor fluctuations",
            "acute ischemic stroke",
        ],
        "intervention": [
            "lecanemab biweekly infusion", "ocrelizumab maintenance",
            "deep brain stimulation", "thrombectomy within 6 hours",
            "vagus nerve stimulation",
        ],
        "comparator": ["placebo", "standard-of-care immunomodulator", "medical therapy alone"],
        "outcome": [
            "Clinical Dementia Rating Scale change", "annualised relapse rate",
            "modified Rankin score at 90 days", "seizure-frequency reduction",
        ],
        "background": (
            "Disease-modifying options for {condition} remain limited, and the "
            "effect size of available interventions is modest."
        ),
    },
    "endocrinology": {
        "condition": [
            "type 2 diabetes with obesity", "polycystic ovary syndrome",
            "primary hypothyroidism", "Cushing's syndrome",
            "diabetic nephropathy",
        ],
        "intervention": [
            "semaglutide weekly injection", "tirzepatide titration",
            "metformin plus pioglitazone", "finerenone add-on",
            "bariatric surgery",
        ],
        "comparator": ["placebo", "standard-of-care insulin", "lifestyle intervention alone"],
        "outcome": [
            "HbA1c reduction at 24 weeks", "body-weight change",
            "cardiovascular composite endpoint", "renal-function preservation",
        ],
        "background": (
            "Despite multiple available agents, achieving sustained control of "
            "{condition} remains challenging in real-world settings."
        ),
    },
}

DESIGN_TEMPLATES = [
    "We conducted a randomised controlled trial enrolling {n} adults with {condition}.",
    "In this multicentre prospective cohort study, {n} participants with {condition} were followed.",
    "A double-blind, placebo-controlled trial randomised {n} patients with {condition} to {intervention} or {comparator}.",
    "We performed a retrospective analysis of electronic health records for {n} patients with {condition}.",
]

RESULTS_TEMPLATES = [
    (
        "{intervention} reduced {outcome} by {effect:.1f}% compared with {comparator} "
        "(hazard ratio {hr:.2f}, 95% CI {lo:.2f} to {hi:.2f}; p={p:.3f})."
    ),
    (
        "Treatment with {intervention} was associated with a {effect:.1f}% improvement "
        "in {outcome} versus {comparator} (relative risk {hr:.2f}; p={p:.3f})."
    ),
    (
        "Compared with {comparator}, {intervention} produced a {effect:.1f}-point "
        "absolute change in {outcome} (mean difference {hr:.2f}; p={p:.3f})."
    ),
]

CONCLUSION_TEMPLATES = [
    "{intervention} significantly improved {outcome} compared with {comparator} in patients with {condition}.",
    "These findings suggest {intervention} is an effective option for {condition}, with a meaningful effect on {outcome}.",
    "Adding {intervention} to standard care improved {outcome} versus {comparator} in {condition}.",
]

SAFETY_TEMPLATES = [
    "Adverse events were reported in {ae:.0f}% of the {intervention} arm and {aec:.0f}% of the {comparator} arm.",
    "Serious adverse events occurred in {ae:.0f}% versus {aec:.0f}% of patients, with no new safety signals identified.",
]


# ----- helpers ---------------------------------------------------------------


def _pick(rng, lst):
    return str(rng.choice(lst))


def _make_one(rng: np.random.Generator, paper_id: int) -> dict:
    topic = _pick(rng, TOPICS)
    v = TOPIC_VOCAB[topic]
    cond = _pick(rng, v["condition"])
    inter = _pick(rng, v["intervention"])
    comp = _pick(rng, v["comparator"])
    out = _pick(rng, v["outcome"])
    n = int(rng.integers(80, 4001))
    effect = float(rng.uniform(8.0, 35.0))
    hr = float(rng.uniform(0.55, 0.95))
    lo = max(0.30, hr - rng.uniform(0.10, 0.20))
    hi = min(1.10, hr + rng.uniform(0.10, 0.20))
    p = float(rng.uniform(0.001, 0.049))
    ae = float(rng.uniform(8.0, 30.0))
    aec = float(rng.uniform(8.0, 30.0))

    background = v["background"].format(condition=cond)
    design = _pick(rng, DESIGN_TEMPLATES).format(
        n=n, condition=cond, intervention=inter, comparator=comp
    )
    results = _pick(rng, RESULTS_TEMPLATES).format(
        intervention=inter, comparator=comp, outcome=out,
        effect=effect, hr=hr, lo=lo, hi=hi, p=p,
    )
    safety = _pick(rng, SAFETY_TEMPLATES).format(
        intervention=inter, comparator=comp, ae=ae, aec=aec,
    )
    conclusion = _pick(rng, CONCLUSION_TEMPLATES).format(
        intervention=inter, comparator=comp, outcome=out, condition=cond,
    )

    abstract = " ".join([background, design, results, safety, conclusion])
    summary = (
        f"{conclusion} The intervention was associated with a {effect:.1f}% "
        f"change in {out} (HR {hr:.2f})."
    )

    return dict(
        paper_id=f"P-{paper_id:05d}",
        topic=topic,
        abstract=abstract,
        summary=summary,
        tokens_count=len(abstract.split()),
    )


def make_papers(n: int = N_PAPERS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [_make_one(rng, i) for i in range(1, n + 1)]
    return pd.DataFrame(rows)


def make_corpus() -> pd.DataFrame:
    df = make_papers()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "papers.parquet", index=False)
    return df


if __name__ == "__main__":
    df = make_corpus()
    print(f"wrote {len(df):,} papers -> data/processed/papers.parquet")
    print(df["topic"].value_counts().to_string())
    print(f"avg abstract tokens: {df['tokens_count'].mean():.0f}")
