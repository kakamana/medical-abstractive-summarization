from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


SAMPLE_ABSTRACT = (
    "Cardiovascular disease remains a leading cause of mortality, and optimal "
    "management of heart failure with reduced ejection fraction is still a "
    "clinical challenge. We conducted a randomised controlled trial enrolling "
    "1200 adults with heart failure with reduced ejection fraction. SGLT2 "
    "inhibitor add-on reduced major adverse cardiovascular events by 18.4% "
    "compared with standard of care (hazard ratio 0.78, 95% CI 0.65 to 0.92; "
    "p=0.004). Adverse events were reported in 14% of the SGLT2 arm and 12% "
    "of the standard-of-care arm. SGLT2 inhibitor add-on significantly "
    "improved major adverse cardiovascular events compared with standard of "
    "care in patients with heart failure with reduced ejection fraction."
)
SAMPLE_REFERENCE = (
    "SGLT2 inhibitor add-on significantly improved major adverse cardiovascular "
    "events compared with standard of care in patients with heart failure with "
    "reduced ejection fraction. The intervention was associated with an 18.4% "
    "change in major adverse cardiovascular events (HR 0.78)."
)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_summarize_returns_summary_and_rouge():
    r = client.post(
        "/summarize",
        json={"abstract": SAMPLE_ABSTRACT, "reference": SAMPLE_REFERENCE, "top_k": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["summary"], str) and len(body["summary"]) > 10
    assert isinstance(body["top_extractive_sentences"], list)
    assert body["rouge_l_target"] is not None
    assert 0 <= body["rouge_l_target"] <= 1


def test_data_generator_deterministic():
    from med_summarize.data import make_papers

    a = make_papers(n=20, seed=7)
    b = make_papers(n=20, seed=7)
    assert (a["paper_id"].tolist() == b["paper_id"].tolist())
    assert (a["abstract"].tolist() == b["abstract"].tolist())


def test_rouge_l_self_match():
    from med_summarize.features import rouge_l

    r = rouge_l("the cat sat on the mat", "the cat sat on the mat")
    assert abs(r["f1"] - 1.0) < 1e-6


def test_textrank_picks_k_sentences():
    from med_summarize.models import extractive_summary

    summary, sents = extractive_summary(SAMPLE_ABSTRACT, top_k=2)
    assert len(sents) == 2
    assert all(s in SAMPLE_ABSTRACT for s in sents)
