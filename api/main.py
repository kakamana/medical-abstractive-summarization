"""FastAPI for the Medical Abstractive Summarization project.

Endpoints:
    GET  /health
    POST /summarize  — abstract in, summary + ROUGE-L out
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Medical Abstractive Summarization", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "This summary is for research-assistant purposes and is not clinical decision "
    "support. Always consult the full abstract and original paper before any "
    "clinical conclusion."
)


class SummarizeRequest(BaseModel):
    abstract: str = Field(min_length=20, description="Medical paper abstract")
    reference: str | None = Field(
        None, description="Optional reference summary for ROUGE-L scoring"
    )
    top_k: int = Field(2, ge=1, le=5, description="Number of sentences to extract")


class SummarizeResponse(BaseModel):
    summary: str
    top_extractive_sentences: list[str]
    rouge_l_target: float | None = None
    rouge_1: float | None = None
    rouge_2: float | None = None
    disclaimer: str = DISCLAIMER


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize_endpoint(req: SummarizeRequest) -> SummarizeResponse:
    try:
        from med_summarize.serve import summarize

        result = summarize(req.abstract, reference=req.reference, top_k=req.top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    return SummarizeResponse(**result)
