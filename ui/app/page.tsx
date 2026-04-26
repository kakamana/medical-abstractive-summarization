"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

type SummarizeResponse = {
  summary: string;
  top_extractive_sentences: string[];
  rouge_l_target: number | null;
  rouge_1: number | null;
  rouge_2: number | null;
  disclaimer: string;
};

const DEMO_ABSTRACT = `Cardiovascular disease remains a leading cause of mortality, and optimal management of heart failure with reduced ejection fraction is still a clinical challenge. We conducted a randomised controlled trial enrolling 1200 adults with heart failure with reduced ejection fraction. SGLT2 inhibitor add-on reduced major adverse cardiovascular events by 18.4% compared with standard of care (hazard ratio 0.78, 95% CI 0.65 to 0.92; p=0.004). Adverse events were reported in 14% of the SGLT2 arm and 12% of the standard-of-care arm. SGLT2 inhibitor add-on significantly improved major adverse cardiovascular events compared with standard of care in patients with heart failure with reduced ejection fraction.`;

const DEMO_REFERENCE = `SGLT2 inhibitor add-on significantly improved major adverse cardiovascular events compared with standard of care in patients with heart failure with reduced ejection fraction. The intervention was associated with an 18.4% change in major adverse cardiovascular events (HR 0.78).`;

export default function Home() {
  const [abstract, setAbstract] = useState(DEMO_ABSTRACT);
  const [reference, setReference] = useState(DEMO_REFERENCE);
  const [topK, setTopK] = useState(2);
  const [out, setOut] = useState<SummarizeResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    const res = await fetch(`${API}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        abstract,
        reference: reference.trim() || null,
        top_k: topK,
      }),
    });
    setOut(await res.json());
    setLoading(false);
  }

  return (
    <main className="min-h-screen p-8 max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold">Medical Abstract Summarizer</h1>
      <p className="opacity-70 mb-6">
        Paste an abstract on the left and (optionally) a reference summary on the
        right. Get an extractive summary plus ROUGE-L vs reference.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <div className="text-xs uppercase opacity-60 tracking-wide mb-1">
            Abstract
          </div>
          <textarea
            value={abstract}
            onChange={(e) => setAbstract(e.target.value)}
            rows={12}
            className="w-full rounded-xl border p-3 text-sm"
          />
        </div>
        <div>
          <div className="text-xs uppercase opacity-60 tracking-wide mb-1">
            Reference summary (optional)
          </div>
          <textarea
            value={reference}
            onChange={(e) => setReference(e.target.value)}
            rows={12}
            className="w-full rounded-xl border p-3 text-sm"
          />
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-4 items-center">
        <label className="text-sm">
          Top-k sentences:
          <input
            type="number"
            min={1}
            max={5}
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value || "2"))}
            className="ml-2 w-20 rounded border px-2 py-1"
          />
        </label>
        <button
          onClick={run}
          disabled={loading || abstract.length < 20}
          className="rounded-xl px-4 py-2 bg-black text-white disabled:opacity-50"
        >
          {loading ? "Summarising..." : "Summarise"}
        </button>
      </div>

      {out && (
        <>
          <div className="mt-8 rounded-2xl border p-4">
            <div className="text-xs uppercase opacity-60 tracking-wide">
              Generated summary
            </div>
            <div className="mt-2 text-base leading-relaxed">{out.summary}</div>
          </div>

          <div className="mt-4 grid grid-cols-3 gap-3">
            <Stat label="ROUGE-1" value={fmt(out.rouge_1)} />
            <Stat label="ROUGE-2" value={fmt(out.rouge_2)} />
            <Stat label="ROUGE-L" value={fmt(out.rouge_l_target)} />
          </div>

          <div className="mt-6 rounded-2xl border p-4">
            <div className="text-xs uppercase opacity-60 tracking-wide">
              Top extractive sentences
            </div>
            <ol className="mt-2 list-decimal list-inside space-y-1">
              {out.top_extractive_sentences.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ol>
          </div>

          <p className="mt-6 text-xs opacity-60 italic">{out.disclaimer}</p>
        </>
      )}
    </main>
  );
}

function fmt(v: number | null): string {
  return v == null ? "—" : v.toFixed(3);
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border p-4">
      <div className="text-xs uppercase tracking-wide opacity-60">{label}</div>
      <div className="text-2xl font-mono mt-1">{value}</div>
    </div>
  );
}
