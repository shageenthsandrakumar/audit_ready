# AuditReady

**An explainable, multi-agent data-rescue copilot.**
Built for the **M-AGENTS / vibeFORWARD** hackathon (NYC Tech Week, June 2026) — **Track 01: Data Rescue**.

🔗 **Live dashboard:** https://shageenthsandrakumar.github.io/audit_ready/
📐 **Model methodology:** https://shageenthsandrakumar.github.io/audit_ready/model.html

> A manufacturer's data is corrupted days before a regulatory audit — duplicates, unit
> conflicts, weights off by a factor of 100. The person who has to sign off is a
> compliance officer who has never opened a database. AuditReady finds every broken
> record, **says how sure it is**, and explains *why* — so a non-technical person can
> walk into the audit able to defend every number.

---

## What it does

A four-agent pipeline runs over the real 5,000-record benchmark and produces a
downloadable, sign-off-ready audit:

1. **Find It** — six detectors catch the six real failure modes (exact duplicates,
   near-duplicate variants, unit-format drift, orphaned customer refs, decimal-shift
   weights, impossible values) → **742 findings**.
2. **Rank It** — orders findings worst- and most-certain-first.
3. **Act On It** — recommends *fix / flag / escalate*, every action with a logged reason.
4. **Explain It** — a human-readable report + a machine-readable JSON evidence trail.

The differentiator: **every finding carries a calibrated Bayesian confidence**, which
routes it into a triage lane — **auto-fix** (decisive), **quick-check** (a glance), or
**needs-review** (the agent genuinely can't tell an error from legitimate data). The
officer reviews the handful the model is unsure about, not all 5,000 rows.

---

## Architecture

```
src/                         Detection + memory  — Chhiring
  memory_store.py              SQLite handoff backbone (shared JSON contract)
  detectors.py                 6 benchmark-tuned detectors -> 742 findings
my-first-agent/              The agent product
  app/agent.py                 4-agent ADK pipeline (Find/Rank/Act/Explain) — Anastasiya
  app/confidence.py            Hierarchical Bayesian confidence + triage — Shageenth
  app/fast_api_app.py          FastAPI product API
  local_demo.py                Run the whole pipeline locally, no GCP/keys — Shageenth
docs/                        Showcase UI (GitHub Pages) — Shageenth
  index.html                   Trust score, confidence x impact quadrant, triage queues
  model.html                   Rendered Bayesian methodology + open questions
data/
  track01_*.csv                The benchmark datasets
  audit_findings.json          Live findings + confidence the UI consumes
```

**Memory & handoff:** only Agent 1 reads the raw CSV; Agents 2–4 operate entirely on
records recalled from the `MemoryStore` via a shared JSON contract — the collaboration
is auditable, no re-ingestion. *(Cognee was the intended memory layer; its API was down
during the event and a judge cleared us to substitute our own behind the same interface.)*

**LLM:** Google ADK orchestrator on **Groq** (`llama-3.1-8b-instant`) via LiteLLM.

---

## Run it

**Whole pipeline locally (no GCP, no API key):**
```bash
pip install pandas numpy scipy fastapi uvicorn
python my-first-agent/local_demo.py   # -> http://127.0.0.1:8000
```

**The agent (ADK + Groq):**
```bash
cd my-first-agent
cp .env.example .env          # add GROQ_API_KEY
uv sync                       # or pip install -e .
# run the ADK playground / POST /pipeline/run
```

**Detection only:**
```bash
python src/run_agent1.py      # 742 findings into the MemoryStore
```

---

## The confidence model (in one line)

For weight errors, a hierarchical Bayesian model (per-part partial pooling, PyMC with a
scipy empirical-Bayes fallback) scores each value as a posterior **P(error)**; structural
classes use evidence-graded priors. Result: mean confidence **0.80**, honest spread
**0.55–1.00**, all three triage lanes populated. Full derivation + open questions:
[`docs/model.html`](https://shageenthsandrakumar.github.io/audit_ready/model.html).

---

## Contributors

| | |
|---|---|
| **Shageenth Sandrakumar** | Bayesian confidence model (`confidence.py`), full pipeline integration, showcase UI + methodology doc, local runner, repo & data bridge |
| **Chhiring** | `MemoryStore` + the six Find-It detectors (`src/`) |
| **Anastasiya Chabotska** | ADK 4-agent pipeline + FastAPI product (`my-first-agent/app/`) |
| **TaliZ** | Demo / presentation |
| **mozywang** | Domain research + QA |
