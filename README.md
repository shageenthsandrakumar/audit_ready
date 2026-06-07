# AuditReady

**An explainable, multi-agent data-rescue copilot.**
Built for the **M-AGENTS Hackathon** (NYC Tech Week, June 7 2026) — **Track 01: Data Rescue**.

> Harven Manufacturing faces a regulatory audit in 4 days and can't trust its own
> warehouse data after a rushed two-plant acquisition. AuditReady finds, fixes, and
> **explains** every broken record — so a compliance officer who's never opened a
> database can walk into the audit able to defend every number.

---

## What makes it different

Most teams will ship a data cleaner. AuditReady goes past the obvious tier:

- **Calibrated confidence on every fix** — *"96% sure this is a duplicate"*, not a black-box yes/no (chasing the PyMC special prize).
- **Triage, not a wall of edits** — every finding routes to 🟢 auto-fixed / 🟡 quick-check / 🔴 human decision.
- **Evidence chains, never "the model said so"** — every action carries a plain-English reason an officer can read and sign.
- **An audit-defense file, not a clean CSV** — the output is something you take *to the regulator*.

---

## Pipeline (per the official 5-step flow)

`Step 0 Define` → **Agent 1 Find It** → **Agent 2 Rank It** → **Agent 3 Act On It** → **Agent 4 Explain It** → `Step 5 Show It`

Each agent recalls the previous agent's work through a shared **memory layer**
(`MemoryStore`). *(Cognee was waived by a judge — APIs unavailable — so we run our
own memory layer; agent-to-agent collaboration is still demonstrated and judged.)*

---

## Division of labor

| Person | Role | Owns |
|---|---|---|
| **shageenth** | Lead Builder | Confidence calibration (PyMC) + Agents 2–4 (Rank/Act/Explain) + orchestration |
| **Chhiring** | Builder | `MemoryStore` (handoff backbone) + Agent 1 detectors |
| **Anastasiya** | Designer | Product Brief (Step 0) + the product UI (Step 5) |
| **TaliZ** | Presenter | Demo narrative + Trupeer video; Agent-4 wording |
| **mozywang** | Domain Expert | Geodo research + Step-4 sign-off + cold-QA |

---

## The shared contract

Every finding moves through the pipeline as this exact JSON. **Don't rename fields** —
the detectors, agents, and UI all build against it.

```json
{
  "record_id": "R-4012",
  "issue_type": "decimal_shift_weight",
  "severity": "high",
  "raw_signals": { "signals_agreed": 2, "sigma_off": 3.1, "details": "..." },
  "confidence": 0.83,
  "triage": "needs_review",
  "current_value": "1500 kg",
  "suggested_fix": "15.00 kg",
  "evidence": "Per-part median for Bolt-M8 is 15kg; this is 100x off (3.1 sigma).",
  "impact_usd": 24000,
  "stage": "found",
  "agent_log": []
}
```

`stage`: `found → ranked → acted → explained` · each agent appends its reasoning to `agent_log`.

---

## Data

Kaggle: `quantologist/track01-vibeforward-m-agents`
- `track01_data_rescue.csv` — 5,000 rows
- `track01_customers.csv` — 240 rows
- 850 seeded issues across 6 classes: exact duplicates · near-duplicate variants ·
  unit-format drift · orphaned customer refs · decimal-shift weights · impossible values

```bash
kaggle datasets download quantologist/track01-vibeforward-m-agents
```

---

## Repo layout

```
detection/   # Agent 1 detectors + MemoryStore   (Chhiring)
agents/      # Agents 2-4 + confidence calibration (shageenth)
ui/          # the product front-end              (Anastasiya)
data/        # local datasets (gitignored)
docs/        # Product Brief, demo script
```

## Submission checklist (due 5:00 PM, Devpost)
- [ ] Product Brief PDF
- [ ] GitHub repo link
- [ ] Trupeer demo video URL
- [ ] Track selection (Track 01)
- [ ] Product description
