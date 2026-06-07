"""AuditReady — local demo server.

Runs the *integrated* pipeline (Chhiring's 6 detectors + Bayesian confidence +
the 4-agent report) on localhost, WITHOUT the GCP/ADK deploy stack — so anyone
on the team can see the real output in a browser. No Groq key or GCP auth needed
(the pipeline itself is deterministic Python; the LLM only orchestrates in prod).

Run:
    python my-first-agent/local_demo.py
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Stub the GCP/ADK deps so we can import the pipeline locally.
# --------------------------------------------------------------------------- #
class _Stub:
    def __init__(self, *a, **k):
        pass


def _mk(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


for _n in ("google", "google.adk"):
    _mk(_n)
_mk("google.adk.agents", Agent=_Stub)
_mk("google.adk.apps", App=_Stub)
_mk("google.adk.models", Gemini=_Stub)
_mk("google.adk.models.lite_llm", LiteLlm=_Stub)
_mk("google.genai")
_mk("google.genai.types", HttpRetryOptions=_Stub)
sys.modules["google.genai"].types = sys.modules["google.genai.types"]
_mk("dotenv", load_dotenv=lambda *a, **k: None)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "my-first-agent"))

from app.agent import REPORTS_DIR, run_data_quality_pipeline  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse  # noqa: E402

api = FastAPI(title="AuditReady (local demo)")

_STATE: dict = {}


def _run() -> dict:
    meta = run_data_quality_pipeline()
    audit = json.loads(Path(meta["audit_path"]).read_text(encoding="utf-8"))
    _STATE["meta"] = meta
    _STATE["audit"] = audit
    return meta


_BADGE = {
    "auto_fixed": "#16a34a",
    "quick_check": "#d97706",
    "needs_review": "#dc2626",
}


def _page() -> str:
    if "audit" not in _STATE:
        _run()
    audit = _STATE["audit"]
    ranked = audit["ranked_findings"]
    actions = {a["rank"]: a for a in audit["actions"]}

    confs = [r["finding"]["evidence"].get("confidence", 0) for r in ranked]
    mean_conf = sum(confs) / len(confs) if confs else 0
    triage_counts: dict[str, int] = {}
    exposure = 0.0
    for r in ranked:
        ev = r["finding"]["evidence"]
        triage_counts[ev.get("triage", "?")] = triage_counts.get(ev.get("triage", "?"), 0) + 1
        exposure += float(ev.get("impact_usd") or 0)

    rows = []
    for r in ranked[:300]:  # cap table for snappy render
        f = r["finding"]
        ev = f["evidence"]
        tri = ev.get("triage", "?")
        act = actions.get(r["rank"], {})
        rows.append(
            f"<tr>"
            f"<td>{r['rank']}</td>"
            f"<td>{f['finding_id']}</td>"
            f"<td>{f['finding_type']}</td>"
            f"<td style='text-align:right'>{ev.get('confidence',0):.0%}</td>"
            f"<td><span class='badge' style='background:{_BADGE.get(tri,'#666')}'>{tri}</span></td>"
            f"<td><b>{act.get('action','')}</b></td>"
            f"<td>${float(ev.get('impact_usd') or 0):,.0f}</td>"
            f"<td class='fix'>{ev.get('suggested_fix','')}</td>"
            f"</tr>"
        )

    triage_html = " &nbsp; ".join(
        f"<span class='badge' style='background:{_BADGE.get(k,'#666')}'>{k}: {v}</span>"
        for k, v in sorted(triage_counts.items())
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>AuditReady — local demo</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#0f172a; color:#e2e8f0; }}
  header {{ padding: 24px 32px; background:#1e293b; border-bottom:1px solid #334155; }}
  h1 {{ margin:0; font-size:22px; }} .sub {{ color:#94a3b8; font-size:13px; margin-top:4px; }}
  .cards {{ display:flex; gap:16px; padding:24px 32px; flex-wrap:wrap; }}
  .card {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:16px 20px; min-width:150px; }}
  .card .n {{ font-size:28px; font-weight:700; }} .card .l {{ color:#94a3b8; font-size:12px; text-transform:uppercase; }}
  table {{ width:calc(100% - 64px); margin:0 32px 40px; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:8px 10px; border-bottom:1px solid #1e293b; text-align:left; }}
  th {{ color:#94a3b8; position:sticky; top:0; background:#0f172a; }}
  .badge {{ padding:2px 8px; border-radius:999px; color:white; font-size:11px; font-weight:600; }}
  .fix {{ color:#94a3b8; max-width:340px; }}
  a {{ color:#60a5fa; }}
</style></head><body>
<header>
  <h1>AuditReady &nbsp;·&nbsp; Track 01 Data Rescue</h1>
  <div class="sub">Integrated pipeline: 6 benchmark detectors → Bayesian confidence → Find/Rank/Act/Explain &nbsp;·&nbsp;
  <a href="/report.md">report.md</a> &nbsp;·&nbsp; <a href="/audit.json">audit.json</a> &nbsp;·&nbsp; <a href="/api/run">re-run</a></div>
</header>
<div class="cards">
  <div class="card"><div class="n">{len(ranked)}</div><div class="l">Findings</div></div>
  <div class="card"><div class="n">{mean_conf:.0%}</div><div class="l">Mean confidence</div></div>
  <div class="card"><div class="n">${exposure:,.0f}</div><div class="l">Estimated exposure</div></div>
  <div class="card" style="min-width:320px"><div class="l">Triage</div><div style="margin-top:8px">{triage_html}</div></div>
</div>
<table>
  <thead><tr><th>#</th><th>Record</th><th>Issue</th><th>Conf.</th><th>Triage</th><th>Action</th><th>Impact</th><th>Suggested fix</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
<div style="padding:0 32px 40px; color:#64748b; font-size:12px">Showing top 300 of {len(ranked)} findings. Full set in audit.json.</div>
</body></html>"""


@api.get("/", response_class=HTMLResponse)
def home() -> str:
    return _page()


@api.get("/api/run")
def rerun() -> JSONResponse:
    meta = _run()
    return JSONResponse(meta)


@api.get("/report.md", response_class=PlainTextResponse)
def report_md() -> str:
    if "meta" not in _STATE:
        _run()
    return Path(_STATE["meta"]["report_path"]).read_text(encoding="utf-8")


@api.get("/audit.json")
def audit_json() -> JSONResponse:
    if "audit" not in _STATE:
        _run()
    return JSONResponse(_STATE["audit"])


if __name__ == "__main__":
    import uvicorn

    print("Running pipeline once, then serving on http://127.0.0.1:8000 ...")
    _run()
    print(f"  {_STATE['meta']['findings_count']} findings ready. Open http://127.0.0.1:8000")
    uvicorn.run(api, host="127.0.0.1", port=8000, log_level="warning")
