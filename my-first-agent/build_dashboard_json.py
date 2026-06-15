"""Regenerate the frontend dashboard data (docs/audit_findings.json).

Runs the *real* pipeline (6 detectors -> Bayesian/Fellegi-Sunter confidence ->
rank -> act) and serializes it into the flat shape docs/index.html consumes.
Reuses app.agent's own ranking/action logic so the dashboard can never drift
from the pipeline. ADK is stubbed (as in local_demo) so it runs without GCP.

Run:  python my-first-agent/build_dashboard_json.py
"""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

# --- stub GCP/ADK so the pipeline imports locally (same trick as local_demo) -- #
def _mk(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _Stub:
    def __init__(self, *a, **k):
        pass


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

import pandas as pd  # noqa: E402

from app.agent import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    _agent_1_find_it,
    _agent_2_rank_it,
    _agent_3_act_on_it,
)


def build() -> dict:
    df = pd.read_csv(DEFAULT_DATASET_PATH)
    findings = _agent_1_find_it(df)          # detectors + F-S/Bayesian confidence
    ranked = _agent_2_rank_it(findings)      # severity, confidence ordering
    actions = {a["rank"]: a for a in _agent_3_act_on_it(ranked, df)}

    out_findings, triage_counts, issue_counts = [], {}, {}
    exposure, conf_sum = 0.0, 0.0
    for item in ranked:
        f = item["finding"]
        ev = f.get("evidence", {}) or {}
        act = actions.get(item["rank"], {})
        conf = float(ev.get("confidence", 0.0))
        triage = ev.get("triage", "needs_review")
        issue = f.get("finding_type", "unknown")
        impact = float(ev.get("impact_usd") or 0.0)

        triage_counts[triage] = triage_counts.get(triage, 0) + 1
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
        exposure += impact
        conf_sum += conf

        out_findings.append({
            "rank": item["rank"],
            "record_id": f.get("finding_id", ""),
            "issue_type": issue,
            "severity": ev.get("issue_severity"),
            "confidence": round(conf, 4),
            "triage": triage,
            "action": act.get("action", ""),
            "action_reason": act.get("action_reason", ""),
            "current_value": ev.get("current_value"),
            "suggested_fix": ev.get("suggested_fix"),
            "evidence": f.get("reason", ""),
            "impact_usd": round(impact, 2),
            "part_number": ev.get("part_number"),
            "plant_id": ev.get("plant_id"),
        })

    n = len(out_findings)
    return {
        "summary": {
            "dataset": Path(DEFAULT_DATASET_PATH).name,
            "rows_analyzed": int(len(df)),
            "total_findings": n,
            "mean_confidence": round(conf_sum / n, 4) if n else 0.0,
            "triage_counts": triage_counts,
            "issue_type_counts": dict(sorted(issue_counts.items())),
            "estimated_exposure_usd": round(exposure, 2),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "findings": out_findings,
    }


if __name__ == "__main__":
    data = build()
    for dest in (_ROOT / "docs" / "audit_findings.json", _ROOT / "data" / "audit_findings.json"):
        dest.write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"wrote {dest.relative_to(_ROOT)}")
    s = data["summary"]
    print(f"\n{s['total_findings']} findings | mean conf {s['mean_confidence']:.1%} | "
          f"exposure ${s['estimated_exposure_usd']:,.0f}")
    print("triage:", s["triage_counts"])
    from collections import defaultdict
    by = defaultdict(set)
    for f in data["findings"]:
        by[f["issue_type"]].add(round(f["confidence"], 3))
    print("\nconfidence spread by issue type (was a single constant each):")
    for k, v in sorted(by.items()):
        vs = sorted(v)
        show = f"{len(vs)} distinct, range [{min(vs)}, {max(vs)}]" if len(vs) > 1 else f"constant {vs[0]}"
        print(f"  {k:24} {show}")
