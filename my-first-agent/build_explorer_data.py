"""Build docs/explorer_data.json for the live in-browser Bayesian explorer.

The explorer recomputes the decimal-shift model client-side as the user drags
the prior/decay sliders, so it needs each weight finding's model internals
(theta = log10 part center, y = log10 value, pred_sd). It also needs a compact
view of all findings to re-triage the whole batch as the decision threshold
moves. This script emits exactly that, reusing the real WeightModel + detectors.

Run:  python my-first-agent/build_explorer_data.py
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

# --- stub GCP/ADK so the pipeline imports locally ---------------------------- #
def _mk(name: str, **attrs):
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

from app.confidence import (  # noqa: E402
    FLAGGED_ERROR_PRIOR,
    TRIAGE_AUTO,
    TRIAGE_REVIEW,
    WeightModel,
    _to_float,
    score_findings,
)
from src.detectors import (  # noqa: E402
    detect_decimal_shift_weights,
    detect_exact_duplicates,
    detect_impossible_values,
    detect_near_duplicate_variants,
    detect_orphaned_customers,
    detect_unit_format_drift,
)
from src.memory_store import MemoryStore  # noqa: E402


def build() -> dict:
    df = pd.read_csv(_ROOT / "data" / "track01_data_rescue.csv")
    customers = pd.read_csv(_ROOT / "my-first-agent" / "app" / "track01_customers.csv")

    store = MemoryStore(db_path=str(_ROOT / "my-first-agent" / "app" / "pipeline_memory.db"))
    store.forget()
    detect_exact_duplicates(df, store)
    detect_orphaned_customers(df, customers, store)
    detect_impossible_values(df, store)
    detect_near_duplicate_variants(df, store)
    detect_unit_format_drift(df, store)
    detect_decimal_shift_weights(df, store)
    raw = store.recall()
    scored = score_findings(raw, df)

    model = WeightModel().fit(df, use_pymc=False)

    # Decimal-shift findings: carry the model internals so the client can
    # recompute P(error)/Bayes factor live as the sliders move.
    weight_cases = []
    for f in scored:
        if f.get("issue_type") != "decimal_shift_weight":
            continue
        part = str(f.get("part_number") or (f.get("raw_signals") or {}).get("part_number"))
        weight = _to_float(f.get("current_value"))
        if not (weight and weight > 0):
            continue
        theta, pred_sd = model._part_params(part)
        corr = model.best_decimal_correction(part, weight)
        weight_cases.append({
            "record_id": f.get("record_id"),
            "part_number": part,
            "weight_kg": round(float(weight), 4),
            "y": round(math.log10(weight), 4),          # observed, log10
            "theta": round(float(theta), 4),            # part center, log10
            "pred_sd": round(float(pred_sd), 4),        # predictive sd, log10
            "impact_usd": round(float(f.get("impact_usd") or 0.0), 2),
            "suggested_kg": corr["suggested_kg"] if corr else None,
            "k": corr["k"] if corr else None,
        })

    # Compact view of ALL findings for global re-triage (confidence is fixed for
    # structural classes; only the threshold the user picks moves their lane).
    all_findings = [{
        "issue_type": f.get("issue_type"),
        "confidence": round(float(f.get("confidence", 0.0)), 4),
        "impact_usd": round(float(f.get("impact_usd") or 0.0), 2),
        "is_weight": f.get("issue_type") == "decimal_shift_weight",
    } for f in scored]

    return {
        "defaults": {
            "decay": 0.3,
            "pi": FLAGGED_ERROR_PRIOR,
            "triage_auto": TRIAGE_AUTO,
            "triage_review": TRIAGE_REVIEW,
        },
        "weight_cases": sorted(weight_cases, key=lambda c: -c["impact_usd"]),
        "all_findings": all_findings,
        "n_rows": int(len(df)),
    }


if __name__ == "__main__":
    data = build()
    dest = _ROOT / "docs" / "explorer_data.json"
    dest.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {dest.relative_to(_ROOT)}")
    print(f"  weight cases: {len(data['weight_cases'])}  | all findings: {len(data['all_findings'])}")
    c = data["weight_cases"][0]
    print(f"  top case: {c['record_id']} {c['part_number']} {c['weight_kg']}kg "
          f"(theta={c['theta']}, y={c['y']}, pred_sd={c['pred_sd']})")
