"""Sensitivity analysis for the decimal-shift weight model.

A reviewer's first question about the H1 mixture is: "you hand-picked
decay=0.3 and a prior pi -- doesn't the verdict just follow your knobs?"

This script answers it empirically. It sweeps:
  * decay in {0.1, 0.2, 0.3, 0.5, 0.7}   (the geometric shift prior), and
  * pi    in {0.17, 0.3, 0.5, 0.7}        (P(error | flagged)),
across a grid of displacements s (the value sits at theta + s in log10 space,
i.e. a x10^s error).

The point (MacKay Ch 28): for a genuine decimal shift the Bayes factor f1/f0 is
astronomically large, so P(error) -> 1 regardless of the knobs. The knobs only
move the answer in the genuinely ambiguous middle (s ~ 0.3-0.6), which is
exactly where an honest model SHOULD defer to the prior. Robustness where it
matters; honesty where it doesn't.

Loads confidence.py directly by path so it does not drag in the ADK package.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console is cp1252 by default

_APP = pathlib.Path(__file__).resolve().parent
_ROOT = _APP.parents[1]

# Import confidence.py standalone (avoid app/__init__ -> google.adk). Register
# in sys.modules before exec so @dataclass can resolve the module (py3.12+).
_spec = importlib.util.spec_from_file_location("confidence_mod", _APP / "confidence.py")
confidence = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = confidence
_spec.loader.exec_module(confidence)

DECAYS = [0.1, 0.2, 0.3, 0.5, 0.7]
PIS = [0.17, 0.3, 0.5, 0.7]
# Displacements in log10 units: 0 = clean, 1 = x10, 2 = x100, 3 = x1000;
# 0.3/0.5 sit in the genuinely ambiguous overlap.
SHIFTS = [0.0, 0.3, 0.5, 1.0, 2.0, 3.0]


def _pick_part(model: "confidence.WeightModel", df: pd.DataFrame) -> str:
    """A well-populated part with a typical predictive sd."""
    counts = df.groupby("part_number")["weight_kg"].count()
    common = counts[counts >= counts.median()].index
    # pick the common part whose pred_sd is closest to the median pred_sd
    sds = {p: model.pred_sd[str(p)] for p in common if str(p) in model.pred_sd}
    med = np.median(list(sds.values()))
    return str(min(sds, key=lambda p: abs(sds[p] - med)))


def run() -> dict:
    df = pd.read_csv(_ROOT / "data" / "track01_data_rescue.csv")
    model = confidence.WeightModel().fit(df, use_pymc=False)
    part = _pick_part(model, df)
    theta, pred_sd = model._part_params(part)

    decay_grid, pi_grid = [], []
    for s in SHIFTS:
        weight = float(10 ** (theta + s))  # value displaced by s in log10 space
        # (a) sweep decay at the default prior
        row = {"shift_log10": s, "x_factor": round(10 ** s, 2)}
        for d in DECAYS:
            p, _ = model.error_probability(part, weight, decay=d)
            row[f"decay={d}"] = round(p, 4)
        # spread across decay values = how much the knob moves the verdict
        vals = [row[f"decay={d}"] for d in DECAYS]
        row["decay_spread"] = round(max(vals) - min(vals), 4)
        decay_grid.append(row)

        # (b) sweep the prior at the default decay
        prow = {"shift_log10": s, "x_factor": round(10 ** s, 2)}
        for pi in PIS:
            p, info = model.error_probability(part, weight, pi=pi)
            prow[f"pi={pi}"] = round(p, 4)
        prow["bayes_factor"] = info["bayes_factor"]
        vals = [prow[f"pi={pi}"] for pi in PIS]
        prow["pi_spread"] = round(max(vals) - min(vals), 4)
        pi_grid.append(prow)

    result = {
        "part": part,
        "part_center_kg": round(10 ** theta, 4),
        "pred_sd_log10": round(pred_sd, 4),
        "decays": DECAYS,
        "pis": PIS,
        "decay_sweep": decay_grid,
        "prior_sweep": pi_grid,
        "max_decay_spread": round(max(r["decay_spread"] for r in decay_grid), 4),
    }
    out = _ROOT / "data" / "sensitivity_report.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run()
    print(f"part {r['part']}  center={r['part_center_kg']}kg  pred_sd={r['pred_sd_log10']} (log10)\n")

    print("(a) DECAY sweep -- does the shift-prior knob control the verdict?")
    print(f"{'shift':>6} {'xfac':>7} " + " ".join(f"d={d:<4}" for d in r["decays"]) + "  spread")
    for row in r["decay_sweep"]:
        cells = " ".join(f"{row[f'decay={d}']:<6.3f}" for d in r["decays"])
        print(f"{row['shift_log10']:>6.1f} {row['x_factor']:>7.1f} {cells}  {row['decay_spread']:.4f}")
    print(f"  -> max spread across ALL decay values: {r['max_decay_spread']:.4f}")
    print("     (≈0 at clean & at real shifts; the knob is inert where it matters)\n")

    print("(b) PRIOR sweep -- pi only bites in the ambiguous middle (honest):")
    print(f"{'shift':>6} {'xfac':>7} " + " ".join(f"pi={pi:<4}" for pi in r["pis"]) + "   BF      spread")
    for row in r["prior_sweep"]:
        cells = " ".join(f"{row[f'pi={pi}']:<6.3f}" for pi in r["pis"])
        bf = row["bayes_factor"]
        bf_s = f"{bf:>8.1e}" if bf is not None else "    inf"
        print(f"{row['shift_log10']:>6.1f} {row['x_factor']:>7.1f} {cells} {bf_s}  {row['pi_spread']:.4f}")
