"""Calibration validation harness for the record-linkage model.

The benchmark's real errors are deterministic and separable, so they can't
*test* calibration -- there's no graded ambiguity. This harness manufactures
ground truth: it injects fuzzy near-duplicates at a KNOWN rate (perturbing a
random subset of fields), then asks whether the Fellegi-Sunter model's
P(match) is calibrated against that truth.

This is simulation-based calibration / posterior predictive checking
(BDA3 sec 6.3): if the model is honest, then among pairs it scores P(match)=q,
a fraction ~q should actually be injected duplicates.

Outputs: prevalence recovery, a reliability diagram (predicted vs empirical),
and the Expected Calibration Error (ECE).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from app.record_linkage import FellegiSunterLinker

RNG = np.random.default_rng(7)

# Fields we may perturb when forging a fuzzy duplicate, and how.
_PERTURB = {
    "quantity": lambda v: int(v) + int(RNG.choice([-2, -1, 1, 2])),
    "weight_kg": lambda v: round(float(v) * float(RNG.uniform(1.02, 1.15)), 2),
    "production_date": lambda v: _shift_date(v),
    "ship_date": lambda v: _shift_date(v),
    "last_modified": lambda v: _shift_date(v),
    "plant_id": lambda v: RNG.choice([c for c in "ABCD" if c != v]) if v in "ABCD" else v,
    "status": lambda v: RNG.choice(["pending", "in_qa", "completed", "shipped"]),
}


def _shift_date(v: str) -> str:
    try:
        return (pd.Timestamp(v) + pd.Timedelta(days=int(RNG.choice([-2, -1, 1, 2])))).strftime("%Y-%m-%d")
    except Exception:
        return v


def inject_fuzzy_duplicates(df: pd.DataFrame, n_inject: int = 500) -> tuple[pd.DataFrame, set[frozenset]]:
    """Return (augmented_df, set of true-match id-pairs).

    Each injected record is a copy of a real one with a random subset of fields
    perturbed (0-4 of them), so the agreement pattern -- and thus the honest
    P(match) -- spans the full range.
    """
    base = df.drop_duplicates(subset=["part_number", "quantity", "weight_kg"]).reset_index(drop=True)
    picks = base.sample(n=n_inject, random_state=7).to_dict("records")
    perturbable = list(_PERTURB)
    new_rows, truth = [], set()
    for i, rec in enumerate(picks):
        clone = dict(rec)
        clone["record_id"] = f"R-INJ-{i:04d}"
        k = int(RNG.integers(1, 7))  # perturb 1..6 of 9 fields -> spans clean..heavily-corrupted
        for fld in RNG.choice(perturbable, size=k, replace=False):
            clone[fld] = _PERTURB[fld](rec[fld])
        new_rows.append(clone)
        truth.add(frozenset((rec["record_id"], clone["record_id"])))
    aug = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)
    return aug, truth


def reliability(proba: np.ndarray, is_match: np.ndarray, n_bins: int = 10) -> tuple[list[dict], float]:
    """Reliability-diagram bins + Expected Calibration Error."""
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece, N = [], 0.0, len(proba)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (proba >= lo) & (proba < hi if hi < 1 else proba <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": 0, "pred": None, "emp": None})
            continue
        pred = float(proba[mask].mean())
        emp = float(is_match[mask].mean())
        ece += (n / N) * abs(pred - emp)
        bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": n, "pred": round(pred, 4), "emp": round(emp, 4)})
    return bins, float(ece)


def run(n_inject: int = 500) -> dict:
    root = pathlib.Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "data" / "track01_data_rescue.csv")
    aug, truth = inject_fuzzy_duplicates(df, n_inject=n_inject)

    linker = FellegiSunterLinker()
    Gamma, ids = linker.candidate_pairs(aug)
    linker.fit(Gamma)
    proba = linker.predict_proba(Gamma)
    is_match = np.array([frozenset(p) in truth for p in ids], dtype=float)

    bins, ece = reliability(proba, is_match)
    est_matches = float(proba.sum())
    result = {
        "n_candidate_pairs": len(ids),
        "n_injected_true_matches": int(is_match.sum()),
        "estimated_prevalence": round(linker.p, 5),
        "expected_matches_model": round(est_matches, 1),
        "recovered_at_0.5": int((proba >= 0.5).sum()),
        "true_matches_recalled_at_0.5": int(((proba >= 0.5) & (is_match == 1)).sum()),
        "false_matches_at_0.5": int(((proba >= 0.5) & (is_match == 0)).sum()),
        "ece": round(ece, 4),
        "reliability_bins": bins,
        "field_weights": linker.weights_table().to_dict("records"),
    }
    out = root / "data" / "calibration_report.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run()
    print(f"candidate pairs: {r['n_candidate_pairs']}   injected true matches: {r['n_injected_true_matches']}")
    print(f"estimated prevalence: {r['estimated_prevalence']}  -> expected matches {r['expected_matches_model']}")
    print(f"recall@0.5: {r['true_matches_recalled_at_0.5']}/{r['n_injected_true_matches']}   "
          f"false matches@0.5: {r['false_matches_at_0.5']}")
    print(f"\nExpected Calibration Error (ECE): {r['ece']}\n")
    print(f"{'bin':>12} {'n':>6} {'predicted':>10} {'empirical':>10}")
    for b in r["reliability_bins"]:
        if b["n"]:
            print(f"  [{b['lo']:.1f},{b['hi']:.1f}) {b['n']:6d} {b['pred']:10.3f} {b['emp']:10.3f}")
