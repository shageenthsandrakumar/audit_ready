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


def operating_curve(proba: np.ndarray, is_match: np.ndarray, grid: int = 50) -> list[dict]:
    """False-match rate and recall as a function of the decision threshold.

    This is the operational curve of BDA3 sec 1.7 (their Figure 1.4): as we lower
    the threshold we declare more matches and the error rate rises. It lets a
    decision-maker pick a threshold for an acceptable false-match rate instead of
    guessing a cutoff.
    """
    n_true = max(int(is_match.sum()), 1)
    out = []
    for t in np.linspace(0.01, 0.99, grid):
        declared = proba >= t
        d = int(declared.sum())
        tp = int((declared & (is_match == 1)).sum())
        fmr = (d - tp) / d if d else 0.0
        out.append({"t": round(float(t), 3), "declared": d,
                    "false_match_rate": round(fmr, 4), "recall": round(tp / n_true, 4)})
    return out


def decision_thresholds(curve: list[dict], fmr_tol: float = 0.01) -> dict:
    """Principled triage cutoffs from the operating curve.

    auto-fix  : lowest threshold whose false-match rate stays <= fmr_tol
                (safe to apply automatically with an audit record)
    needs-review floor : 0.5 (below this, the model favors non-match)
    Between the two sits 'quick-check'. This replaces hand-set 0.60/0.90 cutoffs
    with values derived from a target error rate (decision theory, MacKay Ch 36).
    """
    auto = next((c["t"] for c in curve if c["false_match_rate"] <= fmr_tol), 0.99)
    return {"auto_fix": round(auto, 3), "needs_review_floor": 0.5, "fmr_tolerance": fmr_tol}


def posterior_predictive(linker: FellegiSunterLinker, Gamma_obs: np.ndarray, n_rep: int = 300) -> dict:
    """Posterior predictive check (BDA3 sec 6.3) on the REAL data.

    Draw replicated agreement-pattern datasets from the fitted model and compare
    the distribution of agreement scores to the observed one. If the model fits,
    the observed histogram lies inside the replicated band.
    """
    n, K = Gamma_obs.shape
    obs = np.bincount(Gamma_obs.sum(axis=1).astype(int), minlength=K + 1)
    reps = np.zeros((n_rep, K + 1))
    for r in range(n_rep):
        z = RNG.random(n) < linker.p
        G = np.where(z[:, None], RNG.random((n, K)) < linker.m, RNG.random((n, K)) < linker.u)
        reps[r] = np.bincount(G.sum(axis=1).astype(int), minlength=K + 1)
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    return {
        "score": list(range(K + 1)),
        "observed": obs.tolist(),
        "rep_mean": np.round(reps.mean(axis=0), 1).tolist(),
        "rep_lo": np.round(lo, 1).tolist(),
        "rep_hi": np.round(hi, 1).tolist(),
        "inside_band": [bool(lo[i] <= obs[i] <= hi[i]) for i in range(K + 1)],
    }


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
    curve = operating_curve(proba, is_match)
    thresholds = decision_thresholds(curve)
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
        "operating_curve": curve,
        "decision_thresholds": thresholds,
        "field_weights": linker.weights_table().to_dict("records"),
    }

    # Posterior predictive check on the REAL (un-injected) data.
    real_linker = FellegiSunterLinker()
    G_real, _ = real_linker.candidate_pairs(df)
    real_linker.fit(G_real)
    result["ppc_real"] = posterior_predictive(real_linker, G_real)
    result["ppc_real"]["fitted_prevalence"] = round(real_linker.p, 5)
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
