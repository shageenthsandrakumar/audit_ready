"""Record-linkage confidence via the Fellegi-Sunter mixture model.

This replaces the hand-set constant for near-duplicate confidence with a
calibrated posterior P(duplicate | agreement pattern), estimated from the data
with no labels -- exactly the method behind BDA3 sec 1.7 ("Calibration for
record linkage", Belin & Rubin 1995) and the classical Fellegi-Sunter (1969)
record-linkage model.

The idea
--------
For each candidate record pair we compute an *agreement pattern* gamma -- a
binary vector over comparison fields (1 if the two records match on that field).
Each pair is either a true match (M) or a non-match (U), a latent indicator z.

    P(gamma_k = 1 | M) = m_k     (fields usually agree for a true duplicate)
    P(gamma_k = 1 | U) = u_k     (fields agree only by chance otherwise)

Assuming conditional independence of fields given z (the standard F-S
assumption), the pair likelihood under each class is a product, and the whole
collection of pairs is a two-component mixture with prevalence p = P(M). We fit
(p, m, u) by EM on the *unlabelled* pairs; the E-step responsibility is the
calibrated confidence:

    P(M | gamma) = p * L_M(gamma) / [ p * L_M(gamma) + (1 - p) * L_U(gamma) ].

Because the two components are mostly separated (a clean valley in the agreement
score), the mixture is identifiable and the responsibilities are sharp except in
the genuine overlap region -- which is precisely where an honest "maybe" belongs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

# Comparison fields for the agreement pattern (block key = part_number; part_name
# is dropped as redundant with part_number).
COMPARISON_FIELDS = [
    "plant_id",
    "quantity",
    "unit",
    "weight_kg",
    "production_date",
    "ship_date",
    "customer_id",
    "status",
    "last_modified",
]

_EPS = 1e-6


def _clip(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, _EPS, 1.0 - _EPS)


@dataclass
class FellegiSunterLinker:
    """Two-class (match / non-match) mixture over candidate-pair agreement patterns."""

    fields: list[str] = field(default_factory=lambda: list(COMPARISON_FIELDS))
    p: float = 0.05                      # prevalence P(match)
    m: np.ndarray | None = None          # m_k = P(agree | match)
    u: np.ndarray | None = None          # u_k = P(agree | non-match)
    n_iter_: int = 0
    loglik_: float = float("-nan")

    # ---- candidate generation ------------------------------------------- #
    def candidate_pairs(self, df: pd.DataFrame, block: str = "part_number") -> tuple[np.ndarray, list[tuple[str, str]]]:
        """All within-block pairs -> (Gamma [n_pairs x n_fields] agreement matrix, ids)."""
        rows, ids = [], []
        for _, grp in df.groupby(block):
            if len(grp) < 2:
                continue
            recs = grp[self.fields + ["record_id"]].to_dict("records")
            for a, b in combinations(recs, 2):
                rows.append([1 if a[f] == b[f] else 0 for f in self.fields])
                ids.append((a["record_id"], b["record_id"]))
        return np.array(rows, dtype=float), ids

    # ---- EM fit ---------------------------------------------------------- #
    def fit(self, Gamma: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> FellegiSunterLinker:
        n, K = Gamma.shape
        # Sensible init: matches agree almost always; non-matches at empirical base rate.
        self.m = np.full(K, 0.9)
        self.u = _clip(Gamma.mean(axis=0) * 0.5)
        self.p = 0.05
        prev_ll = -np.inf

        for it in range(1, max_iter + 1):
            # E-step: log-likelihood of each pattern under each class.
            lm = Gamma @ np.log(_clip(self.m)) + (1 - Gamma) @ np.log(_clip(1 - self.m))
            lu = Gamma @ np.log(_clip(self.u)) + (1 - Gamma) @ np.log(_clip(1 - self.u))
            a = np.log(_clip(self.p)) + lm
            b = np.log(_clip(1 - self.p)) + lu
            mx = np.maximum(a, b)
            denom = mx + np.log(np.exp(a - mx) + np.exp(b - mx))   # logsumexp
            r = np.exp(a - denom)                                  # responsibilities P(M | gamma)

            ll = float(denom.sum())
            # M-step.
            sr = r.sum()
            self.p = float(_clip(sr / n))
            self.m = _clip((r[:, None] * Gamma).sum(axis=0) / max(sr, _EPS))
            self.u = _clip(((1 - r)[:, None] * Gamma).sum(axis=0) / max(n - sr, _EPS))

            self.n_iter_, self.loglik_ = it, ll
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll
        return self

    # ---- scoring --------------------------------------------------------- #
    def predict_proba(self, Gamma: np.ndarray) -> np.ndarray:
        """Calibrated P(match | agreement pattern) for each pair."""
        lm = Gamma @ np.log(_clip(self.m)) + (1 - Gamma) @ np.log(_clip(1 - self.m))
        lu = Gamma @ np.log(_clip(self.u)) + (1 - Gamma) @ np.log(_clip(1 - self.u))
        a = np.log(_clip(self.p)) + lm
        b = np.log(_clip(1 - self.p)) + lu
        mx = np.maximum(a, b)
        return np.exp(a - (mx + np.log(np.exp(a - mx) + np.exp(b - mx))))

    def weights_table(self) -> pd.DataFrame:
        """Per-field m_k, u_k and the F-S log2 agreement weight log2(m/u)."""
        return pd.DataFrame(
            {
                "field": self.fields,
                "m_k": np.round(self.m, 4),
                "u_k": np.round(self.u, 4),
                "log2(m/u)": np.round(np.log2(self.m / self.u), 2),
            }
        )


# --------------------------------------------------------------------------- #
#  Standalone diagnostic
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    df = pd.read_csv(root / "data" / "track01_data_rescue.csv")

    linker = FellegiSunterLinker()
    Gamma, ids = linker.candidate_pairs(df)
    linker.fit(Gamma)
    proba = linker.predict_proba(Gamma)
    score = Gamma.sum(axis=1).astype(int)

    print(f"candidate pairs: {len(ids)}   EM iters: {linker.n_iter_}   loglik: {linker.loglik_:.1f}")
    print(f"estimated prevalence p = {linker.p:.4f}  ->  expected matches = {linker.p*len(ids):.1f}")
    print(f"sum of responsibilities (calibration identity) = {proba.sum():.1f}")
    print("\nper-field agreement weights (Fellegi-Sunter):")
    print(linker.weights_table().to_string(index=False))

    print("\ncalibrated P(match) by agreement score y:")
    for y in range(int(score.max()) + 1):
        mask = score == y
        if mask.any():
            print(f"  y={y:2d}: n={mask.sum():6d}   P(match) mean={proba[mask].mean():.4f}"
                  f"   [{proba[mask].min():.3f}, {proba[mask].max():.3f}]")

    declared = proba >= 0.5
    print(f"\npairs declared match (P>=0.5): {declared.sum()}  "
          f"(detector's exact-copy count was 130)")
    mid = (proba > 0.05) & (proba < 0.95)
    print(f"genuinely ambiguous pairs (0.05 < P < 0.95): {mid.sum()}  "
          f"-> these are the honest 'needs-review' near-dups the old detector missed")
