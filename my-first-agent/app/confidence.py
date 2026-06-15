"""Bayesian confidence layer for AuditReady (M-AGENTS, Track 01).

Sits between detection (Agent 1 / Chhiring) and the agents (Anastasiya).
Reads each finding's raw signals + value, returns a CALIBRATED probability that
the flagged issue is real -- "96% sure this is a duplicate", not a yes/no.
This is the PyMC special-prize differentiator.

Pipeline position:
    detection -> writes finding {raw_signals, current_value} to MemoryStore
       -> THIS layer: score_findings() -> adds {confidence, triage}
          -> agents rank / act / explain using confidence

Two tiers:
  Tier 1  fast signal->probability mapping per issue class (always available).
  Tier 2  hierarchical Bayesian per-part weight model (PyMC) giving a real
          posterior probability of error + the most-likely true value.
          Falls back to a scipy empirical-Bayes approximation if PyMC is absent,
          so the module is fully functional either way.

The contract field names match README.md -- do not rename.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# PyMC is optional. If present we fit a full hierarchical model; if not, we use
# the empirical-Bayes fallback below. Either way the public API is identical.
try:
    import pymc as pm  # noqa: F401

    _HAS_PYMC = True
except Exception:  # pragma: no cover - import guard
    _HAS_PYMC = False


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Evidence-grade priors: P(this flagged pattern is a REAL, actionable error) --
# NOT merely "did the pattern match". Deterministic/physical issues are near-
# certain; genuinely ambiguous ones (a possibly-legitimate new customer, a
# case-only spelling, a 2-row near-duplicate that could be a real repeat order)
# are honestly hedged so they fall into the human-review lane. Severity (set by
# the detector) encodes that ambiguity, so some priors vary by severity.
STRUCTURAL_PRIORS: dict[str, Any] = {
    "exact_duplicate": 0.97,            # exact copy -> near-certain
    "impossible_value": 0.96,           # negative qty / >500kg -> physically impossible
    "orphaned_customer": 0.55,          # could be a legit new customer -> defer to a human
    "near_duplicate_variant": {         # by clone-group size (detector severity)
        "high": 0.82,                   # 3+ clones -> very likely a duplicate
        "med": 0.58,                    # 2 clones -> could be a real repeat order
        "low": 0.52,
    },
    "unit_format_drift": {              # whitespace (breaks joins) vs case-only (often benign)
        "med": 0.74,
        "low": 0.45,
    },
}
# Detector-name aliases so either naming resolves.
_ALIASES = {
    "orphaned_customer_ref": "orphaned_customer",
    "near_duplicate": "near_duplicate_variant",
}
# Weight-statistical issue handled directly by the Bayesian model.
STATISTICAL_ISSUES = {"decimal_shift_weight"}

# Base rate of genuine errors in the benchmark (850 / 5000).
ERROR_PRIOR = 0.17

# Prior used INSIDE the weight model's H0/H1 comparison. This is subtle and
# matters: error_probability() only ever scores records the *detector already
# flagged*, so the relevant prior is P(real error | flagged), NOT the
# unconditional 0.17. A flagged record has already passed a screen, so a neutral
# 0.5 ("the data decide") is the honest, weakly-informative choice. Per MacKay
# Ch 28 (p.347) this prior is a decision-theoretic input kept separate from the
# evidence; we therefore also report the Bayes factor f1/f0 so the verdict can
# be read independently of this prior.
FLAGGED_ERROR_PRIOR = 0.5

# Triage thresholds on final confidence.
TRIAGE_AUTO = 0.90      # >= -> auto_fixed
TRIAGE_REVIEW = 0.60    # >= -> quick_check, else needs_review


def to_triage(confidence: float) -> str:
    """Map a confidence in [0, 1] to a triage lane.

    Args:
        confidence: Calibrated probability the flagged issue is real.

    Returns:
        One of "auto_fixed", "quick_check", "needs_review".
    """
    if confidence >= TRIAGE_AUTO:
        return "auto_fixed"
    if confidence >= TRIAGE_REVIEW:
        return "quick_check"
    return "needs_review"


# --------------------------------------------------------------------------- #
# Tier 1 -- signal -> probability for structural issue classes
# --------------------------------------------------------------------------- #

def tier1_confidence(finding: dict[str, Any]) -> float:
    """Fast confidence for an issue from its detector signals.

    Used directly for structural classes (duplicates, orphans, unit drift) and
    as a prior the Bayesian model refines for weight issues.

    Args:
        finding: A finding dict in the shared contract shape. Reads
            "issue_type" and "raw_signals" {signals_agreed, sigma_off}.

    Returns:
        Confidence in [0, 1].
    """
    issue = finding.get("issue_type", "")
    issue = _ALIASES.get(issue, issue)
    severity = str(finding.get("severity", "med")).lower()
    signals = finding.get("raw_signals") or {}
    agreed = int(signals.get("signals_agreed", 1) or 1)
    sigma_off = float(signals.get("sigma_off", 0.0) or 0.0)

    prior = STRUCTURAL_PRIORS.get(issue, 0.60)
    base = prior.get(severity, 0.55) if isinstance(prior, dict) else float(prior)

    # Extra *independent* agreeing signals raise confidence (diminishing, and
    # gentler than before so we don't saturate everything near 1.0).
    conf = base
    for _ in range(max(0, agreed - 1)):
        conf += (1.0 - conf) * 0.25

    # Genuine statistical distance from the per-part baseline raises it.
    if sigma_off:
        conf += (1.0 - conf) * (1.0 - math.exp(-abs(sigma_off) / 3.0)) * 0.4

    return float(min(max(conf, 0.01), 0.99))


# --------------------------------------------------------------------------- #
# Tier 2 -- hierarchical Bayesian per-part weight model
# --------------------------------------------------------------------------- #

@dataclass
class WeightModel:
    """Hierarchical model of log10(weight_kg) per part_number.

    Partial pooling lets rare parts borrow strength from the global
    distribution, so a part seen 3 times still gets a sensible baseline.
    Scores a flagged weight with a Bayesian posterior P(error) and proposes the
    most-likely true value (catching decimal shifts as +/- integer shifts in
    log10 space).
    """

    mu0: float = 0.0                 # global mean of log10(weight)
    tau: float = 1.0                 # between-part sd
    sigma: float = 0.3               # within-part sd
    theta: dict[str, float] = field(default_factory=dict)   # per-part center
    pred_sd: dict[str, float] = field(default_factory=dict)  # per-part predictive sd
    fitted_with: str = "unfitted"

    # ---- fitting ---------------------------------------------------------- #
    def fit(self, df: pd.DataFrame, use_pymc: bool = True) -> WeightModel:
        """Fit the model on the production records.

        Args:
            df: track01_data_rescue rows; needs "part_number" and "weight_kg".
            use_pymc: Use full PyMC MCMC if available; else empirical Bayes.

        Returns:
            self (fitted).
        """
        work = df[["part_number", "weight_kg"]].copy()
        work = work[pd.to_numeric(work["weight_kg"], errors="coerce").notna()]
        work["weight_kg"] = work["weight_kg"].astype(float)
        work = work[work["weight_kg"] > 0]
        work["y"] = np.log10(work["weight_kg"])

        if _HAS_PYMC and use_pymc:
            try:
                return self._fit_pymc(work)
            except Exception:
                pass  # fall through to robust EB
        return self._fit_empirical_bayes(work)

    def _fit_empirical_bayes(self, work: pd.DataFrame) -> WeightModel:
        """Robust empirical-Bayes fit (median/MAD based, scipy only).

        Robust statistics resist the seeded errors that contaminate the data.
        """
        grp = work.groupby("part_number")["y"]
        med = grp.median()
        counts = grp.count()

        # Within-part spread via pooled MAD -> sd (1.4826 * MAD).
        def _mad(v: np.ndarray) -> float:
            return float(np.median(np.abs(v - np.median(v)))) if len(v) else 0.0

        mads = grp.apply(lambda v: _mad(v.to_numpy()))
        sigma = float(np.median(mads[mads > 0]) * 1.4826) if (mads > 0).any() else 0.25
        sigma = max(sigma, 0.05)

        self.mu0 = float(med.median())
        self.tau = float(max((med.quantile(0.84) - med.quantile(0.16)) / 2.0, 0.1))
        self.sigma = sigma

        for part, n in counts.items():
            ybar = float(med[part])
            # Shrink the part center toward the global mean (partial pooling).
            w_data = n / sigma**2
            w_prior = 1.0 / self.tau**2
            theta = (w_data * ybar + w_prior * self.mu0) / (w_data + w_prior)
            var_theta = 1.0 / (w_data + w_prior)
            self.theta[str(part)] = float(theta)
            self.pred_sd[str(part)] = float(math.sqrt(sigma**2 + var_theta))

        self.fitted_with = "empirical_bayes"
        return self

    def _fit_pymc(self, work: pd.DataFrame) -> WeightModel:
        """Full hierarchical Bayesian fit with PyMC (robust StudentT likelihood)."""
        import pymc as pm

        parts = work["part_number"].astype(str).to_numpy()
        uniq = sorted(set(parts))
        idx = {p: i for i, p in enumerate(uniq)}
        part_idx = np.array([idx[p] for p in parts])
        y = work["y"].to_numpy()

        with pm.Model() as model:
            mu0 = pm.Normal("mu0", mu=float(np.median(y)), sigma=2.0)
            tau = pm.HalfNormal("tau", sigma=1.0)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            theta = pm.Normal("theta", mu=mu0, sigma=tau, shape=len(uniq))
            nu = pm.Exponential("nu", 1 / 10.0) + 1  # heavy tails -> robust
            pm.StudentT(
                "obs", nu=nu, mu=theta[part_idx], sigma=sigma, observed=y
            )
            # Fast config for hackathon timing; still a real posterior.
            trace = pm.sample(
                draws=500, tune=500, chains=2, cores=1,
                target_accept=0.9, progressbar=False,
                compute_convergence_checks=False,
            )

        post = trace.posterior
        self.mu0 = float(post["mu0"].mean())
        self.tau = float(post["tau"].mean())
        self.sigma = float(post["sigma"].mean())
        theta_mean = post["theta"].mean(dim=("chain", "draw")).to_numpy()
        theta_sd = post["theta"].std(dim=("chain", "draw")).to_numpy()
        for p, i in idx.items():
            self.theta[p] = float(theta_mean[i])
            self.pred_sd[p] = float(math.sqrt(self.sigma**2 + theta_sd[i] ** 2))

        self.fitted_with = "pymc"
        return self

    # ---- scoring ---------------------------------------------------------- #
    def _part_params(self, part_number: str) -> tuple[float, float]:
        """Center + predictive sd for a part, backing off to global for unseen."""
        p = str(part_number)
        if p in self.theta:
            return self.theta[p], self.pred_sd[p]
        # Unseen part: global mean, inflated sd.
        return self.mu0, float(math.sqrt(self.sigma**2 + self.tau**2))

    def error_probability(
        self, part_number: str, weight: float, *, decay: float = 0.3, pi: float = FLAGGED_ERROR_PRIOR
    ) -> tuple[float, dict[str, Any]]:
        """Posterior probability that a weight is an error (Bayesian model comparison).

        H0 (genuine): y = log10(weight) ~ Normal(theta_part, pred_sd)
        H1 (error):   y ~ mixture of decimal shifts, sum_k w_k Normal(theta+k, pred_sd)
        P(error) = pi*f1 / (pi*f1 + (1-pi)*f0),  pi = P(error | flagged).

        The Bayes factor BF = f1/f0 is the prior-free evidence (MacKay Ch 28); the
        verdict P(error) only depends on `pi` in the genuinely ambiguous middle,
        because for a true decimal shift BF is astronomically large.

        Args:
            part_number: The part the record belongs to.
            weight: The observed weight_kg (> 0).

        Returns:
            (p_error, info) where info carries evidence for the audit trail.
        """
        if weight is None or weight <= 0 or not np.isfinite(weight):
            return 0.99, {"reason": "non-positive or non-finite weight"}

        theta, pred_sd = self._part_params(part_number)
        y = math.log10(weight)

        f0 = stats.norm.pdf(y, loc=theta, scale=max(pred_sd, 1e-3))

        # Error model: decimal shifts (truth ± powers of 10).
        # H1 = mixture of shifts ±1, ±2, ±3 with geometric prior (decay=0.3).
        # This models the actual error mechanism: a weight that's the true value
        # displaced by an integer in log10-space.
        shifts = [-3, -2, -1, 1, 2, 3]
        weights = [decay ** abs(k) for k in shifts]
        Z = sum(weights)
        weights = [w / Z for w in weights]  # normalize to sum to 1
        f1 = sum(
            w * stats.norm.pdf(y, loc=theta + k, scale=max(pred_sd, 1e-3))
            for w, k in zip(weights, shifts)
        )

        # Conditional prior `pi` (default FLAGGED_ERROR_PRIOR): these records were
        # already flagged by the detector, so 0.5 ("the data decide"), not the
        # unconditional 0.17 base rate.
        denom = pi * f1 + (1 - pi) * f0
        p_error = float(pi * f1 / denom) if denom > 0 else 0.99

        # Prior-free evidence (MacKay Ch 28): the Bayes factor for H1 over H0.
        # f0 can underflow for large shifts; treat that as decisive evidence.
        with np.errstate(over="ignore", divide="ignore"):
            bayes_factor = float(f1 / f0) if f0 > 0 else float("inf")

        sigma_off = float((y - theta) / max(pred_sd, 1e-3))
        info = {
            "log10_weight": round(y, 3),
            "part_center_kg": round(10**theta, 4),
            "sigma_off": round(sigma_off, 2),
            "bayes_factor": round(bayes_factor, 2) if np.isfinite(bayes_factor) else None,
            "fitted_with": self.fitted_with,
        }
        return p_error, info

    def best_decimal_correction(
        self, part_number: str, weight: float
    ) -> dict[str, Any] | None:
        """Find the most likely true value if this looks like a decimal shift.

        A decimal shift is a +/- integer shift in log10 space: tests x10^k for
        k in [-3..3] and returns the correction whose value is most plausible
        under the part baseline -- but only if it is far more plausible than the
        original.

        Returns:
            {"k", "suggested_kg", "plausibility_gain"} or None.
        """
        if weight is None or weight <= 0 or not np.isfinite(weight):
            return None
        theta, pred_sd = self._part_params(part_number)
        y = math.log10(weight)
        f_orig = stats.norm.pdf(y, loc=theta, scale=max(pred_sd, 1e-3))

        best = None
        for k in (-3, -2, -1, 1, 2, 3):
            f_k = stats.norm.pdf(y + k, loc=theta, scale=max(pred_sd, 1e-3))
            if best is None or f_k > best[1]:
                best = (k, f_k)
        k, f_best = best
        # Only call it a decimal shift if the correction is decisively better.
        if f_best > f_orig * 100 and f_best > 0.05:
            return {
                "k": k,
                "suggested_kg": round(weight * (10**k), 4),
                "plausibility_gain": round(float(f_best / max(f_orig, 1e-12)), 1),
            }
        return None


# --------------------------------------------------------------------------- #
# Near-duplicate confidence via the Fellegi-Sunter record-linkage model
# --------------------------------------------------------------------------- #
# Issue types whose confidence is the F-S posterior P(match) rather than a
# hand-set structural prior.
DUP_ISSUES = {"near_duplicate_variant", "near_duplicate"}


def _load_linker_cls():
    """Import FellegiSunterLinker, tolerating the ADK-free standalone case."""
    try:  # normal path (production: app/__init__ -> agent -> google.adk is present)
        from app.record_linkage import FellegiSunterLinker

        return FellegiSunterLinker
    except Exception:  # standalone (no ADK): load the module file directly
        import importlib.util
        import pathlib
        import sys

        p = pathlib.Path(__file__).resolve().parent / "record_linkage.py"
        spec = importlib.util.spec_from_file_location("record_linkage_mod", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # let @dataclass resolve the module (py3.12+)
        spec.loader.exec_module(mod)
        return mod.FellegiSunterLinker


def duplicate_posteriors(df: pd.DataFrame) -> dict[str, float]:
    """Calibrated P(this record has a true duplicate) per record_id.

    Fits the Fellegi-Sunter mixture (BDA3 sec 1.7) on all within-part candidate
    pairs and assigns each record the max P(match) over the pairs it appears in
    -- the probability its best-matching partner is a genuine duplicate. This
    replaces the hand-set near_duplicate constants with a posterior estimated
    from the data with no labels.
    """
    linker = _load_linker_cls()()
    Gamma, ids = linker.candidate_pairs(df)
    if len(ids) == 0:
        return {}
    linker.fit(Gamma)
    proba = linker.predict_proba(Gamma)
    best: dict[str, float] = {}
    for (a, b), p in zip(ids, proba):
        p = float(p)
        if p > best.get(a, 0.0):
            best[a] = p
        if p > best.get(b, 0.0):
            best[b] = p
    return best


# --------------------------------------------------------------------------- #
# Scoring a finding / batch
# --------------------------------------------------------------------------- #

def score_finding(
    finding: dict[str, Any],
    model: WeightModel | None = None,
    dup_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Add calibrated `confidence` + `triage` (+ better fix) to one finding.

    Args:
        finding: A finding in the shared contract shape (mutated copy returned).
        model: A fitted WeightModel for weight issues. Optional for structural.

    Returns:
        The finding with `confidence`, `triage`, and possibly a refined
        `suggested_fix` / enriched `evidence`.
    """
    out = dict(finding)
    issue = out.get("issue_type", "")
    canon = _ALIASES.get(issue, issue)
    rec_id = str(out.get("record_id") or (out.get("raw_signals") or {}).get("record_id") or "")

    if issue in STATISTICAL_ISSUES and model is not None:
        part = out.get("part_number") or (out.get("raw_signals") or {}).get("part_number")
        weight = _to_float(out.get("current_value"))
        p_error, info = model.error_probability(str(part), weight)
        # Use the Bayesian posterior directly so confidence scales honestly with
        # magnitude: a 100x shift -> ~1.0, a borderline 2-3x outlier -> mid-range.
        out["confidence"] = round(float(p_error), 4)
        corr = model.best_decimal_correction(str(part), weight)
        if corr:
            out["suggested_fix"] = f"{corr['suggested_kg']} kg"
            out["evidence"] = (
                f"Part {part} baseline ~{info['part_center_kg']}kg; value is "
                f"{info['sigma_off']}sigma off ({corr['k']:+d} decimal shift, "
                f"{corr['plausibility_gain']}x more plausible corrected)."
            )
        out.setdefault("raw_signals", {})
        out["raw_signals"] = {**(out.get("raw_signals") or {}), **info}
    elif canon in DUP_ISSUES and dup_scores is not None and rec_id in dup_scores:
        # Calibrated Fellegi-Sunter posterior instead of a structural constant.
        p = float(dup_scores[rec_id])
        out["confidence"] = round(p, 4)
        out.setdefault("raw_signals", {})
        out["raw_signals"] = {
            **(out.get("raw_signals") or {}),
            "fs_p_match": round(p, 4),
            "confidence_source": "fellegi_sunter",
        }
    else:
        out["confidence"] = round(tier1_confidence(out), 4)

    out["triage"] = to_triage(out["confidence"])
    return out


def score_findings(
    findings: list[dict[str, Any]],
    df: pd.DataFrame | None = None,
    use_pymc: bool = False,
) -> list[dict[str, Any]]:
    """Calibrate a batch of findings; fits the weight model once if needed.

    Args:
        findings: List of findings in the shared contract shape.
        df: The production records, required only if any weight issues present.
        use_pymc: Full PyMC hierarchical fit (slower, the calibration showcase)
            vs the fast scipy empirical-Bayes fit (default, ~instant). Both give
            near-identical scores here; EB keeps dev/demo loops snappy.

    Returns:
        New list of findings with `confidence` + `triage` populated.
    """
    model = None
    if df is not None and any(
        f.get("issue_type") in STATISTICAL_ISSUES for f in findings
    ):
        model = WeightModel().fit(df, use_pymc=use_pymc)

    # Fellegi-Sunter posteriors for near-duplicate findings (BDA3 sec 1.7).
    dup_scores: dict[str, float] | None = None
    if df is not None and any(
        _ALIASES.get(f.get("issue_type", ""), f.get("issue_type", "")) in DUP_ISSUES
        for f in findings
    ):
        try:
            dup_scores = duplicate_posteriors(df)
        except Exception:
            dup_scores = None  # fall back to tier-1 constants if the fit fails

    return [score_finding(f, model, dup_scores) for f in findings]


def assess_confidence(record_json: str) -> str:
    """ADK tool: score a single finding's confidence and triage lane.

    Give this to an agent so it can ask "how sure are we about this finding?"
    The input is one finding as a JSON string (shared contract shape); the
    output is the same finding with `confidence` and `triage` filled in.

    Args:
        record_json: A finding serialized as a JSON string.

    Returns:
        A JSON string of the finding with confidence + triage added.
    """
    import json

    finding = json.loads(record_json)
    scored = score_finding(finding, _GLOBAL_MODEL)
    return json.dumps(scored)


# A process-wide fitted model the ADK tool can use (set by the pipeline).
_GLOBAL_MODEL: WeightModel | None = None


def set_global_model(model: WeightModel) -> None:
    """Register a fitted WeightModel for the assess_confidence ADK tool."""
    global _GLOBAL_MODEL
    _GLOBAL_MODEL = model


# --------------------------------------------------------------------------- #
# Calibration -- the PyMC-prize demo artifact
# --------------------------------------------------------------------------- #

def calibration_table(
    predictions: list[float], labels: list[int], n_bins: int = 10
) -> pd.DataFrame:
    """Reliability table: predicted confidence vs observed correctness.

    Run on a small hand-labeled sample to show "when we say 90%, we're right
    ~90% of the time." This is the calibration-curve mic-drop for judges.

    Args:
        predictions: Predicted confidences in [0, 1].
        labels: 1 if the finding was a genuine error, else 0.
        n_bins: Number of probability bins.

    Returns:
        DataFrame with bin, n, mean_predicted, observed_fraction.
    """
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(labels, dtype=int)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": int(mask.sum()),
                "mean_predicted": round(float(p[mask].mean()), 3),
                "observed_fraction": round(float(y[mask].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def _to_float(value: Any) -> float:
    """Parse a possibly-stringy value like '1500 kg' to a float."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = "".join(c for c in str(value) if c.isdigit() or c in ".-")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- #
# Smoke test -- runs on synthetic data so it works before real data lands
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Synthetic warehouse: 40 parts, each with a true log10 weight.
    parts = [f"P-{i:03d}" for i in range(40)]
    true_logw = {p: rng.uniform(-2, 2) for p in parts}  # 0.01kg .. 100kg
    rows = []
    for p in parts:
        n = rng.integers(5, 40)
        for _ in range(n):
            rows.append({"part_number": p, "weight_kg": 10 ** (true_logw[p] + rng.normal(0, 0.08))})
    df = pd.DataFrame(rows)

    print(f"PyMC available: {_HAS_PYMC}")
    model = WeightModel().fit(df, use_pymc=_HAS_PYMC)
    print(f"Fitted with: {model.fitted_with}  | parts: {len(model.theta)}  "
          f"mu0={model.mu0:.2f} tau={model.tau:.2f} sigma={model.sigma:.2f}\n")

    # Build findings: some genuine decimal shifts, some clean values.
    findings, labels = [], []
    for p in parts[:12]:
        good = 10 ** true_logw[p]
        shifted = good * 100  # 100x decimal shift = a real error
        findings.append({"record_id": f"R-{p}-bad", "issue_type": "decimal_shift_weight",
                         "part_number": p, "current_value": f"{shifted:.2f} kg",
                         "raw_signals": {"signals_agreed": 1}})
        labels.append(1)
        findings.append({"record_id": f"R-{p}-ok", "issue_type": "decimal_shift_weight",
                         "part_number": p, "current_value": f"{good:.2f} kg",
                         "raw_signals": {"signals_agreed": 1}})
        labels.append(0)
    # A couple of structural findings.
    findings.append({"record_id": "R-dup", "issue_type": "exact_duplicate",
                     "raw_signals": {"signals_agreed": 3}})
    labels.append(1)
    findings.append({"record_id": "R-orph", "issue_type": "orphaned_customer_ref",
                     "raw_signals": {"signals_agreed": 1}})
    labels.append(1)

    scored = score_findings(findings, df)
    print("Sample scored findings:")
    for f in scored[:6]:
        print(f"  {f['record_id']:12} {f['issue_type']:22} "
              f"conf={f['confidence']:.3f} [{f['triage']}]")

    preds = [f["confidence"] for f in scored]
    print("\nCalibration table:")
    print(calibration_table(preds, labels).to_string(index=False))
