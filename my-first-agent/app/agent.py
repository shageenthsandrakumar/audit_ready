# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ... [License text omitted for brevity] ...

import os
import sys
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from dotenv import load_dotenv

load_dotenv()




APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]  # audit_ready/ — lets us import src.detectors etc.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Track 01 datasets. The 5,000-row production file is the primary records set;
# the customer lookup is used for orphaned-reference detection.
DEFAULT_DATASET_PATH = APP_DIR / "track01_data_rescue.csv"
CUSTOMERS_PATH = APP_DIR / "track01_customers.csv"
FALLBACK_DATASET_PATH = APP_DIR / "track01_customers.csv"
ALLOWED_DATASETS: dict[str, Path] = {
    "track01_data_rescue.csv": DEFAULT_DATASET_PATH,
    "track01_customers.csv": CUSTOMERS_PATH,
}
REPORTS_DIR = APP_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


@dataclass
class Finding:
    finding_id: str
    finding_type: str
    severity: int
    reason: str
    evidence: dict[str, Any]


def _normalize_dataset_key(path: str) -> str:
    return Path(path.strip()).name.lower()


def _resolve_dataset_path(path: str | None = None) -> Path:
    if path and path.strip():
        key = _normalize_dataset_key(path)
        if key not in ALLOWED_DATASETS:
            allowed = ", ".join(sorted(ALLOWED_DATASETS.keys()))
            raise ValueError(
                f"Unsupported dataset '{path}'. Allowed datasets: {allowed}."
            )
        resolved = ALLOWED_DATASETS[key]
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Dataset not found: {resolved}")

    if DEFAULT_DATASET_PATH.exists():
        return DEFAULT_DATASET_PATH
    if FALLBACK_DATASET_PATH.exists():
        return FALLBACK_DATASET_PATH

    raise FileNotFoundError(
        f"No default dataset found. Expected one of: {DEFAULT_DATASET_PATH}, {FALLBACK_DATASET_PATH}"
    )


def _load_dataset(path: str | None = None) -> pd.DataFrame:
    dataset_path = _resolve_dataset_path(path)
    if dataset_path.suffix.lower() == ".csv":
        return pd.read_csv(dataset_path)
    if dataset_path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(dataset_path)
    raise ValueError("Unsupported dataset type. Use CSV or Excel.")


def _severity_to_int(severity: str, confidence: float) -> int:
    """Map the detector's low/med/high to an int, nudged by confidence so a
    high-confidence finding sorts above an uncertain one within the same tier."""
    base = {"high": 8, "med": 5, "low": 2}.get(str(severity).lower(), 3)
    return min(10, base + round(confidence * 2))


def _to_finding(scored: dict[str, Any]) -> Finding:
    """Convert a shared-contract finding (already scored with Bayesian
    confidence) into the report pipeline's Finding object."""
    conf = float(scored.get("confidence", 0.0))
    triage = scored.get("triage", "needs_review")
    fix = scored.get("suggested_fix", "")
    base_reason = scored.get("evidence", "")
    return Finding(
        finding_id=str(scored.get("record_id", "")),
        finding_type=str(scored.get("issue_type", "unknown")),
        severity=_severity_to_int(scored.get("severity", "med"), conf),
        reason=f"{base_reason} (Confidence {conf:.0%}; suggested fix: {fix})",
        evidence={
            "confidence": round(conf, 4),
            "triage": triage,
            "issue_severity": scored.get("severity"),
            "current_value": scored.get("current_value"),
            "suggested_fix": fix,
            "impact_usd": scored.get("impact_usd"),
            **(scored.get("raw_signals") or {}),
        },
    )


def _agent_1_find_it(df: pd.DataFrame) -> list[Finding]:
    """Find It (Step 1).

    Runs the team's six benchmark-tuned detectors over the production records,
    persists each finding to the shared MemoryStore (so the agent-to-agent
    handoff is auditable), then assigns every finding a *calibrated Bayesian
    confidence*. Returns Findings ready for ranking.
    """
    from src.memory_store import MemoryStore
    from src.detectors import (
        detect_decimal_shift_weights,
        detect_exact_duplicates,
        detect_impossible_values,
        detect_near_duplicate_variants,
        detect_orphaned_customers,
        detect_unit_format_drift,
    )
    from app.confidence import score_findings

    # The detectors need the production records (part_number, weight_kg, ...).
    # If the caller passed the customer lookup by mistake, load the real file.
    if "part_number" not in df.columns:
        df = pd.read_csv(DEFAULT_DATASET_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)

    store = MemoryStore(db_path=str(APP_DIR / "pipeline_memory.db"))
    store.forget()  # fresh run each pipeline invocation

    detect_exact_duplicates(df, store)
    detect_orphaned_customers(df, customers, store)
    detect_impossible_values(df, store)
    detect_near_duplicate_variants(df, store)
    detect_unit_format_drift(df, store)
    detect_decimal_shift_weights(df, store)

    raw = store.recall()                    # shared-contract findings (~742)
    scored = score_findings(raw, df)        # adds calibrated confidence + triage
    return [_to_finding(f) for f in scored]


def _agent_2_rank_it(findings: list[Finding]) -> list[dict[str, Any]]:
    def _conf(f: Finding) -> float:
        return float(f.evidence.get("confidence", 0.0)) if isinstance(f.evidence, dict) else 0.0

    sorted_findings = sorted(
        findings, key=lambda f: (-f.severity, -_conf(f), f.finding_id)
    )
    ranked: list[dict[str, Any]] = []
    for rank, f in enumerate(sorted_findings, start=1):
        c = _conf(f)
        ranked.append(
            {
                "rank": rank,
                "finding": asdict(f),
                "ranking_reason": (
                    f"Ranked #{rank}: severity={f.severity}, confidence={c:.0%}. "
                    f"The worst and most-certain issues are surfaced first."
                ),
            }
        )
    return ranked


def _agent_3_act_on_it(ranked: list[dict[str, Any]], df: pd.DataFrame) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in ranked:
        finding = item["finding"]
        ev = finding.get("evidence", {}) or {}
        triage = ev.get("triage", "needs_review")
        conf = float(ev.get("confidence", 0.0))
        fix = ev.get("suggested_fix") or "Apply the suggested correction."

        if triage == "auto_fixed":
            action = "fix"
            detail = fix
            reason = (
                f"Confidence {conf:.0%} (auto-fix tier): the evidence is decisive, "
                f"so the correction is applied with a logged audit record."
            )
        elif triage == "needs_review":
            action = "escalate"
            detail = "Escalate to the compliance officer before any change."
            reason = (
                f"Confidence {conf:.0%} (needs-review tier): too uncertain to auto-fix; "
                f"a human must decide."
            )
        else:  # quick_check
            action = "flag"
            detail = f"Flag for a 10-second human confirmation. {fix}"
            reason = (
                f"Confidence {conf:.0%} (quick-check tier): likely a real issue, but "
                f"worth a quick human glance before fixing."
            )

        actions.append(
            {
                "rank": item["rank"],
                "finding_id": finding["finding_id"],
                "action": action,
                "action_detail": detail,
                "action_reason": reason,
            }
        )

    return actions


def _agent_4_explain_it(
    dataset_name: str,
    row_count: int,
    ranked: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Data Quality Review Summary")
    lines.append("")
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Rows analyzed: {row_count}")
    lines.append(f"Generated at (UTC): {datetime.now(UTC).isoformat()}")
    lines.append("")

    # Confidence-calibrated summary (Bayesian / PyMC layer).
    confs = [
        float(it["finding"]["evidence"].get("confidence", 0.0))
        for it in ranked
        if isinstance(it["finding"].get("evidence"), dict)
    ]
    triage_counts: dict[str, int] = {}
    total_impact = 0.0
    for it in ranked:
        ev = it["finding"].get("evidence", {}) or {}
        t = ev.get("triage", "unknown")
        triage_counts[t] = triage_counts.get(t, 0) + 1
        total_impact += float(ev.get("impact_usd") or 0.0)
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    lines.append("## Summary")
    lines.append(f"- Findings: {len(ranked)}")
    lines.append(f"- Mean confidence: {mean_conf:.0%}")
    lines.append(
        "- Triage: " + ", ".join(f"{k}={v}" for k, v in sorted(triage_counts.items()))
    )
    lines.append(f"- Estimated exposure: ${total_impact:,.0f}")
    lines.append("")
    lines.append("## Prioritized Findings")
    lines.append("")

    # Map by rank (unique) — record_ids can repeat across issue types.
    action_by_rank = {a["rank"]: a for a in actions}
    for item in ranked:
        finding = item["finding"]
        action = action_by_rank.get(item["rank"], {})
        lines.append(
            f"- Rank {item['rank']}: {finding['finding_id']} ({finding['finding_type']}, severity={finding['severity']})"
        )
        ev = finding.get("evidence", {}) or {}
        lines.append(
            f"  - Confidence: {float(ev.get('confidence', 0.0)):.0%} | "
            f"Triage: {ev.get('triage', 'n/a')} | "
            f"Impact: ${float(ev.get('impact_usd') or 0.0):,.0f}"
        )
        lines.append(f"  - Finding reason: {finding['reason']}")
        lines.append(f"  - Ranking reason: {item['ranking_reason']}")
        lines.append(
            f"  - Action: {action.get('action', 'n/a')} | Reason: {action.get('action_reason', 'n/a')}"
        )

    lines.append("")
    lines.append("## Domain Expert Sign-off")
    lines.append("")
    lines.append("I have reviewed the findings and actions above.")
    lines.append("")
    lines.append("Name: ____________________")
    lines.append("Date: ____________________")
    lines.append("Signature: _______________")
    lines.append("")
    return "\n".join(lines)


def run_data_quality_pipeline(dataset_path: str | None = None) -> dict[str, Any]:
    resolved_dataset_path = _resolve_dataset_path(dataset_path)
    df = _load_dataset(str(resolved_dataset_path))
    findings = _agent_1_find_it(df)
    ranked = _agent_2_rank_it(findings)
    actions = _agent_3_act_on_it(ranked, df)
    report = _agent_4_explain_it(
        dataset_name=resolved_dataset_path.name,
        row_count=len(df),
        ranked=ranked,
        actions=actions,
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_id = f"report-{ts}"
    report_path = REPORTS_DIR / f"{report_id}.md"
    audit_path = REPORTS_DIR / f"{report_id}.json"
    report_path.write_text(report, encoding="utf-8")
    audit_payload = {
        "report_id": report_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "findings": [asdict(f) for f in findings],
        "ranked_findings": ranked,
        "actions": actions,
        "report_path": str(report_path),
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

    return {
        "report_id": report_id,
        "report_path": str(report_path),
        "audit_path": str(audit_path),
        "findings_count": len(findings),
    }


def read_dataset(file_path: str) -> str:
    """Reads a local dataset and returns its raw content for analysis.

    Args:
        file_path: The local path to the dataset file (CSV or Excel).

    Returns:
        The content of the dataset as a CSV string.
    """
    try:
        dataset_path = _resolve_dataset_path(file_path)
    except Exception as e:
        return f"Error: {str(e)}"

    try:
        if dataset_path.suffix.lower() == '.csv':
            df = pd.read_csv(dataset_path)
        elif dataset_path.suffix.lower() in ('.xls', '.xlsx'):
            df = pd.read_excel(dataset_path)
        else:
            return "Error: Unsupported file format. Please provide a .csv or .xlsx file."
        
        # Convert the dataframe to a CSV string to pass to the LLM.
        return df.to_csv(index=False)
    except Exception as e:
        return f"Error reading the dataset: {str(e)}"

root_agent = Agent(
    name="root_agent",
    # Groq via LiteLLM — free tier, high rate limits. Set GROQ_API_KEY in .env.
    model=LiteLlm(model="groq/llama-3.1-8b-instant"),
    instruction="""
You orchestrate a 4-step data-quality workflow:
1) Find It: detect duplicates, anomalies, suspicious patterns.
2) Rank It: prioritize findings worst-first with explicit ranking reasons.
3) Act On It: fix, flag, or escalate every finding with action rationale.
4) Explain It: generate a human-readable sign-off summary.

The dataset files are track01_customers.csv and track01_data_rescue.csv

Use available tools to run the pipeline and reference report artifacts.
""",
    tools=[read_dataset, run_data_quality_pipeline],
)

app = App(
    root_agent=root_agent,
    name="app",
)