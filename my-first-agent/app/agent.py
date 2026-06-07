# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ... [License text omitted for brevity] ...

import os
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types

from dotenv import load_dotenv

load_dotenv()




APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = APP_DIR / "track01_customers.csv"
FALLBACK_DATASET_PATH = APP_DIR / "track01_data_rescue.csv"
ALLOWED_DATASETS: dict[str, Path] = {
    "track01_customers.csv": DEFAULT_DATASET_PATH,
    "track01_data_rescue.csv": FALLBACK_DATASET_PATH,
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


def _agent_1_find_it(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []

    dup_mask = df.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        findings.append(
            Finding(
                finding_id="F-001",
                finding_type="duplicate_rows",
                severity=min(10, 4 + dup_count),
                reason=(
                    f"Detected {dup_count} duplicated records; duplicates can distort risk metrics and downstream actions."
                ),
                evidence={
                    "duplicate_count": dup_count,
                    "sample_indices": df.index[dup_mask].tolist()[:10],
                },
            )
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = int(outlier_mask.fillna(False).sum())
        if outlier_count > 0:
            findings.append(
                Finding(
                    finding_id=f"F-OUT-{col}",
                    finding_type="numeric_anomaly",
                    severity=min(10, 3 + outlier_count),
                    reason=(
                        f"Column '{col}' has {outlier_count} outliers by IQR rule; values outside [{lower:.2f}, {upper:.2f}] are suspicious."
                    ),
                    evidence={
                        "column": col,
                        "outlier_count": outlier_count,
                        "bounds": {"lower": float(lower), "upper": float(upper)},
                    },
                )
            )

    missing_by_col = df.isna().sum()
    missing_cols = missing_by_col[missing_by_col > 0]
    for col, cnt in missing_cols.items():
        ratio = float(cnt) / max(1, len(df))
        if ratio >= 0.2:
            findings.append(
                Finding(
                    finding_id=f"F-MISS-{col}",
                    finding_type="missing_data",
                    severity=min(10, 2 + int(ratio * 10)),
                    reason=(
                        f"Column '{col}' is missing in {cnt}/{len(df)} rows ({ratio:.1%}), reducing confidence in decisions."
                    ),
                    evidence={
                        "column": col,
                        "missing_count": int(cnt),
                        "missing_ratio": ratio,
                    },
                )
            )

    return findings


def _agent_2_rank_it(findings: list[Finding]) -> list[dict[str, Any]]:
    sorted_findings = sorted(findings, key=lambda f: (-f.severity, f.finding_id))
    ranked: list[dict[str, Any]] = []
    for rank, f in enumerate(sorted_findings, start=1):
        ranked.append(
            {
                "rank": rank,
                "finding": asdict(f),
                "ranking_reason": (
                    f"Ranked #{rank} because severity={f.severity}. Higher severity indicates greater business and data quality risk."
                ),
            }
        )
    return ranked


def _agent_3_act_on_it(ranked: list[dict[str, Any]], df: pd.DataFrame) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in ranked:
        finding = item["finding"]
        severity = int(finding["severity"])
        ftype = finding["finding_type"]

        if ftype == "duplicate_rows":
            action = "fix"
            detail = "Remove duplicate rows and retain first occurrence."
            reason = (
                "Duplicates are deterministic data-quality defects; safe auto-fix improves metric integrity."
            )
        elif severity >= 8:
            action = "escalate"
            detail = "Escalate to domain expert for approval before data changes."
            reason = (
                "High-severity findings can materially affect compliance or customer impact and need human oversight."
            )
        else:
            action = "flag"
            detail = "Flag record set for manual review in next QA cycle."
            reason = (
                "Medium/low severity does not justify unattended mutation; keeping an auditable flag is safer."
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
    lines.append("## Prioritized Findings")
    lines.append("")

    action_by_id = {a["finding_id"]: a for a in actions}
    for item in ranked:
        finding = item["finding"]
        action = action_by_id.get(finding["finding_id"], {})
        lines.append(
            f"- Rank {item['rank']}: {finding['finding_id']} ({finding['finding_type']}, severity={finding['severity']})"
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
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
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