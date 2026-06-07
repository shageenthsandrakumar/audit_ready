"""
Find-It detectors (Agent 1 / Step 1).

Six data-quality detectors that emit the shared JSON contract into MemoryStore.

EASY tier:
  1. exact_duplicates     — all columns (except record_id) identical
  2. orphaned_customers   — customer_id not in lookup table
  3. impossible_values    — negative quantity, weight > 500 kg

SUBTLE tier (per-part statistical baselining):
  4. near_duplicate_variants — same part_number + quantity + weight_kg
  5. unit_format_drift       — whitespace / case drift in part_number
  6. decimal_shift_weights   — weight_kg 10x or 0.1x part median

Each detector returns a list of dicts matching the shared contract.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import numpy as np

from src.memory_store import MemoryStore


# --------------------------------------------------------------------------- #
#  shared contract helper
# --------------------------------------------------------------------------- #
def _finding(
    record_id: str,
    issue_type: str,
    severity: str,
    raw_signals: dict[str, Any],
    current_value: str,
    suggested_fix: str,
    evidence: str,
    impact_usd: float | None = None,
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "issue_type": issue_type,
        "severity": severity,
        "raw_signals": raw_signals,
        "current_value": current_value,
        "suggested_fix": suggested_fix,
        "evidence": evidence,
        "impact_usd": impact_usd,
        "stage": "found",
        "agent_log": [],
    }


# --------------------------------------------------------------------------- #
#  1. exact_duplicates
# --------------------------------------------------------------------------- #
def detect_exact_duplicates(df: pd.DataFrame, store: MemoryStore) -> int:
    """
    Rows where every column except record_id is identical.
    Each duplicate row gets its own finding.
    """
    cols = [c for c in df.columns if c != "record_id"]
    dup_mask = df.duplicated(subset=cols, keep=False)
    dups = df[dup_mask]

    count = 0
    for _, row in dups.iterrows():
        # build a fingerprint of the duplicate group
        fingerprint = " | ".join(f"{c}={row[c]}" for c in cols[:6])
        finding = _finding(
            record_id=str(row["record_id"]),
            issue_type="exact_duplicate",
            severity="high",
            raw_signals={
                "signals_agreed": 2,
                "sigma_off": 0.0,
                "details": fingerprint,
                "part_number": row["part_number"],
                "plant_id": row["plant_id"],
            },
            current_value=f"qty={row['quantity']}, weight={row['weight_kg']}kg",
            suggested_fix="Merge duplicate records; keep one master record_id.",
            evidence=(
                f"Another record shares identical plant, part, quantity, weight, "
                f"customer, and dates. This inflates inventory counts."
            ),
            impact_usd=round(row["quantity"] * 12.50, 2),  # rough unit cost proxy
        )
        store.remember(finding)
        count += 1
    return count


# --------------------------------------------------------------------------- #
#  2. orphaned_customers
# --------------------------------------------------------------------------- #
def detect_orphaned_customers(
    df: pd.DataFrame, customers_df: pd.DataFrame, store: MemoryStore
) -> int:
    """
    customer_id values present in data but missing from customer lookup.
    """
    valid_ids = set(customers_df["customer_id"].unique())
    orphan_mask = ~df["customer_id"].isin(valid_ids)
    orphans = df[orphan_mask]

    count = 0
    for _, row in orphans.iterrows():
        finding = _finding(
            record_id=str(row["record_id"]),
            issue_type="orphaned_customer",
            severity="med",
            raw_signals={
                "signals_agreed": 1,
                "sigma_off": 0.0,
                "details": f"customer_id={row['customer_id']} not in lookup",
                "part_number": row["part_number"],
                "plant_id": row["plant_id"],
            },
            current_value=str(row["customer_id"]),
            suggested_fix="Verify customer in CRM; add to lookup or reassign.",
            evidence=(
                f"Customer ID '{row['customer_id']}' does not exist in the "
                f"customer master file. Shipments to unknown customers risk "
                f"revenue-recognition errors."
            ),
            impact_usd=round(row["quantity"] * 25.0, 2),
        )
        store.remember(finding)
        count += 1
    return count


# --------------------------------------------------------------------------- #
#  3. impossible_values
# --------------------------------------------------------------------------- #
def detect_impossible_values(df: pd.DataFrame, store: MemoryStore) -> int:
    """
    Negative quantities or weights exceeding physical limits (>500 kg).
    """
    count = 0

    # negative quantity
    neg_qty = df[df["quantity"] < 0]
    for _, row in neg_qty.iterrows():
        finding = _finding(
            record_id=str(row["record_id"]),
            issue_type="impossible_value",
            severity="high",
            raw_signals={
                "signals_agreed": 1,
                "sigma_off": abs(float(row["quantity"])),
                "details": f"negative_quantity={row['quantity']}",
                "part_number": row["part_number"],
                "plant_id": row["plant_id"],
            },
            current_value=str(row["quantity"]),
            suggested_fix="Investigate source transaction; flip sign or void record.",
            evidence=(
                f"Quantity is negative ({row['quantity']}), which is impossible "
                f"for a physical part. Likely a data-entry sign error."
            ),
            impact_usd=abs(round(row["quantity"] * 15.0, 2)),
        )
        store.remember(finding)
        count += 1

    # extreme weight
    extreme_w = df[df["weight_kg"] > 500]
    for _, row in extreme_w.iterrows():
        finding = _finding(
            record_id=str(row["record_id"]),
            issue_type="impossible_value",
            severity="high",
            raw_signals={
                "signals_agreed": 1,
                "sigma_off": float(row["weight_kg"]),
                "details": f"extreme_weight={row['weight_kg']}kg",
                "part_number": row["part_number"],
                "plant_id": row["plant_id"],
            },
            current_value=f"{row['weight_kg']} kg",
            suggested_fix="Re-weigh part or correct unit (grams vs kilograms).",
            evidence=(
                f"Weight {row['weight_kg']} kg exceeds the 500 kg plant-handling "
                f"limit for any single part in this dataset."
            ),
            impact_usd=round(row["weight_kg"] * 10.0, 2),
        )
        store.remember(finding)
        count += 1

    return count


# --------------------------------------------------------------------------- #
#  4. near_duplicate_variants  (subtle — per-part baselining)
# --------------------------------------------------------------------------- #
def detect_near_duplicate_variants(df: pd.DataFrame, store: MemoryStore) -> int:
    """
    Rows with identical part_number + quantity + weight_kg but different
    record_id. These are near-duplicates that may represent accidental
    re-entries with tiny metadata differences.
    """
    # group by the business-key triple
    grouped = df.groupby(["part_number", "quantity", "weight_kg"])
    count = 0

    for (part, qty, wgt), group in grouped:
        if len(group) < 2:
            continue
        # severity scales with how many clones exist
        severity = "high" if len(group) >= 3 else "med"
        for _, row in group.iterrows():
            finding = _finding(
                record_id=str(row["record_id"]),
                issue_type="near_duplicate_variant",
                severity=severity,
                raw_signals={
                    "signals_agreed": len(group),
                    "sigma_off": 0.0,
                    "details": f"clone_group_size={len(group)}",
                    "part_number": part,
                    "plant_id": row["plant_id"],
                },
                current_value=f"qty={qty}, weight={wgt}kg",
                suggested_fix="Deduplicate: retain newest record_id, archive clones.",
                evidence=(
                    f"Record shares exact part_number, quantity, and weight with "
                    f"{len(group)-1} other record(s). This suggests a re-entry or "
                    f"EDI duplicate rather than a true separate shipment."
                ),
                impact_usd=round(qty * 12.50 * (len(group) - 1), 2),
            )
            store.remember(finding)
            count += 1
    return count


# --------------------------------------------------------------------------- #
#  5. unit_format_drift  (subtle — whitespace / case drift)
# --------------------------------------------------------------------------- #
def detect_unit_format_drift(df: pd.DataFrame, store: MemoryStore) -> int:
    """
    part_number values with leading/trailing whitespace or inconsistent
    casing. These break joins and cause inventory splits.
    """
    # whitespace drift
    ws_mask = df["part_number"].str.contains(r"^\s|\s$", na=False, regex=True)
    ws_rows = df[ws_mask]

    count = 0
    for _, row in ws_rows.iterrows():
        clean = row["part_number"].strip()
        finding = _finding(
            record_id=str(row["record_id"]),
            issue_type="unit_format_drift",
            severity="med",
            raw_signals={
                "signals_agreed": 1,
                "sigma_off": 0.0,
                "details": f"raw_part_number={repr(row['part_number'])}, clean={clean}",
                "part_number": row["part_number"],
                "plant_id": row["plant_id"],
            },
            current_value=row["part_number"],
            suggested_fix=f"Trim whitespace → '{clean}'.",
            evidence=(
                f"part_number contains leading/trailing spaces: "
                f"{repr(row['part_number'])}. This causes JOIN mismatches "
                f"and splits the same physical part into two inventory buckets."
            ),
            impact_usd=round(row["quantity"] * 5.0, 2),
        )
        store.remember(finding)
        count += 1

    # case drift (lowercase variants that don't match uppercase norm)
    case_mask = (df["part_number"] != df["part_number"].str.upper()) & (~ws_mask)
    case_rows = df[case_mask]
    for _, row in case_rows.iterrows():
        canonical = row["part_number"].strip().upper()
        finding = _finding(
            record_id=str(row["record_id"]),
            issue_type="unit_format_drift",
            severity="low",
            raw_signals={
                "signals_agreed": 1,
                "sigma_off": 0.0,
                "details": f"raw_part_number={repr(row['part_number'])}, canonical={canonical}",
                "part_number": row["part_number"],
                "plant_id": row["plant_id"],
            },
            current_value=row["part_number"],
            suggested_fix=f"Normalize case → '{canonical}'.",
            evidence=(
                f"part_number uses lowercase or mixed case: {repr(row['part_number'])}. "
                f"Standard is ALL-CAPS with hyphen/underscore. Case drift breaks "
                f"exact-match lookups in ERP systems."
            ),
            impact_usd=round(row["quantity"] * 3.0, 2),
        )
        store.remember(finding)
        count += 1

    return count


# --------------------------------------------------------------------------- #
#  6. decimal_shift_weights  (subtle — per-part σ baselining)
# --------------------------------------------------------------------------- #
def detect_decimal_shift_weights(df: pd.DataFrame, store: MemoryStore) -> int:
    """
    Weights that are ~10x or ~0.1x the per-part median, signalling a
    decimal-place error (e.g. 65.1 kg entered as 651.0 kg).
    """
    # compute per-part median and std
    stats = df.groupby("part_number")["weight_kg"].agg(["median", "std"]).reset_index()
    stats.columns = ["part_number", "median_w", "std_w"]
    merged = df.merge(stats, on="part_number", how="left")

    # ratio to median
    merged["ratio"] = merged["weight_kg"] / merged["median_w"]

    # flag 10x up or 10x down (with a tolerance band)
    shift_mask = (merged["ratio"] > 8.0) | (merged["ratio"] < 0.15)
    shifted = merged[shift_mask]

    count = 0
    for _, row in shifted.iterrows():
        ratio = float(row["ratio"])
        direction = "high" if ratio > 1 else "low"
        # sigma_off = how many stds away from median
        sigma = float(row["std_w"]) if pd.notna(row["std_w"]) and row["std_w"] > 0 else 1.0
        sigma_off = abs(row["weight_kg"] - row["median_w"]) / sigma

        finding = _finding(
            record_id=str(row["record_id"]),
            issue_type="decimal_shift_weight",
            severity="high",
            raw_signals={
                "signals_agreed": 1,
                "sigma_off": round(sigma_off, 2),
                "details": (
                    f"weight={row['weight_kg']}kg, median={round(row['median_w'],2)}kg, "
                    f"ratio={round(ratio,2)}"
                ),
                "part_number": row["part_number"],
                "plant_id": row["plant_id"],
            },
            current_value=f"{row['weight_kg']} kg",
            suggested_fix=(
                f"Divide weight by 10 → {round(row['weight_kg']/10,2)} kg "
                f"(matches part median {round(row['median_w'],2)} kg)."
            ),
            evidence=(
                f"Weight {row['weight_kg']} kg is ~{round(ratio,1)}x the part's "
                f"historical median ({round(row['median_w'],2)} kg). This is a "
                f"classic decimal-place shift (e.g. 651.2 entered as 65.12 or vice versa)."
            ),
            impact_usd=round(row["quantity"] * abs(row["weight_kg"] - row["median_w"]) * 2.5, 2),
        )
        store.remember(finding)
        count += 1
    return count
