"""
Agent 1 — Find-It runner.

Loads the Kaggle dataset, runs all 6 detectors, and writes findings
to the MemoryStore. Downstream agents (Rank/Act/Explain) recall()
from this same store.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory_store import MemoryStore
from src.detectors import (
    detect_exact_duplicates,
    detect_orphaned_customers,
    detect_impossible_values,
    detect_near_duplicate_variants,
    detect_unit_format_drift,
    detect_decimal_shift_weights,
)


def run(data_dir: str = "data", db_path: str = "data/memory.db") -> dict[str, int]:
    """
    Run all detectors and return counts per issue_type.
    """
    # ------------------------------------------------------------------ #
    #  load data
    # ------------------------------------------------------------------ #
    data_path = Path(data_dir)
    df = pd.read_csv(data_path / "track01_data_rescue.csv")
    customers = pd.read_csv(data_path / "track01_customers.csv")

    print(f"[Agent 1] Loaded {len(df):,} production records + {len(customers):,} customers")

    # ------------------------------------------------------------------ #
    #  init memory
    # ------------------------------------------------------------------ #
    store = MemoryStore(db_path=db_path)
    # clean slate for fresh run
    wiped = store.forget()
    if wiped:
        print(f"[Agent 1] Cleared {wiped} previous findings from memory")

    # ------------------------------------------------------------------ #
    #  run detectors
    # ------------------------------------------------------------------ #
    counts: dict[str, int] = {}

    counts["exact_duplicate"] = detect_exact_duplicates(df, store)
    counts["orphaned_customer"] = detect_orphaned_customers(df, customers, store)
    counts["impossible_value"] = detect_impossible_values(df, store)
    counts["near_duplicate_variant"] = detect_near_duplicate_variants(df, store)
    counts["unit_format_drift"] = detect_unit_format_drift(df, store)
    counts["decimal_shift_weight"] = detect_decimal_shift_weights(df, store)

    total = sum(counts.values())
    print(f"[Agent 1] Findings written: {total}")
    for issue, n in counts.items():
        print(f"  - {issue}: {n}")

    # ------------------------------------------------------------------ #
    #  sanity check: recall() works for downstream agents
    # ------------------------------------------------------------------ #
    sample = store.recall(issue_type="exact_duplicate")
    sample = sample[:3]
    print(f"[Agent 1] recall() sanity check: {len(sample)} exact_duplicate rows returned")

    return counts


if __name__ == "__main__":
    run()
