"""
Unit tests for the 6 Find-It detectors.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.memory_store import MemoryStore
from src.detectors import (
    detect_exact_duplicates,
    detect_orphaned_customers,
    detect_impossible_values,
    detect_near_duplicate_variants,
    detect_unit_format_drift,
    detect_decimal_shift_weights,
)


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        s = MemoryStore(str(db))
        yield s


def test_exact_duplicates(store):
    df = pd.DataFrame({
        "record_id": ["R-1", "R-2", "R-3"],
        "plant_id": ["A", "A", "B"],
        "part_number": ["BOLT-100", "BOLT-100", "BOLT-101"],
        "part_name": ["Bolt", "Bolt", "Bolt"],
        "quantity": [100, 100, 200],
        "unit": ["mm", "mm", "mm"],
        "weight_kg": [10.0, 10.0, 20.0],
        "production_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "ship_date": ["2024-01-05", "2024-01-05", "2024-01-06"],
        "customer_id": ["CU-1", "CU-1", "CU-2"],
        "status": ["shipped", "shipped", "pending"],
        "last_modified": ["2024-01-06", "2024-01-06", "2024-01-07"],
    })
    n = detect_exact_duplicates(df, store)
    assert n == 2  # R-1 and R-2 are exact duplicates
    assert store.count() == 2


def test_orphaned_customers(store):
    df = pd.DataFrame({
        "record_id": ["R-1", "R-2"],
        "plant_id": ["A", "A"],
        "part_number": ["BOLT-100", "BOLT-101"],
        "part_name": ["Bolt", "Bolt"],
        "quantity": [100, 200],
        "unit": ["mm", "mm"],
        "weight_kg": [10.0, 20.0],
        "production_date": ["2024-01-01", "2024-01-02"],
        "ship_date": ["2024-01-05", "2024-01-06"],
        "customer_id": ["CU-1", "CX-BAD"],
        "status": ["shipped", "pending"],
        "last_modified": ["2024-01-06", "2024-01-07"],
    })
    customers = pd.DataFrame({
        "customer_id": ["CU-1"],
        "company_name": ["Acme"],
        "region": ["North"],
    })
    n = detect_orphaned_customers(df, customers, store)
    assert n == 1
    rows = store.recall(issue_type="orphaned_customer")
    assert rows[0]["record_id"] == "R-2"


def test_impossible_values(store):
    df = pd.DataFrame({
        "record_id": ["R-1", "R-2", "R-3"],
        "plant_id": ["A", "A", "A"],
        "part_number": ["BOLT-100", "BOLT-101", "BOLT-102"],
        "part_name": ["Bolt", "Bolt", "Bolt"],
        "quantity": [-5, 100, 200],
        "unit": ["mm", "mm", "mm"],
        "weight_kg": [10.0, 600.0, 20.0],
        "production_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "ship_date": ["2024-01-05", "2024-01-06", "2024-01-07"],
        "customer_id": ["CU-1", "CU-1", "CU-2"],
        "status": ["shipped", "pending", "completed"],
        "last_modified": ["2024-01-06", "2024-01-07", "2024-01-08"],
    })
    n = detect_impossible_values(df, store)
    assert n == 2  # negative qty + extreme weight


def test_near_duplicate_variants(store):
    df = pd.DataFrame({
        "record_id": ["R-1", "R-2", "R-3"],
        "plant_id": ["A", "B", "A"],
        "part_number": ["BOLT-100", "BOLT-100", "BOLT-101"],
        "part_name": ["Bolt", "Bolt", "Bolt"],
        "quantity": [100, 100, 200],
        "unit": ["mm", "mm", "mm"],
        "weight_kg": [10.0, 10.0, 20.0],
        "production_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "ship_date": ["2024-01-05", "2024-01-06", "2024-01-07"],
        "customer_id": ["CU-1", "CU-2", "CU-3"],
        "status": ["shipped", "pending", "completed"],
        "last_modified": ["2024-01-06", "2024-01-07", "2024-01-08"],
    })
    n = detect_near_duplicate_variants(df, store)
    assert n == 2  # R-1 and R-2 share part+qty+weight


def test_unit_format_drift(store):
    df = pd.DataFrame({
        "record_id": ["R-1", "R-2", "R-3"],
        "plant_id": ["A", "A", "A"],
        "part_number": [" bolt-100 ", "BOLT-100", "bolt-100"],
        "part_name": ["Bolt", "Bolt", "Bolt"],
        "quantity": [100, 100, 100],
        "unit": ["mm", "mm", "mm"],
        "weight_kg": [10.0, 10.0, 10.0],
        "production_date": ["2024-01-01", "2024-01-01", "2024-01-01"],
        "ship_date": ["2024-01-05", "2024-01-05", "2024-01-05"],
        "customer_id": ["CU-1", "CU-1", "CU-1"],
        "status": ["shipped", "shipped", "shipped"],
        "last_modified": ["2024-01-06", "2024-01-06", "2024-01-06"],
    })
    n = detect_unit_format_drift(df, store)
    assert n == 2  # whitespace + lowercase


def test_decimal_shift_weights(store):
    df = pd.DataFrame({
        "record_id": ["R-1", "R-2", "R-3", "R-4"],
        "plant_id": ["A", "A", "A", "A"],
        "part_number": ["BOLT-100", "BOLT-100", "BOLT-100", "BOLT-101"],
        "part_name": ["Bolt", "Bolt", "Bolt", "Bolt"],
        "quantity": [100, 100, 100, 200],
        "unit": ["mm", "mm", "mm", "mm"],
        "weight_kg": [10.0, 10.0, 105.0, 20.0],
        "production_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "ship_date": ["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"],
        "customer_id": ["CU-1", "CU-1", "CU-1", "CU-2"],
        "status": ["shipped", "shipped", "shipped", "pending"],
        "last_modified": ["2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09"],
    })
    n = detect_decimal_shift_weights(df, store)
    assert n == 1  # R-3 is 10.5x median
    rows = store.recall(issue_type="decimal_shift_weight")
    assert rows[0]["record_id"] == "R-3"
