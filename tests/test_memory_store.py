"""
Unit tests for MemoryStore — the handoff backbone.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.memory_store import MemoryStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        s = MemoryStore(str(db))
        yield s


def test_remember_and_recall(store):
    finding = {
        "record_id": "R-TEST-01",
        "issue_type": "exact_duplicate",
        "severity": "high",
        "raw_signals": {"signals_agreed": 2, "sigma_off": 0.0, "details": "x"},
        "current_value": "qty=100",
        "suggested_fix": "merge",
        "evidence": "dup",
        "impact_usd": 123.45,
        "stage": "found",
        "agent_log": [],
    }
    rid = store.remember(finding)
    assert rid > 0

    rows = store.recall(record_id="R-TEST-01")
    assert len(rows) == 1
    assert rows[0]["issue_type"] == "exact_duplicate"
    assert rows[0]["raw_signals"]["signals_agreed"] == 2


def test_recall_by_issue_type(store):
    for i in range(3):
        store.remember(
            {
                "record_id": f"R-{i}",
                "issue_type": "orphaned_customer",
                "severity": "med",
                "raw_signals": {},
                "current_value": "x",
                "suggested_fix": "y",
                "evidence": "z",
                "stage": "found",
                "agent_log": [],
            }
        )
    assert len(store.recall(issue_type="orphaned_customer")) == 3
    assert len(store.recall(issue_type="exact_duplicate")) == 0


def test_update_stage_and_agent_log(store):
    store.remember(
        {
            "record_id": "R-UP",
            "issue_type": "impossible_value",
            "severity": "high",
            "raw_signals": {},
            "current_value": "-5",
            "suggested_fix": "flip",
            "evidence": "neg",
            "stage": "found",
            "agent_log": [],
        }
    )
    store.update("R-UP", {"stage": "ranked", "agent_note": {"agent": "agent_2", "note": "scored 0.9"}})
    row = store.recall(record_id="R-UP")[0]
    assert row["stage"] == "ranked"
    assert len(row["agent_log"]) == 1
    assert row["agent_log"][0]["agent"] == "agent_2"


def test_forget(store):
    store.remember(
        {
            "record_id": "R-DEL",
            "issue_type": "unit_format_drift",
            "severity": "low",
            "raw_signals": {},
            "current_value": "x",
            "suggested_fix": "y",
            "evidence": "z",
            "stage": "found",
            "agent_log": [],
        }
    )
    assert store.count() == 1
    store.forget(record_id="R-DEL")
    assert store.count() == 0


def test_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "persist.db"
        s1 = MemoryStore(str(db))
        s1.remember(
            {
                "record_id": "R-P",
                "issue_type": "decimal_shift_weight",
                "severity": "high",
                "raw_signals": {"sigma_off": 9.5},
                "current_value": "650kg",
                "suggested_fix": "divide by 10",
                "evidence": "shift",
                "stage": "found",
                "agent_log": [],
            }
        )

        s2 = MemoryStore(str(db))
        rows = s2.recall(record_id="R-P")
        assert len(rows) == 1
        assert rows[0]["raw_signals"]["sigma_off"] == 9.5
