"""
MemoryStore — the handoff backbone for all 4 agents.

Simple, reliable SQLite backend. No external service.
Survives restarts. Supports full recall() for downstream agents.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any
from datetime import datetime


class MemoryStore:
    """
    SQLite-backed store for findings / agent notes.

    Each row holds a JSON blob so the shared schema can evolve
    without ALTER TABLE every time a teammate adds a field.
    """

    def __init__(self, db_path: str = "data/memory.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    #  internals
    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS findings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id   TEXT NOT NULL,
                    issue_type  TEXT NOT NULL,
                    severity    TEXT NOT NULL CHECK(severity IN ('low','med','high')),
                    raw_signals TEXT NOT NULL,          -- JSON
                    current_value TEXT,
                    suggested_fix TEXT,
                    evidence    TEXT,
                    impact_usd  REAL,
                    stage       TEXT NOT NULL DEFAULT 'found',
                    agent_log   TEXT NOT NULL DEFAULT '[]', -- JSON array
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_record_id ON findings(record_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_issue_type ON findings(issue_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_stage ON findings(stage)"
            )
            conn.commit()

    # ------------------------------------------------------------------ #
    #  public API
    # ------------------------------------------------------------------ #
    def remember(self, obj: dict[str, Any]) -> int:
        """
        Store a finding / agent note.

        Returns the auto-generated row id.
        """
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO findings
                    (record_id, issue_type, severity, raw_signals,
                     current_value, suggested_fix, evidence, impact_usd,
                     stage, agent_log, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    obj["record_id"],
                    obj["issue_type"],
                    obj["severity"],
                    json.dumps(obj.get("raw_signals", {})),
                    obj.get("current_value"),
                    obj.get("suggested_fix"),
                    obj.get("evidence"),
                    obj.get("impact_usd"),
                    obj.get("stage", "found"),
                    json.dumps(obj.get("agent_log", [])),
                    now,
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def recall(
        self,
        *,
        record_id: str | None = None,
        part_number: str | None = None,
        plant_id: str | None = None,
        issue_type: str | None = None,
        stage: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query findings by any combination of filters.
        Returns list of full JSON objects (newest first).
        """
        # part_number and plant_id live inside raw_signals JSON or record_id.
        # We support them by post-filtering for now (dataset is 5k rows).
        clauses: list[str] = []
        params: list[Any] = []

        if record_id:
            clauses.append("record_id = ?")
            params.append(record_id)
        if issue_type:
            clauses.append("issue_type = ?")
            params.append(issue_type)
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        sql = f"SELECT * FROM findings {where} ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            blob = dict(row)
            blob["raw_signals"] = json.loads(blob["raw_signals"])
            blob["agent_log"] = json.loads(blob["agent_log"])
            # post-filter part_number / plant_id if requested
            if part_number and blob.get("raw_signals", {}).get("part_number") != part_number:
                continue
            if plant_id and blob.get("raw_signals", {}).get("plant_id") != plant_id:
                continue
            results.append(blob)
        return results

    def update(self, record_id: str, patch: dict[str, Any]) -> int:
        """
        Append agent reasoning or change stage.
        Returns number of rows updated (0 or 1).
        """
        # fetch existing
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM findings WHERE record_id = ? ORDER BY created_at DESC LIMIT 1",
                (record_id,),
            ).fetchone()
            if not row:
                return 0

            blob = dict(row)
            agent_log: list[dict[str, Any]] = json.loads(blob["agent_log"])

            # append to agent_log if provided
            if "agent_note" in patch:
                agent_log.append(
                    {
                        "agent": patch["agent_note"].get("agent", "unknown"),
                        "note": patch["agent_note"].get("note", ""),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            # stage transition
            new_stage = patch.get("stage", blob["stage"])
            new_severity = patch.get("severity", blob["severity"])
            new_impact = patch.get("impact_usd", blob["impact_usd"])
            new_suggested = patch.get("suggested_fix", blob["suggested_fix"])

            conn.execute(
                """
                UPDATE findings
                SET stage = ?, severity = ?, impact_usd = ?,
                    suggested_fix = ?, agent_log = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_stage,
                    new_severity,
                    new_impact,
                    new_suggested,
                    json.dumps(agent_log),
                    datetime.utcnow().isoformat(),
                    blob["id"],
                ),
            )
            conn.commit()
            return 1

    def forget(self, **filters: Any) -> int:
        """
        Clear findings before a fresh run.
        If no filters given, wipes the entire table.
        Returns number of rows deleted.
        """
        if not filters:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM findings")
                conn.commit()
                return cur.rowcount

        # build WHERE from same keys as recall
        clauses: list[str] = []
        params: list[Any] = []
        for k, v in filters.items():
            if k in ("record_id", "issue_type", "stage", "severity"):
                clauses.append(f"{k} = ?")
                params.append(v)
        if not clauses:
            return 0

        where = " AND ".join(clauses)
        with self._connect() as conn:
            cur = conn.execute(f"DELETE FROM findings WHERE {where}", params)
            conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------ #
    #  helpers for detectors
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM findings").fetchone()
            return row[0] if row else 0

    def get_by_issue_type(self, issue_type: str) -> list[dict[str, Any]]:
        return self.recall(issue_type=issue_type)
