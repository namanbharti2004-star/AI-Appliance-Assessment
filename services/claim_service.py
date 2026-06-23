"""
Feature 9: Claim History

SQLite-backed claim history storage.
Every inspection is saved with claim_id, timestamp, appliance, damage,
severity, fraud_score, repair_cost, etc.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "claim_history.db")


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db() -> None:
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            claim_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            source_type TEXT DEFAULT 'image',
            appliance TEXT,
            appliance_confidence REAL DEFAULT 0.0,
            damage_detected INTEGER DEFAULT 0,
            damage_type TEXT,
            damage_confidence REAL DEFAULT 0.0,
            damage_percentage REAL DEFAULT 0.0,
            severity TEXT DEFAULT 'None',
            missing_part_detected INTEGER DEFAULT 0,
            missing_part TEXT,
            ela_score REAL DEFAULT 0.0,
            metadata_risk_score REAL DEFAULT 0.0,
            fraud_score INTEGER DEFAULT 0,
            fraud_risk_level TEXT DEFAULT 'Low',
            condition_score INTEGER DEFAULT 100,
            grade TEXT DEFAULT 'A',
            repair_cost_min INTEGER DEFAULT 0,
            repair_cost_max INTEGER DEFAULT 0,
            repair_cost_display TEXT,
            repair_impact TEXT DEFAULT '',
            repairability TEXT DEFAULT '',
            recommended_action TEXT DEFAULT '',
            claim_score INTEGER DEFAULT 0,
            claim_risk TEXT DEFAULT 'Low',
            decision TEXT DEFAULT 'APPROVE',
            full_report TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_claim(report: Dict[str, Any]) -> str:
    _init_db()
    claim_id = report.get("claim_id") or str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    conn = _get_connection()

    full_report = json.dumps(report, default=str)
    repair = report.get("repair_cost_display") or report.get("repair_estimate", {}).get("estimated_repair_cost", "")

    conn.execute(
        """INSERT OR REPLACE INTO claims (
            claim_id, timestamp, source_type, appliance, appliance_confidence,
            damage_detected, damage_type, damage_confidence, damage_percentage, severity,
            missing_part_detected, missing_part,
            ela_score, metadata_risk_score, fraud_score, fraud_risk_level,
            condition_score, grade,
            repair_cost_min, repair_cost_max, repair_cost_display,
            repair_impact, repairability, recommended_action,
            claim_score, claim_risk, decision, full_report
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            claim_id,
            now,
            report.get("source_type", "image"),
            report.get("appliance"),
            report.get("appliance_confidence", 0.0),
            1 if report.get("damage_detected") else 0,
            report.get("damage_type"),
            report.get("damage_confidence", 0.0),
            report.get("damage_percentage", 0.0),
            report.get("severity", "None"),
            1 if report.get("missing_part_detected") else 0,
            report.get("missing_part"),
            report.get("ela_score", 0.0),
            report.get("metadata_risk_score", 0.0),
            report.get("fraud_score", 0),
            report.get("fraud_risk_level", "Low"),
            report.get("condition_score", 100),
            report.get("grade", "A"),
            report.get("repair_cost_min", 0),
            report.get("repair_cost_max", 0),
            repair,
            report.get("repair_impact", ""),
            report.get("repairability", ""),
            report.get("recommended_action", ""),
            report.get("claim_score", 0),
            report.get("claim_risk", "Low"),
            report.get("decision", "APPROVE"),
            full_report,
        ),
    )
    conn.commit()
    conn.close()
    return claim_id


def get_claims(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    _init_db()
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM claims ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_claim_by_id(claim_id: str) -> Optional[Dict[str, Any]]:
    _init_db()
    conn = _get_connection()
    row = conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_claim_stats() -> Dict[str, Any]:
    _init_db()
    conn = _get_connection()
    total = conn.execute("SELECT COUNT(*) as c FROM claims").fetchone()["c"]
    high_risk = conn.execute("SELECT COUNT(*) as c FROM claims WHERE claim_risk = 'High'").fetchone()["c"]
    avg_fraud = conn.execute("SELECT COALESCE(AVG(fraud_score),0) as a FROM claims").fetchone()["a"]
    avg_condition = conn.execute("SELECT COALESCE(AVG(condition_score),100) as a FROM claims").fetchone()["a"]
    conn.close()
    return {
        "total_claims": total,
        "high_risk_claims": high_risk,
        "avg_fraud_score": round(avg_fraud, 1),
        "avg_condition_score": round(avg_condition, 1),
    }
