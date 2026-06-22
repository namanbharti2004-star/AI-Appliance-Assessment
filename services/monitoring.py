"""
Monitoring & Observability Module.

Tracks:
- Inference timing per module
- Model decisions with confidence
- Error rates and failure types
- Processing time trends
- Model version tracking
- Audit trail for every prediction
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional
from collections import defaultdict


MONITOR_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "monitor.db",
)


def _ensure_db() -> None:
    os.makedirs(os.path.dirname(MONITOR_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(MONITOR_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inference_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            module TEXT,
            operation TEXT,
            duration_ms REAL,
            success INTEGER,
            error TEXT,
            model_version TEXT,
            confidence REAL,
            metadata TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_name TEXT,
            model_version TEXT,
            operation TEXT,
            avg_confidence REAL,
            total_calls INTEGER,
            error_rate REAL,
            avg_duration_ms REAL
        )
    """)
    conn.commit()
    conn.close()


_ensure_db()


@dataclass
class InferenceEvent:
    module: str
    operation: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    model_version: str = "3.0.0"
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class Monitor:
    _instance: Optional[Monitor] = None
    _session_stats: Dict[str, list] = defaultdict(list)

    def __new__(cls) -> Monitor:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        _ensure_db()

    def log_inference(self, event: InferenceEvent) -> None:
        try:
            conn = sqlite3.connect(MONITOR_DB_PATH)
            conn.execute(
                "INSERT INTO inference_log (timestamp, module, operation, duration_ms, success, error, model_version, confidence, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(),
                    event.module,
                    event.operation,
                    round(event.duration_ms, 2),
                    1 if event.success else 0,
                    event.error,
                    event.model_version,
                    event.confidence,
                    json.dumps(event.metadata) if event.metadata else None,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        key = f"{event.module}:{event.operation}"
        self._session_stats[key].append({
            "duration_ms": event.duration_ms,
            "success": event.success,
            "confidence": event.confidence,
        })

    @contextmanager
    def track(self, module: str, operation: str, model_version: str = "3.0.0") -> Iterator[None]:
        start = time.perf_counter()
        error: Optional[str] = None
        success = True
        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            duration = (time.perf_counter() - start) * 1000
            self.log_inference(InferenceEvent(
                module=module,
                operation=operation,
                duration_ms=duration,
                success=success,
                error=error,
                model_version=model_version,
            ))

    def get_session_stats(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, events in self._session_stats.items():
            durations = [e["duration_ms"] for e in events]
            successes = [e["success"] for e in events]
            confidences = [e["confidence"] for e in events if e["confidence"] is not None]
            result[key] = {
                "total_calls": len(events),
                "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0,
                "max_duration_ms": round(max(durations), 2) if durations else 0,
                "min_duration_ms": round(min(durations), 2) if durations else 0,
                "success_rate": round(sum(successes) / len(successes) * 100, 1) if successes else 0,
                "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            }
        return result

    def get_recent_errors(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            conn = sqlite3.connect(MONITOR_DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM inference_log WHERE success = 0 ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def get_performance_summary(self) -> Dict[str, Any]:
        try:
            conn = sqlite3.connect(MONITOR_DB_PATH)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT module, operation, COUNT(*) as calls, AVG(duration_ms) as avg_dur, SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as errors FROM inference_log GROUP BY module, operation ORDER BY calls DESC"
            ).fetchall()
            conn.close()
            return {
                "modules": [dict(r) for r in rows],
                "total_calls": sum(r["calls"] for r in rows),
                "total_errors": sum(r["errors"] for r in rows),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_module_timing(self, module: str, operation: str) -> Dict[str, float]:
        key = f"{module}:{operation}"
        events = self._session_stats.get(key, [])
        durations = [e["duration_ms"] for e in events]
        if not durations:
            return {"avg": 0, "max": 0, "min": 0, "count": 0}
        return {
            "avg": round(sum(durations) / len(durations), 2),
            "max": round(max(durations), 2),
            "min": round(min(durations), 2),
            "count": len(durations),
        }


monitor = Monitor()
