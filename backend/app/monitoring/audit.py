"""Retraining audit log.

Every retrain attempt — accepted or rejected — appends one JSON line to
``reports/retraining/history.jsonl``. The log is the durable answer to
the examiner's "show me your governance trail" question: for any
deployed model you can point to the run that promoted it, the prior
champion's metrics, and the McNemar p-value that justified the swap.

We deliberately keep the storage primitive (one JSONL file) so the file
is grep-able from a shell, parseable in pandas, and trivial to back up.
For high-volume production this would graduate to a database, but for a
final-year project a flat file is the right level of complexity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEntry:
    """One line in ``history.jsonl``. Keep fields stable — UI parses these."""

    timestamp: str
    trigger: str
    promoted: bool
    reason: str
    champion_macro_f1: float
    challenger_macro_f1: float
    macro_f1_delta: float
    per_class_deltas: dict[str, float]
    mcnemar_p_value: float | None
    mcnemar_b: int | None
    mcnemar_c: int | None
    mcnemar_significant: bool | None
    n_test: int
    registered_model_name: str | None = None
    registered_model_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "promoted": self.promoted,
            "reason": self.reason,
            "champion_macro_f1": self.champion_macro_f1,
            "challenger_macro_f1": self.challenger_macro_f1,
            "macro_f1_delta": self.macro_f1_delta,
            "per_class_deltas": self.per_class_deltas,
            "mcnemar_p_value": self.mcnemar_p_value,
            "mcnemar_b": self.mcnemar_b,
            "mcnemar_c": self.mcnemar_c,
            "mcnemar_significant": self.mcnemar_significant,
            "n_test": self.n_test,
            "registered_model_name": self.registered_model_name,
            "registered_model_version": self.registered_model_version,
        }


def append_audit(entry: AuditEntry, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.as_dict()) + "\n")


def read_audit(log_path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Return audit entries newest-first; ``limit`` caps the response size."""
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines rather than fail — we want the audit
                # endpoint to stay responsive even if a write was truncated.
                continue
    rows.reverse()
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
