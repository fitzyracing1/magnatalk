"""Append-only JSONL decision log.

Every routing decision is recorded so you can audit what was picked and why,
and later fill in the ``actual`` field with what really happened (real cost,
runtime, provider behaviour) to compare against the estimate.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .router import Decision

DEFAULT_LOG_FILENAME = "magnatalk_decisions.jsonl"


def default_log_path() -> Path:
    return Path(os.environ.get("MAGNATALK_LOG_FILE", DEFAULT_LOG_FILENAME))


def decision_record(decision: Decision) -> dict:
    rec = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **decision.to_dict(),
        "actual": None,
    }
    return rec


def append_decision(decision: Decision, path: str | Path | None = None) -> dict:
    """Append a decision to the JSONL log and return the written record."""
    p = Path(path) if path else default_log_path()
    rec = decision_record(decision)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def read_decisions(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else default_log_path()
    if not p.exists():
        return []
    records = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def annotate_decision(
    decision_id: str, actual: dict, path: str | Path | None = None
) -> dict:
    """Set the ``actual`` field on a logged decision (e.g. real cost/runtime)."""
    p = Path(path) if path else default_log_path()
    records = read_decisions(p)
    hit = None
    for rec in records:
        if rec.get("id") == decision_id:
            rec["actual"] = actual
            hit = rec
    if hit is None:
        raise KeyError(f"no decision with id {decision_id!r} in {p}")
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    return hit
