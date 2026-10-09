"""Diff the labeled facility data against the previous snapshot.

Appends alert entries to the alerts JSON when facilities appear,
disappear, change priority, or move emissions past a threshold.

Cron ready: argparse interface, exit 0 on success, exit 1 on error,
no side effects beyond the snapshot and alerts files.

Email delivery is not implemented. That is a follow up.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

ID_COL = "facility_id"
NAME_COL = "facility_name"
EMISSIONS_COL = "avg_annual_emissions"
PRIORITY_COL = "priority"


def _read_rows(path: str) -> dict:
    """Read the labeled CSV keyed by facility id."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            key = str(rec.get(ID_COL, "")).strip()
            if key:
                rows[key] = rec
    return rows


def _write_csv_atomic(path: str, fieldnames: list, rows: list) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".snapshot_",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _append_alerts(path: str, alerts: list) -> None:
    existing = []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            existing = data
    except (OSError, ValueError):
        existing = []
    existing.extend(alerts)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".alerts_",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _pct_change(old: float, new: float) -> float:
    if old <= 0:
        return 0.0
    return (new - old) / old * 100.0


def diff_snapshots(current: dict, previous: dict,
                   jump_pct: float) -> list:
    """Build alert entries from a current vs previous snapshot diff."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    alerts = []
    for key, rec in current.items():
        name = rec.get(NAME_COL, key)
        prev = previous.get(key)
        if prev is None:
            alerts.append({
                "ts": now,
                "severity": "med",
                "text": f"New facility above threshold: {name}.",
                "facility_id": key,
            })
            continue
        if rec.get(PRIORITY_COL) != prev.get(PRIORITY_COL):
            alerts.append({
                "ts": now,
                "severity": "med",
                "text": (f"Priority changed from {prev.get(PRIORITY_COL)} "
                         f"to {rec.get(PRIORITY_COL)} for {name}."),
                "facility_id": key,
            })
        try:
            old_e = float(prev.get(EMISSIONS_COL) or 0)
            new_e = float(rec.get(EMISSIONS_COL) or 0)
        except ValueError:
            continue
        move = _pct_change(old_e, new_e)
        if abs(move) >= jump_pct:
            direction = "rose" if move > 0 else "fell"
            alerts.append({
                "ts": now,
                "severity": "high" if move > 0 else "med",
                "text": (f"Emissions {direction} {abs(move):.1f} percent "
                         f"for {name}."),
                "facility_id": key,
            })
    for key, rec in previous.items():
        if key not in current:
            alerts.append({
                "ts": now,
                "severity": "med",
                "text": (f"Facility no longer above threshold: "
                         f"{rec.get(NAME_COL, key)}."),
                "facility_id": key,
            })
    return alerts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Diff facility data against the previous snapshot "
                    "and append alerts.")
    parser.add_argument("--data", default=str(BASE / "data"
                                              / "facility_labeled_national.csv"),
                        help="Labeled facility CSV to check.")
    parser.add_argument("--snapshot",
                        default=str(BASE / "data" / "watchlist_snapshot.csv"),
                        help="Snapshot CSV from the previous run.")
    parser.add_argument("--alerts",
                        default=str(BASE / "data" / "alerts.json"),
                        help="Alerts JSON to append to.")
    parser.add_argument("--jump-pct", type=float, default=5.0,
                        help="Emissions move percent that triggers an alert.")
    args = parser.parse_args(argv)

    try:
        current = _read_rows(args.data)
    except OSError as exc:
        print(f"error: cannot read data file: {exc}", file=sys.stderr)
        return 1

    if not os.path.exists(args.snapshot):
        fieldnames = [ID_COL, NAME_COL, EMISSIONS_COL, PRIORITY_COL]
        _write_csv_atomic(args.snapshot, fieldnames,
                          [{k: rec.get(k, "") for k in fieldnames}
                           for _, rec in sorted(current.items())])
        print(f"initialized snapshot with {len(current)} facilities, "
              "no alerts on first run")
        return 0

    try:
        previous = _read_rows(args.snapshot)
    except OSError as exc:
        print(f"error: cannot read snapshot: {exc}", file=sys.stderr)
        return 1

    alerts = diff_snapshots(current, previous, args.jump_pct)
    if alerts:
        _append_alerts(args.alerts, alerts)
    fieldnames = [ID_COL, NAME_COL, EMISSIONS_COL, PRIORITY_COL]
    _write_csv_atomic(args.snapshot, fieldnames,
                      [{k: rec.get(k, "") for k in fieldnames}
                       for _, rec in sorted(current.items())])
    print(f"checked {len(current)} facilities, {len(alerts)} new alerts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
