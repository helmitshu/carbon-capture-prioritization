"""End-to-end refresh runner for one GHGRP release.

Runs the full v3 refresh in order:
  10_clean_national  -> schema, validation gate, cleaning v1,
                        quality report (without the model section)
  11_features_model_national -> facility aggregation, labels
  12_train_v3        -> model, panel tiers, reasons, metrics
Then patches the quality report with the model section and records
a refresh summary for tier-shift comparison on the next run.

Manual staging gate: this script prepares everything locally. Nothing
is pushed. Hakim reads the data quality report before any push.

Usage:
  python3 scripts/refresh_release.py [--release-id 2024]
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIDDEN = PROJECT_ROOT / "hidden_files"
SUMMARY = HIDDEN / "last_refresh_summary.json"

os.chdir(PROJECT_ROOT)


def run_step(script: str) -> None:
    print(f"--- {script} ---")
    r = subprocess.run([sys.executable, f"scripts/{script}"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(r.stdout[-1500:] if len(r.stdout) > 1500 else r.stdout)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError(f"{script} failed (exit {r.returncode})")


def main(release_id: str | None = None) -> None:
    from quality_report import write_quality_report

    # Step 1: clean.
    mod = importlib.import_module("10_clean_national")
    qr = mod.main(release_id)
    release_id = qr["release_id"]

    # Steps 2 and 3: features and model.
    run_step("11_features_model_national.py")
    run_step("12_train_v3.py")

    # Step 3b: re-register v3 so the registry never goes stale.
    # The app serves model/model_v3.pkl directly; the registry is the
    # lineage record. Idempotent: same version, new card and hashes.
    print("--- 05_register_model.py (v3, staging) ---")
    r = subprocess.run(
        [sys.executable, "scripts/05_register_model.py",
         "--version", "v3", "--stage", "staging",
         "--training-date", datetime.date.today().isoformat(),
         "--source-artifact", "model/model_v3.pkl",
         "--source-data", "data/facility_labeled_national.csv",
         "--training-script", "scripts/12_train_v3.py",
         "--metrics-json", "hidden_files/v3_metrics.json"],
        cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(r.stdout[-800:] if len(r.stdout) > 800 else r.stdout)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError("05_register_model.py failed")

    # Step 4: model section for the quality report.
    import pandas as pd
    fac = pd.read_csv(PROJECT_ROOT / "model" / "facilities_v3.csv")
    tiers = fac["verdict_tier"].value_counts().to_dict()
    tier_shift_note = ""
    if SUMMARY.exists():
        prev = json.loads(SUMMARY.read_text())
        ptiers = prev.get("tiers", {})
        parts = []
        for t in ("unanimous", "majority", "contested"):
            old = int(ptiers.get(t, 0))
            new = int(tiers.get(t, 0))
            if old != new:
                parts.append(f"{t}: {old} -> {new}")
        if parts:
            tier_shift_note = ("Tier shifts vs the previous refresh "
                               "(" + prev.get("release_id", "?") + "): "
                               + ", ".join(parts) + ".")
        else:
            tier_shift_note = ("Tier counts unchanged vs the previous "
                               "refresh (" + prev.get("release_id", "?")
                               + ").")
    qr["model"] = {
        "facilities_above_cutoff": int(len(fac)),
        "tiers": {k: int(v) for k, v in tiers.items()},
        "tier_shift_note": tier_shift_note,
    }
    md_path, json_path = write_quality_report(qr, release_id, HIDDEN)
    print(f"rewrote {md_path} with model section")

    summary = {
        "release_id": release_id,
        "facilities_above_cutoff": int(len(fac)),
        "tiers": {k: int(v) for k, v in tiers.items()},
        "clean_rows": qr["cleaning"]["rows_out"],
        "quarantine_rate": qr["validation"]["quarantine_rate"],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nFiles awaiting Hakim's push approval:")
    for f in qr["files"] + [
            "model/model_v3.pkl",
            "model/facilities_v3.csv",
            "data/facility_labeled_national.csv",
            "hidden_files/v3_metrics.json",
            "hidden_files/last_refresh_summary.json",
            str(md_path.relative_to(PROJECT_ROOT)),
            str(json_path.relative_to(PROJECT_ROOT)),
    ]:
        print(f"  {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-id", default=None)
    args = ap.parse_args()
    main(args.release_id)
