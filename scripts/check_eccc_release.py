"""Yearly ECCC release watch for the GHGRP facility dataset.

Checks the ECCC Data Mart API for the GHGRP facility file and reports
whether the file on ECCC's side is newer than our latest raw snapshot
in data/raw/manifest.json.

No browser needed: the Data Mart exposes a plain JSON file API.
Network failures are reported as a check failure (exit 2), never as a
false "no update".

Manual fallback (if this script ever fails): open
https://open.canada.ca/data/en/dataset/a8ba14b7-7f23-462a-bdbb-83b0ef629823
in a browser, follow "View ECCC Data Mart", open the
greenhouse-gas-reporting-program-ghgrp-facility-greenhouse-gas-ghg-data
folder, and compare the last-modified date of
PDGES-GHGRP-GHGEmissionsGES-2004-Present.csv against the manifest.

Usage: python3 scripts/check_eccc_release.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_releases import load_manifest, latest_release

DATASET_PATH = ("substances/monitor/"
                "greenhouse-gas-reporting-program-ghgrp-facility-"
                "greenhouse-gas-ghg-data")
TARGET_FILE = "PDGES-GHGRP-GHGEmissionsGES-2004-Present.csv"
API = ("https://data-donnees.ec.gc.ca/api/path_contents?path="
       + DATASET_PATH.replace("/", "%2F"))


def eccc_last_modified() -> str | None:
    req = urllib.request.Request(API, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    for item in data.get("path_contents", []):
        if item.get("name") == TARGET_FILE:
            return item.get("last_modified")
    return None


def main() -> int:
    try:
        remote = eccc_last_modified()
    except Exception as e:  # noqa: BLE001 - network is inherently flaky
        print(f"RELEASE WATCH: check failed ({e}). "
              f"Use the manual fallback in this script's docstring.")
        return 2
    if not remote:
        print("RELEASE WATCH: target file not found in the ECCC listing. "
              "Use the manual fallback in this script's docstring.")
        return 2
    latest = latest_release()
    local = latest["retrieval_date"]
    print(f"ECCC file last modified: {remote}")
    print(f"Our latest snapshot: {latest['release_id']} "
          f"(retrieved {local})")
    if remote > local:
        print("RELEASE WATCH: a newer ECCC release exists. "
              "Plan the next refresh.")
        return 1
    print("RELEASE WATCH: our snapshot is current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
