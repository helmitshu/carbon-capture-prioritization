"""Phase 5d: promote a staging model version to production (backlog item 3).

The staging slot: a version sits in staging, gets exercised there
(validation gate + staging API traffic), and only then may the
production pointer move. This script is the ONLY sanctioned way to move
it. It refuses to touch the registry unless every precondition holds.

Preconditions, in order:
  1. The version exists in the registry and its stage is "staging".
     (No skipping the line: archived/production versions cannot be
     promoted directly.)
  2. The validation gate (06_validate_model.py) PASSES for the version.
     A FAIL aborts before anything is written.

On success:
  - the current production version (if any) moves to "archived",
  - the candidate's stage becomes "production",
  - the registry's top-level "production" pointer moves to it.

The deploy step is deliberately NOT automated: after promotion, set
MODEL_VERSION to the new version on the production API service and
redeploy it. The script prints exactly that reminder.

Usage:
    .venv/bin/python scripts/07_promote_model.py --version v2 --dry-run
    .venv/bin/python scripts/07_promote_model.py --version v2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = PROJECT_ROOT / "model" / "registry"
REGISTRY_INDEX = REGISTRY_DIR / "registry.json"


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "validate_model", PROJECT_ROOT / "scripts" / "06_validate_model.py")
    module = importlib.util.module_from_spec(spec)
    # Must be registered before exec: the module uses @dataclass, which
    # looks up sys.modules[cls.__module__] at class-creation time.
    sys.modules["validate_model"] = module
    spec.loader.exec_module(module)
    return module


def _read_index() -> dict:
    return json.loads(REGISTRY_INDEX.read_text())


def _write_index(index: dict) -> None:
    """Atomic write: temp file + rename, so a crash can never leave a
    half-written registry index behind."""
    tmp = REGISTRY_INDEX.with_suffix(REGISTRY_INDEX.suffix + ".tmp")
    tmp.write_text(json.dumps(index, indent=2) + "\n")
    os.replace(tmp, REGISTRY_INDEX)


def _set_card_stage(version: str, stage: str) -> None:
    """Keep the version's card in sync with the registry index stage.

    Without this the card keeps saying "staging" after promotion and the
    two records disagree about what is live.
    """
    card_path = REGISTRY_DIR / f"model_{version}.card.json"
    if not card_path.exists():
        return
    card = json.loads(card_path.read_text())
    card["stage"] = stage
    card_path.write_text(json.dumps(card, indent=2) + "\n")


def promote(version: str, dry_run: bool = False) -> dict:
    """Promote a staging version. Returns a result dict; raises on refusal."""
    index = _read_index()
    versions = index.get("versions", {})
    if version not in versions:
        raise SystemExit(f"refusing: version '{version}' not in registry")
    entry = versions[version]
    if entry.get("stage") != "staging":
        raise SystemExit(
            f"refusing: version '{version}' is '{entry.get('stage')}', "
            f"not 'staging'")

    gate = _load_gate()
    # Bars come from the gate module: one source of truth, no local copy
    # to drift out of sync with ACCEPTANCE_CRITERIA.md.
    bars = dict(gate.DEFAULT_BARS)

    previous = index.get("production")
    if previous == version:
        raise SystemExit(f"refusing: '{version}' is already production")

    report = gate.run_gate(version, bars)
    if not report.passed:
        failed = [c.name for c in report.checks if not c.passed]
        raise SystemExit(
            f"refusing: validation gate FAILED for '{version}': "
            f"{', '.join(failed)}")

    result = {"promoted": version, "previous_production": previous,
              "dry_run": dry_run}
    if dry_run:
        print(f"dry run: gate PASSED for '{version}'; "
              f"would move production pointer "
              f"{previous!r} -> '{version}'")
        return result

    if previous and previous in versions:
        versions[previous]["stage"] = "archived"
        _set_card_stage(previous, "archived")
    entry["stage"] = "production"
    entry["promoted_date"] = date.today().isoformat()
    _set_card_stage(version, "production")
    index["production"] = version
    _write_index(index)
    print(f"promoted '{version}' to production "
          f"(previous: {previous!r})")
    print(f"NEXT: set MODEL_VERSION={version} on the production API "
          f"service and redeploy it.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a staging registry version to production, "
                    "gated on the validation gate.")
    parser.add_argument("--version", required=True,
                        help="staging version to promote (e.g. v2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="check preconditions without writing")
    args = parser.parse_args()
    promote(args.version, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
