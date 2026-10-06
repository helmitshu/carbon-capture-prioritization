"""Data quality report writer.

Every refresh writes two files:
  hidden_files/data_quality_report_<release>.md    human readable
  hidden_files/data_quality_report_<release>.json  machine readable

The markdown report is a consulting asset: it reads as a plain account
of what came in, what was checked, what was held back, and what
changed. Commas and periods only, no em-dashes.
"""
from __future__ import annotations

import json
from pathlib import Path


def _fmt(n) -> str:
    return f"{n:,}"


def write_quality_report(report: dict, release_id: str,
                         out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"data_quality_report_{release_id}.md"
    json_path = out_dir / f"data_quality_report_{release_id}.json"

    schema = report["schema"]
    val = report["validation"]
    cleaning = report["cleaning"]
    recon = report["reconciliation"]
    outliers = report["outliers"]
    model = report.get("model", {})

    lines = []
    lines.append(f"# Data quality report, GHGRP release {release_id}")
    lines.append("")
    lines.append(f"Source: {report['source']}")
    lines.append(f"Retrieved: {report['retrieval_date']}")
    lines.append(f"Cleaning rules: version {report['cleaning_version']}")
    lines.append("")

    lines.append("## What came in")
    lines.append("")
    lines.append(f"Raw rows: {_fmt(val['rows_in'])}. "
                 f"Years covered: {report['years_covered'][0]} to "
                 f"{report['years_covered'][1]}.")
    lines.append(f"Schema check: {schema['actual_columns']} columns, "
                 f"expected {schema['expected_columns']}. "
                 f"Missing: {len(schema['missing_columns'])}. "
                 f"Extra: {len(schema['extra_columns'])}.")
    lines.append("")

    lines.append("## Validation gate")
    lines.append("")
    lines.append(f"Rows quarantined: {_fmt(val['rows_quarantined'])} "
                 f"({val['quarantine_rate']:.2%} of raw). "
                 f"Stop threshold: "
                 f"{val['quarantine_stop_threshold']:.0%}.")
    if val["quarantine_reasons"]:
        lines.append("Reasons:")
        for reason, count in sorted(val["quarantine_reasons"].items(),
                                    key=lambda kv: -kv[1]):
            lines.append(f"  {reason}: {_fmt(count)}")
    else:
        lines.append("No rows quarantined. Every row passed validation.")
    lines.append(f"Rows into cleaning: {_fmt(val['rows_clean'])}.")
    lines.append("")

    lines.append("## Cleaning")
    lines.append("")
    lines.append(f"Rows out: {_fmt(cleaning['rows_out'])}. "
                 f"Rows removed by cleaning rules: "
                 f"{_fmt(cleaning['rows_removed'])}.")
    if cleaning["removals"]:
        for rule, count in cleaning["removals"].items():
            lines.append(f"  {rule}: {_fmt(count)}")
    lines.append("")

    lines.append("## Facility reconciliation")
    lines.append("")
    lines.append(f"Facilities reporting in {report['years_covered'][1]}: "
                 f"{_fmt(recon['facilities_in_release_year'])}.")
    lines.append(f"New facilities this release: "
                 f"{_fmt(len(recon['new_facilities']))}.")
    if recon["new_facilities"]:
        for f in recon["new_facilities"][:10]:
            lines.append(f"  {f['facility_id']}: {f['facility_name']} "
                         f"({f['province']})")
        if len(recon["new_facilities"]) > 10:
            lines.append(f"  ... and "
                         f"{_fmt(len(recon['new_facilities']) - 10)} more "
                         f"(full list in the JSON report).")
    lines.append(f"Facilities that stopped reporting: "
                 f"{_fmt(len(recon['retired_candidates']))}.")
    if recon["retired_candidates"]:
        for f in recon["retired_candidates"][:10]:
            lines.append(f"  {f['facility_id']}: {f['facility_name']} "
                         f"({f['province']}), last seen "
                         f"{f['last_year']}")
        if len(recon["retired_candidates"]) > 10:
            lines.append(f"  ... and "
                         f"{_fmt(len(recon['retired_candidates']) - 10)} "
                         f"more (full list in the JSON report).")
    lines.append("")

    lines.append("## Year over year outliers for review")
    lines.append("")
    lines.append("Facilities whose total emissions swung more than 50 "
                 "percent against their own prior year, with an absolute "
                 "swing above 10,000 tonnes. Flagged for "
                 "human review, never auto removed.")
    if outliers["flagged"]:
        lines.append(f"Flagged: {_fmt(outliers['flagged_count'])}. "
                     f"Top movers:")
        for o in outliers["flagged"][:10]:
            direction = "up" if o["pct_change"] > 0 else "down"
            lines.append(f"  {o['facility_id']} {o['facility_name']} "
                         f"({o['year']}): {direction} "
                         f"{abs(o['pct_change']):.0%}, "
                         f"{_fmt(round(o['prev_emissions']))} to "
                         f"{_fmt(round(o['curr_emissions']))} tonnes.")
        if outliers["flagged_count"] > 10:
            lines.append(f"  ... and "
                         f"{_fmt(outliers['flagged_count'] - 10)} more "
                         f"(full list in the JSON report).")
    else:
        lines.append("No outliers flagged.")
    lines.append("")

    if model:
        lines.append("## Model refresh")
        lines.append("")
        lines.append(f"Facilities above the 100kt screening cutoff: "
                     f"{_fmt(model['facilities_above_cutoff'])}.")
        lines.append(f"Panel tiers: "
                     f"{model['tiers'].get('unanimous', 0)} unanimous, "
                     f"{model['tiers'].get('majority', 0)} majority, "
                     f"{model['tiers'].get('contested', 0)} contested.")
        if model.get("tier_shift_note"):
            lines.append(model["tier_shift_note"])
        lines.append("")

    lines.append("## Files")
    lines.append("")
    for f in report["files"]:
        lines.append(f"  {f}")
    lines.append("")

    md_path.write_text("\n".join(lines))
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    return md_path, json_path
