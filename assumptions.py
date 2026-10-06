"""Single source of truth for every assumption in the v3.1 economics layer.

Before this module, the same numbers lived in three places: economics.py,
memo.py, and the app copy. When a policy moved, nothing flagged the rot.
Now every assumption lives here once, with its source, its dates, and its
review period. The app shows them in an Assumptions panel, and a monthly
cron flags anything past its review date.

Each entry carries: key, name, value, unit, source (with the publication
or report name), source_date, last_verified, review (plain language), and
review_days (an integer when the review is calendar driven, None when it
is event driven, for example a federal budget).

Writing rules for user-facing copy: commas and periods only, plain
phrasing, no personal names.
"""
from __future__ import annotations

import datetime as _dt

ASSUMPTIONS = [
    {
        "key": "fx_usd_cad",
        "name": "USD to CAD exchange rate",
        "value": 1.43,
        "unit": "CAD per USD",
        "source": ("Illustrative assumption set by the project, October "
                   "2026. Paid memos use the prevailing rate."),
        "source_date": "October 2026",
        "last_verified": "2026-10-06",
        "review": "yearly",
        "review_days": 365,
    },
    {
        "key": "capture_band_high_purity",
        "name": "Capture cost band, high purity streams",
        "value": (15.0, 25.0, 35.0),
        "unit": "USD per tonne of CO2, low, base and high",
        "source": "High purity streams over 90% CO2, USD 15-35 per tonne.",
        "source_date": "2025, IEA reporting year",
        "last_verified": "2026-10-06",
        "review": "yearly, when the IEA publishes fresh cost data",
        "review_days": 365,
    },
    {
        "key": "capture_band_hydrogen_precombustion",
        "name": "Capture cost band, pre-combustion hydrogen",
        "value": (38.0, 45.0, 52.0),
        "unit": "USD per tonne of CO2, low, base and high",
        "source": "IEA 2025, pre-combustion hydrogen, USD 38-52 per tonne.",
        "source_date": "2025, IEA reporting year",
        "last_verified": "2026-10-06",
        "review": "yearly, when the IEA publishes fresh cost data",
        "review_days": 365,
    },
    {
        "key": "capture_band_power_postcombustion",
        "name": "Capture cost band, post-combustion power",
        "value": (57.0, 62.0, 68.0),
        "unit": "USD per tonne of CO2, low, base and high",
        "source": ("Post-combustion retrofit combined cycle, "
                   "USD 57-68 per tonne."),
        "source_date": "2025, IEA reporting year",
        "last_verified": "2026-10-06",
        "review": "yearly, when the IEA publishes fresh cost data",
        "review_days": 365,
    },
    {
        "key": "capture_band_cement",
        "name": "Capture cost band, cement",
        "value": (68.0, 82.0, 95.0),
        "unit": "USD per tonne of CO2, low, base and high",
        "source": "IEA 2025, post-combustion cement, USD 68-95 per tonne.",
        "source_date": "2025, IEA reporting year",
        "last_verified": "2026-10-06",
        "review": "yearly, when the IEA publishes fresh cost data",
        "review_days": 365,
    },
    {
        "key": "capture_band_industry_default",
        "name": "Capture cost band, industry default",
        "value": (60.0, 90.0, 120.0),
        "unit": "USD per tonne of CO2, low, base and high",
        "source": ("Illustrative industry band, USD 60-120 per tonne."),
        "source_date": "October 2026",
        "last_verified": "2026-10-06",
        "review": "yearly",
        "review_days": 365,
    },
    {
        "key": "sector_band_map",
        "name": "Sector to cost band mapping",
        "value": {
            "Cement Manufacturing": "cement",
            "Lime Manufacturing": "cement",
            "Glass Manufacturing": "cement",
            "Fossil-Fuel Electric Power Generation": "power_postcombustion",
            "Other Electric Power Generation": "power_postcombustion",
            "Steam and Air-Conditioning Supply": "power_postcombustion",
            "Chemical Fertilizer (except Potash) Manufacturing": "high_purity",
            "Industrial Gas Manufacturing": "high_purity",
        },
        "unit": "GHGRP sector name to band name",
        "source": ("Project mapping of GHGRP sector names to the cost "
                   "bands above. Sectors not listed fall back to the "
                   "industry default band."),
        "source_date": "October 2026",
        "last_verified": "2026-10-06",
        "review": "yearly, alongside the cost bands",
        "review_days": 365,
    },
    {
        "key": "ts_adder_usd",
        "name": "Transport plus storage adder",
        "value": (7.0, 12.0),
        "unit": "USD per tonne of CO2, low and high",
        "source": ("Global CCS Institute: pipeline USD 3 to 4 per tonne "
                   "plus storage USD 3.50 to 7.50 per tonne. Benchmarks "
                   "checked October 2026."),
        "source_date": "October 2026",
        "last_verified": "2026-10-06",
        "review": "yearly",
        "review_days": 365,
    },
    {
        "key": "carbon_price_deck",
        "name": "Federal carbon price schedule",
        "value": {2026: 95, 2027: 100, 2028: 100, 2029: 100, 2030: 115},
        "unit": "CAD per tonne of CO2e",
        "source": ("Federal carbon price benchmark, revised May 15, 2026. "
                   "The revision ended the $170 by 2030 path."),
        "source_date": "2026-05-15",
        "last_verified": "2026-10-06",
        "review": "yearly, or on any federal carbon price announcement",
        "review_days": 365,
    },
    {
        "key": "ccus_itc",
        "name": "Federal CCUS investment tax credit rates",
        "value": {
            "capture_dac": 0.60,
            "capture_other": 0.50,
            "transport_storage_use": 0.375,
            "eor_factor": 0.5,
            "halve_from_year": 2036,
        },
        "unit": "fraction of eligible capital cost",
        "source": ("IEA policy tracker, June 2026. Standard rates for "
                   "dedicated geological storage, 2022 to 2035. EOR at half "
                   "rates from April 28, 2026 per the 2026 Spring Economic "
                   "Update. All rates halve from 2036 to 2040."),
        "source_date": "June 2026",
        "last_verified": "2026-10-06",
        "review": "on each federal budget",
        "review_days": None,
    },
    {
        "key": "capture_rate",
        "name": "Illustrative capture rate",
        "value": 0.90,
        "unit": "fraction of the CO2 stream captured",
        "source": ("Illustrative build assumption, labeled wherever it "
                   "surfaces. Paid memos use the client commercial case."),
        "source_date": "October 2026",
        "last_verified": "2026-10-06",
        "review": "yearly",
        "review_days": 365,
    },
    {
        "key": "capex_per_tpa_cad",
        "name": "Illustrative build cost",
        "value": 800.0,
        "unit": "CAD per tonne per annum of capture capacity",
        "source": ("Illustrative build assumption, labeled wherever it "
                   "surfaces. Paid memos use the client commercial case."),
        "source_date": "October 2026",
        "last_verified": "2026-10-06",
        "review": "yearly",
        "review_days": 365,
    },
    {
        "key": "capex_split_capture",
        "name": "Capex split, capture equipment share",
        "value": 0.70,
        "unit": "share of capex treated as capture equipment",
        "source": ("Illustrative build assumption, labeled wherever it "
                   "surfaces. The rest is transport, storage and use."),
        "source_date": "October 2026",
        "last_verified": "2026-10-06",
        "review": "yearly",
        "review_days": 365,
    },
]

def _by_key(key: str) -> dict:
    for entry in ASSUMPTIONS:
        if entry["key"] == key:
            return entry
    raise KeyError(f"Unknown assumption: {key}")


def get_assumption(key: str) -> dict:
    """Return the assumption entry for key. Raises KeyError if missing."""
    return _by_key(key)


def assumption_value(key: str):
    """Return just the value of the assumption for key."""
    return _by_key(key)["value"]


# The carbon price deck in the shape the memo and economics modules
# have always used. Import it from here, never redefine it.
CARBON_PRICE_DECK = {
    "currency": "CAD",
    "unit": "per tonne CO2e",
    "schedule": dict(_by_key("carbon_price_deck")["value"]),
    "source": "Federal benchmark revised May 15, 2026",
}


def format_value(entry: dict) -> str:
    """Plain language rendering of an assumption value."""
    value = entry["value"]
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            label = str(k).replace("_", " ")
            parts.append(f"{label} {v}")
        return ", ".join(parts)
    if isinstance(value, (tuple, list)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float) and 0 < value < 1:
        return f"{value:.0%}"
    return str(value)


def _parse(date_str: str) -> _dt.date:
    return _dt.date.fromisoformat(date_str)


def assumption_status(as_of: _dt.date | None = None) -> list:
    """Status of every assumption as of a date (default today).

    Each item carries the entry fields plus days_since_verified,
    overdue (True when past its calendar review period), and a plain
    language status_text. Event driven assumptions never flag as
    overdue automatically; their review trigger is shown instead.
    """
    today = as_of or _dt.date.today()
    out = []
    for entry in ASSUMPTIONS:
        verified = _parse(entry["last_verified"])
        days = (today - verified).days
        review_days = entry["review_days"]
        overdue = review_days is not None and days > review_days
        if overdue:
            late_by = days - review_days
            status_text = (
                f"Due for review. Last verified {entry['last_verified']}, "
                f"review {entry['review']}, {late_by} days overdue.")
        elif review_days is None:
            status_text = (
                f"Review {entry['review']}. "
                f"Last verified {entry['last_verified']}.")
        else:
            status_text = (
                f"Current. Last verified {entry['last_verified']}, "
                f"review {entry['review']}.")
        out.append({
            **entry,
            "value_text": format_value(entry),
            "days_since_verified": days,
            "overdue": overdue,
            "status_text": status_text,
        })
    return out


def overdue_assumptions(as_of: _dt.date | None = None) -> list:
    """Only the assumptions past their review period."""
    return [a for a in assumption_status(as_of) if a["overdue"]]
