"""Tests for the BD Toolkit: bd_toolkit.py and scripts/check_watchlist.py.

Pure functions only; no Streamlit runner is needed.
"""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import bd_toolkit
from bd_toolkit import (canonical_city, haversine_km, load_city_coords,
                        load_hubs, normalize_company, outreach_brief,
                        owner_portfolios, read_alerts, read_watchlist,
                        readiness_score, scenario_row, why_this_account,
                        write_watchlist)
from check_watchlist import diff_snapshots, main as checker_main
from economics import carbon_liability_cad, facility_economics


def _facilities():
    return pd.DataFrame([
        {"facility_id": 1, "facility_name": "Alpha Plant",
         "province": "Alberta", "city": "CALGARY",
         "sector": "Cement Manufacturing", "company_trade": "Alpha Corp",
         "avg_annual_emissions": 2_000_000.0, "co2_share": 0.995,
         "verdict_tier": "unanimous"},
        {"facility_id": 2, "facility_name": "Beta Plant",
         "province": "Alberta", "city": "EDMONTON",
         "sector": "Cement Manufacturing", "company_trade": "Alpha Corp",
         "avg_annual_emissions": 1_000_000.0, "co2_share": 0.99,
         "verdict_tier": "majority"},
        {"facility_id": 3, "facility_name": "Gamma Plant",
         "province": "Ontario", "city": "HAMILTON",
         "sector": "Cement Manufacturing",
         "company_trade": "not applicable",
         "avg_annual_emissions": 500_000.0, "co2_share": 0.98,
         "verdict_tier": "contested"},
        {"facility_id": 4, "facility_name": "Delta Plant",
         "province": "Alberta", "city": "Unknown",
         "sector": "Cement Manufacturing", "company_trade": "",
         "avg_annual_emissions": 300_000.0, "co2_share": 0.97,
         "verdict_tier": "unanimous"},
    ])


def _hubs():
    return [{"name": "Quest CCS", "type": "ccs_site", "lat": 53.77,
             "lon": -113.08, "source": "test", "note": ""}]


def _cities():
    return {"CALGARY": {"lat": 51.0447, "lon": -113.0719, "note": "",
                        "source": "test"},
            "EDMONTON": {"lat": 53.5461, "lon": -113.4938, "note": "",
                         "source": "test"},
            "HAMILTON": {"lat": 43.26, "lon": -79.87, "note": "",
                         "source": "test"}}


# ---------------------------------------------------------------------------
# Owner portfolios
# ---------------------------------------------------------------------------

def test_company_blank_mapping():
    assert normalize_company("not applicable") == "Independent operator"
    assert normalize_company("") == "Independent operator"
    assert normalize_company("Unknown") == "Independent operator"
    assert normalize_company(None) == "Independent operator"
    assert normalize_company("  Alpha Corp ") == "Alpha Corp"


def test_owner_aggregation_math():
    owners = owner_portfolios(_facilities())
    alpha = owners[owners["company"] == "Alpha Corp"].iloc[0]
    assert alpha["n_facilities"] == 2
    assert alpha["total_emissions_t"] == pytest.approx(3_000_000.0)
    expected_liability = (carbon_liability_cad(2_000_000.0, 2030)
                          + carbon_liability_cad(1_000_000.0, 2030))
    assert alpha["total_liability_2030_cad"] == pytest.approx(
        expected_liability)
    assert alpha["tier_unanimous"] == 1
    assert alpha["tier_majority"] == 1
    assert alpha["tier_contested"] == 0
    # Blank company names roll up together.
    indie = owners[owners["company"] == "Independent operator"].iloc[0]
    assert indie["n_facilities"] == 2
    # Sorted by liability, highest first.
    assert owners.iloc[0]["company"] == "Alpha Corp"


def test_why_this_account_bullets():
    owners = owner_portfolios(_facilities())
    alpha = owners[owners["company"] == "Alpha Corp"].iloc[0]
    bullets = why_this_account(alpha)
    assert any("1 of 2 facilities carry a unanimous verdict" in b
               for b in bullets)
    assert any("One conversation covers" in b for b in bullets)
    assert any("ITC" in b for b in bullets)


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

def test_haversine_calgary_edmonton():
    d = haversine_km(51.0447, -113.0719, 53.5461, -113.4938)
    assert 250 < d < 310


def test_readiness_known_city_full_score():
    s = readiness_score(2_000_000.0, 0.995, "CALGARY", _hubs(), _cities())
    assert s["denominator"] == 100
    assert s["scale_pts"] == pytest.approx(8.0)
    assert s["purity_pts"] == pytest.approx(29.85, abs=0.06)
    assert s["prox_pts"] is not None
    assert s["nearest_hub"] == "Quest CCS"
    assert s["prox_km"] == pytest.approx(303.0, abs=2.0)
    assert s["prox_pts"] == 0.0  # just over the 300 km band
    assert s["total"] == pytest.approx(8.0 + 29.85, abs=0.06)


def test_readiness_near_hub_scores_full_proximity():
    s = readiness_score(2_000_000.0, 0.995, "EDMONTON", _hubs(), _cities())
    assert s["prox_km"] == pytest.approx(36.9, abs=2.0)
    assert s["prox_pts"] == 30.0


def test_readiness_unknown_city_scores_out_of_70():
    s = readiness_score(2_000_000.0, 0.995, "Nowhereville", _hubs(),
                        _cities())
    assert s["denominator"] == 70
    assert s["prox_pts"] is None
    assert s["prox_km"] is None
    assert s["nearest_hub"] is None
    assert "unknown" in s["note"]
    assert s["total"] == pytest.approx(8.0 + 29.85, abs=0.06)


def test_readiness_city_alias_never_guessed():
    assert canonical_city("FT MCMURRAY") == "FORT MCMURRAY"
    assert canonical_city("Unknown") is None
    assert canonical_city("") is None
    assert canonical_city(None) is None


def test_hub_and_city_files_load():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hubs = load_hubs(os.path.join(root, "data", "hubs.csv"))
    cities = load_city_coords(os.path.join(root, "data", "city_coords.csv"))
    assert len(hubs) >= 3
    assert all(h["source"] for h in hubs)
    assert "CALGARY" in cities
    assert "FORT MCMURRAY" in cities


# ---------------------------------------------------------------------------
# Scenario lab
# ---------------------------------------------------------------------------

def test_scenario_base_matches_economics():
    s = scenario_row(2_000_000.0, "Cement Manufacturing", 0.995,
                     price=115, itc_rate_pct=50, cost_mult=1.0)
    econ = facility_economics(2_000_000.0, "Cement Manufacturing",
                             co2_share=0.995)
    assert s["liability_cad"] == pytest.approx(
        econ["liability_2030_cad"])
    assert s["margin_cad"] == pytest.approx(
        econ["margin_vs_abatable_cad"]["base"])
    assert s["itc_credit_cad"] == pytest.approx(
        econ["itc"]["credit_cad"])


def test_scenario_bull_beats_base_beats_bear():
    base = scenario_row(2_000_000.0, "Cement Manufacturing", 0.995,
                        115, 50, 1.0)
    bull = scenario_row(2_000_000.0, "Cement Manufacturing", 0.995,
                        200, 60, 0.7)
    bear = scenario_row(2_000_000.0, "Cement Manufacturing", 0.995,
                        50, 0, 1.5)
    assert bull["margin_cad"] > base["margin_cad"] > bear["margin_cad"]
    assert bull["liability_cad"] > base["liability_cad"]
    assert bull["itc_credit_cad"] > base["itc_credit_cad"]
    assert bear["itc_credit_cad"] == pytest.approx(0.0)


def test_scenario_verdict_thresholds():
    econ = scenario_row(2_000_000.0, "Cement Manufacturing", 0.995,
                        200, 60, 0.7)
    assert econ["verdict"] == "Economic"
    poor = scenario_row(2_000_000.0, "Cement Manufacturing", 0.995,
                        50, 0, 1.5)
    assert poor["verdict"] == "Underwater"
    assert "method" in poor and poor["method"]


# ---------------------------------------------------------------------------
# Outreach kit
# ---------------------------------------------------------------------------

def test_outreach_email_purity_conditional():
    high = {"facility_name": "P", "company_trade": "C",
            "province": "Alberta", "sector": "Cement Manufacturing",
            "avg_annual_emissions": 1_000_000.0, "co2_share": 0.995,
            "verdict_tier": "unanimous"}
    low = dict(high, co2_share=0.90)
    rd = {"total": 80.0, "denominator": 100, "scale_pts": 16.0,
          "purity_pts": 27.0}
    assert "well suited to capture" in outreach_brief(high, rd)["email_body"]
    assert "well suited to capture" not in outreach_brief(low, rd)[
        "email_body"]
    brief = outreach_brief(high, rd)
    assert len(brief["hooks"]) == 3
    assert len(brief["talking_points"]) == 3
    assert brief["email_subject"]


# ---------------------------------------------------------------------------
# Watchlist persistence
# ---------------------------------------------------------------------------

def test_watchlist_round_trip(tmp_path):
    path = str(tmp_path / "watchlist.json")
    assert read_watchlist(path) == set()
    write_watchlist({"G10003", "G10001", "G10002"}, path)
    assert read_watchlist(path) == {"G10001", "G10002", "G10003"}
    write_watchlist(set(), path)
    assert read_watchlist(path) == set()


def test_watchlist_corrupt_file_reads_empty(tmp_path):
    path = str(tmp_path / "watchlist.json")
    with open(path, "w") as fh:
        fh.write("not json{{{")
    assert read_watchlist(path) == set()


def test_alerts_missing_reads_empty(tmp_path):
    assert read_alerts(str(tmp_path / "alerts.json")) == []


# ---------------------------------------------------------------------------
# Checker script
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["facility_id", "facility_name",
                                           "avg_annual_emissions",
                                           "priority"])
        w.writeheader()
        w.writerows(rows)


def test_checker_first_run_initializes(tmp_path):
    data = str(tmp_path / "data.csv")
    snap = str(tmp_path / "snap.csv")
    alerts = str(tmp_path / "alerts.json")
    _write_csv(data, [{"facility_id": "1", "facility_name": "A",
                       "avg_annual_emissions": "1000000",
                       "priority": "CCS Candidate"}])
    code = checker_main(["--data", data, "--snapshot", snap,
                         "--alerts", alerts])
    assert code == 0
    assert os.path.exists(snap)
    assert not os.path.exists(alerts)


def test_checker_diff_logic(tmp_path):
    data = str(tmp_path / "data.csv")
    snap = str(tmp_path / "snap.csv")
    alerts = str(tmp_path / "alerts.json")
    _write_csv(data, [
        {"facility_id": "1", "facility_name": "A",
         "avg_annual_emissions": "1000000", "priority": "CCS Candidate"},
        {"facility_id": "2", "facility_name": "B",
         "avg_annual_emissions": "2000000", "priority": "CCS Candidate"},
    ])
    assert checker_main(["--data", data, "--snapshot", snap,
                         "--alerts", alerts]) == 0
    # Second run: one new facility, one removed, one jumped, one priority
    # change.
    _write_csv(data, [
        {"facility_id": "1", "facility_name": "A",
         "avg_annual_emissions": "1100000", "priority": "CCS Candidate"},
        {"facility_id": "3", "facility_name": "C",
         "avg_annual_emissions": "500000",
         "priority": "Potential CU Candidate"},
    ])
    assert checker_main(["--data", data, "--snapshot", snap,
                         "--alerts", alerts]) == 0
    with open(alerts) as fh:
        found = json.load(fh)
    texts = [a["text"] for a in found]
    assert any("New facility above threshold: C." in t for t in texts)
    assert any("no longer above threshold: B." in t for t in texts)
    assert any("Emissions rose 10.0 percent for A." in t for t in texts)
    assert all(a["severity"] in {"high", "med", "low"} for a in found)
    # Third run with no changes: no new alerts.
    before = len(found)
    assert checker_main(["--data", data, "--snapshot", snap,
                         "--alerts", alerts]) == 0
    with open(alerts) as fh:
        assert len(json.load(fh)) == before


def test_diff_snapshots_pure():
    prev = {"1": {"facility_id": "1", "facility_name": "A",
                  "avg_annual_emissions": "1000000",
                  "priority": "CCS Candidate"}}
    cur = {"1": {"facility_id": "1", "facility_name": "A",
                 "avg_annual_emissions": "900000",
                 "priority": "Potential CU Candidate"}}
    alerts = diff_snapshots(cur, prev, 5.0)
    texts = [a["text"] for a in alerts]
    assert any("Priority changed" in t for t in texts)
    assert any("Emissions fell 10.0 percent" in t for t in texts)
