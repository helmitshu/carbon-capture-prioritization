"""Tests for the centralized assumptions registry.

Every assumption in the v3.1 economics layer lives in assumptions.py.
These tests pin the registry shape, the staleness logic, and prove the
refactor changed no computed number.
"""
import datetime

import pytest

import assumptions
from assumptions import (ASSUMPTIONS, CARBON_PRICE_DECK, assumption_status,
                         assumption_value, format_value, get_assumption,
                         overdue_assumptions)
from economics import (CCUS_ITC, FX_USD_CAD, capture_cost_band,
                       carbon_liability_cad, facility_economics,
                       itc_credit_cad)
from memo import CARBON_PRICE_DECK as MEMO_DECK

REQUIRED_FIELDS = {"key", "name", "value", "unit", "source", "source_date",
                   "last_verified", "review", "review_days"}


def test_every_assumption_has_required_fields():
    assert len(ASSUMPTIONS) >= 10
    keys = [a["key"] for a in ASSUMPTIONS]
    assert len(keys) == len(set(keys)), "assumption keys must be unique"
    for a in ASSUMPTIONS:
        missing = REQUIRED_FIELDS - set(a)
        assert not missing, f"{a.get('key')}: missing {missing}"
        assert a["name"] and a["unit"] and a["source"]
        assert a["review_days"] is None or isinstance(
            a["review_days"], int)
        datetime.date.fromisoformat(a["last_verified"])


def test_get_assumption_unknown_key_raises():
    with pytest.raises(KeyError):
        get_assumption("no_such_assumption")


def test_carbon_price_deck_has_single_source_of_truth():
    assert MEMO_DECK is CARBON_PRICE_DECK
    assert CARBON_PRICE_DECK["schedule"] == {
        2026: 95, 2027: 100, 2028: 100, 2029: 100, 2030: 115}
    assert CARBON_PRICE_DECK["source"] == \
        "Federal benchmark revised May 15, 2026"
    assert assumption_value("carbon_price_deck") == \
        CARBON_PRICE_DECK["schedule"]


def test_economics_public_values_unchanged():
    assert FX_USD_CAD == 1.43
    assert assumption_value("fx_usd_cad") == 1.43
    assert CCUS_ITC["capture_other"] == 0.50
    assert CCUS_ITC["transport_storage_use"] == 0.375
    assert CCUS_ITC["capture_dac"] == 0.60
    assert CCUS_ITC["eor_factor"] == 0.5
    assert CCUS_ITC["halve_from_year"] == 2036
    assert CCUS_ITC["source"]
    assert assumption_value("capture_rate") == 0.90
    assert assumption_value("capex_per_tpa_cad") == 800.0
    assert assumption_value("capex_split_capture") == 0.70
    assert tuple(assumption_value("ts_adder_usd")) == (7.0, 12.0)


def test_capture_bands_match_documented_ranges():
    expected = {
        "capture_band_high_purity": (15.0, 25.0, 35.0),
        "capture_band_hydrogen_precombustion": (38.0, 45.0, 52.0),
        "capture_band_power_postcombustion": (57.0, 62.0, 68.0),
        "capture_band_cement": (68.0, 82.0, 95.0),
        "capture_band_industry_default": (60.0, 90.0, 120.0),
    }
    for key, band in expected.items():
        assert tuple(assumption_value(key)) == band


def test_nothing_overdue_today():
    assert overdue_assumptions() == []


def test_yearly_items_flag_overdue_after_review_period():
    future = datetime.date(2028, 6, 1)
    overdue = {a["key"] for a in overdue_assumptions(future)}
    assert "fx_usd_cad" in overdue
    assert "carbon_price_deck" in overdue
    assert "capture_band_cement" in overdue
    # Event driven reviews never flag automatically.
    assert "ccus_itc" not in overdue
    for a in overdue_assumptions(future):
        assert "Due for review" in a["status_text"]
        assert "overdue" in a["status_text"]


def test_event_driven_assumption_shows_its_trigger():
    itc = [a for a in assumption_status()
           if a["key"] == "ccus_itc"][0]
    assert not itc["overdue"]
    assert "federal budget" in itc["status_text"].lower()


def test_status_boundary_day_is_not_overdue():
    verified = datetime.date.fromisoformat("2026-10-06")
    boundary = verified + datetime.timedelta(days=365)
    assert overdue_assumptions(boundary) == []
    assert overdue_assumptions(boundary + datetime.timedelta(days=1))


def test_snapshot_computed_outputs_unchanged():
    """Pin the full economics output for a reference facility.

    Captured before the assumptions refactor. Any drift means the
    refactor changed a number, which must never happen silently.
    """
    e = facility_economics(1200000.0, "Cement Manufacturing",
                           co2_share=0.999)
    assert e["captured_tonnes"] == pytest.approx(1078920.0)
    assert e["capex_cad"] == pytest.approx(863136000.0)
    assert e["capex_capture_cad"] == pytest.approx(604195200.0)
    assert e["capex_tsu_cad"] == pytest.approx(258940800.0)
    assert e["net_capex_cad"] == pytest.approx(463935600.0)
    assert e["liability_2030_cad"] == pytest.approx(138000000.0)
    assert e["abatable_liability_2030_cad"] == pytest.approx(137862000.0)
    assert e["itc"]["credit_cad"] == pytest.approx(399200400.0)
    assert e["annual_capture_cost_cad"]["low"] == \
        pytest.approx(104914180.8)
    assert e["annual_capture_cost_cad"]["base"] == \
        pytest.approx(126514159.2)
    assert e["annual_capture_cost_cad"]["high"] == \
        pytest.approx(146571282.0)
    assert e["margin_vs_abatable_cad"]["low"] == pytest.approx(32947819.2)
    assert e["margin_vs_abatable_cad"]["base"] == \
        pytest.approx(11347840.8)
    assert e["margin_vs_abatable_cad"]["high"] == pytest.approx(-8709282.0)
    band = capture_cost_band("Cement Manufacturing")
    assert band["low_cad"] == pytest.approx(97.24)
    assert band["base_cad"] == pytest.approx(117.26)
    assert band["high_cad"] == pytest.approx(135.85)
    assert e["ts_adder_cad"] == pytest.approx((10.01, 17.16))
    assert carbon_liability_cad(1000000, 2026) == pytest.approx(95000000.0)
    assert carbon_liability_cad(1000000, 2029) == pytest.approx(100000000.0)
    assert itc_credit_cad(100.0, 100.0, eor=True, year=2036)[
        "credit_cad"] == pytest.approx(21.88)


def test_assumption_copy_has_no_em_dashes():
    for a in ASSUMPTIONS:
        for field in ("name", "source", "review"):
            assert "\u2014" not in a[field] and "\u2013" not in a[field]
    for a in assumption_status():
        assert "\u2014" not in a["status_text"]
        assert "\u2014" not in format_value(a)
    assert isinstance(assumptions.ASSUMPTIONS, list)
