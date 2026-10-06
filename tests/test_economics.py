"""Tests for the v3.1 economics layer.

Pure functions only; no Streamlit runner is needed.
"""
import pytest

from economics import (CCUS_ITC, FX_USD_CAD, capture_cost_band,
                       carbon_liability_cad, econ_chart_data,
                       facility_economics, itc_credit_cad)
from memo import CARBON_PRICE_DECK, render_memo_html


def _facility(**over):
    d = {
        "facility_name": "Test Cement Plant",
        "city": "Exshaw",
        "province": "Alberta",
        "operator": "Test Corp",
        "sector": "Cement Manufacturing",
        "avg_annual_emissions": 1_200_000.0,
        "co2_share": 0.999,
        "years_reported": 20,
        "priority": "CCS Candidate",
        "model_prediction": "CCS Candidate",
        "data_through": "2023",
    }
    d.update(over)
    return d


def test_capture_band_cement_uses_iea_range():
    band = capture_cost_band("Cement Manufacturing")
    assert band["band"] == "cement"
    # IEA 2025 cement USD 68-95/t converted at the labeled FX rate.
    assert band["low_cad"] == pytest.approx(68 * FX_USD_CAD)
    assert band["high_cad"] == pytest.approx(95 * FX_USD_CAD)
    assert band["source"]


def test_capture_band_power_and_high_purity():
    power = capture_cost_band("Fossil-Fuel Electric Power Generation")
    assert power["band"] == "power_postcombustion"
    fert = capture_cost_band(
        "Chemical Fertilizer (except Potash) Manufacturing")
    assert fert["band"] == "high_purity"
    assert fert["low_cad"] < power["low_cad"]


def test_capture_band_unknown_sector_falls_back():
    band = capture_cost_band("Some Sector Nobody Listed")
    assert band["band"] == "industry_default"
    assert band["low_cad"] < band["base_cad"] < band["high_cad"]


def test_itc_standard_rates():
    out = itc_credit_cad(100.0, 100.0)
    assert out["credit_cad"] == pytest.approx(50.0 + 37.5)
    assert out["capture_rate"] == pytest.approx(CCUS_ITC["capture_other"])
    assert not out["eor"]


def test_itc_eor_halves_rates():
    std = itc_credit_cad(100.0, 100.0)["credit_cad"]
    eor = itc_credit_cad(100.0, 100.0, eor=True)["credit_cad"]
    assert eor == pytest.approx(std * 0.5)


def test_itc_halves_from_2036():
    now = itc_credit_cad(100.0, 100.0, year=2030)["credit_cad"]
    later = itc_credit_cad(100.0, 100.0, year=2036)["credit_cad"]
    assert later == pytest.approx(now * 0.5)


def test_liability_uses_revised_deck():
    assert carbon_liability_cad(1_000_000, 2030) == pytest.approx(
        1_000_000 * CARBON_PRICE_DECK["schedule"][2030])


def test_facility_economics_labels_every_assumption():
    econ = facility_economics(1_200_000.0, "Cement Manufacturing")
    assert econ["captured_tonnes"] == pytest.approx(1_200_000.0 * 0.90)
    assert econ["net_capex_cad"] == pytest.approx(
        econ["capex_cad"] - econ["itc"]["credit_cad"])
    assert econ["annual_capture_cost_cad"]["low"] < \
        econ["annual_capture_cost_cad"]["base"] < \
        econ["annual_capture_cost_cad"]["high"]
    joined = " ".join(econ["assumptions"]).lower()
    assert "illustrative" in joined
    assert "benchmark" in joined


def test_facility_economics_scales_by_co2_share():
    full = facility_economics(1_000_000.0, "Cement Manufacturing",
                             co2_share=1.0)
    part = facility_economics(1_000_000.0, "Cement Manufacturing",
                             co2_share=0.26)
    # The regulatory liability uses total CO2e in both cases.
    assert part["liability_2030_cad"] == pytest.approx(
        full["liability_2030_cad"])
    # Everything capture related scales by the CO2 share.
    assert part["captured_tonnes"] == pytest.approx(
        full["captured_tonnes"] * 0.26)
    assert part["abatable_liability_2030_cad"] == pytest.approx(
        full["liability_2030_cad"] * 0.26)
    assert part["capex_cad"] == pytest.approx(full["capex_cad"] * 0.26)
    # The margin is measured against the abatable slice, not the bill.
    assert part["margin_vs_abatable_cad"]["base"] == pytest.approx(
        part["abatable_liability_2030_cad"]
        - part["annual_capture_cost_cad"]["base"])
    joined = " ".join(part["assumptions"]).lower()
    assert "co2 share" in joined
    assert "abatable slice" in joined


def test_memo_v31_shows_abatable_liability():
    html = render_memo_html(_facility(co2_share=0.26), [], "", 0,
                            economics="v31")
    assert "Abatable slice of the liability" in html
    assert "CO2 share of emissions" in html


def test_memo_v31_has_sourced_economics():
    html = render_memo_html(_facility(), ["Reason one."], "unanimous", 4,
                            economics="v31")
    assert "Indicative economics envelope" in html
    assert "investment tax credit" in html
    assert "Illustrative economics envelope" not in html


def test_memo_legacy_unchanged_for_v2():
    html = render_memo_html(_facility(), ["Reason one."], "unanimous", 4)
    assert "Illustrative economics envelope" in html
    # The legacy economics table has no ITC row (the next steps section
    # may still mention the credit, which is correct in both versions).
    assert "Federal CCUS investment tax credit</td>" not in html


def test_economics_copy_has_no_em_dashes():
    econ = facility_economics(500_000.0, "Petroleum Refineries")
    for a in econ["assumptions"]:
        assert "\u2014" not in a and "\u2013" not in a
    html = render_memo_html(_facility(), [], "", 0, economics="v31")
    assert "\u2014" not in html and "\u2013" not in html


def test_chart_data_mix_sums_to_emissions():
    d = econ_chart_data(219341.0, "Animal (except Poultry) Slaughtering",
                        co2_share=0.26)
    assert sum(m["co2e"] for m in d["mix"]) == pytest.approx(
        d["total_co2e"])
    assert sum(m["share"] for m in d["mix"]) == pytest.approx(1.0)
    assert d["mix"][0]["label"] == "CO2"
    assert d["mix"][0]["share"] == pytest.approx(0.26)
    assert d["mix"][0]["co2e"] == pytest.approx(219341.0 * 0.26)


def test_chart_data_liability_split_adds_up():
    d = econ_chart_data(1_000_000.0, "Cement Manufacturing",
                        co2_share=0.92)
    liab = d["liability"]
    assert liab["abatable"] + liab["locked_in"] == pytest.approx(
        liab["total"])
    assert liab["abatable"] == pytest.approx(liab["total"] * 0.92)
    assert liab["price_per_tonne"] == CARBON_PRICE_DECK["schedule"][2030]
    assert liab["price_source"]


def test_chart_data_cost_band_ordered_against_carbon_price():
    d = econ_chart_data(500_000.0, "Petroleum Refineries")
    cb = d["cost_band"]
    assert cb["low"] < cb["base"] < cb["high"]
    assert cb["carbon_price"] == pytest.approx(115.0)
    assert cb["band_source"]


def test_chart_figures_build_without_error():
    import matplotlib.pyplot as plt
    from econ_charts import (cost_band_chart, emissions_mix_donut,
                            liability_split_bar)
    args = (219341.0, "Animal (except Poultry) Slaughtering", 0.26)
    for builder in (emissions_mix_donut, liability_split_bar,
                    cost_band_chart):
        fig = builder(*args)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    # Full CO2 share edge case: no zero-division, no empty figure.
    fig = emissions_mix_donut(1_000_000.0, "Cement Manufacturing", 1.0)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
