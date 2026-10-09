"""Tests for the v3 MVP UI wiring: memo generation and v3 data contracts.

Pure functions only; no Streamlit runner is needed.
"""
import builtins

import joblib
import numpy as np
import pandas as pd
import pytest

from features import encoder_categories, encoder_values
from memo import CARBON_PRICE_DECK, render_memo_html, render_memo_pdf


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


def test_memo_contains_facility_tier_reasons_and_deck():
    html = render_memo_html(
        _facility(), ["Reason one.", "Reason two."], "majority", 3)
    assert "Test Cement Plant" in html
    assert "majority" in html
    assert "Reason one." in html
    assert "Reason two." in html
    assert "$115" in html  # revised 2030 deck figure
    assert "May 15, 2026" in html
    assert "STRONG CANDIDATE" in html


def test_memo_cu_verdict_grade():
    html = render_memo_html(
        _facility(priority="Potential CU Candidate",
                  model_prediction="Potential CU Candidate"),
        [], "contested", 2)
    assert "UTILIZATION CANDIDATE" in html


def test_memo_has_no_em_dashes():
    html = render_memo_html(
        _facility(), ["Reason one."], "unanimous", 4)
    for bad in ["\u2014", "\u2013", "&mdash;", "&ndash;"]:
        assert bad not in html, f"found {bad!r} in memo HTML"


def test_memo_sanitizes_em_dashes_in_inputs():
    html = render_memo_html(
        _facility(facility_name="Plant \u2014 North"),
        ["Rate \u2013 high."], "unanimous", 4)
    for bad in ["\u2014", "\u2013", "&mdash;", "&ndash;"]:
        assert bad not in html


@pytest.mark.parametrize("tier,cls", [
    ("unanimous", "tier-unanimous"),
    ("majority", "tier-majority"),
    ("contested", "tier-contested"),
])
def test_tier_badge_css_class_mapping(tier, cls):
    html = render_memo_html(_facility(), [], tier, 4)
    assert cls in html


def test_sample_banner_present_without_client():
    html = render_memo_html(_facility(), [], "unanimous", 4, client_name="")
    assert "SAMPLE DOCUMENT" in html


def test_sample_banner_absent_with_client():
    html = render_memo_html(
        _facility(), [], "unanimous", 4, client_name="Acme Carbon")
    assert "SAMPLE DOCUMENT" not in html
    assert "Acme Carbon" in html


def test_memo_handles_empty_reasons_and_tier():
    html = render_memo_html(_facility(), [], "", 0)
    assert "Test Cement Plant" in html
    assert 'class="tier-badge' not in html


def test_carbon_price_deck_source_note():
    assert (CARBON_PRICE_DECK["source"]
            == "Federal benchmark revised May 15, 2026")
    assert CARBON_PRICE_DECK["schedule"][2030] == 115


def test_v3_facilities_columns():
    fac = pd.read_csv("model/facilities_v3.csv")
    for col in ["facility_id", "facility_name", "province", "city",
                "company_trade", "avg_annual_emissions", "co2_share",
                "years_reported", "priority", "panel_ccs_votes",
                "panel_majority", "verdict_tier",
                "reason_1", "reason_2", "reason_3",
                "log_emissions", "naics_sector_encoded"]:
        assert col in fac.columns, f"missing v3 column: {col}"
    assert set(fac["verdict_tier"].unique()) <= {
        "unanimous", "majority", "contested"}
    assert fac["panel_ccs_votes"].between(0, 4).all()
    assert len(fac) == 457  # 2024 refresh: 457 above the 100kt cutoff


def test_encoder_helpers_match_legacy_v2_walkthrough():
    """The app.py walkthrough swap must render v2 output identically."""
    le = joblib.load("model/model.pkl")["encoder"]  # raw LabelEncoder
    cats = np.array(encoder_categories(le))
    vals = encoder_values(le)
    for thr in [0.5, 5.0, 12.0, 24.0]:
        for go_left in (True, False):
            if go_left:
                legacy = le.classes_[le.transform(le.classes_) <= thr]
                new = cats[vals <= thr]
            else:
                legacy = le.classes_[le.transform(le.classes_) > thr]
                new = cats[vals > thr]
            assert list(new) == [str(c) for c in legacy]


def test_pdf_raises_import_error_without_weasyprint(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "weasyprint" or name.startswith("weasyprint."):
            raise ImportError("mocked missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="weasyprint not installed"):
        render_memo_pdf("<html></html>")


def test_pdf_builds_when_weasyprint_available():
    pytest.importorskip("weasyprint")
    pdf = render_memo_pdf(
        render_memo_html(_facility(), ["Reason one."], "unanimous", 4))
    assert pdf[:4] == b"%PDF"


def test_screening_tab_detail_calls_memo_section():
    """Regression: the Screening tab detail view must wire the memo button.

    The memo section used to exist only in facility_profile (Facilities tab),
    so the primary detail view had no 'Generate screening memo' button.
    Static check on app.py source: _memo_section must be invoked inside the
    Screening section (between 'if nav == "Screening":' and
    'if nav == "Facilities":').
    """
    import pathlib
    src = pathlib.Path(__file__).resolve().parent.parent.joinpath(
        "app.py").read_text()
    start = src.index('if nav == "Screening":')
    end = src.index('if nav == "Facilities":')
    screening_block = src[start:end]
    assert "_memo_section(" in screening_block, (
        "Screening tab detail view does not call _memo_section")


def test_screening_tab_detail_shows_tier_badge():
    """The Screening tab detail view must show the panel tier badge too."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parent.parent.joinpath(
        "app.py").read_text()
    start = src.index('if nav == "Screening":')
    end = src.index('if nav == "Facilities":')
    screening_block = src[start:end]
    assert "tier_badge(row)" in screening_block, (
        "Screening tab detail view does not show the tier badge")
