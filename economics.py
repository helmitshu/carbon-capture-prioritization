"""Indicative CCS economics for the v3.1 product layer.

Pure functions: no Streamlit, no I/O. All figures are clearly labeled
estimates built from sourced public benchmarks, never site engineering.
Every assumption is labeled where it surfaces in the memo or the app.

Writing rules for user-facing copy: commas and periods only, plain
phrasing, no personal names.

Sources (checked October 2026): see assumptions.py, the single source
of truth. The per band detail:
- Capture cost bands: IEA 2025 reporting (pre-combustion hydrogen
  USD 38-52/t, post-combustion cement USD 68-95/t), post-combustion
  retrofit combined cycle USD 57-68/t, high purity streams
  (ethanol style, over 90% CO2) USD 15-35/t.
- Transport and storage adder: Global CCS Institute, pipeline
  USD 3-4/t plus storage USD 3.50-7.50/t.
- CCUS Investment Tax Credit: IEA policy tracker updated June 2026.
  Standard rates for dedicated geological storage, 2022 to 2035:
  60% direct air capture equipment, 50% other capture equipment,
  37.5% transportation storage and use equipment. Rates halve from
  2036 to 2040. The 2026 Spring Economic Update extends the credit
  to enhanced oil recovery at half rates (30%, 25%, 18.75%) for
  expenditures from April 28, 2026, in jurisdictions meeting the
  95% permanence rule.
- Carbon price: federal benchmark revised May 15, 2026
  (see assumptions.CARBON_PRICE_DECK).
"""
from __future__ import annotations

from assumptions import CARBON_PRICE_DECK, assumption_value, get_assumption

# Illustrative FX assumption, October 2026. Paid memos use the
# prevailing rate.
FX_USD_CAD = assumption_value("fx_usd_cad")

# Capture cost bands in USD per tonne of CO2, (low, base, high).
# These are levelized capture-only costs from public benchmarks.
_BAND_ORDER = ("high_purity", "hydrogen_precombustion",
               "power_postcombustion", "cement", "industry_default")
_CAPTURE_BANDS_USD = {
    band: tuple(assumption_value(f"capture_band_{band}"))
    for band in _BAND_ORDER
}

_BAND_SOURCES = {
    band: get_assumption(f"capture_band_{band}")["source"]
    for band in _BAND_ORDER
}

# Keyword map from facility sector names to cost bands. Sectors not
# listed fall back to industry_default.
_SECTOR_BANDS = dict(assumption_value("sector_band_map"))

# Transport plus storage adder, USD per tonne (pipeline 3-4 plus
# storage 3.50-7.50, Global CCS Institute).
TS_ADDER_USD = tuple(assumption_value("ts_adder_usd"))

_ITC_RATES = assumption_value("ccus_itc")
CCUS_ITC = {
    "capture_dac": _ITC_RATES["capture_dac"],
    "capture_other": _ITC_RATES["capture_other"],
    "transport_storage_use": _ITC_RATES["transport_storage_use"],
    "eor_factor": _ITC_RATES["eor_factor"],
    "halve_from_year": _ITC_RATES["halve_from_year"],
    "source": get_assumption("ccus_itc")["source"],
}

# Illustrative build assumptions, labeled wherever they surface.
CAPTURE_RATE = assumption_value("capture_rate")
CAPEX_PER_TPA_CAD = assumption_value("capex_per_tpa_cad")
CAPEX_SPLIT_CAPTURE = assumption_value("capex_split_capture")  # share of capex treated as capture equipment


def _s(value) -> str:
    return (str(value).replace("\u2014", ",").replace("\u2013", ",").strip())


def capture_cost_band(sector: str) -> dict:
    """Return the capture cost band for a sector, in CAD per tonne.

    Returns low, base, high, the band name, and a source note.
    Unknown sectors fall back to industry_default.
    """
    key = _SECTOR_BANDS.get(_s(sector), "industry_default")
    low, base, high = _CAPTURE_BANDS_USD[key]
    return {
        "band": key,
        "low_cad": round(low * FX_USD_CAD, 2),
        "base_cad": round(base * FX_USD_CAD, 2),
        "high_cad": round(high * FX_USD_CAD, 2),
        "low_usd": low,
        "base_usd": base,
        "high_usd": high,
        "source": _BAND_SOURCES[key],
    }


def carbon_liability_cad(emissions_tonnes: float, year: int) -> float:
    """Annual carbon liability in CAD at the federal benchmark price."""
    schedule = CARBON_PRICE_DECK["schedule"]
    price = schedule.get(year, schedule[max(schedule)])
    return float(emissions_tonnes or 0) * price


def itc_credit_cad(capex_capture_cad: float, capex_tsu_cad: float,
                   eor: bool = False, year: int = 2026) -> dict:
    """Refundable CCUS ITC on eligible capex, in CAD.

    Uses the standard dedicated storage rates. EOR halves them.
    All rates halve again from 2036.
    """
    factor = 1.0
    if eor:
        factor *= CCUS_ITC["eor_factor"]
    if year >= CCUS_ITC["halve_from_year"]:
        factor *= 0.5
    credit = (float(capex_capture_cad or 0) * CCUS_ITC["capture_other"]
              + float(capex_tsu_cad or 0) * CCUS_ITC["transport_storage_use"])
    credit *= factor
    return {
        "credit_cad": round(credit, 2),
        "capture_rate": CCUS_ITC["capture_other"] * factor,
        "tsu_rate": CCUS_ITC["transport_storage_use"] * factor,
        "eor": eor,
        "source": CCUS_ITC["source"],
    }


def facility_economics(avg_annual_emissions: float, sector: str,
                       co2_share: float = 1.0,
                       capture_rate: float = CAPTURE_RATE,
                       capex_per_tpa_cad: float = CAPEX_PER_TPA_CAD,
                       eor: bool = False, year: int = 2026) -> dict:
    """Indicative per-facility CCS economics, all figures in CAD.

    The carbon liability is the true regulatory figure: the federal
    price applies per tonne of CO2e, so it uses total emissions.
    Everything capture related scales by co2_share, because only the
    CO2 fraction can be captured, and the margin is measured against
    the abatable slice of the liability, not the full bill.
    Every input that is not measured is an illustrative assumption and
    is labeled as such in the returned assumption notes.
    """
    emissions = float(avg_annual_emissions or 0)
    co2_share = min(max(float(co2_share or 0), 0.0), 1.0)
    band = capture_cost_band(sector)
    captured = emissions * co2_share * capture_rate
    capex = captured * capex_per_tpa_cad
    capex_capture = capex * CAPEX_SPLIT_CAPTURE
    capex_tsu = capex * (1 - CAPEX_SPLIT_CAPTURE)
    itc = itc_credit_cad(capex_capture, capex_tsu, eor=eor, year=year)
    net_capex = capex - itc["credit_cad"]
    liability_2030 = carbon_liability_cad(emissions, 2030)
    abatable_2030 = liability_2030 * co2_share

    annual_cost = {}
    margin = {}
    for level in ("low", "base", "high"):
        cost = captured * band[f"{level}_cad"]
        annual_cost[level] = round(cost, 2)
        margin[level] = round(abatable_2030 - cost, 2)

    ts_low = round(TS_ADDER_USD[0] * FX_USD_CAD, 2)
    ts_high = round(TS_ADDER_USD[1] * FX_USD_CAD, 2)

    return {
        "emissions_tonnes": emissions,
        "co2_share": co2_share,
        "captured_tonnes": round(captured, 2),
        "band": band,
        "ts_adder_cad": (ts_low, ts_high),
        "capex_cad": round(capex, 2),
        "capex_capture_cad": round(capex_capture, 2),
        "capex_tsu_cad": round(capex_tsu, 2),
        "itc": itc,
        "net_capex_cad": round(net_capex, 2),
        "annual_capture_cost_cad": annual_cost,
        "liability_2030_cad": round(liability_2030, 2),
        "abatable_liability_2030_cad": round(abatable_2030, 2),
        "margin_vs_abatable_cad": margin,
        "assumptions": [
            f"Capture rate {capture_rate:.0%} of the CO2 fraction, illustrative.",
            (f"CO2 share {co2_share:.0%} of reported emissions. Only this "
             "fraction is treated as capturable."),
            (f"Build cost ${capex_per_tpa_cad:,.0f} per tonne per annum, "
             "illustrative."),
            (f"Capex split {CAPEX_SPLIT_CAPTURE:.0%} capture equipment, "
             f"{1 - CAPEX_SPLIT_CAPTURE:.0%} transport storage and use, "
             "illustrative."),
            (f"FX {FX_USD_CAD} CAD per USD, October 2026, illustrative."),
            "Capture cost bands are public benchmarks, not site engineering.",
            ("Margin is measured against the abatable slice of the carbon "
             "liability, not the full regulatory bill."),
        ],
    }


def econ_chart_data(avg_annual_emissions: float, sector: str,
                    co2_share: float = 1.0) -> dict:
    """Chart-ready numbers for the v3 economics visuals.

    Everything derives from facility_economics, so the charts can
    never disagree with the text figures. The emissions mix is in
    CO2e tonnes: non-CO2 gases are shown at their CO2-equivalent
    weight, which is also how the federal carbon price treats them.
    """
    econ = facility_economics(avg_annual_emissions, sector,
                             co2_share=co2_share)
    emissions = econ["emissions_tonnes"]
    share = econ["co2_share"]
    co2_co2e = emissions * share
    other_co2e = emissions - co2_co2e
    price_2030 = CARBON_PRICE_DECK["schedule"][2030]
    band = econ["band"]
    return {
        "econ": econ,
        "total_co2e": round(emissions, 2),
        "mix": [
            {"label": "CO2", "co2e": round(co2_co2e, 2), "share": share,
             "note": "Capturable fraction"},
            {"label": "Other GHGs", "co2e": round(other_co2e, 2),
             "share": 1 - share,
             "note": "Mostly methane, shown in CO2-equivalent tonnes"},
        ],
        "liability": {
            "total": econ["liability_2030_cad"],
            "abatable": econ["abatable_liability_2030_cad"],
            "locked_in": round(econ["liability_2030_cad"]
                               - econ["abatable_liability_2030_cad"], 2),
            "price_per_tonne": price_2030,
            "price_source": CARBON_PRICE_DECK["source"],
        },
        "cost_band": {
            "low": band["low_cad"],
            "base": band["base_cad"],
            "high": band["high_cad"],
            "carbon_price": float(price_2030),
            "band_source": band["source"],
        },
    }
